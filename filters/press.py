"""Search a press-release corpus (DOJ/OIG) for names, so each lead can be checked for prior public disclosure.

WHIP-style rewards need ORIGINAL information: a provider or person already named in a DOJ/OIG release is a
known case. Matching is by normalised phrase with an inverted index; a hit is "corroborated" when the same
release also mentions the provider's state or city. Name-only hits are notes, not exclusions.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

from clean.normalize import norm_name, person_key, clean_str

_WORD = re.compile(r"[a-z0-9]+")
GENERIC = {"home", "health", "care", "hospice", "services", "service", "medical", "healthcare", "agency", "group",
           "nursing", "center", "centers", "clinic", "the", "of", "and", "inc", "llc", "corp", "co", "at", "for",
           "palliative", "community", "family", "senior", "seniors", "visiting", "regional", "county", "first",
           "american", "united", "national", "premier", "quality", "advanced", "new", "life", "angels", "comfort"}
STATES = {"AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
          "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
          "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
          "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
          "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
          "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
          "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
          "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas",
          "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
          "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia"}


def _toks(s: str) -> list[str]:
    return _WORD.findall(s.lower())


class PressIndex:
    def __init__(self, releases: list[dict]):
        self.docs = releases
        self.norm: list[str] = []
        self.inv: dict[str, set[int]] = defaultdict(set)
        for i, r in enumerate(releases):
            text = " ".join(_toks(r["title"] + " " + r["text"]))
            self.norm.append(" " + text + " ")
            for t in set(text.split()):
                if len(t) >= 3:
                    self.inv[t].add(i)

    @classmethod
    def load(cls, path: str | Path) -> "PressIndex":
        recs = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
        return cls(recs)

    def _candidates(self, toks: list[str]) -> set[int] | None:
        sets = [self.inv.get(t, set()) for t in toks if len(t) >= 3]
        if not sets:
            return None
        sets.sort(key=len)
        out = set(sets[0])
        for s in sets[1:]:
            out &= s
            if not out:
                break
        return out

    def _hits(self, pattern: re.Pattern, cand: set[int] | None, state: str, city: str, limit: int = 5) -> list[dict]:
        out = []
        for i in sorted(cand or []):
            m = pattern.search(self.norm[i])
            if not m:
                continue
            doc = self.docs[i]
            st = STATES.get(clean_str(state).upper(), "")
            state_match = bool(st and f" {st} " in self.norm[i])
            city_match = bool(city and f" {' '.join(_toks(city))} " in self.norm[i])
            s = max(0, m.start() - 80)
            out.append({"title": doc["title"], "url": doc["url"], "date": doc["date"],
                        "corroborated": state_match or city_match, "state_match": state_match, "city_match": city_match,
                        "snippet": self.norm[i][s:m.end() + 80].strip()})
            if len(out) >= limit:
                break
        return out

    def find_person(self, name: str, state: str = "", city: str = "") -> list[dict]:
        """First + last name adjacent, allowing one middle name/initial between them."""
        key = person_key(name)
        if not key or key.startswith("ORG:"):
            return []
        first, last = key.split()[0].lower(), key.split()[-1].lower()
        if len(first) < 2 or len(last) < 3:
            return []
        pat = re.compile(rf" {re.escape(first)}(?: [a-z])?(?: [a-z]+)? {re.escape(last)} ")
        # allow either "first last" or "first X last"; verify tokens adjacent within 2 words
        pat = re.compile(rf" {re.escape(first)}(?: [a-z]+)?(?: [a-z])? {re.escape(last)} ")
        return self._hits(pat, self._candidates([first, last]), state, city)

    def find_org(self, name: str, state: str = "", city: str = "") -> list[dict]:
        """Organisation name (suffixes stripped). Needs a distinctive (non-generic) token to avoid noise."""
        n = " ".join(_toks(norm_name(name)))
        toks = n.split()
        if len(toks) < 2 or not (set(toks) - GENERIC):
            return []
        pat = re.compile(rf" {re.escape(n)} ")
        return self._hits(pat, self._candidates(toks), state, city)


def load_latest(cfg: dict) -> "PressIndex | None":
    files = sorted((cfg["paths"]["raw"] / "press_doj").glob("press_doj_*.jsonl")) if (cfg["paths"]["raw"] / "press_doj").exists() else []
    return PressIndex.load(files[-1]) if files else None


def fmt(hits: list[dict], n: int = 2) -> str:
    return " | ".join(f"{h['date']} {h['title']} <{h['url']}>" + ("" if h["corroborated"] else " [name only]") for h in hits[:n])


def apply_press(df, idx: "PressIndex", cfg: dict):
    """Adds press_provider_refs / press_owner_refs / press_note; marks a provider known_case when its name
    appears in a release that also mentions its CITY; owners named in a release (same state) -> linked_known_case."""
    import pandas as pd
    df = df.copy()
    for c, v in (("press_provider_refs", ""), ("press_owner_refs", ""), ("press_note", "")):
        df[c] = v
    top_n = cfg["press"]["check_top_n"]
    order = df.sort_values("score", ascending=False).index[:top_n] if "score" in df else df.index
    check_owners = set(order) | set(df.index[df.get("n_flags", 0) >= 3]) if "n_flags" in df else set(order)
    for i, r in df.iterrows():
        ph = idx.find_org(r["name"], r["state"], clean_str(r.get("city", "")))
        if ph:
            df.at[i, "press_provider_refs"] = fmt(ph)
            if any(h["city_match"] for h in ph):
                df.at[i, "known_case"] = True
                df.at[i, "known_case_reason"] = (clean_str(r.get("known_case_reason", "")) + " | " if clean_str(r.get("known_case_reason", "")) else "") + \
                    "provider name appears in a DOJ press release that mentions its city: " + fmt([h for h in ph if h["city_match"]], 1)
            else:
                df.at[i, "press_note"] = "provider name appears in a DOJ press release (state/no city match): name-only, verify"
        if i in check_owners:
            infos = r["owner_info"] if isinstance(r.get("owner_info"), dict) else {}
            refs = []
            for k, inf in infos.items():
                if k.startswith("ORG:") or inf.get("role", "").upper().startswith("CORPORATE DIRECTOR BOARD"):
                    continue
                oh = [h for h in idx.find_person(inf.get("name", ""), r["state"], clean_str(r.get("city", ""))) if h["city_match"]]
                if oh:
                    refs.append(f"{inf.get('name')}: {fmt(oh, 1)}")
            if refs:
                df.at[i, "press_owner_refs"] = " ; ".join(refs[:3])
                df.at[i, "linked_known_case"] = True
                cur = clean_str(r.get("linked_known_case_detail", ""))
                df.at[i, "linked_known_case_detail"] = (cur + "; " if cur else "") + "owner/officer named in DOJ press release: " + refs[0]
    return df
