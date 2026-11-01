"""Known-case filter. WHIP pays only for ORIGINAL information, so providers already on LEIE/SAM,
named in DOJ/OIG press releases, or under an existing same-conduct state sanction are flagged
and EXCLUDED from the target ranking. Their owners/addresses stay as linked-to-known-case signals.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz, process
from clean.normalize import person_key, norm_name, split_people, clean_str, is_org


def _truthy(x) -> bool:
    return clean_str(x).upper() in ("Y", "YES", "TRUE", "1", "T")


class KnownSets:
    def __init__(self):
        self.npis: dict[str, str] = {}
        self.persons: dict[str, list[str]] = defaultdict(list)    # key -> sources
        self.orgs: dict[str, list[str]] = defaultdict(list)
        self.last_blocks: dict[str, list[str]] = defaultdict(list)

    def add_name(self, raw: str, src: str):
        k = person_key(raw)
        if not k:
            return
        if k.startswith("ORG:"):
            self.orgs[k[4:]].append(src)
        else:
            self.persons[k].append(src)
            self.last_blocks[k.split()[-1]].append(k)

    def match_person(self, key: str, thr: int):
        if key in self.persons:
            return key, 100, self.persons[key]
        for cand in set(self.last_blocks.get(key.split()[-1], [])):
            if cand[0] == key[0] and fuzz.ratio(cand, key) >= thr:
                return cand, fuzz.ratio(cand, key), self.persons[cand]
        return None

    def match_org(self, nm: str, thr: int):
        if not nm or not self.orgs:
            return None
        if nm in self.orgs:
            return nm, 100, self.orgs[nm]
        hit = process.extractOne(nm, list(self.orgs), scorer=fuzz.token_sort_ratio, score_cutoff=thr)
        return (hit[0], hit[1], self.orgs[hit[0]]) if hit else None


def build_known_sets(frames: dict, log) -> KnownSets:
    ks = KnownSets()
    leie = frames.get("leie")
    if leie is not None:
        raw = leie
        for _, r in raw.iterrows():
            src = f"LEIE {r.get('_rec_id','')}"
            if clean_str(r.get("npi", "")).strip("0"):
                ks.npis[clean_str(r["npi"])] = src
            if clean_str(r.get("busname", "")):
                ks.add_name(r["busname"], src)
            if clean_str(r.get("firstname", "")) and clean_str(r.get("lastname", "")):
                ks.add_name(f"{r['firstname']} {r['lastname']}", src)
    sam = frames.get("sam")
    if sam is not None:
        for _, r in sam.iterrows():
            src = f"SAM {r.get('_rec_id','')}"
            for c in ("name", "entity_name"):
                if clean_str(r.get(c, "")):
                    ks.add_name(r[c], src)
            if clean_str(r.get("npi", "")):
                ks.npis[clean_str(r["npi"])] = src
            fn, ln = clean_str(r.get("firstname", r.get("first", ""))), clean_str(r.get("lastname", r.get("last", "")))
            if fn and ln:
                ks.add_name(f"{fn} {ln}", src)
    log.info(f"known-case sets: {len(ks.npis)} NPIs, {len(ks.persons)} persons, {len(ks.orgs)} orgs (LEIE/SAM)")
    return ks


def split_press(press: pd.DataFrame | None, ks: KnownSets, log):
    """Press-release names go into a separate KnownSets (they link, they are not exclusions)."""
    pk = KnownSets()
    if press is None:
        return pk
    for _, r in press.iterrows():
        src = f"PRESS {clean_str(r.get('url','')) or r.get('_rec_id','')}"
        for nm in split_people(r.get("names", "")) + split_people(r.get("name", "")):
            pk.add_name(nm, src)
    log.info(f"press releases: {len(pk.persons)} persons, {len(pk.orgs)} orgs")
    return pk


def apply_known_case(df: pd.DataFrame, edges: list[dict], excl: KnownSets, press: KnownSets, cfg: dict):
    pthr, othr = cfg["match"]["person_threshold"], cfg["match"]["org_threshold"]
    df = df.copy()
    df["known_case"] = False
    df["known_case_reason"] = ""
    df["owner_exclusion_match"] = False
    df["owner_exclusion_detail"] = ""
    df["linked_known_case"] = False
    df["linked_known_case_detail"] = ""

    for i, r in df.iterrows():
        reasons = []
        npi = clean_str(r["npi"])
        if npi and npi in excl.npis:
            reasons.append(f"provider NPI {npi} on exclusion list ({excl.npis[npi]})")
        nm = norm_name(r["name"])
        for label, ks in (("LEIE/SAM", excl), ("press release", press)):
            m = ks.match_org(nm, othr)
            if m:
                reasons.append(f"provider name matches {label} entity '{m[0]}' ({m[1]:.0f}) [{', '.join(m[2][:2])}]")
        if _truthy(r.get("sanction_same_conduct", "")):
            reasons.append("existing state sanction for same conduct (sanction_same_conduct=Y)")
        if reasons:
            df.at[i, "known_case"] = True
            df.at[i, "known_case_reason"] = " | ".join(reasons)

        # owners / officials vs exclusion lists and press names
        hits, press_hits = [], []
        for key in r["people"]:
            if key.startswith("ORG:"):
                for ks, lst in ((excl, hits), (press, press_hits)):
                    m = ks.match_org(key[4:], othr)
                    if m:
                        lst.append(f"{key[4:]}~{m[0]} ({m[1]:.0f}; {m[2][0]})")
            else:
                for ks, lst in ((excl, hits), (press, press_hits)):
                    m = ks.match_person(key, pthr)
                    if m:
                        lst.append(f"{key}~{m[0]} ({m[1]:.0f}; {m[2][0]})")
        if hits:
            df.at[i, "owner_exclusion_match"] = True
            df.at[i, "owner_exclusion_detail"] = "; ".join(hits) + "  -- fuzzy name match only, verify identity"
        if press_hits:
            df.at[i, "linked_known_case"] = True
            df.at[i, "linked_known_case_detail"] = "named in press release: " + "; ".join(press_hits)

    # link via shared address/owner to providers that ARE known cases
    known_ids = set(df.index[df["known_case"]])
    ids = df["provider_id"].to_dict()
    for e in edges:
        for x, y in ((e["a"], e["b"]), (e["b"], e["a"])):
            if y in known_ids and x not in known_ids:
                df.at[x, "linked_known_case"] = True
                msg = f"shares {e['type']} '{e['key']}' with known-case {ids[y]}"
                cur = df.at[x, "linked_known_case_detail"]
                df.at[x, "linked_known_case_detail"] = (cur + "; " + msg).strip("; ")
    return df
