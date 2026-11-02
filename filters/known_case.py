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
        self.loc: dict[str, list[tuple]] = defaultdict(list)   # key -> [(state, zip5, middle initial)] per source record
        self.rec: dict[str, dict] = {}                          # source id -> details for human review

    def add_name(self, raw: str, src: str, state: str = "", zip5: str = "", mid: str = ""):
        k = person_key(raw)
        if not k:
            return
        z = "".join(c for c in clean_str(zip5) if c.isdigit())[:5]
        loc = (clean_str(state).upper(), z, clean_str(mid).upper()[:1])      # (state, zip5, middle initial)
        if k.startswith("ORG:"):
            self.orgs[k[4:]].append(src)
            self.loc["O:" + k[4:]].append(loc)
        else:
            self.persons[k].append(src)
            self.loc["P:" + k].append(loc)
            self.last_blocks[k.split()[-1]].append(k)

    def match_person(self, key: str, thr: int):
        if key in self.persons:
            return key, 100, self.persons[key], self.loc["P:" + key]
        for cand in set(self.last_blocks.get(key.split()[-1], [])):
            if cand[0] == key[0] and fuzz.ratio(cand, key) >= thr:
                return cand, fuzz.ratio(cand, key), self.persons[cand], self.loc["P:" + cand]
        return None

    def match_org(self, nm: str, thr: int):
        if not nm or not self.orgs:
            return None
        if nm in self.orgs:
            return nm, 100, self.orgs[nm], self.loc["O:" + nm]
        hit = process.extractOne(nm, list(self.orgs), scorer=fuzz.token_sort_ratio, score_cutoff=thr)
        return (hit[0], hit[1], self.orgs[hit[0]], self.loc["O:" + hit[0]]) if hit else None


def build_known_sets(frames: dict, log) -> KnownSets:
    ks = KnownSets()
    leie = frames.get("leie")
    if leie is not None:
        raw = leie
        for _, r in raw.iterrows():
            src = f"LEIE {r.get('_rec_id','')}"
            st, zp = r.get("state", ""), r.get("zip", "")
            if clean_str(r.get("npi", "")).strip("0"):
                ks.npis[clean_str(r["npi"])] = src
            if clean_str(r.get("busname", "")):
                ks.add_name(r["busname"], src, st, zp)
            ks.rec[src] = {"name": " ".join(x for x in (r.get("firstname", ""), r.get("midname", ""), r.get("lastname", ""), r.get("busname", "")) if x),
                           "city": r.get("city", ""), "state": st, "specialty": r.get("specialty", ""), "general": r.get("general", ""),
                           "excl_type": r.get("excltype", ""), "excl_date": r.get("excldate", ""), "reinstated": r.get("reindate", ""),
                           "dob_year": clean_str(r.get("dob", ""))[:4]}
            if clean_str(r.get("firstname", "")) and clean_str(r.get("lastname", "")):
                ks.add_name(f"{r['firstname']} {r['lastname']}", src, st, zp, r.get("midname", ""))
    sam = frames.get("sam")
    if sam is not None:
        for _, r in sam.iterrows():
            src = f"SAM {r.get('_rec_id','')}"
            for c in ("name", "entity_name"):
                if clean_str(r.get(c, "")):
                    ks.add_name(r[c], src, r.get("state", ""), r.get("zip", ""))
            if clean_str(r.get("npi", "")):
                ks.npis[clean_str(r["npi"])] = src
            fn, ln = clean_str(r.get("firstname", r.get("first", ""))), clean_str(r.get("lastname", r.get("last", "")))
            if fn and ln:
                ks.add_name(f"{fn} {ln}", src, r.get("state", ""), r.get("zip", ""), r.get("middle", r.get("midname", "")))
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
            pk.add_name(nm, src, r.get("state", ""), r.get("zip", ""))
    log.info(f"press releases: {len(pk.persons)} persons, {len(pk.orgs)} orgs")
    return pk


def _loc_ok(locs, state, zip5, need_zip: bool, mid: str = "") -> list[int]:
    """Indices of source records that corroborate the match: same state (and ZIP when need_zip), and no
    conflicting middle initial (both known and different => a different person)."""
    mid = clean_str(mid).upper()[:1]
    out = []
    for i, (st, z, m) in enumerate(locs):
        if not (st and st == state and (not need_zip or (z and z == zip5))):
            continue
        if mid and m and mid != m:
            continue
        out.append(i)
    return out


def apply_known_case(df: pd.DataFrame, edges: list[dict], excl: KnownSets, press: KnownSets, cfg: dict):
    """A provider is a KNOWN CASE only on an NPI match, or a name match corroborated by location
    (same state AND ZIP as the list entry). Name-only matches are recorded in `known_case_note`
    but do not exclude the provider (national franchise/common names produce false matches)."""
    pthr, othr = cfg["match"]["person_threshold"], cfg["match"]["org_threshold"]
    df = df.copy()
    df["owner_exclusion_records"] = [[] for _ in range(len(df))]
    for c, v in (("owner_exclusion_strength", 0.0), ("known_case", False), ("known_case_reason", ""), ("known_case_note", ""),
                 ("owner_exclusion_match", False), ("owner_exclusion_detail", ""),
                 ("linked_known_case", False), ("linked_known_case_detail", "")):
        df[c] = v

    for i, r in df.iterrows():
        reasons, notes = [], []
        npi = clean_str(r["npi"])
        if npi and npi in excl.npis:
            reasons.append(f"provider NPI {npi} on exclusion list ({excl.npis[npi]})")
        nm = norm_name(r["name"])
        for label, ks in (("LEIE/SAM", excl), ("press release", press)):
            m = ks.match_org(nm, othr)
            if not m:
                continue
            txt = f"provider name matches {label} entity '{m[0]}' ({m[1]:.0f}) [{', '.join(m[2][:2])}]"
            if _loc_ok(m[3], r["state"], clean_str(r["zip5"]), need_zip=True):
                reasons.append(txt + " at same ZIP")
            else:
                notes.append(txt + " but different location: name-only, NOT excluded, verify")
        if _truthy(r.get("sanction_same_conduct", "")):
            reasons.append("existing state sanction for same conduct (sanction_same_conduct=Y)")
        if reasons:
            df.at[i, "known_case"] = True
            df.at[i, "known_case_reason"] = " | ".join(reasons)
        if notes:
            df.at[i, "known_case_note"] = " | ".join(notes)

        # owners / officials vs exclusion lists and press names (state must agree)
        hits, press_hits, recs, strengths = [], [], [], []
        mids = r["owner_mid"] if isinstance(r.get("owner_mid"), dict) else {}
        infos = r["owner_info"] if isinstance(r.get("owner_info"), dict) else {}
        for key in r["people"]:
            for ks, lst in ((excl, hits), (press, press_hits)):
                m = ks.match_org(key[4:], othr) if key.startswith("ORG:") else ks.match_person(key, pthr)
                if not m:
                    continue
                idx = _loc_ok(m[3], r["state"], "", need_zip=False, mid=mids.get(key, ""))
                if idx:   # cite the record(s) that actually corroborate, not just the first with that name
                    cited = ", ".join(m[2][j] for j in idx[:3])
                    lst.append(f"{key.replace('ORG:', '')}~{m[0]} ({m[1]:.0f}; {cited}; {len(idx)} same-state record(s))")
                    if ks is excl:
                        inf = infos.get(key, {})
                        mid_ok = bool(mids.get(key)) and any(m[3][j][2] == mids.get(key) for j in idx)
                        nc = inf.get("name_count", 0)
                        # strength: identity evidence (middle initial agrees) x name ambiguity (people sharing the name)
                        base = 1.0 if (mid_ok or key.startswith("ORG:")) else 0.5
                        amb = 1.0 if nc <= 3 else (0.5 if nc <= 10 else 0.25)
                        strengths.append(base * amb)
                        for j in idx[:3]:
                            recs.append({"provider_id": r["provider_id"], "provider": r["name"], "sector": r["sector"],
                                         "owner": inf.get("name", key), "owner_role": inf.get("role", ""),
                                         "ownership_pct": inf.get("pct", ""), "owner_since": inf.get("since", ""),
                                         "owner_middle": mids.get(key, ""), "people_with_same_name_nationally": nc,
                                         "middle_initial_status": "agrees" if mid_ok else "unknown (cannot rule out)",
                                         "match_strength": base * amb, "leie_source": m[2][j],
                                         **{f"leie_{k}": v for k, v in ks.rec.get(m[2][j], {}).items()}})
        if hits:
            df.at[i, "owner_exclusion_strength"] = max(strengths) if strengths else 1.0
            df.at[i, "owner_exclusion_records"] = recs
            df.at[i, "owner_exclusion_match"] = True
            df.at[i, "owner_exclusion_detail"] = "; ".join(hits) + "  -- fuzzy name + same-state match only, verify identity"
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
