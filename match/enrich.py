"""Enrich providers with corporate records and NPPES data."""
from __future__ import annotations
import pandas as pd
from rapidfuzz import fuzz
from clean.normalize import norm_name, clean_str, person_key
from fetch import nppes_lookup


_BOARD_WORDS = {"", "BOARD MEMBER", "BOARD", "DIRECTOR", "MEMBER", "BOARD OF DIRECTORS", "CHAIR", "CHAIRMAN",
                "CHAIRPERSON", "VICE CHAIR", "BOARD CHAIR", "BOARD CHAIRMAN", "BOARD DIRECTOR", "TRUSTEE", "BOARD TRUSTEE"}


def is_board_role(role: str) -> bool:
    """PECOS role text is '<ROLE> <TITLE>'. Board-only = a corporate director with no officer or management
    title (e.g. 'CORPORATE DIRECTOR BOARD MEMBER'). 'CORPORATE DIRECTOR PRESIDENT' is an officer, not board-only."""
    r = " ".join(clean_str(role).upper().split())
    if not r.startswith("CORPORATE DIRECTOR"):
        return False
    return r[len("CORPORATE DIRECTOR"):].strip() in _BOARD_WORDS


def _join(*parts) -> str:
    seen, out = set(), []
    for p in parts:
        for x in [y.strip() for y in clean_str(p).split(";") if y.strip()]:
            if x.upper() not in seen:
                seen.add(x.upper())
                out.append(x)
    return "; ".join(out)


def enrich_corp(prov: pd.DataFrame, corp: pd.DataFrame | None, res, cfg, log) -> pd.DataFrame:
    prov = prov.copy()
    for c in ("registered_agent", "corp_formation_date", "corp_status"):
        if c not in prov:
            prov[c] = pd.NA
    if "board_members" not in prov:
        prov["board_members"] = ""
    for c in ("owner_natl", "owner_mid", "owner_info"):
        if c not in prov:
            prov[c] = [{} for _ in range(len(prov))]
    if corp is None or corp.empty or "name" not in corp:
        return prov
    corp = corp.copy()
    for c in ("ccn", "npi", "officer_name", "officer_title"):
        if c not in corp:
            corp[c] = ""
    corp["_n"] = corp["name"].map(norm_name)
    admin_w, agent_w = cfg["clean"]["admin_title_words"], cfg["clean"]["agent_title_words"]
    ents, by_ccn, by_npi = {}, {}, {}
    for n, g in corp.groupby("_n"):
        owners, admins, boards = [], [], []
        roles_by_person: dict[str, list[str]] = {}
        for _, r in g.iterrows():
            k = person_key(clean_str(r.get("officer_name", "")))
            if k:
                roles_by_person.setdefault(k, []).append(clean_str(r.get("officer_title", "")))
        # board-only individuals are kept for the exclusion-list check but not used to link providers
        board_keys = {k for k, ts in roles_by_person.items() if not k.startswith("ORG:") and all(is_board_role(t) for t in ts)}
        for _, r in g.iterrows():
            title = clean_str(r.get("officer_title", "")).upper()
            who = clean_str(r.get("officer_name", ""))
            if not who or any(w in title for w in agent_w):
                continue
            if person_key(who) in board_keys:
                boards.append(who)
                continue
            (admins if any(w in title for w in admin_w) else owners).append(who)
        mids = {}
        if "officer_middle" in g.columns:
            for _, rr in g.iterrows():
                k = person_key(clean_str(rr.get("officer_name", "")))
                mi = clean_str(rr.get("officer_middle", "")).upper()[:1]
                if k and mi:
                    mids[k] = mi
        info = {}
        for _, rr in g.iterrows():
            k = person_key(clean_str(rr.get("officer_name", "")))
            if k:
                nc = rr.get("owner_name_count")
                info[k] = {"name": clean_str(rr.get("officer_name", "")), "role": clean_str(rr.get("officer_title", "")),
                           "pct": clean_str(rr.get("ownership_pct", "")), "since": clean_str(rr.get("owner_since", "")),
                           "name_count": int(nc) if pd.notna(nc) else 0}
        natl = {}
        if "owner_national" in g.columns:
            for _, rr in g.iterrows():
                k = person_key(clean_str(rr.get("officer_name", "")))
                v = rr.get("owner_national")
                if k and pd.notna(v):
                    natl[k] = max(natl.get(k, 0), int(v))
        ents[n] = {"boards": _join(*boards), "info": info, "mids": mids, "natl": natl, "owners": _join(*owners), "admins": _join(*admins),
                   "agent": clean_str(g.get("registered_agent", pd.Series([""])).iloc[0]),
                   "formed": g["formation_date"].dropna().min() if "formation_date" in g else pd.NaT,
                   "status": clean_str(g["status"].iloc[0]) if "status" in g else "",
                   "rec": g["_rec_id"].iloc[0]}
        for c in g["ccn"].dropna().unique():
            if clean_str(c):
                by_ccn[clean_str(c)] = n
        for c in g["npi"].dropna().unique():
            if clean_str(c):
                by_npi[clean_str(c)] = n
    keys = list(ents)
    thr = cfg["match"]["corp_name_threshold"]
    hits = 0
    for i, r in prov.iterrows():
        n = norm_name(r["name"])
        k = by_ccn.get(clean_str(r["ccn"])) or by_npi.get(clean_str(r["npi"])) or (n if n in ents else None)
        if k is None and n:
            best = max(keys, key=lambda x: fuzz.token_sort_ratio(n, x), default=None)
            if best and fuzz.token_sort_ratio(n, best) >= thr:
                k = best
        if k is None:
            continue
        e = ents[k]
        hits += 1
        prov.at[i, "owners"] = _join(r["owners"] if pd.notna(r["owners"]) else "", e["owners"])
        prov.at[i, "administrator"] = _join(r["administrator"] if pd.notna(r["administrator"]) else "", e["admins"])
        cur = prov.at[i, "owner_natl"] if isinstance(prov.at[i, "owner_natl"], dict) else {}
        prov.at[i, "owner_natl"] = {**cur, **e["natl"]}
        curi = prov.at[i, "owner_info"] if isinstance(prov.at[i, "owner_info"], dict) else {}
        prov.at[i, "owner_info"] = {**curi, **e["info"]}
        curm = prov.at[i, "owner_mid"] if isinstance(prov.at[i, "owner_mid"], dict) else {}
        prov.at[i, "owner_mid"] = {**curm, **e["mids"]}
        prov.at[i, "board_members"] = _join(r["board_members"] if pd.notna(r["board_members"]) else "", e["boards"])
        prov.at[i, "registered_agent"] = e["agent"]
        prov.at[i, "corp_formation_date"] = e["formed"]
        prov.at[i, "corp_status"] = e["status"]
        prov.at[i, "sources"] = list(r["sources"]) + [{"source": "corp", "file": res.path and str(res.path),
                                                       "download_date": res.date, "record_id": e["rec"]}]
    log.info(f"corp enrichment: {hits}/{len(prov)} providers matched")
    return prov


def enrich_nppes(prov: pd.DataFrame, manual: pd.DataFrame | None, res, cfg, log) -> pd.DataFrame:
    prov = prov.copy()
    prov["npi_enum_date"] = pd.NaT
    info = nppes_lookup(prov["npi"].dropna().astype(str).tolist(), cfg, log)
    src_file, src_date = ("NPPES API", cfg["_as_of"].isoformat())
    if not info and manual is not None and not manual.empty:
        info = {clean_str(r["npi"]): {"auth_official": clean_str(r.get("auth_official", "")),
                                      "enumeration_date": r.get("enumeration_date", pd.NaT)}
                for _, r in manual.iterrows() if clean_str(r.get("npi", ""))}
        src_file, src_date = (str(res.path) if res and res.path else "manual", res.date if res else None)
    n = 0
    for i, r in prov.iterrows():
        d = info.get(clean_str(r["npi"]))
        if not d:
            continue
        n += 1
        if d.get("auth_official"):
            prov.at[i, "auth_official"] = _join(r["auth_official"] if pd.notna(r["auth_official"]) else "", d["auth_official"])
        prov.at[i, "npi_enum_date"] = pd.to_datetime(d.get("enumeration_date"), errors="coerce")
        prov.at[i, "sources"] = list(r["sources"]) + [{"source": "nppes", "file": src_file,
                                                       "download_date": src_date, "record_id": f"NPI {r['npi']}"}]
    log.info(f"NPPES enrichment: {n}/{len(prov)} providers")
    return prov
