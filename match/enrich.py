"""Enrich providers with corporate records and NPPES data."""
from __future__ import annotations
import pandas as pd
from rapidfuzz import fuzz
from clean.normalize import norm_name, clean_str
from fetch import nppes_lookup


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
        prov[c] = pd.NA
    if corp is None or corp.empty:
        return prov
    corp = corp.copy()
    corp["_n"] = corp["name"].map(norm_name)
    admin_w, agent_w = cfg["clean"]["admin_title_words"], cfg["clean"]["agent_title_words"]
    ents = {}
    for n, g in corp.groupby("_n"):
        owners, admins = [], []
        for _, r in g.iterrows():
            title = clean_str(r.get("officer_title", "")).upper()
            who = clean_str(r.get("officer_name", ""))
            if not who or any(w in title for w in agent_w):
                continue
            (admins if any(w in title for w in admin_w) else owners).append(who)
        ents[n] = {"owners": _join(*owners), "admins": _join(*admins),
                   "agent": clean_str(g.get("registered_agent", pd.Series([""])).iloc[0]),
                   "formed": g["formation_date"].dropna().min() if "formation_date" in g else pd.NaT,
                   "status": clean_str(g["status"].iloc[0]) if "status" in g else "",
                   "rec": g["_rec_id"].iloc[0]}
    keys = list(ents)
    thr = cfg["match"]["corp_name_threshold"]
    hits = 0
    for i, r in prov.iterrows():
        n = norm_name(r["name"])
        k = n if n in ents else None
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
