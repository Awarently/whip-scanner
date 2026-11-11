"""National check: individuals/entities on the OIG exclusion list (LEIE) who appear in Medicare enrollment
records (PECOS) as owners, officers or managers of home health agencies and hospices, in any state.

Public data only. A name match is NOT proof of identity: results are tiered by identity evidence and every
row carries what a human needs to verify it (role, ownership %, dates, LEIE record). Nothing here uses
demographic attributes of patients; the only personal attributes used are name, middle initial and the
LEIE's own state/city for the excluded person.

Tiers (strongest first)
  A  name + middle initial agree (both known) + same state, or provider NPI itself on the list
  B  name + same state, middle initial unknown on one side
  C  name + middle initial agree, but different state
  (name-only matches in other states, or with conflicting middle initials, are not reported)
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path

import pandas as pd

from clean.normalize import person_key, norm_name, clean_str
from match.enrich import is_board_role

TIER_BASE = {"A": 3.0, "B": 2.0, "C": 1.0}


def _cityn(x) -> str:
    return " ".join("".join(c for c in clean_str(x).upper() if c.isalpha() or c == " ").split())


def role_class(title: str) -> str:
    t = clean_str(title).upper()
    if is_board_role(t):
        return "board"
    if "OWNERSHIP" in t or "PARTNERSHIP" in t or "SECURITY INTEREST" in t:
        return "owner"
    if "OFFICER" in t or "DIRECTOR" in t:
        return "officer"
    if "MANAGING EMPLOYEE" in t or "OPERATIONAL" in t:
        return "manager"
    return "other"


ROLE_WEIGHT = {"owner": 1.0, "officer": 0.8, "manager": 0.8, "other": 0.3, "board": 0.3}


def _d(x):
    s = clean_str(x)
    if not s or s in ("0", "00000000"):
        return pd.NaT
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce") if s.isdigit() and len(s) == 8 else pd.to_datetime(s, errors="coerce")


def prepare_leie(leie: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Active (not reinstated) exclusions. Returns (persons, npi->record)."""
    l = leie.copy()
    l.columns = [c.lower() for c in l.columns]
    reinst = l["reindate"].map(_d)
    l = l[reinst.isna()].copy()                       # reinstated => no longer excluded
    l["excl_date"] = l["excldate"].map(_d)
    l["key"] = (l["firstname"] + " " + l["lastname"]).map(person_key)
    l["mid"] = l["midname"].str.upper().str[:1]
    l["orgkey"] = l["busname"].map(lambda x: ("ORG:" + norm_name(x)) if clean_str(x) else "")
    npis = {r["npi"]: r for _, r in l[l["npi"].str.strip("0") != ""].iterrows()}
    return l, npis


def match_exclusions(owners: pd.DataFrame, leie: pd.DataFrame, cert: pd.DataFrame | None = None) -> pd.DataFrame:
    """owners: one row per owner/officer role per enrollment (columns: name, ccn, npi, state, city, officer_name,
    officer_title, ownership_pct, owner_id, owner_since, officer_middle, owner_name_count, sector).
    leie: raw LEIE frame. cert: optional (ccn, cert_date). Returns one row per (owner, enrollment, LEIE record)."""
    l, npis = prepare_leie(leie)
    by_key: dict[str, list[int]] = {}
    for i, k in l["key"].items():
        if k:
            by_key.setdefault(k, []).append(i)
    by_org: dict[str, list[int]] = {}
    for i, k in l["orgkey"].items():
        if k:
            by_org.setdefault(k, []).append(i)
    certmap = {}
    if cert is not None and len(cert):
        certmap = dict(zip(cert["ccn"], pd.to_datetime(cert["cert_date"], errors="coerce")))
    n_enroll = owners.groupby("owner_id")["ccn"].nunique().to_dict()
    rows = []
    for _, o in owners.iterrows():
        pk = person_key(o["officer_name"])
        state = clean_str(o["state"]).upper()
        mid = clean_str(o.get("officer_middle", "")).upper()[:1]
        cands = []
        if pk and not pk.startswith("ORG:"):
            for i in by_key.get(pk, []):
                r = l.loc[i]
                if mid and r["mid"] and mid != r["mid"]:
                    continue                                   # conflicting middle initial: different person
                same_state = clean_str(r["state"]).upper() == state
                mid_agree = bool(mid and r["mid"] and mid == r["mid"])
                tier = "A" if (mid_agree and same_state) else "B" if same_state else "C" if mid_agree else None
                if tier:
                    cands.append((tier, i))
        elif pk.startswith("ORG:"):
            for i in by_org.get(pk, []):
                if clean_str(l.at[i, "state"]).upper() == state:
                    cands.append(("A", i))
        npi = clean_str(o.get("npi", ""))
        if npi and npi in npis:
            cands.append(("A", npis[npi].name))
        if not cands:
            continue
        # same-state LEIE people sharing this name (after middle filter): more than one = ambiguous
        n_same_state = sum(1 for t, i in cands if t in ("A", "B"))
        for tier, i in cands:
            r = l.loc[i]
            since = _d(o.get("owner_since", ""))
            cdate = certmap.get(clean_str(o["ccn"]), pd.NaT)
            rc = role_class(o["officer_title"])
            after = bool(pd.notna(since) and pd.notna(r["excl_date"]) and since > r["excl_date"])
            cafter = bool(pd.notna(cdate) and pd.notna(r["excl_date"]) and cdate > r["excl_date"])
            nc = int(float(o["owner_name_count"])) if clean_str(o.get("owner_name_count", "")) else 0
            amb = 1.0 if nc <= 3 else 0.5 if nc <= 10 else 0.25
            pct = pd.to_numeric(o.get("ownership_pct"), errors="coerce")
            same_city = bool(clean_str(o.get("city", "")) and _cityn(o.get("city", "")) == _cityn(r["city"]))
            spec = clean_str(r["specialty"]).upper()
            relevant = any(w in spec for w in ("HOME HEALTH", "HOSPICE", "OWNER", "NURS", "HEALTH CARE AIDE", "PERSONAL CARE"))
            score = (TIER_BASE[tier] * amb * ROLE_WEIGHT[rc] * (1.0 if n_same_state <= 1 else 0.5)
                     + (1.0 if after else 0) + (1.0 if cafter else 0)
                     + (1.0 if same_city else 0) + (0.5 if relevant else 0))
            rows.append({
                "tier": tier, "priority_score": round(score, 2), "sector": o["sector"], "provider": o["name"],
                "ccn": o["ccn"], "provider_npi": npi, "provider_state": state, "city": o.get("city", ""),
                "owner": o["officer_name"], "owner_role": clean_str(o["officer_title"]), "role_class": rc,
                "ownership_pct": pct, "owner_since": since, "owner_middle": mid, "owner_associate_id": o.get("owner_id", ""),
                "owner_enrollments_nationally": n_enroll.get(o.get("owner_id", ""), 0),
                "provider_cert_date": cdate, "leie_name": " ".join(x for x in (r["firstname"], r["midname"], r["lastname"], r["busname"]) if x),
                "leie_state": r["state"], "leie_city": r["city"], "leie_general": r["general"], "leie_specialty": r["specialty"],
                "leie_excl_type": r["excltype"], "leie_excl_date": r["excl_date"], "leie_dob_year": clean_str(r["dob"])[:4],
                "same_city": same_city, "specialty_relevant": relevant,
                "owner_after_exclusion": after, "cert_after_exclusion": cafter,
                "months_exclusion_to_cert": (round((cdate - r["excl_date"]).days / 30.4, 1) if pd.notna(cdate) and pd.notna(r["excl_date"]) else None),
                "months_exclusion_to_owner_since": (round((since - r["excl_date"]).days / 30.4, 1) if pd.notna(since) and pd.notna(r["excl_date"]) else None),
                "people_same_name_in_pecos": nc, "leie_same_name_same_state": n_same_state})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["priority_score", "tier"], ascending=[False, True]).reset_index(drop=True)
    return out


def annotate_press(df: pd.DataFrame, idx) -> pd.DataFrame:
    """Mark candidates already named in DOJ press releases (so they are not original information).
    already_public needs the release to mention the provider's CITY; state-only hits go to press_possible."""
    from filters.press import fmt
    df = df.copy()
    owner_refs, prov_refs, possible, public = [], [], [], []
    cache: dict[tuple, list] = {}
    for _, r in df.iterrows():
        k = ("p", r["owner"], r["provider_state"], r["city"])
        if k not in cache:
            cache[k] = idx.find_person(r["owner"], r["provider_state"], r["city"])
        oh = cache[k]
        k2 = ("o", r["provider"], r["provider_state"], r["city"])
        if k2 not in cache:
            cache[k2] = idx.find_org(r["provider"], r["provider_state"], clean_str(r["city"]))
        ph = cache[k2]
        oh_city = [h for h in oh if h["city_match"]]
        oh_state = [h for h in oh if h["state_match"] and not h["city_match"]]
        owner_refs.append(fmt(oh_city, 2))
        possible.append(fmt(oh_state, 1))
        prov_refs.append(fmt(ph, 2))
        public.append(bool(oh_city) or any(h["city_match"] for h in ph))
    df["press_owner_refs"], df["press_possible_state_only"] = owner_refs, possible
    df["press_provider_refs"], df["already_public"] = prov_refs, public
    return df


def run_national(cfg: dict, log, resolve_source, feeds, offline: bool = False) -> Path | None:
    """Download national PECOS owners (HH + hospice), CMS certification dates and LEIE; write the review file."""
    from clean.build import load_source
    day = cfg["_as_of"].strftime("%Y%m%d")
    out_dir = cfg["paths"]["out"] / day / "national_exclusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = resolve_source("leie", cfg["sources"]["leie"], "shared", "all", cfg, log, offline)
    leie = load_source(res, cfg["sources"]["leie"], cfg, "shared", "all")
    if leie is None:
        log.warn("LEIE unavailable; cannot run the national check")
        return None
    owners_all, cert_all = [], []
    for sector, sid, pdc_sid in (("home_health", "owners_hh", "cms_provider_hh"), ("hospice", "owners_hospice", "cms_provider_hospice")):
        spec = cfg["sources"][sid]
        dest = cfg["paths"]["raw"] / f"{sid}_national" / f"{sid}_national_{day}.csv"
        if not dest.exists():
            feeds.pecos(spec, cfg, log, "ALL", dest)
        o = pd.read_csv(dest, dtype=str).fillna("")
        o["sector"] = sector
        owners_all.append(o)
        cdest = cfg["paths"]["raw"] / f"{pdc_sid}_national" / f"{pdc_sid}_national_{day}.csv"
        if not cdest.exists():
            feeds.pdc(cfg["sources"][pdc_sid], cfg, log, cdest)
        c = pd.read_csv(cdest, dtype=str).fillna("")
        c.columns = [x.strip().lower() for x in c.columns]
        ccn = next(x for x in c.columns if "certification number" in x)
        cd = next(x for x in c.columns if x == "certification date")
        cert_all.append(pd.DataFrame({"ccn": c[ccn], "cert_date": c[cd]}))
    owners = pd.concat(owners_all, ignore_index=True)
    cert = pd.concat(cert_all, ignore_index=True)
    log.info(f"national: {len(owners)} owner/officer rows across {owners['ccn'].nunique()} providers; LEIE rows {len(leie)}")
    res_df = match_exclusions(owners, leie, cert)
    from filters.press import load_latest
    pidx = load_latest(cfg)
    if pidx is not None and len(res_df):
        res_df = annotate_press(res_df, pidx)
        log.info(f"press check: {int(res_df['already_public'].sum())} of {len(res_df)} candidate rows already named in DOJ releases")
    else:
        log.warn("no DOJ press corpus; originality not checked (python main.py --refresh-press)")
    res_df.to_csv(out_dir / "national_identity_review.csv", index=False)
    log.info(f"national exclusion check: {len(res_df)} candidate rows -> {out_dir}")
    return out_dir
