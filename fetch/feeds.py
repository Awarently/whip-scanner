"""Structured public feeds: Georgia DCH facility feed, CMS Provider Data Catalog, CMS data-api, PECOS owners.

All are public, unauthenticated endpoints that the sites' own pages use. One request per page of data,
`fetch.delay_seconds` between requests, results written to dated raw files and never overwritten.
"""
from __future__ import annotations
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

PDC_META = "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{id}"
API_DATA = "https://data.cms.gov/data-api/v1/dataset/{id}/data"
GA_FEED = "https://forms.dch.georgia.gov/hfrd/FACSEARCH.aspx"


def _get(url, cfg, log, params=None, stream=False, ua=True):
    """ua=False sends the library's default User-Agent. data.cms.gov's Provider Data Catalog metastore
    returns 403 to our custom research User-Agent but serves the same public data to a plain script
    client; the repo owner explicitly approved this for data.cms.gov only (2026-11-01)."""
    f = cfg["fetch"]
    time.sleep(f["delay_seconds"])
    r = requests.get(url, params=params, headers={"User-Agent": f["user_agent"]} if ua else None,
                     timeout=f["timeout_seconds"] * (4 if stream else 1), stream=stream)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code} for {url}")
    return r


# ------------------------------------------------------------------ Georgia DCH
def ga_dch(spec, cfg, log, ddir: Path) -> list[Path]:
    """Statewide facility feed -> one CSV per facility type (file stem sets facility_type)."""
    files = spec["feed_files"]                      # {stem: FACTYPE}
    if ddir.exists() and all((ddir / f"{s}.csv").exists() for s in files):
        return [ddir / f"{s}.csv" for s in files]    # same-day files exist: never overwrite
    r = _get(spec.get("feed_url", GA_FEED), cfg, log)
    root = ET.fromstring(r.content)
    df = pd.DataFrame([dict(e.attrib) for e in root]).fillna("")
    if df.empty or "FACTYPE" not in df:
        raise ValueError("GA feed returned no facility rows")
    t = df["FACTYPE"].str.upper().str.strip()
    ddir.mkdir(parents=True, exist_ok=True)
    out = []
    for stem, ftype in files.items():
        s = df[t == ftype.upper()]
        c = pd.DataFrame({
            "Facility Name": s.get("FACNAME", "").astype(str).str.strip(), "License Number": s["FACID"],
            "Address": s["Address"].astype(str).str.strip(), "City": s["City"].astype(str).str.strip(),
            "Zip": s["Zip"], "County": s["County"].astype(str).str.strip().str.upper(),
            "License Date": s.get("LICENSE_DATE", "").astype(str).str.replace(r"\s+12:00:00 AM", "", regex=True),
            "Facility Type": s["FACTYPE"].str.title(), "Phone": s.get("Phone", "")})
        p = ddir / f"{stem}.csv"
        c.to_csv(p, index=False)
        out.append(p)
        log.info(f"GA feed: {len(c)} rows of type {ftype} -> {p.name}")
    return out


# ------------------------------------------------------------------ CMS PDC
def pdc(spec, cfg, log, dest: Path) -> Path:
    """CMS Provider Data Catalog dataset: resolve the current CSV link via the metastore API."""
    meta = _get(PDC_META.format(id=spec["dataset_id"]), cfg, log, ua=False).json()
    url = meta["distribution"][0]["downloadURL"]
    log.info(f"{spec['dataset_id']}: '{meta.get('title')}' modified {meta.get('modified')} -> {url.rsplit('/', 1)[-1]}")
    r = _get(url, cfg, log, stream=True, ua=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest


# ------------------------------------------------------------------ CMS data-api
def _api_rows(dataset_id, cfg, log, state_param=None, state=None, size=5000) -> list[dict]:
    rows, offset = [], 0
    while True:
        params = {"size": size, "offset": offset}
        if state_param and state:
            params[f"filter[{state_param}]"] = state
        batch = _get(API_DATA.format(id=dataset_id), cfg, log, params=params).json()
        rows += batch
        if len(batch) < size:
            return rows
        offset += size


def cms_api(spec, cfg, log, state: str, dest: Path) -> Path:
    rows = _api_rows(spec["dataset_id"], cfg, log, spec.get("state_param"), state)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"data-api dataset {spec['dataset_id']} returned no rows for {state}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    log.info(f"{spec['dataset_id']}: {len(df)} rows for {state}")
    return dest


def pecos(spec, cfg, log, state: str, dest: Path) -> Path:
    """PECOS enrollments (state-filtered) joined to All-Owners -> one row per owner, keyed by CCN/NPI."""
    enr = pd.DataFrame(_api_rows(spec["enroll_id"], cfg, log, spec.get("state_param", "ENROLLMENT STATE"), state))
    if enr.empty:
        raise ValueError(f"no PECOS enrollments for {state}")
    own = pd.DataFrame(_api_rows(spec["owners_id"], cfg, log))
    # how many Medicare providers of this type each owner (by PECOS associate id) is tied to NATIONALLY;
    # state-level counts hide corporate parents (e.g. a private-equity fund with 68 agencies, 5 in one state)
    natl = own.groupby("ASSOCIATE ID - OWNER")["ENROLLMENT ID"].nunique()
    own = own[own["ENROLLMENT ID"].isin(set(enr["ENROLLMENT ID"]))].copy()
    own["natl"] = own["ASSOCIATE ID - OWNER"].map(natl)
    j = own.merge(enr, on="ENROLLMENT ID", how="inner", suffixes=("_own", ""))
    person = (j["FIRST NAME - OWNER"].fillna("") + " " + j["LAST NAME - OWNER"].fillna("")).str.strip()
    out = pd.DataFrame({
        "name": j["ORGANIZATION NAME"], "dba": j["DOING BUSINESS AS NAME"], "ccn": j["CCN"], "npi": j["NPI"],
        "address": (j["ADDRESS LINE 1"].fillna("") + " " + j["ADDRESS LINE 2"].fillna("")).str.strip(), "city": j["CITY"], "state": j["STATE"], "zip": j["ZIP CODE"],
        "formation_date": j["INCORPORATION DATE"],
        "officer_name": person.where(person != "", j["ORGANIZATION NAME - OWNER"]),
        "officer_title": j["ROLE TEXT - OWNER"].fillna("") + " " + j["TITLE - OWNER"].fillna(""),
        "ownership_pct": j["PERCENTAGE OWNERSHIP"], "enrollment_id": j["ENROLLMENT ID"],
        "owner_id": j["ASSOCIATE ID - OWNER"], "owner_national_providers": j["natl"],
        "officer_middle": j["MIDDLE NAME - OWNER"]})
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    log.info(f"PECOS: {len(enr)} enrollments, {len(out)} owner rows for {state}")
    return dest
