"""Turn resolved source files into canonical per-source frames."""
from __future__ import annotations
import pandas as pd
from .loaders import read_table, canon, hkey
from .normalize import norm_address, clean_str


_TYPE_FROM_STEM = {"hha": "Home Health Agency", "phcp": "Private Home Care Provider"}


def load_source(res, spec, cfg, state, sector) -> pd.DataFrame | None:
    """Load one source; sources with several files (res.paths) are concatenated."""
    if res.path is None:
        return None
    paths = res.paths or [res.path]
    if len(paths) == 1:
        return _load_one(paths[0], res, spec, cfg, state, sector, tag=False)
    parts = [_load_one(p, res, spec, cfg, state, sector, tag=True) for p in paths]
    parts = [x for x in parts if x is not None and len(x)]
    return pd.concat(parts, ignore_index=True) if parts else None


def _load_one(path, res, spec, cfg, state, sector, tag) -> pd.DataFrame | None:
    role = spec["role"]
    if role in ("leie", "sam", "press"):          # list-style files keep their own headers
        df = read_table(path)
        df.columns = [hkey(c) for c in df.columns]
        for c, d in (("_src", res.source), ("_file", str(path)), ("_dl_date", res.date)):
            df[c] = d
        df["_rec_id"] = [f"row{i + 2}" for i in range(len(df))]
        return df
    raw = read_table(path)
    if role == "cms_quality" and "measure_code" in {hkey(c) for c in raw.columns}:
        df = _pivot_quality(raw, state)
    else:
        df = canon(raw, cfg["columns"])
    if df.empty:
        return df
    df["_src"] = res.source
    df["_file"] = str(path)
    df["_dl_date"] = res.date
    df["_rec_id"] = [(f"{path.stem}:" if tag else "") + f"row{i + 2}" for i in range(len(df))]
    if role == "license":
        if "facility_type" not in df:
            stem = path.stem.lower().replace("license", "").strip("_- ")
            df["facility_type"] = _TYPE_FROM_STEM.get(stem, stem.upper()) if stem else ""
    if role in ("license", "cms_provider", "cms_util", "cms_quality"):
        if "state" in df:
            df = df[df["state"].str.upper().isin(["", state])].copy()
        df["state"] = state
        if role == "cms_util":
            if "smry_ctgry" in df:                       # drop NATION / STATE total rows
                df = df[df["smry_ctgry"].str.upper() == "PROVIDER"].copy()
            if "year" in df:
                key = "ccn" if "ccn" in df else "npi"
                df = df.sort_values("year").drop_duplicates(key, keep="last")
            if sector == "hospice" and "service_days" in df and "episodes" in df:
                df["avg_los"] = df["service_days"] / df["episodes"].where(df["episodes"] > 0)
            if "neoplasm_pct" in df and "noncancer_share" not in df:
                df["noncancer_share"] = 100 - df["neoplasm_pct"]
    if role == "license":
        words = cfg["clean"]["closed_status_words"]
        st = df.get("status", pd.Series("", index=df.index)).str.upper()
        closed = st.map(lambda s: any(w in s for w in words))
        if "close_date" not in df:
            df["close_date"] = pd.NaT
        df["is_closed"] = closed | df["close_date"].notna()
    return df


def add_address_fields(df: pd.DataFrame, cfg) -> pd.DataFrame:
    markers = cfg["clean"]["virtual_markers"]
    rows = [norm_address(a, z, markers) for a, z in zip(df.get("address", ""), df.get("zip", ""))]
    adf = pd.DataFrame(rows, index=df.index)
    for c in adf.columns:
        df[c] = adf[c]
    if "virtual_flag" in df:
        vf = df["virtual_flag"].str.upper().isin(["Y", "YES", "TRUE", "1"])
        df["is_virtual"] = df["is_virtual"] | vf
    return df


_QUALITY_MEASURES = {"H_011_01_OBSERVED": "final_days_visit_pct", "Pct_Pts_w_Cancer": "cancer_pct"}


def _pivot_quality(raw: pd.DataFrame, state: str) -> pd.DataFrame:
    """CMS Hospice - Provider Data is long (one row per measure); pivot the measures we use."""
    raw = raw.copy()
    raw.columns = [hkey(c) for c in raw.columns]
    raw = raw[raw["state"].str.upper() == state]
    lut = {k.lower(): v for k, v in _QUALITY_MEASURES.items()}
    raw["_m"] = raw["measure_code"].str.lower().map(lut)
    raw = raw[raw["_m"].notna()].copy()
    raw["score"] = raw["score"].map(to_num)
    wide = raw.pivot_table(index="cms_certification_number_ccn", columns="_m", values="score", aggfunc="first")
    wide = wide.reset_index().rename(columns={"cms_certification_number_ccn": "ccn"})
    names = raw.drop_duplicates("cms_certification_number_ccn").set_index("cms_certification_number_ccn")["facility_name"]
    wide["name"] = wide["ccn"].map(names)
    wide["state"] = state
    if "cancer_pct" in wide:
        wide["noncancer_share"] = 100 - wide["cancer_pct"]
    return wide
