"""Turn resolved source files into canonical per-source frames."""
from __future__ import annotations
import pandas as pd
from .loaders import read_table, canon, hkey
from .normalize import norm_address, clean_str


def load_source(res, spec, cfg, state, sector) -> pd.DataFrame | None:
    if res.path is None:
        return None
    role = spec["role"]
    if role in ("leie", "sam", "press"):          # list-style files keep their own headers
        df = read_table(res.path)
        df.columns = [hkey(c) for c in df.columns]
        for c, d in (("_src", res.source), ("_file", str(res.path)), ("_dl_date", res.date)):
            df[c] = d
        df["_rec_id"] = [f"row{i + 2}" for i in range(len(df))]
        return df
    df = canon(read_table(res.path), cfg["columns"])
    if df.empty:
        return df
    df["_src"] = res.source
    df["_file"] = str(res.path)
    df["_dl_date"] = res.date
    df["_rec_id"] = [f"row{i + 2}" for i in range(len(df))]       # 1-based incl. header row
    if role in ("license", "cms_provider", "cms_util", "cms_quality"):
        if "state" in df:
            df = df[df["state"].str.upper().isin(["", state])].copy()
        df["state"] = state
        if "year" in df and role == "cms_util":
            key = "ccn" if "ccn" in df else "npi"
            df = df.sort_values("year").drop_duplicates(key, keep="last")
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
