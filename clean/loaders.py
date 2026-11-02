"""Read raw/manual files and map headers to canonical columns."""
from __future__ import annotations
import re
import pandas as pd
from .normalize import to_num, to_date, clean_str

NUMERIC = ["beneficiaries", "episodes", "total_payment", "avg_los", "live_discharge_pct", "noncancer_share",
           "cap_utilization_pct", "cap_hit_years", "final_days_visit_pct", "inspection_failed", "year",
           "service_days", "neoplasm_pct", "ownership_pct", "owner_national"]
DATES = ["license_date", "cert_date", "close_date", "formation_date", "enumeration_date"]


def hkey(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h).lower()).strip("_")


def read_table(path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def canon(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename to canonical columns (first alias present wins); drop everything else."""
    cols = {hkey(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon_name, alist in aliases.items():
        for a in alist:
            if a in cols:
                out[canon_name] = df[cols[a]]
                break
    for c in NUMERIC:
        if c in out:
            out[c] = out[c].map(to_num)
    for c in DATES:
        if c in out:
            out[c] = out[c].map(to_date)
    for c in out.columns:
        if c not in NUMERIC and c not in DATES:
            out[c] = out[c].map(clean_str)
    return out
