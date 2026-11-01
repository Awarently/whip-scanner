"""Peer comparison: percentile rank within county peers (if >= min_peer), else state."""
from __future__ import annotations
import pandas as pd


def peer_stats(df: pd.DataFrame, values: pd.Series, min_peer: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "state": df["state"],
                      "county": df["county"].where(df["county"].fillna("").astype(str).str.strip() != "")})
    st_pct = d.groupby("state")["v"].rank(pct=True) * 100
    st_med = d.groupby("state")["v"].transform("median")
    st_n = d.groupby("state")["v"].transform("count")
    cg = d.dropna(subset=["county"]).groupby(["state", "county"])["v"]
    c_pct = (cg.rank(pct=True) * 100).reindex(d.index)
    c_med = cg.transform("median").reindex(d.index)
    c_n = cg.transform("count").reindex(d.index)
    use_c = c_n.fillna(0) >= min_peer
    out = pd.DataFrame({
        "pct": c_pct.where(use_c, st_pct),
        "median": c_med.where(use_c, st_med),
        "n": c_n.where(use_c, st_n),
        "group": pd.Series(["county" if u else "state" for u in use_c], index=d.index),
    })
    for c in ("pct", "median", "n"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out.loc[d["v"].isna(), ["pct", "median", "n"]] = float("nan")
    return out


def desc(v, st, i, fmt="{:,.2f}", unit="") -> str:
    p = st.at[i, "pct"]
    return (f"{fmt.format(v)}{unit} = {p:.0f}th percentile of {int(st.at[i, 'n'])} {st.at[i, 'group']} peers "
            f"(peer median {fmt.format(st.at[i, 'median'])}{unit})")
