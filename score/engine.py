"""Scoring engine: sector/state are parameters; flags + weights come from config."""
from __future__ import annotations
import pandas as pd
from .flags import FLAGS, _ppb
from .peers import peer_stats


def score_sector(df: pd.DataFrame, sector: str, cfg: dict) -> pd.DataFrame:
    """df = providers of ONE sector and ONE state (peer percentiles are per state)."""
    df = df.copy()
    sc = cfg["scoring"][sector]
    ctx = {"as_of": cfg["_as_of"], "min_peer": cfg["scoring"]["min_peer_group"]}
    score = pd.Series(0.0, index=df.index)
    names = []
    for fl in FLAGS[sector]:
        mask, detail = fl.fn(df, sc["thresholds"], ctx)
        strength = mask.fillna(0).astype(float).clip(0, 1)          # booleans -> 0/1; graded flags 0.5 or 1
        df[f"flag_{fl.name}"] = (strength > 0).astype(int)
        df[f"detail_{fl.name}"] = detail.fillna("")
        score += strength * sc["weights"].get(fl.name, 1)
        names.append(fl.name)
    df["score"] = score
    df["n_flags"] = df[[f"flag_{n}" for n in names]].sum(axis=1)
    df["flags_triggered"] = df.apply(lambda r: "; ".join(n for n in names if r[f"flag_{n}"]), axis=1)
    mx = sum(sc["weights"].get(n, 1) for n in names)
    df["score_pct_of_max"] = (100 * df["score"] / mx).round(1)

    # estimated dollar exposure: excess payment/beneficiary over peer median x beneficiaries
    ppb = _ppb(df)
    st = peer_stats(df, ppb, ctx["min_peer"])
    ex = ((ppb - st["median"].astype(float)).clip(lower=0) * df["beneficiaries"].astype(float)).astype(float).round(0)
    df["peer_median_payment_per_bene"] = st["median"].astype(float)
    df["payment_per_bene"] = ppb
    df["est_exposure_usd"] = ex.where(df["n_flags"] > 0)

    # rank: known cases excluded from target ranking
    elig = (~df["known_case"]) & (df["score"] >= cfg["scoring"]["min_score_for_lead"])
    order = df[elig].sort_values(["score", "est_exposure_usd"], ascending=False, na_position="last")
    df["rank"] = pd.NA
    df.loc[order.index, "rank"] = range(1, len(order) + 1)
    return df


def rerank(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recompute rank after known-case changes (known cases are excluded from the target ranking)."""
    df = df.copy()
    elig = (~df["known_case"]) & (df["score"] >= cfg["scoring"]["min_score_for_lead"])
    order = df[elig].sort_values(["score", "est_exposure_usd"], ascending=False, na_position="last")
    df["rank"] = pd.NA
    df.loc[order.index, "rank"] = range(1, len(order) + 1)
    return df
