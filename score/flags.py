"""Red-flag definitions. One engine; each sector registers its own flag functions.

A flag function receives (df, thresholds, ctx) and returns (mask, detail) Series indexed like df.
`uses` lists the source ids whose records are cited as evidence for that flag.
Adding a sector (DME, lab testing) = new FLAGS entry + config block; engine is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from .peers import peer_stats, desc


@dataclass
class Flag:
    name: str
    label: str
    fn: Callable
    uses: tuple = ()


def _blank(df):
    return pd.Series(False, index=df.index), pd.Series("", index=df.index)


def _age_months(df, asof):
    return ((pd.Timestamp(asof) - df["license_date"]).dt.days / 30.4375)


def _ppb(df):
    return df["total_payment"] / df["beneficiaries"].where(df["beneficiaries"] > 0)


def _eppb(df):
    return df["episodes"] / df["beneficiaries"].where(df["beneficiaries"] > 0)


# ---------------------------------------------------------------- common flags
def f_shared_address(df, t, ctx):
    col = "n_same_addr_same_sector"
    m = (df[col] + 1) >= t["shared_address_min"]
    d = df.apply(lambda r: f"{int(r[col]) + 1} agencies at normalised address '{r['street_norm']}"
                           f"{' STE ' + r['unit'] if r['unit'] else ''}' ({r['zip5']}); {r['link_detail']}", axis=1)
    return m, d.where(m, "")


def f_shared_person(df, t, ctx):
    col = "n_shared_people_same_sector"
    m = (df[col] + 1) >= t["shared_person_min"]
    return m, df["link_detail"].where(m, "").map(lambda s: ("shared owner/administrator/official: " + s) if s else "")


def f_shared_addr_or_owner_any(df, t, ctx):
    m = (df["n_same_addr_all"] > 0) | (df["n_shared_people_all"] > 0)
    return m, df["link_detail"].where(m, "")


def f_new_high_volume(df, t, ctx):
    age = _age_months(df, ctx["as_of"])
    pb = peer_stats(df, df["beneficiaries"], ctx["min_peer"])
    pp = peer_stats(df, df["total_payment"], ctx["min_peer"])
    thr = t["new_license_volume_pct"]
    m = (age < t["new_license_months"]) & ((pb["pct"].astype(float) >= thr) | (pp["pct"].astype(float) >= thr))
    m = m.fillna(False)
    d = pd.Series("", index=df.index)
    for i in df.index[m]:
        r = df.loc[i]
        d[i] = (f"licensed {r['license_date']:%Y-%m-%d} ({age[i]:.0f} months old); "
                f"beneficiaries {r['beneficiaries']:,.0f} ({pb.at[i, 'pct']:.0f}th pct), "
                f"payment ${r['total_payment']:,.0f} ({pp.at[i, 'pct']:.0f}th pct)")
    return m, d


def f_virtual(df, t, ctx):
    m = df["is_virtual"].fillna(False).astype(bool)
    return m, df["address"].fillna("").map(lambda a: f"mailbox/virtual-office style address: '{a}'").where(m, "")


def f_owner_excl(df, t, ctx):
    m = df["owner_exclusion_match"].astype(bool)
    return m, df["owner_exclusion_detail"].where(m, "")


def f_linked_known(df, t, ctx):
    m = df["linked_known_case"].astype(bool)
    return m, df["linked_known_case_detail"].where(m, "")


def f_chain(df, t, ctx):
    m = df["chain_owner_count"] > 0
    return m, ("chain/corporate ownership (low weight): " + df["chain_owner_detail"]).where(m, "")


def f_cross_sector(df, t, ctx):
    m = df["cluster_cross_sector"].astype(bool)
    return m, df.apply(lambda r: f"cluster {r['cluster_id']} spans home health + hospice ({int(r['cluster_size'])} providers)", axis=1).where(m, "")


def f_churn(df, t, ctx):
    m = df["license_churn"].astype(bool)
    return m, df["churn_detail"].where(m, "")


# ---------------------------------------------------------------- home health
def _top_pct_flag(series_fn, label, fmt, unit=""):
    def fn(df, t, ctx):
        s = series_fn(df)
        st = peer_stats(df, s, ctx["min_peer"])
        m = (st["pct"].astype(float) >= t["top_pct"]).fillna(False)
        d = pd.Series([f"{label}: " + desc(s[i], st, i, fmt, unit) if m[i] else "" for i in df.index], index=df.index)
        return m, d
    return fn


def f_sanction(df, t, ctx):
    sf = df["sanction_flag"].fillna("").astype(str).str.upper().isin(["Y", "YES", "TRUE", "1", "T"])
    insp = pd.to_numeric(df["inspection_failed"], errors="coerce").fillna(0) > 0
    m = sf | insp
    d = df.apply(lambda r: "; ".join(x for x in (
        "state sanction/enforcement action on record" if str(r["sanction_flag"]).upper() in ("Y", "YES", "TRUE", "1", "T") else "",
        f"failed inspections/deficiencies: {r['inspection_failed']}" if pd.to_numeric(r["inspection_failed"], errors="coerce") > 0 else "") if x), axis=1)
    return m, d.where(m, "")


HOME_HEALTH = [
    Flag("shared_address", "2+ agencies at same address/suite", f_shared_address, ("license", "cms_provider")),
    Flag("shared_person", "Shared owner/administrator/official", f_shared_person, ("license", "corp", "nppes")),
    Flag("new_license_high_volume", "License <24 mo with high volume", f_new_high_volume, ("license", "cms_util")),
    Flag("high_payment_per_bene", "Payment per beneficiary top 5%", _top_pct_flag(_ppb, "payment/beneficiary", "${:,.0f}"), ("cms_util",)),
    Flag("high_episodes_per_bene", "Episodes per beneficiary top 5%", _top_pct_flag(_eppb, "episodes/beneficiary", "{:,.2f}"), ("cms_util",)),
    Flag("license_churn", "Closure then new agency (same owner/address) <12 mo", f_churn, ("license",)),
    Flag("virtual_address", "Virtual/mailbox/PO box address", f_virtual, ("license", "cms_provider")),
    Flag("owner_exclusion_match", "Owner/official fuzzy-matches LEIE/SAM", f_owner_excl, ("leie", "sam", "corp", "nppes")),
    Flag("linked_known_case", "Linked to known case", f_linked_known, ("press", "leie", "sam")),
    Flag("sanctions_or_failed_inspection", "Sanctions or failed inspections", f_sanction, ("license",)),
    Flag("cross_sector_network", "Network spans home health + hospice", f_cross_sector, ("license", "corp")),
    Flag("chain_ownership", "Chain/corporate owner (many providers)", f_chain, ("corp",)),
]

# ---------------------------------------------------------------- hospice
def f_cap(df, t, ctx):
    cu = pd.to_numeric(df["cap_utilization_pct"], errors="coerce")
    cy = pd.to_numeric(df["cap_hit_years"], errors="coerce")
    m = (cu >= t["cap_utilization_pct"]) | (cy >= t["cap_hit_years"])
    d = df.apply(lambda r: f"aggregate cap utilisation {r['cap_utilization_pct']}%, years at/over cap {r['cap_hit_years']}", axis=1)
    return m.fillna(False), d.where(m.fillna(False), "")


def f_low_final(df, t, ctx):
    s = pd.to_numeric(df["final_days_visit_pct"], errors="coerce")
    st = peer_stats(df, s, ctx["min_peer"])
    m = (st["pct"].astype(float) <= t["final_days_bottom_pct"]).fillna(False)
    d = pd.Series([("visits in final days of life: " + desc(s[i], st, i, "{:,.1f}", "%")) if m[i] else "" for i in df.index], index=df.index)
    return m, d


HOSPICE = [
    Flag("long_length_of_stay", "Average length of stay top 5%", _top_pct_flag(lambda d: d["avg_los"], "average length of stay", "{:,.0f}", " days"), ("cms_util",)),
    Flag("high_live_discharge", "High live-discharge rate", _top_pct_flag(lambda d: d["live_discharge_pct"], "live-discharge rate", "{:,.1f}", "%"), ("cms_util",)),
    Flag("high_noncancer_share", "High non-cancer/dementia diagnosis share", _top_pct_flag(lambda d: d["noncancer_share"], "non-cancer/dementia share", "{:,.1f}", "%"), ("cms_util",)),
    Flag("frequent_cap", "Frequently hits aggregate cap", f_cap, ("cms_util",)),
    Flag("low_final_days_visits", "Low visit intensity in final days", f_low_final, ("cms_quality",)),
    Flag("new_hospice_high_billing", "New hospice with high billing", f_new_high_volume, ("license", "cms_util")),
    Flag("shared_address_or_owner", "Shared address/owner with hospice or HHA", f_shared_addr_or_owner_any, ("license", "corp", "nppes")),
    Flag("owner_exclusion_match", "Owner/official fuzzy-matches LEIE/SAM", f_owner_excl, ("leie", "sam", "corp", "nppes")),
    Flag("linked_known_case", "Linked to known case", f_linked_known, ("press", "leie", "sam")),
    Flag("cross_sector_network", "Network spans home health + hospice", f_cross_sector, ("license", "corp")),
    Flag("chain_ownership", "Chain/corporate owner (many providers)", f_chain, ("corp",)),
]

FLAGS = {"home_health": HOME_HEALTH, "hospice": HOSPICE}
