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
    return ((pd.Timestamp(asof) - df["operating_since"]).dt.days / 30.4375)


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
    m = ((age < t["new_license_months"]) & ((pb["pct"].astype(float) >= thr) | (pp["pct"].astype(float) >= thr))
         & (pd.to_numeric(df["beneficiaries"], errors="coerce") >= t["min_beneficiaries"]))
    m = m.fillna(False)
    d = pd.Series("", index=df.index)
    for i in df.index[m]:
        r = df.loc[i]
        cert = "" if pd.isna(r["cert_date"]) else f", CMS certified {r['cert_date']:%Y-%m-%d}"
        d[i] = (f"operating since {r['operating_since']:%Y-%m-%d} (state license {r['license_date']:%Y-%m-%d}{cert}) "
                f"= {age[i]:.0f} months old; "
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
def _outlier_flag(series_fn, label, fmt, unit, tkey, mode, needs_benes=True):
    """Graded peer-outlier flag. Fires only if the rank test AND a minimum size of gap AND a minimum
    caseload all hold, so a top-5% rank with a tiny real gap does not count.

      mode "ratio_high": value / peer median  >= weak (strength 0.5) or >= strong (1.0)
      mode "gap_high"  : value - peer median  >= weak / strong (percentage points)
      mode "ratio_low" : value / peer median  <= weak / strong (e.g. 0.67 / 0.5)
    """
    def fn(df, t, ctx):
        s = pd.to_numeric(series_fn(df), errors="coerce")
        st = peer_stats(df, s, ctx["min_peer"])
        weak, strong = t[tkey]
        if mode == "ratio_low":
            rank_ok = st["pct"] <= t["final_days_bottom_pct"]
            measure = s / st["median"]
            ok_w, ok_s = measure <= weak, measure <= strong
        else:
            rank_ok = st["pct"] >= t["top_pct"]
            measure = (s / st["median"]) if mode == "ratio_high" else (s - st["median"])
            ok_w, ok_s = measure >= weak, measure >= strong
        benes = pd.to_numeric(df["beneficiaries"], errors="coerce")
        gate = benes >= t["min_beneficiaries"]
        if not needs_benes:
            gate = gate | benes.isna()
        base = (rank_ok & gate).fillna(False)
        strength = pd.Series(0.0, index=df.index)
        strength[base & ok_w.fillna(False)] = 0.5
        strength[base & ok_s.fillna(False)] = 1.0
        detail = pd.Series("", index=df.index)
        for i in df.index[strength > 0]:
            gap = s[i] - st.at[i, "median"]
            z = st.at[i, "robust_z"]
            detail[i] = (f"{label}: {fmt.format(s[i])}{unit} = {st.at[i, 'pct']:.0f}th percentile of {int(st.at[i, 'n'])} "
                         f"{st.at[i, 'group']} peers; peer median {fmt.format(st.at[i, 'median'])}{unit}; "
                         f"{s[i] / st.at[i, 'median']:.2f}x median (gap {gap:+,.2f}{unit}); "
                         f"robust z {z:.1f}; {'STRONG' if strength[i] == 1 else 'weak'} signal "
                         f"(thresholds {weak}/{strong}), {int(benes[i]) if pd.notna(benes[i]) else 'n/a'} beneficiaries")
        return strength, detail
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
    Flag("high_payment_per_bene", "Payment per beneficiary top 5%", _outlier_flag(_ppb, "payment/beneficiary", "${:,.0f}", "", "payment_ratio", "ratio_high"), ("cms_util",)),
    Flag("high_episodes_per_bene", "Episodes per beneficiary top 5%", _outlier_flag(_eppb, "episodes/beneficiary", "{:,.2f}", "", "episodes_ratio", "ratio_high"), ("cms_util",)),
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


HOSPICE = [
    Flag("long_length_of_stay", "Average length of stay top 5%", _outlier_flag(lambda d: d["avg_los"], "average length of stay", "{:,.0f}", " days", "los_ratio", "ratio_high"), ("cms_util",)),
    Flag("high_live_discharge", "High live-discharge rate", _outlier_flag(lambda d: d["live_discharge_pct"], "live-discharge rate", "{:,.1f}", "%", "live_discharge_gap_pts", "gap_high"), ("cms_util",)),
    Flag("high_noncancer_share", "High non-cancer/dementia diagnosis share", _outlier_flag(lambda d: d["noncancer_share"], "non-cancer share", "{:,.1f}", "%", "noncancer_gap_pts", "gap_high"), ("cms_util",)),
    Flag("frequent_cap", "Frequently hits aggregate cap", f_cap, ("cms_util",)),
    Flag("low_final_days_visits", "Low visit intensity in final days", _outlier_flag(lambda d: d["final_days_visit_pct"], "visits in final days of life", "{:,.1f}", "%", "final_days_ratio", "ratio_low", needs_benes=False), ("cms_quality",)),
    Flag("new_hospice_high_billing", "New hospice with high billing", f_new_high_volume, ("license", "cms_util")),
    Flag("shared_address_or_owner", "Shared address/owner with hospice or HHA", f_shared_addr_or_owner_any, ("license", "corp", "nppes")),
    Flag("owner_exclusion_match", "Owner/official fuzzy-matches LEIE/SAM", f_owner_excl, ("leie", "sam", "corp", "nppes")),
    Flag("linked_known_case", "Linked to known case", f_linked_known, ("press", "leie", "sam")),
    Flag("cross_sector_network", "Network spans home health + hospice", f_cross_sector, ("license", "corp")),
    Flag("chain_ownership", "Chain/corporate owner (many providers)", f_chain, ("corp",)),
]

FLAGS = {"home_health": HOME_HEALTH, "hospice": HOSPICE}
