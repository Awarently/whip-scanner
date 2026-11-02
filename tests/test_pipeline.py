import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import load_config
from clean.normalize import norm_address, norm_name, person_key
from main import run
from tools.make_synthetic import generate


def test_address_normalisation():
    a = norm_address("100 Main Street Suite 200", "33101-1234")
    b = norm_address("100 MAIN ST., STE 200", "33101")
    assert a["addr_key"] == b["addr_key"]
    assert norm_address("PO Box 4412", "33102")["is_virtual"]
    assert not norm_address("1 Oak Ave", "33102")["is_virtual"]


def test_name_keys():
    assert norm_name("Foo Health, L.L.C.") == norm_name("FOO HEALTH LLC")
    assert person_key("Quill, Zed") == person_key("Zed Q. Quill")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    generate(root)
    cfg = load_config()
    cfg["paths"].update({"manual": root / "manual", "raw": root / "raw", "cache": root / "cache", "out": root / "out"})
    cfg["fetch"]["nppes_enabled"] = False
    comb, clusters, outdir = run(cfg, ["FL"], ["home_health", "hospice"], offline=True)
    return comb.set_index("name"), clusters, outdir


def flags(res, name):
    return set(res.at[name, "flags_triggered"].split("; "))


def test_expected_flags(result):
    r, _, _ = result
    assert "shared_address" in flags(r, "Suite Twin One LLC")
    assert "shared_person" in flags(r, "Owner Share Beta LLC")
    assert "shared_person" in flags(r, "Owner Share Alpha LLC")          # via corp record
    assert {"new_license_high_volume", "high_payment_per_bene"} <= flags(r, "Brand New Fast Billing LLC")
    assert "high_payment_per_bene" in flags(r, "Extreme Payment Outlier LLC")
    assert "license_churn" in flags(r, "New Agency Reopened LLC")
    assert "virtual_address" in flags(r, "Mailbox Agency LLC")
    assert "owner_exclusion_match" in flags(r, "Owner Excluded Match LLC")
    assert "linked_known_case" in flags(r, "Neighbor Of Known LLC")
    assert "linked_known_case" in flags(r, "Press Linked Owner LLC")
    assert "sanctions_or_failed_inspection" in flags(r, "Sanctioned Agency LLC")
    assert {"long_length_of_stay", "frequent_cap", "low_final_days_visits"} <= flags(r, "Long Stay Hospice LLC")
    assert "new_hospice_high_billing" in flags(r, "New Hospice Big Billing LLC")
    assert "cross_sector_network" in flags(r, "Cross Sector Hospice LLC")
    assert "cross_sector_network" in flags(r, "Cross Sector Home Health LLC")


def test_known_case_excluded_from_ranking(result):
    r, _, _ = result
    k = r.loc["Already Convicted Agency LLC"]
    assert k["known_case"] and pd.isna(k["rank"])
    assert r["rank"].dropna().is_unique


def test_siblings_not_merged(result):
    r, _, _ = result
    assert "Suite Twin One LLC" in r.index and "Suite Twin Two LLC" in r.index


def test_outputs(result):
    _, clusters, out = result
    for p in ("combined/red_flags.csv", "combined/clusters.csv", "combined/report.html", "run_log.txt",
              "FL/home_health/red_flags.csv", "FL/hospice/report.html"):
        assert (out / p).exists(), p
    assert any((out / "combined/evidence").glob("*.md"))
    assert clusters["cross_sector"].any()


@pytest.fixture(scope="module")
def ga(tmp_path_factory):
    root = tmp_path_factory.mktemp("ga")
    generate(root)
    cfg = load_config()
    cfg["paths"].update({"manual": root / "manual", "raw": root / "raw", "cache": root / "cache", "out": root / "out"})
    cfg["fetch"]["nppes_enabled"] = False
    comb, _, out = run(cfg, ["GA"], ["home_health", "hospice"], offline=True)
    return comb.set_index("name"), out


def test_ga_loads_both_license_types(ga):
    r, out = ga
    hh = r[r["sector"] == "home_health"]
    assert set(hh["facility_type"]) == {"Home Health Agency", "Private Home Care Provider"}
    assert len(hh) == 15 + 15 + 5
    assert all(r["state"] == "GA")
    log = (out / "run_log.txt").read_text()
    assert "license_hh_GA" in log and "license_hha.csv" in log and "license_phcp.csv" in log


def test_ga_flags(ga):
    r, _ = ga
    assert "shared_address" in flags(r, "Ga Suite Share HHA Inc")           # across HHA + PHCP files
    assert "shared_address" in flags(r, "Ga Suite Share Home Care LLC")
    assert "shared_person" in flags(r, "Ga Owner Share HHA Inc")
    assert "virtual_address" in flags(r, "Ga Mailbox Home Care LLC")
    assert "cross_sector_network" in flags(r, "Ga Hospice Owner Share Inc")
    assert (r["provider_id"].str.startswith("GA-")).all()


def test_chain_owner_downweighted(result):
    r, clusters, _ = result
    chain = r[r.index.str.startswith("Sample Home Health 0") & (r["chain_owner_count"] > 0)]
    assert len(chain) == 8                                   # 8 agencies under one corporate parent
    assert not chain["flags_triggered"].str.contains("shared_person").any()
    assert chain["flags_triggered"].str.contains("chain_ownership").all()
    assert chain["score"].max() <= 0.5                       # chain flag alone is a low-weight signal
    assert not clusters["provider_names"].str.contains("Sample Home Health 00").any()
    assert "shared_person" in flags(r, "Owner Share Beta LLC")   # small 2-agency network still flagged


def test_relicensed_agency_not_new(result):
    r, _, _ = result
    row = r.loc["Relicensed Old Agency LLC"]
    assert row["operating_since"].year == 2010                       # CMS certification date wins
    assert "new_license_high_volume" not in flags(r, "Relicensed Old Agency LLC")
    assert "new_license_high_volume" in flags(r, "Brand New Fast Billing LLC")


def test_graph_clusters_have_hubs(result):
    r, clusters, out = result
    assert {"hub_node", "hub_type", "hub_degree", "bridge_nodes", "n_owner_nodes", "n_address_nodes"} <= set(clusters.columns)
    cross = clusters[clusters["cross_sector"]].iloc[0]
    assert "CROSS OWNER" in cross["hub_node"] and cross["hub_degree"] >= 2
    assert clusters.iloc[0]["cross_sector"]                                  # cross-sector networks listed first
    suite = clusters[clusters["provider_names"].str.contains("Suite Twin One")].iloc[0]
    assert suite["hub_type"] == "address" and "100 MAIN ST" in suite["hub_node"]
    assert "Largest networks" in (out / "combined" / "report.html").read_text()


def test_graded_outlier_flags(result):
    r, _, _ = result
    strong = r.loc["Brand New Fast Billing LLC", "detail_high_payment_per_bene"]
    assert "STRONG" in strong and "x median" in strong and "robust z" in strong
    mild = r.loc["Mild Payment Outlier LLC"]
    assert mild["flag_high_payment_per_bene"] == 1 and "weak signal" in mild["detail_high_payment_per_bene"]
    assert mild["score"] == 1.5                                   # weight 3 x 0.5 for a weak signal


def test_rank_alone_is_not_enough():
    import numpy as np
    from score.flags import HOME_HEALTH
    n = 40
    df = pd.DataFrame({"state": "FL", "county": "X", "beneficiaries": 100.0,
                       "total_payment": [100000.0] * (n - 1) + [110000.0]})   # top of the group but only 1.1x median
    fl = next(f for f in HOME_HEALTH if f.name == "high_payment_per_bene")
    cfg = load_config()["scoring"]["home_health"]["thresholds"]
    strength, _ = fl.fn(df, cfg, {"min_peer": 10, "as_of": None})
    assert (strength == 0).all()
    df["beneficiaries"] = 10.0                                            # tiny caseload is ignored too
    df["total_payment"] = [100000.0] * (n - 1) + [300000.0]
    strength, _ = fl.fn(df, cfg, {"min_peer": 10, "as_of": None})
    assert (strength == 0).all()


def test_national_chain_owner_is_chain_even_with_few_local_providers():
    from match.network import build_edges, add_people
    cfg = load_config()
    df = pd.DataFrame({"provider_id": ["a", "b"], "sector": "home_health", "addr_key": "", "is_virtual": False,
                       "owners": ["ADVENT FUND LP", "ADVENT FUND LP"], "administrator": "", "auth_official": ""})
    df = add_people(df, cfg)
    edges, chain = build_edges(df, cfg)
    assert len(edges) == 1 and not chain                                   # two local providers: a small network
    key = next(iter(df.at[0, "people"]))
    df["owner_natl"] = [{key: 68}, {key: 68}]
    edges, chain = build_edges(df, cfg)
    assert not edges and chain and chain[0]["size"] == 68                   # 68 providers nationally: chain


def test_middle_initial_conflict_rules_out_match():
    from filters.known_case import _loc_ok
    locs = [("GA", "30301", "B"), ("TX", "75001", "")]
    assert _loc_ok(locs, "GA", "", False, mid="L") == []          # Amy L vs Amy B: different people
    assert _loc_ok(locs, "GA", "", False, mid="B") == [0]
    assert _loc_ok(locs, "GA", "", False, mid="") == [0]          # middle unknown on our side: cannot rule out
    assert _loc_ok(locs, "TX", "", False, mid="Z") == [1]         # no middle on the list entry: cannot rule out
    assert _loc_ok(locs, "FL", "", False) == []                   # other state


def test_owner_exclusion_strength_and_review_file(result):
    r, _, out = result
    row = r.loc["Owner Excluded Match LLC"]
    assert row["owner_exclusion_strength"] == 0.5          # middle initial unknown, name not ambiguous
    assert row["score"] >= 2.5                              # weight 5 x 0.5
    rv = pd.read_csv(out / "combined" / "identity_review.csv")
    assert {"owner", "leie_excl_type", "middle_initial_status", "match_strength"} <= set(rv.columns)
    assert "Owner Excluded Match LLC" in set(rv["provider"])
