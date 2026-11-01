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
