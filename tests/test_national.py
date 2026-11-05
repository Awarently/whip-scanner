import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exclusion.national import match_exclusions, role_class


def _leie():
    base = dict(BUSNAME="", GENERAL="X", SPECIALTY="Y", CITY="AUSTIN", ZIP="1", DOB="19700101", WAIVERDATE="0", NPI="0")
    return pd.DataFrame([
        dict(base, LASTNAME="ROE", FIRSTNAME="ALEX", MIDNAME="QUINN", STATE="TX", EXCLTYPE="1128a1", EXCLDATE="20200101", REINDATE="0"),
        dict(base, LASTNAME="DOE", FIRSTNAME="PAT", MIDNAME="", STATE="TX", EXCLTYPE="1128a1", EXCLDATE="20100101", REINDATE="20150101")])


def _row(name, mid, state, since, ccn, title="5% OR GREATER DIRECT OWNERSHIP INTEREST"):
    return dict(name="AGENCY " + ccn, ccn=ccn, npi="", state=state, city="C", officer_name=name, officer_title=title,
                ownership_pct="100", owner_id="id" + ccn, owner_since=since, officer_middle=mid, owner_name_count="1",
                sector="home_health")


def test_tiers_and_exclusions():
    own = pd.DataFrame([_row("ALEX ROE", "Q", "TX", "2021-05-01", "1"),      # A: middle agrees, same state, joined after exclusion
                        _row("ALEX ROE", "R", "TX", "2021-05-01", "2"),      # conflicting middle initial: ruled out
                        _row("ALEX ROE", "", "TX", "2019-05-01", "3"),       # B: middle unknown, same state
                        _row("ALEX ROE", "Q", "FL", "2021-05-01", "4"),      # C: middle agrees, other state
                        _row("PAT DOE", "", "TX", "2021-05-01", "5")])       # reinstated exclusion: ignored
    r = match_exclusions(own, _leie(), pd.DataFrame({"ccn": ["1"], "cert_date": ["2021-09-01"]})).set_index("ccn")
    assert set(r.index) == {"1", "3", "4"}
    assert r.loc["1", "tier"] == "A" and r.loc["1", "owner_after_exclusion"] and r.loc["1", "cert_after_exclusion"]
    assert r.loc["3", "tier"] == "B" and not r.loc["3", "owner_after_exclusion"]
    assert r.loc["4", "tier"] == "C"
    assert r["priority_score"].idxmax() == "1"
    assert abs(r.loc["1", "months_exclusion_to_cert"] - 20.0) < 0.6      # cert 2021-09 vs exclusion 2020-01


def test_role_class():
    assert role_class("5% OR GREATER INDIRECT OWNERSHIP INTEREST") == "owner"
    assert role_class("CORPORATE OFFICER CEO") == "officer"
    assert role_class("W-2 MANAGING EMPLOYEE ADMINISTRATOR") == "manager"
    assert role_class("CORPORATE DIRECTOR BOARD MEMBER") == "board"
