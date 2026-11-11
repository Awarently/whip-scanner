import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filters.press import PressIndex, apply_press
from exclusion.national import annotate_press

REL = [{"title": "Two Arrested for Hospice Fraud Conspiracy", "url": "https://example.invalid/pr1", "date": "2021-12-02",
        "text": "Jane Q. Roe, 38, of Glendale, California, and John Doe were charged. Sunny Day Hospice in Van Nuys, California billed Medicare."},
       {"title": "Florida Man Sentenced", "url": "https://example.invalid/pr2", "date": "2020-01-01",
        "text": "Pat Smith of Miami, Florida was sentenced."}]


def test_person_and_org_search():
    idx = PressIndex(REL)
    h = idx.find_person("JANE ROE", "CA", "Glendale")
    assert h and h[0]["corroborated"] and h[0]["url"].endswith("pr1")      # middle initial between first and last is allowed
    assert idx.find_person("Jane Roe", "TX", "Dallas")[0]["corroborated"] is False   # name only: other state
    assert idx.find_person("Janet Roe") == []                               # different first name
    assert idx.find_org("Sunny Day Hospice LLC", "CA", "Van Nuys")[0]["city_match"]
    assert idx.find_org("Home Health Care Inc") == []                       # only generic words: not searched


def test_apply_press_marks_known_case_and_linked_owner():
    idx = PressIndex(REL)
    df = pd.DataFrame({"name": ["SUNNY DAY HOSPICE", "Quiet Agency"], "state": "CA", "city": ["Van Nuys", "Glendale"],
                       "score": [5.0, 4.0], "n_flags": [3, 3], "known_case": False, "known_case_reason": "",
                       "linked_known_case": False, "linked_known_case_detail": "",
                       "owner_info": [{}, {"JANE ROE": {"name": "Jane Roe", "role": "5% OR GREATER DIRECT OWNERSHIP INTEREST"}}]})
    cfg = {"press": {"check_top_n": 10}}
    out = apply_press(df, idx, cfg)
    assert out.loc[0, "known_case"] and "pr1" in out.loc[0, "press_provider_refs"]
    assert out.loc[1, "linked_known_case"] and "Jane Roe" in out.loc[1, "press_owner_refs"] and not out.loc[1, "known_case"]


def test_national_annotation():
    idx = PressIndex(REL)
    df = pd.DataFrame({"owner": ["JANE ROE", "PAT SMITH"], "provider": ["X", "Y"], "provider_state": ["CA", "TX"], "city": ["Glendale", "Austin"]})
    out = annotate_press(df, idx)
    assert list(out["already_public"]) == [True, False]
    state_only = annotate_press(pd.DataFrame({"owner": ["JANE ROE"], "provider": ["X"], "provider_state": ["CA"], "city": ["Fresno"]}), idx)
    assert not state_only.loc[0, "already_public"] and state_only.loc[0, "press_possible_state_only"]   # same state, other city: only "possible"
