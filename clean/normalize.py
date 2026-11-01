"""Normalisation of addresses, names, people, dates and numbers."""
from __future__ import annotations
import re
import pandas as pd

_ABBR = {"STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "ROAD": "RD", "DRIVE": "DR", "LANE": "LN",
         "COURT": "CT", "HIGHWAY": "HWY", "PARKWAY": "PKWY", "CIRCLE": "CIR", "PLACE": "PL", "TRAIL": "TRL",
         "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "NORTHEAST": "NE", "NORTHWEST": "NW",
         "SOUTHEAST": "SE", "SOUTHWEST": "SW", "SUITE": "STE", "BUILDING": "BLDG", "FLOOR": "FL",
         "APARTMENT": "APT", "ROOM": "RM", "HWY.": "HWY", "US": "US"}
_UNIT_RE = re.compile(r"\b(?:STE|SUITE|UNIT|APT|APARTMENT|RM|ROOM|FLOOR|BLDG|BUILDING|LOT|SPC)\b\.?\s*#?\s*([A-Z0-9][A-Z0-9\-]*)"
                      r"|#\s*([A-Z0-9][A-Z0-9\-]*)")
_POBOX_RE = re.compile(r"\bP\.?\s?O\.?\s?BOX\b|\bPOST OFFICE BOX\b|\bPMB\b|\bMAIL\s?BOX\b|\bBOX\s+\d+\b|\bDRAWER\s+\d+\b")
_ORG_WORDS = {"LLC", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LP", "LLP", "PA", "PLLC", "PC",
              "HOLDINGS", "GROUP", "TRUST", "ENTERPRISES", "PARTNERS", "ASSOCIATES", "SERVICES", "HEALTH",
              "HEALTHCARE", "CARE", "HOSPICE", "AGENCY"}
_ORG_SUFFIX = {"LLC", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LP", "LLP", "PLLC", "PC", "PA", "THE", "OF"}
_PERSON_NOISE = {"JR", "SR", "II", "III", "IV", "MD", "DO", "RN", "DR", "MR", "MRS", "MS", "NP", "PHD"}


def clean_str(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()


def norm_address(street: str, zip_code: str = "", virtual_markers: list[str] | None = None) -> dict:
    raw = clean_str(street).upper()
    virtual = bool(_POBOX_RE.search(raw)) or any(m in raw for m in (virtual_markers or []))
    s = re.sub(r"[.,;]", " ", raw)
    unit = ""
    m = _UNIT_RE.search(s)
    if m:
        unit = (m.group(1) or m.group(2) or "").lstrip("0") or "0"
        s = (s[:m.start()] + " " + s[m.end():])
    toks = [_ABBR.get(t, t) for t in re.sub(r"[^A-Z0-9# \-]", " ", s).split()]
    street_norm = " ".join(toks)
    z5 = re.sub(r"\D", "", clean_str(zip_code))[:5]
    key = f"{street_norm}|{z5}" if street_norm else ""
    return {"street_norm": street_norm, "unit": unit, "zip5": z5, "street_key": key,
            "addr_key": f"{key}|{unit}" if key else "", "is_virtual": virtual}


def norm_name(name: str) -> str:
    s = clean_str(name).upper().replace("&", " AND ").replace(".", "")   # "L.L.C." -> "LLC"
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _ORG_SUFFIX]
    return " ".join(toks)


def is_org(name: str) -> bool:
    toks = set(re.sub(r"[^A-Z0-9 ]", " ", clean_str(name).upper().replace(".", "")).split())
    return bool(toks & _ORG_WORDS)


def person_key(name: str) -> str:
    """'ORG:<norm>' for companies, else 'FIRST LAST' (middle names/initials dropped)."""
    raw = clean_str(name).upper()
    if not raw:
        return ""
    if is_org(raw):
        return "ORG:" + norm_name(raw)
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first} {last}"
    toks = [t for t in re.sub(r"[^A-Z \-']", " ", raw).split() if t not in _PERSON_NOISE]
    toks = [t for t in toks if len(t) > 1] or toks
    if len(toks) < 2:
        return ""
    return f"{toks[0]} {toks[-1]}"


def split_people(s: str) -> list[str]:
    parts = re.split(r"\s*(?:;|\||/|\band\b)\s*", clean_str(s), flags=re.I)
    return [p for p in (x.strip() for x in parts) if p]


def to_date(x):
    s = clean_str(x)
    if not s:
        return pd.NaT
    return pd.to_datetime(s, errors="coerce")


def to_num(x):
    if isinstance(x, (int, float)):
        return x
    s = re.sub(r"[$,%\s]", "", clean_str(x))
    if s in ("", "-", "NA", "N/A", "*", "Not Available"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")
