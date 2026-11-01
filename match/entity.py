"""Entity resolution: merge per-source frames into one row per provider.

Exact keys (CCN, NPI, license #) first, then fuzzy name + normalised address. A cluster may
hold at most ONE row per source, so look-alike sibling agencies at the same suite are never
merged into one (that would hide the shared-address signal).
"""
from __future__ import annotations
import hashlib
import pandas as pd
from rapidfuzz import fuzz
from clean.normalize import norm_name, clean_str

PRIORITY = ["license", "cms_provider", "cms_util", "cms_quality"]
CANON_COLS = ["name", "dba", "license_no", "facility_type", "npi", "ccn", "address", "city", "state", "zip", "county",
              "license_date", "close_date", "is_closed", "status", "owners", "administrator", "auth_official",
              "sanction_flag", "sanction_same_conduct", "inspection_failed", "virtual_flag", "beneficiaries",
              "episodes", "total_payment", "avg_los", "live_discharge_pct", "noncancer_share",
              "cap_utilization_pct", "cap_hit_years", "final_days_visit_pct",
              "street_norm", "unit", "zip5", "street_key", "addr_key", "is_virtual"]


class UF:
    def __init__(self, n, srcs):
        self.p = list(range(n))
        self.srcs = [{s} for s in srcs]

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.srcs[ra] & self.srcs[rb]:
            return False
        self.p[rb] = ra
        self.srcs[ra] |= self.srcs[rb]
        return True


def merge_sector(frames: dict[str, pd.DataFrame], roles: dict[str, str], state: str, sector: str,
                 cfg: dict) -> pd.DataFrame:
    parts = []
    for sid, df in frames.items():
        if df is None or df.empty or roles[sid] not in PRIORITY:
            continue
        d = df.copy()
        d["_prio"] = PRIORITY.index(roles[sid])
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    allr = pd.concat(parts, ignore_index=True).sort_values("_prio", kind="stable").reset_index(drop=True)
    for c in CANON_COLS:
        if c not in allr:
            allr[c] = pd.NA
    allr["_nname"] = allr["name"].map(norm_name)
    uf = UF(len(allr), allr["_src"])

    # exact keys
    for col in ("ccn", "npi", "license_no"):
        seen: dict[str, int] = {}
        for i, v in allr[col].items():
            v = clean_str(v)
            if not v:
                continue
            if v in seen:
                uf.union(seen[v], i)
            else:
                seen[v] = i

    # fuzzy: block by street_key, then by zip5
    m = cfg["match"]
    for col, thr in (("street_key", m["name_threshold"]), ("zip5", m["name_threshold_zip_only"])):
        blocks = allr[allr[col].fillna("") != ""].groupby(col).groups
        for _, idx in blocks.items():
            idx = list(idx)
            if len(idx) < 2 or len(idx) > 200:
                continue
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if allr.at[i, "_src"] == allr.at[j, "_src"]:
                        continue
                    if fuzz.token_sort_ratio(allr.at[i, "_nname"], allr.at[j, "_nname"]) >= thr:
                        uf.union(i, j)

    allr["_cluster"] = [uf.find(i) for i in range(len(allr))]
    recs = []
    for _, g in allr.groupby("_cluster", sort=False):
        row = {}
        for c in CANON_COLS:
            vals = g[c].dropna()
            vals = vals[vals.astype(str).str.strip() != ""] if len(vals) else vals
            row[c] = vals.iloc[0] if len(vals) else pd.NA
        row["is_closed"] = bool(g["is_closed"].fillna(False).astype(bool).any()) if "is_closed" in g else False
        row["sources"] = [{"source": r["_src"], "file": r["_file"], "download_date": r["_dl_date"],
                           "record_id": r["_rec_id"]} for _, r in g.iterrows()]
        row["match_method"] = "single-source" if len(g) == 1 else "ccn/npi/license or fuzzy name+address"
        recs.append(row)
    out = pd.DataFrame(recs)
    for c in ("name", "dba", "license_no", "facility_type", "npi", "ccn", "address", "city", "zip", "county", "status", "owners",
              "administrator", "auth_official", "sanction_flag", "sanction_same_conduct", "virtual_flag",
              "street_norm", "unit", "zip5", "street_key", "addr_key"):
        out[c] = out[c].astype(object).where(out[c].notna(), "")
    for c in ("inspection_failed", "beneficiaries", "episodes", "total_payment", "avg_los", "live_discharge_pct",
              "noncancer_share", "cap_utilization_pct", "cap_hit_years", "final_days_visit_pct"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["is_virtual"] = out["is_virtual"].fillna(False).astype(bool)
    for c in ("license_date", "close_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["state"] = state
    out["sector"] = sector
    pre = {"home_health": "HH", "hospice": "HO"}[sector]

    def pid(r):
        for k in ("ccn", "license_no", "npi"):
            if clean_str(r[k]):
                return f"{state}-{pre}-{clean_str(r[k])}"
        h = hashlib.md5(f"{r['name']}|{r['addr_key']}".encode()).hexdigest()[:8]
        return f"{state}-{pre}-H{h}"

    out["provider_id"] = out.apply(pid, axis=1)
    dup = out["provider_id"].duplicated(keep=False)
    if dup.any():
        out.loc[dup, "provider_id"] = out.loc[dup, "provider_id"] + "-" + out.loc[dup].groupby("provider_id").cumcount().astype(str)
    return out
