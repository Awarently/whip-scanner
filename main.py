"""Healthcare fraud red-flag scanner (home health + hospice). Output = leads only.

    python main.py --states FL --sectors home_health hospice
    python main.py --demo            # synthetic data, no network
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

from core import load_config, RunLog
from fetch import resolve_source
from clean.build import load_source, add_address_fields
from match.entity import merge_sector
from match.enrich import enrich_corp, enrich_nppes
from match.network import annotate
from filters.known_case import build_known_sets, split_press, apply_known_case
from score.engine import score_sector
from report import write_all


def specs_for(cfg, role, state, sector):
    for sid, sp in cfg["sources"].items():
        if sp["role"] != role:
            continue
        if sp["sectors"] != "all" and sector not in sp["sectors"]:
            continue
        if "states" in sp and state not in sp["states"]:
            continue
        yield sid, sp


def run(cfg, states, sectors, offline=False, log=None):
    log = log or RunLog()
    run_dir = cfg["paths"]["out"] / cfg["_as_of"].strftime("%Y%m%d")
    all_frames, shared = [], {}

    # shared exclusion / press sources (state/sector irrelevant)
    for role in ("leie", "sam", "press"):
        for sid, sp in specs_for(cfg, role, states[0], sectors[0]):
            res = resolve_source(sid, sp, "shared", "all", cfg, log, offline)
            shared[role] = load_source(res, sp, cfg, "shared", "all")

    corp_by_state, nppes_by_state = {}, {}
    for st in states:
        if st not in cfg["run"]["supported_states"]:
            log.warn(f"state {st} not in supported_states; skipping")
            continue
        for role, store in (("corp", corp_by_state), ("nppes", nppes_by_state)):
            for sid, sp in specs_for(cfg, role, st, sectors[0]):
                res = resolve_source(sid, sp, st, "all", cfg, log, offline)
                store[st] = (load_source(res, sp, cfg, st, "all"), res)
        for sector in sectors:
            frames, roles = {}, {}
            for role in ("license", "cms_provider", "cms_util", "cms_quality"):
                for sid, sp in specs_for(cfg, role, st, sector):
                    res = resolve_source(sid, sp, st, sector, cfg, log, offline)
                    df = load_source(res, sp, cfg, st, sector)
                    if df is not None:
                        df = add_address_fields(df, cfg) if role in ("license", "cms_provider") else df
                        frames[sid], roles[sid] = df, role
            if not frames:
                log.warn(f"{st}/{sector}: no provider data available; skipped (see manual instructions above)")
                continue
            prov = merge_sector(frames, roles, st, sector, cfg)
            if prov.empty:
                continue
            corp, cres = corp_by_state.get(st, (None, None))
            prov = enrich_corp(prov, corp, cres, cfg, log)
            npp, nres = nppes_by_state.get(st, (None, None))
            prov = enrich_nppes(prov, npp, nres, cfg, log)
            log.info(f"{st}/{sector}: {len(prov)} providers after entity matching")
            all_frames.append(prov)

    if not all_frames:
        log.warn("No provider data loaded for any state/sector. Place files per the instructions above and re-run.")
        log.write(run_dir / "run_log.txt")
        return None
    prov = pd.concat(all_frames, ignore_index=True)
    prov["county"] = prov["county"].fillna("")

    prov, edges, clusters = annotate(prov, cfg, log)
    excl = build_known_sets({k: v for k, v in shared.items() if k in ("leie", "sam")}, log)
    press = split_press(shared.get("press"), excl, log)
    prov = apply_known_case(prov, edges, excl, press, cfg)

    scored = []
    for (st, sector), g in prov.groupby(["state", "sector"]):
        scored.append(score_sector(g, sector, cfg))
    allp = pd.concat(scored).sort_index()
    info = f"Run {cfg['_as_of']} · {len(allp)} providers · {int((allp['rank'].notna()).sum())} ranked leads · {int(allp['known_case'].sum())} known-case exclusions"
    for (st, sector), g in allp.groupby(["state", "sector"]):
        g = g.copy()
        write_all(g, clusters, cfg, run_dir / st / sector, f"Fraud red-flag leads - {st} {sector}", info)
    # combined: ranks recomputed across everything by score
    comb = allp.copy()
    elig = comb[comb["rank"].notna()].sort_values(["score", "est_exposure_usd"], ascending=False, na_position="last")
    comb["rank"] = pd.NA
    comb.loc[elig.index, "rank"] = range(1, len(elig) + 1)
    write_all(comb, clusters, cfg, run_dir / "combined", "Fraud red-flag leads - combined", info)
    log.write(run_dir / "run_log.txt")
    log.info(f"outputs written to {run_dir}")
    return comb, clusters, run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--states", nargs="+")
    ap.add_argument("--sectors", nargs="+")
    ap.add_argument("--offline", action="store_true", help="skip automated downloads; use manual/cache only")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data in ./data_demo and run on it")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.demo:
        from tools.make_synthetic import generate
        root = Path(__file__).parent / "data_demo"
        generate(root)
        cfg["paths"].update({"manual": root / "manual", "raw": root / "raw", "cache": root / "cache",
                             "out": root / "output"})
        cfg["fetch"]["nppes_enabled"] = False
        a.offline = True
    states = a.states or cfg["run"]["states"]
    sectors = a.sectors or cfg["run"]["sectors"]
    out = run(cfg, states, sectors, a.offline)
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
