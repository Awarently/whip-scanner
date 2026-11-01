"""Resolve each data source: automated download -> manual file -> older cache -> missing.

Rules: dated raw files are never overwritten; delays between requests; no login or
CAPTCHA circumvention (sources that need either have url: null and go manual).
"""
from __future__ import annotations
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from core import RunLog
from . import feeds

_printed: set[str] = set()


@dataclass
class FetchResult:
    source: str
    path: Path | None
    mode: str               # auto | cache | manual | missing
    date: str | None
    paths: list | None = None   # several files (e.g. GA HHA + Private Home Care exports)


def _manual_candidates(cfg, spec, state, sector) -> list[Path]:
    base = cfg["paths"]["manual"]
    name = spec["manual_name"]
    return [base / state / sector / name, base / state / name, base / "shared" / name]


def _dated_files(rawdir: Path, stem: str) -> list[Path]:
    return sorted(rawdir.glob(f"{stem}_*.csv"))


def _download(url: str, dest: Path, cfg: dict, log: RunLog) -> bool:
    f = cfg["fetch"]
    try:
        time.sleep(f["delay_seconds"])
        r = requests.get(url, headers={"User-Agent": f["user_agent"]}, timeout=f["timeout_seconds"],
                         stream=True, allow_redirects=True)
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200:
            log.warn(f"HTTP {r.status_code} for {url}")
            return False
        if "text/html" in ctype:
            log.warn(f"{url} returned HTML (login/CAPTCHA/landing page?); not bypassing")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        if tmp.stat().st_size == 0:
            tmp.unlink()
            return False
        tmp.rename(dest)
        return True
    except requests.RequestException as e:
        log.warn(f"download failed for {url}: {type(e).__name__}: {str(e)[:120]}")
        return False


def resolve_source(src_id: str, spec: dict, state: str, sector: str, cfg: dict,
                   log: RunLog, offline: bool = False) -> FetchResult:
    stem = f"{src_id}_{state}"
    rawdir = cfg["paths"]["raw"] / src_id
    today = cfg["_as_of"].strftime("%Y%m%d")
    res: FetchResult | None = None

    # 1a. structured feeds (GA DCH, CMS PDC / data-api, PECOS owners)
    kind = spec.get("kind")
    if kind and not offline:
        sub = rawdir / today
        try:
            if kind == "ga_dch":
                paths = feeds.ga_dch(spec, cfg, log, sub)
                res = FetchResult(src_id, paths[0], "auto", today, paths=paths)
            else:
                dest = rawdir / f"{stem}_{today}.csv"
                if dest.exists():
                    res = FetchResult(src_id, dest, "cache", today)
                    log.info(f"{src_id}: reusing today's download {dest.name}")
                else:
                    fn = {"pdc": lambda: feeds.pdc(spec, cfg, log, dest),
                          "cms_api": lambda: feeds.cms_api(spec, cfg, log, state, dest),
                          "pecos": lambda: feeds.pecos(spec, cfg, log, state, dest)}[kind]
                    res = FetchResult(src_id, fn(), "auto", today)
            log.info(f"{src_id}: {kind} feed OK")
        except Exception as e:                       # noqa: BLE001 - any failure -> manual fallback
            log.warn(f"{src_id}: {kind} feed failed ({type(e).__name__}: {str(e)[:150]}); trying manual file")
            res = None

    # 1b. automated plain URL
    url = spec.get("url")
    if res is None and url and not offline:
        dest = rawdir / f"{stem}_{today}.csv"
        if dest.exists():                       # never overwrite silently
            res = FetchResult(src_id, dest, "cache", today)
            log.info(f"{src_id}: reusing today's download {dest.name}")
        elif _download(url, dest, cfg, log):
            res = FetchResult(src_id, dest, "auto", today)
            log.info(f"{src_id}: downloaded -> {dest}")
    elif not url and not kind:
        log.info(f"{src_id}: no automated bulk URL configured; trying manual file")

    # 2. manual
    if res is None and spec.get("manual_glob"):
        for d in dict.fromkeys(p.parent for p in _manual_candidates(cfg, spec, state, sector)):
            files = sorted(d.glob(spec["manual_glob"])) if d.exists() else []
            if files:
                dmax = max(dt.date.fromtimestamp(f.stat().st_mtime) for f in files).isoformat()
                res = FetchResult(src_id, files[0], "manual", dmax, paths=files)
                log.info(f"{src_id}: using {len(files)} manual file(s) in {d}: {', '.join(f.name for f in files)}")
                break
    if res is None:
        for p in _manual_candidates(cfg, spec, state, sector):
            if p.exists():
                d = dt.date.fromtimestamp(p.stat().st_mtime).isoformat()
                res = FetchResult(src_id, p, "manual", d)
                log.info(f"{src_id}: using manual file {p} (modified {d})")
                break

    # 3. earlier dated cache (GA feed keeps one dated folder per download)
    if res is None and spec.get("kind") == "ga_dch" and rawdir.exists():
        for d in sorted((x for x in rawdir.iterdir() if x.is_dir()), reverse=True):
            files = [d / f"{k}.csv" for k in spec["feed_files"]]
            if all(f.exists() for f in files):
                res = FetchResult(src_id, files[0], "cache", d.name, paths=files)
                log.warn(f"{src_id}: using cached feed from {d.name}")
                break
    if res is None:
        old = _dated_files(rawdir, stem)
        if old:
            d = old[-1].stem.rsplit("_", 1)[-1]
            res = FetchResult(src_id, old[-1], "cache", d)
            log.warn(f"{src_id}: automated + manual unavailable; using older cache {old[-1].name}")

    # 4. missing
    if res is None:
        res = FetchResult(src_id, None, "missing", None)
        key = f"{src_id}:{state}:{sector}"
        if key not in _printed:
            _printed.add(key)
            want = _manual_candidates(cfg, spec, state, sector)[0]
            log.warn(f"{src_id}: MISSING. MANUAL DOWNLOAD NEEDED -> {spec.get('instructions','')}  "
                     f"[expected at {want}]")

    log.record_source(source=src_id, state=state, sector=sector, mode=res.mode,
                      file="; ".join(str(x) for x in (res.paths or [res.path])) if res.path else None, date=res.date)
    return res


# ---------------------------------------------------------------- NPPES API
def nppes_lookup(npis: list[str], cfg: dict, log: RunLog) -> dict[str, dict]:
    """Query the NPPES NPI Registry API (public, no login). Cached per NPI.

    Returns {npi: {auth_official, enumeration_date, address}}. Stops on first
    connection failure (caller falls back to manual nppes.csv).
    """
    f = cfg["fetch"]
    if not f.get("nppes_enabled", True):
        return {}
    cdir = cfg["paths"]["cache"] / "nppes"
    cdir.mkdir(parents=True, exist_ok=True)
    out, lookups = {}, 0
    for npi in dict.fromkeys(n for n in npis if n and str(n).isdigit()):
        cf = cdir / f"{npi}.json"
        if cf.exists():
            data = json.loads(cf.read_text())
        else:
            if lookups >= f["nppes_max_lookups"]:
                log.warn("NPPES: lookup cap reached; remaining NPIs not enriched")
                break
            try:
                time.sleep(f["delay_seconds"])
                r = requests.get("https://npiregistry.cms.hhs.gov/api/",
                                 params={"version": "2.1", "number": npi},
                                 headers={"User-Agent": f["user_agent"]}, timeout=f["timeout_seconds"])
                lookups += 1
                if r.status_code != 200:
                    log.warn(f"NPPES HTTP {r.status_code}; stopping API enrichment")
                    break
                data = r.json()
                cf.write_text(json.dumps({"fetched": dt.date.today().isoformat(), **data}))
            except (requests.RequestException, ValueError) as e:
                log.warn(f"NPPES API unavailable ({type(e).__name__}); falling back to manual nppes.csv")
                break
        for rec in data.get("results", [])[:1]:
            b = rec.get("basic", {})
            ao = " ".join(x for x in (b.get("authorized_official_first_name"),
                                      b.get("authorized_official_last_name")) if x)
            addr = next((a for a in rec.get("addresses", []) if a.get("address_purpose") == "LOCATION"), {})
            out[str(npi)] = {"auth_official": ao, "enumeration_date": b.get("enumeration_date", ""),
                             "address": addr.get("address_1", "")}
    return out
