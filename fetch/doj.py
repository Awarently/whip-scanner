"""DOJ press releases (public JSON API: www.justice.gov/api/v1/press_releases.json).

The API can filter on title only, so we pull every release whose title contains a health-care-fraud keyword
(config press.doj_title_keywords) and keep the body text; names are then searched in the bodies
(filters/press.py). Dated raw file, never overwritten; delay between requests.
"""
from __future__ import annotations
import datetime as dt
import html
import json
import re
import time
from pathlib import Path

import requests

API = "https://www.justice.gov/api/v1/press_releases.json"
_TAG = re.compile(r"<[^>]+>")


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def fetch_doj_press(cfg: dict, log, dest: Path) -> Path:
    pc = cfg["press"]
    f = cfg["fetch"]
    seen: dict[str, dict] = {}
    for kw in pc["doj_title_keywords"]:
        page, total = 0, None
        while True:
            time.sleep(pc.get("delay_seconds", 1.0))
            r = requests.get(API, params={"pagesize": 50, "page": page, "parameters[title]": kw},
                             headers={"User-Agent": f["user_agent"]}, timeout=f["timeout_seconds"])
            if r.status_code != 200:
                raise requests.HTTPError(f"DOJ API HTTP {r.status_code} for keyword '{kw}' page {page}")
            j = r.json()
            total = int(j["metadata"]["resultset"]["count"])
            for x in j["results"]:
                u = x["uuid"]
                if u in seen:
                    continue
                d = x.get("date") or ""
                seen[u] = {"uuid": u, "title": _text(x.get("title", "")), "url": x.get("url", ""),
                           "date": dt.datetime.fromtimestamp(int(d), dt.timezone.utc).date().isoformat() if d.isdigit() else "",
                           "component": "; ".join(c.get("name", "") for c in (x.get("component") or [])),
                           "text": _text(x.get("body", "")) or _text(x.get("teaser", ""))}
            page += 1
            if page * 50 >= total or not j["results"]:
                break
        log.info(f"DOJ press: keyword '{kw}' {total} matches; {len(seen)} unique so far")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in seen.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.rename(dest)
    log.info(f"DOJ press: {len(seen)} releases -> {dest}")
    return dest
