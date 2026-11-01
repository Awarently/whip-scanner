"""Shared helpers: config loading, run log, dates."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    as_of = cfg["run"].get("as_of")
    cfg["_as_of"] = dt.date.fromisoformat(str(as_of)) if as_of else dt.date.today()
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    return cfg


class RunLog:
    """Collects messages; written to run_log.txt at the end of a run."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.sources: list[dict] = []   # one entry per source resolution

    def add(self, level: str, msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {msg}"
        self.lines.append(line)
        print(line)

    info = lambda self, m: self.add("INFO", m)      # noqa: E731
    warn = lambda self, m: self.add("WARN", m)      # noqa: E731

    def record_source(self, **kw) -> None:
        self.sources.append(kw)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = [s for s in self.sources if s["mode"] in ("auto", "cache")]
        man = [s for s in self.sources if s["mode"] == "manual"]
        miss = [s for s in self.sources if s["mode"] == "missing"]
        out = ["SOURCE SUMMARY",
               f"  automated download OK : {len(ok)}",
               f"  manual fallback used  : {len(man)}",
               f"  missing (no data)     : {len(miss)}", ""]
        for s in self.sources:
            out.append(f"  {s['mode']:<8} {s['state']:<7} {s['sector']:<12} {s['source']:<14} "
                       f"{s.get('file') or '-'}  (data date {s.get('date') or '-'})")
        out += ["", "LOG"] + self.lines
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
