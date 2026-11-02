"""CSV, evidence markdown and HTML report writers."""
from __future__ import annotations
import html
import urllib.parse
from pathlib import Path
import pandas as pd
from score.flags import FLAGS

DISCLAIMER = ("LEADS ONLY - statistical red flags from public provider, billing, licensing and corporate data. "
              "Not an accusation or finding of wrongdoing. Every lead needs human verification before any tip is submitted.")

PHRASE = {
    "shared_address": "operates from the same address/suite as other agencies",
    "shared_person": "shares an owner, administrator or authorised official with other providers",
    "new_license_high_volume": "was licensed under 24 months ago yet bills at high volume",
    "high_payment_per_bene": "has Medicare payment per beneficiary in the top 5% of peers",
    "high_episodes_per_bene": "has episodes per beneficiary in the top 5% of peers",
    "license_churn": "appears to be a closure-and-reopen of a related agency within 12 months",
    "virtual_address": "lists a mailbox/virtual-office style address",
    "owner_exclusion_match": "has an owner/official whose name fuzzy-matches an excluded individual or entity",
    "linked_known_case": "is linked by owner/address to a provider or person already in a public enforcement matter",
    "sanctions_or_failed_inspection": "has sanctions or failed inspections on record",
    "chain_ownership": "is owned by a chain/corporate parent with many providers (weak signal on its own)",
    "cross_sector_network": "belongs to a network spanning both home health and hospice",
    "long_length_of_stay": "has an average length of stay in the top 5% of peers",
    "high_live_discharge": "has a high live-discharge rate versus peers",
    "high_noncancer_share": "has an unusually high share of non-cancer/dementia diagnoses versus peers",
    "frequent_cap": "repeatedly approaches or exceeds the Medicare aggregate cap",
    "low_final_days_visits": "delivers few visits in patients' final days compared with peers",
    "new_hospice_high_billing": "is a new hospice with high billing",
    "shared_address_or_owner": "shares an address or owner with other hospices/home health agencies",
}
_META = ["provider_id", "rank", "state", "sector", "name", "dba", "license_no", "facility_type", "npi", "ccn", "address", "city", "zip",
         "county", "license_date", "cert_date", "operating_since", "close_date", "status", "owners", "administrator", "auth_official",
         "known_case", "known_case_reason", "known_case_note", "score", "score_pct_of_max", "n_flags", "flags_triggered",
         "est_exposure_usd", "match_method", "cluster_id", "cluster_size", "cluster_hub", "beneficiaries", "total_payment", "payment_per_bene",
         "peer_median_payment_per_bene"]


def _s(x) -> str:
    return "" if x is None or (not isinstance(x, (list, dict)) and pd.isna(x)) else str(x)


def write_csvs(df: pd.DataFrame, clusters: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    flagcols = [c for c in df.columns if c.startswith("flag_")]
    cols = [c for c in _META if c in df.columns] + flagcols
    d = df.sort_values(["known_case", "rank", "score"], ascending=[True, True, False], na_position="last")
    d[cols].to_csv(outdir / "red_flags.csv", index=False)
    if clusters is not None and len(clusters):
        pid = set(df["provider_id"])
        keep = clusters[clusters["provider_ids"].map(lambda s: any(p in pid for p in s.split("; ")))]
    else:
        keep = pd.DataFrame(columns=["cluster_id"])
    keep.to_csv(outdir / "clusters.csv", index=False)
    return d


def summary_text(r) -> str:
    fl = [f for f in r["flags_triggered"].split("; ") if f]
    if not fl:
        return "No red flags triggered."
    return (f"{r['name']} " + "; it ".join(PHRASE.get(f, f) for f in fl[:6]) +
            ". These are statistical indicators that may have innocent explanations (e.g. a legitimate chain, "
            "a specialised patient mix, a recent change of ownership) and require human verification.")


def write_evidence(df: pd.DataFrame, cfg: dict, outdir: Path, labels: dict) -> int:
    ev = outdir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    roles = {k: v["role"] for k, v in cfg["sources"].items()}
    roles.update({k + "_enrollment": "cms_provider" for k, v in cfg["sources"].items() if v["role"] == "corp"})
    n = 0
    for _, r in df[(df["n_flags"] > 0) | df["known_case"]].iterrows():
        lines = [f"# {r['name']}  ({r['provider_id']})", "", f"> {DISCLAIMER}", "",
                 "## Identifiers", f"- Sector / state: {r['sector']} / {r['state']}",
                 f"- License #: {_s(r['license_no'])}", f"- NPI: {_s(r['npi'])}", f"- CCN: {_s(r['ccn'])}",
                 f"- Address: {_s(r['address'])}, {_s(r['city'])} {_s(r['zip'])}  (normalised: {r['addr_key']})",
                 f"- Licensed: {_s(r['license_date'])[:10]}   Closed: {_s(r['close_date'])[:10]}   Status: {_s(r['status'])}",
                 f"- Owners/officers: {_s(r['owners'])}", f"- Administrator: {_s(r['administrator'])}",
                 f"- Authorised official: {_s(r['auth_official'])}", f"- Registered agent: {_s(r.get('registered_agent'))}",
                 f"- Network cluster: {_s(r['cluster_id']) or 'none'} (size {int(r['cluster_size'])})", "",
                 "## Known-case check"]
        if r["known_case"]:
            lines += [f"**EXCLUDED FROM TARGET RANKING** - {r['known_case_reason']}",
                      "WHIP pays only for original information; this provider is already known. Its owners/addresses are kept as linking signals."]
        else:
            lines += ["No direct match to LEIE/SAM, press releases, or same-conduct state sanction in the supplied data "
                      "(absence of a match is not proof the provider is unknown to enforcement; check manually)."]
            if r["linked_known_case"]:
                lines.append(f"Linked signal: {r['linked_known_case_detail']}")
        lines += ["", f"## Triggered flags (score {r['score']:.0f}, {r['score_pct_of_max']}% of max)"]
        fl = [f for f in FLAGS[r["sector"]] if r[f"flag_{f.name}"]]
        if not fl:
            lines.append("_none_")
        for f in fl:
            lines += [f"### {f.label}", f"- Detail / peer comparison: {r[f'detail_{f.name}']}"]
            cited = [s for s in r["sources"] if roles.get(s["source"], s["source"]) in f.uses]
            for s in cited:
                lines.append(f"- Source: `{s['source']}` file `{s['file']}` downloaded/dated {s['download_date']} record {s['record_id']}")
            if not cited:
                lines.append("- Source: see detail (record IDs cited inline)")
        lines += ["", "## Peer comparison",
                  f"- Payment per beneficiary: {_s(r['payment_per_bene'])} vs peer median {_s(r['peer_median_payment_per_bene'])}",
                  f"- Beneficiaries: {_s(r['beneficiaries'])}   Total payment: {_s(r['total_payment'])}", "",
                  "## Suspected pattern (plain English)", summary_text(r), "",
                  "## Estimated dollar exposure",
                  (f"${r['est_exposure_usd']:,.0f}" if pd.notna(r["est_exposure_usd"]) else "not estimable (no utilisation data)"),
                  f"_{cfg['scoring']['exposure_note']}_", "",
                  "## All source records used"]
        for s in r["sources"]:
            lines.append(f"- {s['source']}: {s['file']} | {s['download_date']} | {s['record_id']}")
        (ev / f"{r['provider_id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    return n


def _links(r) -> str:
    out = []
    if _s(r["npi"]):
        out.append(f'<a href="https://npiregistry.cms.hhs.gov/provider-view/{html.escape(_s(r["npi"]))}">NPI Registry</a>')
    q = urllib.parse.quote(f"{_s(r['address'])} {_s(r['city'])} {_s(r['state'])} {_s(r['zip'])}")
    out.append(f'<a href="https://www.google.com/maps/search/?api=1&query={q}">Map</a>')
    out.append('<a href="https://exclusions.oig.hhs.gov/">LEIE search</a>')
    out.append('<a href="https://sam.gov/search/?index=ex">SAM exclusions</a>')
    if r["state"] == "FL":
        out += ['<a href="https://search.sunbiz.org/Inquiry/CorporationSearch/ByName">Sunbiz</a>',
                '<a href="https://quality.healthfinder.fl.gov/">FL AHCA</a>']
    if r["state"] == "GA":
        out += ['<a href="https://ecorp.sos.ga.gov/BusinessSearch">GA SOS</a>',
                '<a href="https://forms.dch.georgia.gov/HFRD">GA DCH</a>']
    if _s(r["ccn"]):
        out.append('<a href="https://www.medicare.gov/care-compare/">Care Compare</a> (CCN ' + html.escape(_s(r["ccn"])) + ")")
    return " · ".join(out)


def write_html(df: pd.DataFrame, cfg: dict, outdir: Path, title: str, runinfo: str, clusters=None) -> None:
    top = df[df["rank"].notna()].sort_values("rank").head(cfg["report"]["top_n"])
    excl = df[df["known_case"]]
    rows = []
    for _, r in top.iterrows():
        flags = "".join(f"<li><b>{html.escape(f.label)}</b>: {html.escape(_s(r[f'detail_{f.name}']))}</li>"
                        for f in FLAGS[r["sector"]] if r[f"flag_{f.name}"])
        ex = f"${r['est_exposure_usd']:,.0f}" if pd.notna(r["est_exposure_usd"]) else "n/a"
        rows.append(
            f"<tr><td>{int(r['rank'])}</td><td><a href='evidence/{html.escape(r['provider_id'])}.md'>{html.escape(_s(r['name']))}</a>"
            f"<br><small>{html.escape(r['provider_id'])} · {html.escape(_s(r['address']))}, {html.escape(_s(r['city']))}</small></td>"
            f"<td>{r['state']}</td><td>{r['sector']}</td><td>{r['score']:.0f}</td><td>{ex}</td>"
            f"<td><details><summary>{int(r['n_flags'])} flags</summary><ul>{flags}</ul>"
            f"<p>{html.escape(summary_text(r))}</p><p class='lk'>{_links(r)}</p></details></td></tr>")
    net_rows = ""
    if clusters is not None and len(clusters):
        for _, c in clusters.head(10).iterrows():
            net_rows += (f"<tr><td>{html.escape(c['cluster_id'])}</td><td>{int(c['size'])}</td><td>{html.escape(c['sectors'])}</td>"
                         f"<td>{html.escape(str(c['hub_node']))} ({int(c['hub_degree'])} links)</td>"
                         f"<td>${c['total_medicare_payment']:,.0f}</td>"
                         f"<td><small>{html.escape(str(c['provider_names'])[:240])}</small></td></tr>")
    ex_rows = "".join(f"<li>{html.escape(_s(r['name']))} ({html.escape(r['provider_id'])}): {html.escape(r['known_case_reason'])}</li>"
                      for _, r in excl.iterrows())
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1b1f23}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f3f4f6}}.warn{{background:#fff4e5;border:1px solid #f0b46c;padding:10px;border-radius:6px}}small{{color:#555}}
.lk a{{margin-right:2px}}</style>
<h1>{html.escape(title)}</h1><p class="warn"><b>{DISCLAIMER}</b></p><p>{html.escape(runinfo)}</p>
<h2>Top {len(top)} leads</h2><table><tr><th>#</th><th>Provider</th><th>State</th><th>Sector</th><th>Score</th><th>Est. exposure</th><th>Flags / verification</th></tr>
{''.join(rows) or '<tr><td colspan=7>No leads met the minimum score.</td></tr>'}</table>
<p><small>{html.escape(cfg['scoring']['exposure_note'])}</small></p>
<h2>Largest networks (cross-sector first)</h2><table><tr><th>Cluster</th><th>Providers</th><th>Sectors</th><th>Hub owner/address</th><th>Medicare payment</th><th>Members</th></tr>
{net_rows or '<tr><td colspan=6>No networks.</td></tr>'}</table>
<h2>Excluded from ranking: known cases ({len(excl)})</h2><ul>{ex_rows or '<li>none</li>'}</ul></html>"""
    (outdir / "report.html").write_text(page, encoding="utf-8")


def write_all(df, clusters, cfg, outdir: Path, title: str, runinfo: str) -> None:
    d = write_csvs(df, clusters, outdir)
    write_evidence(df, cfg, outdir, {})
    keep = pd.read_csv(outdir / "clusters.csv") if (outdir / "clusters.csv").stat().st_size > 5 else None
    write_html(df, cfg, outdir, title, runinfo, keep)
