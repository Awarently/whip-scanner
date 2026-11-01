# Healthcare Fraud Red-Flag Scanner (Home Health + Hospice)

Pulls public provider data, cross-references it, and ranks providers with statistical red flags.
**Output is leads only.** Uses provider, billing, licensing and corporate data; never demographic, ethnic or
population data.

> **Disclaimer.** Results are statistical leads, not accusations or findings. Each requires human
> verification of the underlying records before anything is submitted to HHS-OIG (tips.hhs.gov) or the
> DOJ. Peer outliers often have innocent explanations (chains, patient mix, ownership changes).
> Fuzzy name matches against LEIE/SAM must be checked for identity.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py --demo                                  # synthetic data, no network
python main.py --states FL --sectors home_health hospice
python main.py --states FL GA --offline                # skip downloads, use data/manual + data/raw
python -m pytest -q tests
```

## Data flow
`fetch -> clean -> match (entity, enrich, network) -> filters (known case) -> score -> report`

1. **fetch/**: tries the automated URL in `config.yaml` (dated files in `data/raw/<source>/`, never overwritten),
   then `data/manual/<STATE>/<sector>/<file>`, `data/manual/<STATE>/<file>`, `data/manual/shared/<file>`,
   then the newest older cache. Missing sources print exact manual instructions and appear in `run_log.txt`.
   No logins, CAPTCHA or terms bypassing; `delay_seconds` between requests.
2. **clean/**: header aliases (`config.yaml: columns`), address/name/person normalisation.
3. **match/**: merge by CCN/NPI/license #, then fuzzy name + normalised address (one row per source per
   provider, so sibling agencies at one suite stay separate); enrich with corporate records and NPPES;
   build address/owner/official links, churn detection and clusters.
4. **filters/**: known-case filter. Providers on LEIE/SAM, named in press releases, or with a same-conduct
   state sanction (`sanction_same_conduct=Y`) are listed but **excluded from ranking**. Their owners and
   addresses mark other providers as `linked_known_case`.
5. **score/**: one engine; per-sector flags in `score/flags.py`, weights/thresholds in `config.yaml`.
   Percentiles use county peers when >= `min_peer_group`, else state.
6. **report/**: per state/sector and `combined/`: `red_flags.csv`, `clusters.csv`, `evidence/<id>.md`,
   `report.html` (top 50), plus `run_log.txt` in the run folder (`output/<YYYYMMDD>/`).

## Manual inputs
See `config.yaml: sources` for each file's name and instructions. Minimum for a useful run, per state and
sector: `license.csv`, `cms_provider.csv`, `cms_util.csv` (hospice also `cms_quality.csv`), plus shared
`leie.csv`, and optionally `sam.csv`, `press_releases.csv`, `corp.csv`, `nppes.csv`.
Percent columns are read as 0-100.

**Georgia license lists.** GA DCH HFR Facility Finder (forms.dch.georgia.gov/HFRD) is an interactive form,
so it is manual-only. Export both lists and save them as
`data/manual/GA/home_health/license_hha.csv` (Home Health Agencies) and
`data/manual/GA/home_health/license_phcp.csv` (Private Home Care Providers); hospice goes in
`data/manual/GA/hospice/license.csv`. All `license*.csv` files in the folder are concatenated and
`facility_type` is set from the file name. Private Home Care Providers are state-licensed and usually have
no CCN or Medicare data, so only the address/owner/churn/virtual/sanction flags can fire for them.
Column names in the GA exports are unverified; extend `config.yaml: columns` if a header is not recognised.
Run with `python main.py --states GA`.

## Caveats
- CMS dataset URLs and column aliases were written without live access and are **unverified**. Fix
  `config.yaml` if a download 404s or a header is not recognised (unmapped columns are dropped silently).
- Estimated exposure = (payment per beneficiary minus peer median) x beneficiaries; a gap vs peers, not an
  overpayment finding.
- Adding a sector (DME, lab testing): add a flag list in `score/flags.py`, a `scoring.<sector>` block and
  `sources` entries in `config.yaml`; the engine is unchanged.
