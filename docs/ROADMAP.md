# Roadmap: statistical / graph additions

Principle: **rule flags stay the primary ranking.** Anything below is a secondary column or a tiebreaker.
A WHIP/OIG tip needs specific, verifiable facts; a model score only helps decide where to look.
Hard rule: features come from provider, billing, licensing and corporate data only. The CMS utilization
files contain demographic columns (race, age, sex); they are never used as features.

## Phases (implement when the trigger is met)

| # | Item | Trigger | Output |
|---|------|---------|--------|
| 1 | **networkx** for `clusters.csv`: nodes = providers, owners, addresses, officials; edges = shared links; connected components = clusters; degree/betweenness centrality = hub owners/addresses | After the first manual review of the top GA leads (confirms the current link rules are right) and before the first FL run | `clusters.csv` gains `hub_node`, `hub_degree`; cross-sector networks reported first |
| 2 | **Anomaly score** (PyOD ECOD + IsolationForest, scikit-learn `RobustScaler`), per state and sector, on peer-normalised features | Hospice live-discharge and cap data loaded, multi-year utilization kept (for volume growth), and >= ~150 providers with complete features per state/sector (GA and FL both run) | One extra column `anomaly_score` (0-100 percentile); tiebreaker after rule score; never used to exclude or add flags |
| 3 | **Explanations** in `evidence/<id>.md`: per-feature contributions in plain terms ("ALOS 3.1x peer median, live-discharge 2.4x") | After phase 2 is stable. ECOD gives per-feature outlier scores natively; use SHAP only if IsolationForest is the scorer | "Why the model flagged this" section citing the underlying numbers and source records |
| 4 | **DBSCAN** on billing profiles to find providers with near-identical profiles; compare against network clusters | After phase 1 and 2 | Informational `profile_group` column; a lead only if it coincides with a graph cluster |
| 5 | *Conditional:* XGBoost + imbalanced-learn on weak labels (LEIE / SAM / press-release providers) | Only if there are >= ~100 positives per sector with usable features; otherwise skip (thin, biased toward what was prosecuted) | Secondary signal at best; the known-case filter stays as is |

## Features

- Home health: payment/beneficiary, episodes/beneficiary, license age, year-over-year volume growth.
- Hospice: average length of stay, live-discharge rate, non-cancer share, cap utilization/years over cap, final-days visit intensity.
- All features are peer-normalised (county if >= 10 peers, else state) before scaling; log-transform heavy-tailed money fields.

## Prerequisites before phase 2

- Keep every utilization year (currently only the latest year per provider is kept) so volume growth exists.
- Load hospice live-discharge and cap sources; otherwise those features are missing for every provider.
- Impute nothing silently: providers missing a feature are scored on available features and marked `anomaly_partial`.

## Explicitly skipped

Hummingbird (bank AML), River (no streaming data), FastAPI (nothing to serve), SMOTE without real labels.

## Dependencies

Add `pyod`, `scikit-learn`, `networkx` (and `shap` only if needed) to `requirements.txt` when the phase is implemented, not before.
