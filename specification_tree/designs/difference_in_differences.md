# Design: Difference-in-Differences (DiD)

This design file enumerates **within-design estimator implementations** for DiD-style identification under maintained parallel-trends-type assumptions.

Universal RC, inference, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical DiD estimate for the claim object.
- Record design-defining metadata under `coefficient_vector_json.design.difference_in_differences` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design estimator implementations (`design/difference_in_differences/*`)

Spec ID format:

- `design/difference_in_differences/{axis}/{variant}`

### A) Core DiD estimators (static average effects)

| spec_id | Description |
|---|---|
| `design/difference_in_differences/estimator/twfe` | Two-way fixed effects DiD (paper baseline class) |
| `design/difference_in_differences/estimator/sun_abraham` | Interaction-weighted event-time estimator (staggered adoption) |
| `design/difference_in_differences/estimator/callaway_santanna` | Group-time ATT aggregation (staggered adoption) |
| `design/difference_in_differences/estimator/borusyak_imputation` | Imputation estimator (Borusyak et al.) |
| `design/difference_in_differences/estimator/de_chaisemartin` | de Chaisemartin & D’Haultfoeuille style estimator |

### B) Doubly-robust / reweighted DiD (optional)

Only use when the paper (or replication code) clearly supports propensity-style reweighting as a robustness implementation within the DiD design.

| spec_id | Description |
|---|---|
| `design/difference_in_differences/estimator/dr_did` | Doubly-robust DiD (requires nuisance models) |
| `design/difference_in_differences/estimator/ipw_did` | IPW-style DiD reweighting |

**Bundled estimator note**: DR/IPW DiD uses nuisance models with covariates. If the manuscript/code reveals linked adjustment across components, enforce joint control-set variation. See `specification_tree/REVEALED_SEARCH_SPACE.md`.

### C) Dynamic DiD / event-study objects

If the paper’s headline claim is dynamic (leads/lags), treat it as an event-study design and use:

- `specification_tree/designs/event_study.md`

## Standard DiD diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the DiD estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible DiD reporting.

- `diag/difference_in_differences/pretrends/joint_test` (when pre-treatment periods exist)
- `diag/difference_in_differences/pretrends/linear_trend_test` (as a complementary check, when meaningful)
- `diag/difference_in_differences/weights/bacon_decomposition` (when using TWFE with staggered adoption)
- `diag/difference_in_differences/overlap/cohort_counts` (support diagnostics for staggered/event-time cells)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls: `specification_tree/modules/robustness/controls.md`
- Sample rules: `specification_tree/modules/robustness/sample.md`
- Fixed effects: `specification_tree/modules/robustness/fixed_effects.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`
- Data construction: `specification_tree/modules/robustness/data_construction.md`
- Functional form: `specification_tree/modules/robustness/functional_form.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- SE and clustering: `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Exploration (`explore/*`)

- Treatment definition changes (timing/exposure): `specification_tree/modules/exploration/variable_definitions.md`
- Heterogeneity: `specification_tree/modules/exploration/heterogeneity.md`
- Alternative estimands: `specification_tree/modules/exploration/alternative_estimands.md`
