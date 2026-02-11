# Design: Duration / Survival Models (Time-to-Event)

This design file enumerates **within-design estimator implementations** for time-to-event outcomes (hazard/survival models), common in labor (unemployment duration), health, and IO (firm survival).

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical duration-model estimate for the claim object.
- Record: time scale, censoring rules, event definition, and the paper’s focal parameterization (hazard ratio vs log-hazard coefficient vs AFT time ratio).

## Design estimator implementations (`design/duration_survival/*`)

Spec ID format:

- `design/duration_survival/{axis}/{variant}`

### A) Model family

| spec_id | Description |
|---|---|
| `design/duration_survival/model/cox_ph` | Cox proportional hazards (semi-parametric) |
| `design/duration_survival/model/weibull_aft` | Weibull AFT |
| `design/duration_survival/model/exponential_aft` | Exponential AFT |
| `design/duration_survival/model/log_normal_aft` | Log-normal AFT |
| `design/duration_survival/model/log_logistic_aft` | Log-logistic AFT |
| `design/duration_survival/model/competing_risks` | Competing risks (paper-relevant only) |

### B) Ties handling / baseline-hazard conventions (Cox; paper-relevant only)

| spec_id | Description |
|---|---|
| `design/duration_survival/cox_ties/breslow` | Breslow ties method |
| `design/duration_survival/cox_ties/efron` | Efron ties method |
| `design/duration_survival/cox_ties/exact` | Exact partial likelihood |

### C) Time-varying effects (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/duration_survival/tvc/none` | No time-varying coefficients |
| `design/duration_survival/tvc/treatment_x_duration` | Treatment \u00d7 duration interaction |

### D) Frailty / unobserved heterogeneity (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/duration_survival/frailty/none` | No frailty |
| `design/duration_survival/frailty/shared_gamma` | Shared gamma frailty by group |

## Standard survival diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the duration estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible survival reporting.

- `diag/duration_survival/ph_assumption/schoenfeld` (Cox PH only)
- `diag/duration_survival/fit/concordance_index`
- `diag/duration_survival/censoring/counts` (events vs censored)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Sample rules (censoring-as-sample rules; careful with population changes): `specification_tree/modules/robustness/sample.md`
- Pre-processing/coding (event definition variants when estimand-preserving): `specification_tree/modules/robustness/preprocessing.md`

### Inference (`infer/*`)

- Robust/clustered SE and resampling: `specification_tree/modules/inference/standard_errors.md`
- Resampling (bootstrap): `specification_tree/modules/inference/resampling.md`

### Exploration (`explore/*`)

- Heterogeneity/subpopulations: `specification_tree/modules/exploration/heterogeneity.md`
