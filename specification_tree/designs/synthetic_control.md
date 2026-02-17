# Design: Synthetic Control / Comparative Case Study

This design file enumerates **within-design implementation choices** for comparative-case-study estimators that construct a counterfactual for treated unit(s) using a weighted combination of donors (SCM/SDID/interactive-FE variants).

Synthetic-control outputs are typically a **gap path over time**; the output contract requires a declared scalar focal summary and storage of the full path. See `specification_tree/CONTRACT.md`.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical synthetic-control estimate for the claim object.
- Record: treated unit(s), donor pool definition, predictor/matching set, pre-treatment period, and the paper’s scalar summary rule for the post-treatment effect.
- Record design-defining metadata under `coefficient_vector_json.design.synthetic_control` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design implementation variants (`design/synthetic_control/*`)

Spec ID format:

- `design/synthetic_control/{axis}/{variant}`

### A) Estimator family

| spec_id | Description |
|---|---|
| `design/synthetic_control/estimator/scm` | Classic SCM (Abadie et al.) |
| `design/synthetic_control/estimator/sdid` | Synthetic DiD (Arkhangelsky et al.) |
| `design/synthetic_control/estimator/gsynth` | Generalized synthetic control / interactive fixed effects |

### B) Donor pool / comparison-set definition

| spec_id | Description |
|---|---|
| `design/synthetic_control/donors/baseline` | Paper’s donor pool (baseline) |
| `design/synthetic_control/donors/exclude_neighbors` | Exclude geographic neighbors / close substitutes (spillover guardrail) |
| `design/synthetic_control/donors/exclude_outliers` | Exclude donors with extreme pre-trends / leverage |
| `design/synthetic_control/donors/restricted_to_support` | Restrict to donors within covariate/support bounds |

### C) Predictor / matching set

| spec_id | Description |
|---|---|
| `design/synthetic_control/predictors/baseline` | Paper’s predictor set |
| `design/synthetic_control/predictors/outcomes_only` | Match on lagged outcomes only |
| `design/synthetic_control/predictors/reduced` | Reduced predictors (drop weak/controversial predictors) |
| `design/synthetic_control/predictors/extended` | Extended predictors (add reasonable pre-treatment covariates) |

### D) Pre-treatment fit window

| spec_id | Description |
|---|---|
| `design/synthetic_control/preperiod/full` | Full pre-treatment period |
| `design/synthetic_control/preperiod/short` | Shorter pre-period (recent years only) |
| `design/synthetic_control/preperiod/long` | Longer pre-period (if available) |

### E) Multiple treated units (if applicable)

| spec_id | Description |
|---|---|
| `design/synthetic_control/multi/pooled` | Pool treated units and estimate an average effect |
| `design/synthetic_control/multi/separate` | Separate synthetic control per treated unit + summary |

## Standard synthetic-control diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the comparative-case estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible SCM/SDID reporting.

- `diag/synthetic_control/fit/pre_rmspe`
- `diag/synthetic_control/weights/donor_weight_concentration`
- `diag/synthetic_control/placebo/in_space` (in-space placebos / permutation distribution summary)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Pre-processing/coding (standardization, deflators, scaling, index construction): `specification_tree/modules/robustness/preprocessing.md`
- Data construction (aggregation, denominators, deflators): `specification_tree/modules/robustness/data_construction.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Diagnostics (`diag/*`)

- Synthetic-control diagnostics (fit/placebos): `specification_tree/modules/diagnostics/design_diagnostics.md`

### Post-processing (`post/*`)

- Specification-curve / multiverse summaries: `specification_tree/modules/postprocess/specification_curve.md`

### Exploration (`explore/*`)

- Alternative outcome definitions and timing choices: `specification_tree/modules/exploration/variable_definitions.md`
- Heterogeneity: `specification_tree/modules/exploration/heterogeneity.md`
