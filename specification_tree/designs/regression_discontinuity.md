# Design: Regression Discontinuity (RD)

This design file enumerates **within-design estimator implementations** for RD-style identification at a cutoff, under maintained continuity-type assumptions.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical RD estimate for the claim object.
- Record: running variable, cutoff, bandwidth rule, kernel, polynomial order, and the paper’s chosen scalar summary of the discontinuity.
- Store these design-defining parameters under `coefficient_vector_json.design.regression_discontinuity` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design estimator implementations (`design/regression_discontinuity/*`)

Spec ID format:

- `design/regression_discontinuity/{axis}/{variant}`

### A) RD type (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/regression_discontinuity/type/sharp` | Sharp RD (deterministic treatment at cutoff) |
| `design/regression_discontinuity/type/fuzzy` | Fuzzy RD (imperfect compliance; IV-at-cutoff estimand) |
| `design/regression_discontinuity/type/kink` | Regression kink design (derivative discontinuity) |

### B) Bandwidth selection

| spec_id | Description |
|---|---|
| `design/regression_discontinuity/bandwidth/ik` | Imbens–Kalyanaraman bandwidth |
| `design/regression_discontinuity/bandwidth/ccft` | Calonico–Cattaneo–Farrell–Titiunik bandwidth |
| `design/regression_discontinuity/bandwidth/half_baseline` | Half the paper’s baseline bandwidth |
| `design/regression_discontinuity/bandwidth/double_baseline` | Double the paper’s baseline bandwidth |
| `design/regression_discontinuity/bandwidth/fixed_small` | Fixed “small” bandwidth (paper-aligned units) |
| `design/regression_discontinuity/bandwidth/fixed_large` | Fixed “large” bandwidth |

### C) Local polynomial order

| spec_id | Description |
|---|---|
| `design/regression_discontinuity/poly/local_linear` | Local linear (order 1) |
| `design/regression_discontinuity/poly/local_quadratic` | Local quadratic (order 2) |
| `design/regression_discontinuity/poly/local_cubic` | Local cubic (order 3) |

### D) Kernel choice

| spec_id | Description |
|---|---|
| `design/regression_discontinuity/kernel/triangular` | Triangular (default) |
| `design/regression_discontinuity/kernel/uniform` | Uniform (rectangular) |
| `design/regression_discontinuity/kernel/epanechnikov` | Epanechnikov |

### E) Estimation/inference procedure (RD-specific)

| spec_id | Description |
|---|---|
| `design/regression_discontinuity/procedure/conventional` | Conventional (no bias correction) |
| `design/regression_discontinuity/procedure/robust_bias_corrected` | Robust bias-corrected (Calonico et al. style) |
| `design/regression_discontinuity/procedure/honest_ci` | “Honest” confidence intervals (when implementable) |

## Standard RD diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the RD estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible RD reporting.

- `diag/regression_discontinuity/manipulation/mccrary_density`
- `diag/regression_discontinuity/balance/covariate_continuity`
- `diag/regression_discontinuity/functional_form/bin_sensitivity_plot` (or an equivalent RD plot/binned-sensitivity summary)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls / covariates (when used as precision): `specification_tree/modules/robustness/controls.md`
- Sample rules (includes donut-style exclusions when treated as RC): `specification_tree/modules/robustness/sample.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`
- Data construction: `specification_tree/modules/robustness/data_construction.md`
- Functional form: `specification_tree/modules/robustness/functional_form.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- SE and clustering (when applicable): `specification_tree/modules/inference/standard_errors.md`
- Resampling / randomization inference: `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- RD diagnostics (density, covariate balance, placebo cutoffs): `specification_tree/modules/diagnostics/design_diagnostics.md`
- Placebos: `specification_tree/modules/diagnostics/placebos.md`

### Sensitivity (`sens/*`)

- RD-assumption sensitivity: `specification_tree/modules/sensitivity/assumptions/regression_discontinuity.md`

### Exploration (`explore/*`)

- Heterogeneity / subgroup discontinuities: `specification_tree/modules/exploration/heterogeneity.md`
- Alternative estimands: `specification_tree/modules/exploration/alternative_estimands.md`
