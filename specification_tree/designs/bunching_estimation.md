# Design: Bunching Estimation (Kinks / Notches)

Bunching estimation uses discontinuities in incentive schedules (kinks/notches) to infer behavioral responses (often elasticities) from excess mass around thresholds.

This design file enumerates **within-design implementation choices** for bunching estimators.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical bunching estimate.
- Record: threshold(s), binning, bandwidth/window, counterfactual specification, and the paper’s elasticity/excess-mass mapping.

## Design implementation variants (`design/bunching_estimation/*`)

Spec ID format:

- `design/bunching_estimation/{axis}/{variant}`

### A) Counterfactual specification (polynomial order / flexibility)

| spec_id | Description |
|---|---|
| `design/bunching_estimation/counterfactual/poly_order_3` | Polynomial counterfactual (order 3) |
| `design/bunching_estimation/counterfactual/poly_order_5` | Polynomial counterfactual (order 5; common baseline) |
| `design/bunching_estimation/counterfactual/poly_order_7` | Polynomial counterfactual (order 7) |

### B) Window / bandwidth around threshold

| spec_id | Description |
|---|---|
| `design/bunching_estimation/bandwidth/half_baseline` | Half baseline window |
| `design/bunching_estimation/bandwidth/baseline` | Paper baseline window |
| `design/bunching_estimation/bandwidth/double_baseline` | Double baseline window |

### C) Multiple thresholds (if present)

| spec_id | Description |
|---|---|
| `design/bunching_estimation/threshold/single` | Threshold-specific estimate |
| `design/bunching_estimation/threshold/pooled` | Pooled estimate across thresholds |

### D) Standard-error construction (design-specific)

| spec_id | Description |
|---|---|
| `design/bunching_estimation/se/bootstrap` | Bootstrap SEs (paper-aligned resampling unit) |
| `design/bunching_estimation/se/parametric` | Parametric/delta-method SEs (if used) |

## Standard bunching diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the bunching estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible bunching reporting.

- `diag/bunching_estimation/placebo/thresholds` (placebo thresholds)
- `diag/bunching_estimation/density/continuity` (density continuity around cutoff)
- `diag/bunching_estimation/donut/exclude_threshold_bin` (donut hole: exclude threshold bin)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Sample restrictions (subpopulations are usually exploration unless headline): `specification_tree/modules/robustness/sample.md`
- Pre-processing/coding (binning/top-coding of running variable): `specification_tree/modules/robustness/preprocessing.md`

### Inference (`infer/*`)

- Resampling procedures (bootstrap variants): `specification_tree/modules/inference/resampling.md`

### Sensitivity (`sens/*`)

- Assumption/measurement sensitivity (when available): `specification_tree/modules/sensitivity/unobserved_confounding.md`
