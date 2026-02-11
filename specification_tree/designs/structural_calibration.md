# Design: Structural Calibration / Moment Matching / Simulation-Based Models

This design file enumerates **within-design implementation choices** for papers whose main evidence object is a calibrated or estimated structural model matched to data moments, with counterfactual exercises as the main outputs.

These papers often do not produce a single regression coefficient per “spec”. The output contract still requires a scalar focal summary per run when integrating into the unified pipeline (and storage of the full set of parameters/moments in JSON).

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical calibrated model / structural estimate:
  - parameter values/posteriors,
  - targeted moments,
  - fit statistics,
  - and the headline counterfactual quantity.

## Design implementation variants (`design/structural_calibration/*`)

Spec ID format:

- `design/structural_calibration/{axis}/{variant}`

### A) Moment/target set

| spec_id | Description |
|---|---|
| `design/structural_calibration/targets/baseline` | Paper’s baseline moment conditions |
| `design/structural_calibration/targets/subset` | Subset of baseline moments |
| `design/structural_calibration/targets/extended` | Add plausible additional moments |

### B) Parameter treatment (fixed vs estimated)

| spec_id | Description |
|---|---|
| `design/structural_calibration/parameters/fixed_baseline` | Paper’s baseline fixed/calibrated parameters |
| `design/structural_calibration/parameters/estimate_more` | Estimate previously fixed parameters (if feasible) |
| `design/structural_calibration/parameters/fix_more` | Fix previously estimated parameters (if justified) |

### C) Functional-form alternatives (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/structural_calibration/functional_form/baseline` | Paper’s baseline functional forms/distributions |
| `design/structural_calibration/functional_form/alternative_distributions` | Alternative distributional assumptions |
| `design/structural_calibration/functional_form/alternative_preferences` | Alternative preference/technology parameterizations |

### D) Sample window / calibration dataset

| spec_id | Description |
|---|---|
| `design/structural_calibration/sample/full` | Full sample |
| `design/structural_calibration/sample/pre_crisis` | Pre-crisis subsample (paper-defined) |
| `design/structural_calibration/sample/post_crisis` | Post-crisis subsample (paper-defined) |

### E) Counterfactual definition (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/structural_calibration/counterfactual/baseline` | Paper’s headline counterfactual |
| `design/structural_calibration/counterfactual/alternative_policy` | Alternative counterfactual policy/scenario |

## Standard structural-model diagnostics (record when applicable)

These are **diagnostics**, not estimates of the focal counterfactual quantity. They are not part of the default core surface, but should be computed/recorded when the paper relies on them.

- `diag/structural_calibration/fit/moment_table` (data vs model moments)
- `diag/structural_calibration/fit/overid_test` (when meaningful/available)
- `diag/structural_calibration/uncertainty/bootstrap_or_posterior` (uncertainty source recorded)

## Typed references to universal modules (do not duplicate here)

### Inference (`infer/*`)

- Resampling (bootstrap) and related uncertainty procedures: `specification_tree/modules/inference/resampling.md`

### Exploration (`explore/*`)

- Alternative estimands (welfare/policy objects): `specification_tree/modules/exploration/alternative_estimands.md`
