# Design: Dynamic Panel (Lagged Outcomes; Panel GMM / Related)

This design file enumerates **within-design implementation choices** for dynamic panel models where lagged dependent variables and/or endogenous regressors require specialized estimators (e.g., Arellano–Bond / Blundell–Bond GMM).

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical dynamic-panel estimate for the claim object.
- Record: lag structure, instrument strategy, and the paper’s focal parameterization (short-run vs long-run effects).

## Design estimator implementations (`design/dynamic_panel/*`)

Spec ID format:

- `design/dynamic_panel/{axis}/{variant}`

### A) Estimator family

| spec_id | Description |
|---|---|
| `design/dynamic_panel/estimator/ols_fe` | FE with lagged outcome (biased; comparison) |
| `design/dynamic_panel/estimator/anderson_hsiao` | Anderson–Hsiao IV (first-difference + lagged instruments) |
| `design/dynamic_panel/estimator/diff_gmm` | Difference GMM (Arellano–Bond) |
| `design/dynamic_panel/estimator/sys_gmm` | System GMM (Blundell–Bond) |
| `design/dynamic_panel/estimator/lsdv_bias_corrected` | Bias-corrected LSDV (when implementable) |

### B) Lag structure (paper-relevant grid)

| spec_id | Description |
|---|---|
| `design/dynamic_panel/lags/y_1` | One lag of dependent variable |
| `design/dynamic_panel/lags/y_2` | Two lags of dependent variable |
| `design/dynamic_panel/lags/aic_optimal` | AIC-selected lag length (if used) |
| `design/dynamic_panel/lags/bic_optimal` | BIC-selected lag length (if used) |

### C) Instrument strategy (GMM implementations)

| spec_id | Description |
|---|---|
| `design/dynamic_panel/instruments/full` | Full instrument set (paper baseline class) |
| `design/dynamic_panel/instruments/collapsed` | Collapsed instrument matrix (limits proliferation) |
| `design/dynamic_panel/instruments/lag_2_4` | Use lags 2–4 as instruments |
| `design/dynamic_panel/instruments/lag_2_6` | Use lags 2–6 as instruments |
| `design/dynamic_panel/instruments/limited` | Aggressively limited instruments (few instruments) |

### D) One-step vs two-step weighting (GMM)

| spec_id | Description |
|---|---|
| `design/dynamic_panel/gmm/one_step` | One-step GMM |
| `design/dynamic_panel/gmm/two_step` | Two-step GMM |
| `design/dynamic_panel/gmm/windmeijer` | Two-step with Windmeijer correction (SE adjustment) |

## Standard dynamic-panel diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the dynamic-panel estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible GMM reporting.

- `diag/dynamic_panel/ar/ar1` and `diag/dynamic_panel/ar/ar2`
- `diag/dynamic_panel/overid/hansen` (and/or Sargan, paper-specific)
- `diag/dynamic_panel/instruments/count` (instrument count vs groups; proliferation guardrail)

## Long-run effects (audit requirement)

If the paper reports long-run effects (e.g., \u03b2/(1-\u03b1) in an AR(1)-type model), record:

- the short-run coefficient(s),
- the persistence parameter(s),
- the derived long-run quantity,

in `coefficient_vector_json.long_run` (and ensure scalar fields correspond to the paper’s focal object).

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls / coding / sample rules: `specification_tree/modules/robustness/controls.md`
- Sample restrictions: `specification_tree/modules/robustness/sample.md`
- Fixed effects: `specification_tree/modules/robustness/fixed_effects.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`

### Inference (`infer/*`)

- SE and clustering: `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- Design diagnostics menu: `specification_tree/modules/diagnostics/design_diagnostics.md`
