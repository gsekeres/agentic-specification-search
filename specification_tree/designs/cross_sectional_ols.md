# Design: Cross-Sectional Regression / Selection-on-Observables

This design file enumerates **within-design estimator implementations** for papers whose primary evidence object is a cross-sectional regression interpreted (explicitly or implicitly) as causal under adjustment for observables.

Universal RC, inference, diagnostics, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical cross-sectional estimate for the claim object.
- Record design-defining metadata under `coefficient_vector_json.design.cross_sectional_ols` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design estimator implementations (`design/cross_sectional_ols/*`)

Spec ID format:

- `design/cross_sectional_ols/{axis}/{variant}`

### A) Outcome-regression implementations (linear)

| spec_id | Description |
|---|---|
| `design/cross_sectional_ols/estimator/ols` | Linear outcome regression (paper baseline class) |

### B) Weighting and doubly-robust implementations (binary treatment; unconfoundedness)

Only applicable when the baseline estimand is an ATE/ATT-style object under unconfoundedness and the treatment is binary (or can be treated as such without changing the claim object).

| spec_id | Description |
|---|---|
| `design/cross_sectional_ols/estimator/ipw_ate` | IPW estimator for ATE (propensity model required) |
| `design/cross_sectional_ols/estimator/aipw_ate` | AIPW / doubly-robust ATE |
| `design/cross_sectional_ols/estimator/aipw_att` | AIPW-style ATT (if ATT is the baseline estimand) |

**Linkage constraint (important)**: AIPW/IPW involve bundled components (propensity + outcome models). If the manuscript reveals that covariate adjustment is shared across components, robustness search over controls should vary them **jointly** (do not mix-and-match across components). See `specification_tree/REVEALED_SEARCH_SPACE.md`.

### C) Matching-style implementations (optional; binary treatment)

| spec_id | Description |
|---|---|
| `design/cross_sectional_ols/estimator/matching_nn` | Nearest-neighbor matching (ATE/ATT depending on implementation) |
| `design/cross_sectional_ols/estimator/matching_kernel` | Kernel matching (if implementable) |

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls: `specification_tree/modules/robustness/controls.md`
- Sample rules: `specification_tree/modules/robustness/sample.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`
- Data construction: `specification_tree/modules/robustness/data_construction.md`
- Functional form: `specification_tree/modules/robustness/functional_form.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- SE and clustering: `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- General regression diagnostics: `specification_tree/modules/diagnostics/regression_diagnostics.md`

### Sensitivity (`sens/*`)

- Unobserved confounding sensitivity: `specification_tree/modules/sensitivity/unobserved_confounding.md`

### Exploration (`explore/*`)

- Alternative variable definitions: `specification_tree/modules/exploration/variable_definitions.md`
- Alternative estimands: `specification_tree/modules/exploration/alternative_estimands.md`
- Heterogeneity: `specification_tree/modules/exploration/heterogeneity.md`
- CATE: `specification_tree/modules/exploration/cate_estimation.md`
- Policy learning: `specification_tree/modules/exploration/policy_learning.md`
