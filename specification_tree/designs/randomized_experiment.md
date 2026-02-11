# Design: Randomized Experiment (RCT / Field Experiment)

This design file enumerates **within-design estimator implementations** for randomized assignment under maintained randomization/exchangeability.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical experimental estimate for the claim object (often ITT).
- Record: randomization unit, stratification/blocking, weights, and the paper’s outcome/treatment definitions.

## Design estimator implementations (`design/randomized_experiment/*`)

Spec ID format:

- `design/randomized_experiment/{axis}/{variant}`

### A) ITT implementations (estimand-preserving under random assignment)

| spec_id | Description |
|---|---|
| `design/randomized_experiment/estimator/diff_in_means` | Difference-in-means (minimal adjustment) |
| `design/randomized_experiment/estimator/ancova` | ANCOVA (controls for baseline outcome when available) |
| `design/randomized_experiment/estimator/with_covariates` | Add pre-treatment covariates used by the paper |
| `design/randomized_experiment/estimator/strata_fe` | Include strata/block fixed effects (when randomized within strata) |

### B) Treatment intensity / dose response (only if baseline treatment is continuous/intensity)

| spec_id | Description |
|---|---|
| `design/randomized_experiment/intensity/linear` | Linear dose/intensity model |
| `design/randomized_experiment/intensity/categories` | Categorize intensity into a small number of bins (paper-aligned) |

## Standard RCT diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the experimental estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible RCT reporting.

- `diag/randomized_experiment/balance/covariates`
- `diag/randomized_experiment/attrition/attrition_diff` (when outcomes are missing/attrition is nontrivial)
- `diag/randomized_experiment/noncompliance/first_stage` (only when noncompliance exists and TOT/LATE is analyzed)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls / covariates (selection is typically pre-treatment precision, not identification): `specification_tree/modules/robustness/controls.md`
- Sample rules / attrition-as-sample rules: `specification_tree/modules/robustness/sample.md`
- Pre-processing/coding (includes missingness handling): `specification_tree/modules/robustness/preprocessing.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- SE and clustering (cluster at the randomization unit when appropriate): `specification_tree/modules/inference/standard_errors.md`
- Randomization inference / wild cluster bootstrap: `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- Design diagnostics (balance, attrition diagnostics): `specification_tree/modules/diagnostics/design_diagnostics.md`

### Sensitivity (`sens/*`)

- Randomized-experiment assumption sensitivity (attrition bounds, interference-type issues): `specification_tree/modules/sensitivity/assumptions/randomized_experiment.md`

### Exploration (`explore/*`)

- TOT/LATE instead of ITT (noncompliance): `specification_tree/modules/exploration/alternative_estimands.md`
- Spillovers / alternative exposure mappings: `specification_tree/modules/exploration/variable_definitions.md`
