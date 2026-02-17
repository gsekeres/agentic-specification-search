# Design: Instrumental Variables (IV)

This design file enumerates **within-design estimator implementations** for IV identification under maintained relevance + exclusion-type assumptions.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical IV estimate for the claim object.
- Record design-defining metadata under `coefficient_vector_json.design.instrumental_variables` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

**Bundled estimator note (important)**: IV is a multi-component bundle (first stage + reduced form + second stage). Record a `bundle` block in `coefficient_vector_json` and enforce any revealed linkage constraints for adjustment sets across stages. See `specification_tree/REVEALED_SEARCH_SPACE.md`.

## Design estimator implementations (`design/instrumental_variables/*`)

Spec ID format:

- `design/instrumental_variables/{axis}/{variant}`

### A) Second-stage estimators (holding the instrument set fixed)

These target the same IV estimand *conditional on the instrument set*.

| spec_id | Description |
|---|---|
| `design/instrumental_variables/estimator/2sls` | Two-stage least squares |
| `design/instrumental_variables/estimator/liml` | LIML (often more stable under many/weak instruments) |
| `design/instrumental_variables/estimator/fuller` | Fuller-modified LIML |
| `design/instrumental_variables/estimator/gmm_2step` | Two-step GMM (overidentified settings) |
| `design/instrumental_variables/estimator/gmm_cue` | Continuously updated GMM (CUE) |
| `design/instrumental_variables/estimator/control_function` | Control-function implementation (when appropriate) |

### B) Jackknife / leave-out variants (optional; many instruments)

| spec_id | Description |
|---|---|
| `design/instrumental_variables/estimator/jive` | Jackknife IV (JIVE) |
| `design/instrumental_variables/estimator/leave_one_out` | Leave-one-out IV (when implementable) |
| `design/instrumental_variables/estimator/split_sample` | Split-sample IV (instrument construction / leave-out logic) |

## Standard IV diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the IV estimand. They are not part of the default core surface, but they should be computed and recorded (at least for the baseline and main design variants) because they are standard for credible IV reporting.

- `diag/instrumental_variables/strength/first_stage_f` (or an equivalent weak-IV statistic when robust/clustered)
- `diag/instrumental_variables/validity/overid_test` (only when overidentified)
- `diag/instrumental_variables/endogeneity/hausman` (when an OLS comparison is meaningful)

## IV-specific audit requirements (write to `coefficient_vector_json`)

For all baseline and `design/instrumental_variables/*` rows, include:

- endogenous regressor(s) and instrument(s),
- first-stage coefficient(s) and strength diagnostics (at minimum: first-stage F or Kleibergen–Paap where relevant),
- number of instruments and overidentification degrees of freedom when applicable,
- a `bundle` block recording whether controls/FE are linked across stages.

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

### Diagnostics (`diag/*`)

- Design diagnostics (includes weak-IV, overid, balance-type checks): `specification_tree/modules/diagnostics/design_diagnostics.md`

### Sensitivity (`sens/*`)

- Exclusion-restriction and IV-assumption sensitivity: `specification_tree/modules/sensitivity/assumptions/instrumental_variables.md`

### Exploration (`explore/*`)

- Instrument-set changes (often change LATE): `specification_tree/modules/exploration/variable_definitions.md`
- Alternative IV estimands (LATE/MTE): `specification_tree/modules/exploration/alternative_estimands.md`
- Heterogeneity: `specification_tree/modules/exploration/heterogeneity.md`

### Estimation wrappers

- DML as nuisance-learning layer (including IV-DML where appropriate): `specification_tree/modules/estimation/dml.md`
