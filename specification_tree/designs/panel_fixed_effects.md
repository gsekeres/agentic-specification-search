# Design: Panel Fixed Effects (Within / Additive FE Regression)

This design file enumerates **within-design estimator implementations** for panel regressions that rely on additive fixed effects and within-unit identifying variation under maintained exogeneity conditions.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical panel-FE estimate for the claim object.
- Record: panel index (unit, time), FE structure, clustering/inference, and the paper’s scalar focal parameter.
- Record design-defining metadata under `coefficient_vector_json.design.panel_fixed_effects` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design estimator implementations (`design/panel_fixed_effects/*`)

Spec ID format:

- `design/panel_fixed_effects/{axis}/{variant}`

### A) Baseline-class estimator

| spec_id | Description |
|---|---|
| `design/panel_fixed_effects/estimator/within` | Within estimator / additive FE regression (paper baseline class) |

### B) Differencing-style implementations (optional)

These can target the same causal object in some settings, but they may change identifying variation and finite-sample behavior. Use only when the paper’s code/setting makes them meaningful.

| spec_id | Description |
|---|---|
| `design/panel_fixed_effects/estimator/first_difference` | First-difference estimator |
| `design/panel_fixed_effects/estimator/long_difference` | Long-difference (paper-aligned horizon) |

### C) Correlated random effects (optional comparison)

This changes assumptions (RE-style likelihood + Mundlak controls). Treat as a **comparison** rather than an estimand-preserving robustness check unless the paper explicitly frames it as an implementation alternative.

| spec_id | Description |
|---|---|
| `design/panel_fixed_effects/estimator/correlated_random_effects` | Correlated random effects / Mundlak |

## Standard panel diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the panel-FE estimand. They are not part of the default core surface, but they should be computed and recorded when applicable because they are standard in many panel workflows.

- `diag/panel_fixed_effects/fe_vs_re/hausman` (when an RE comparison is meaningful)
- `diag/panel_fixed_effects/serial_corr/wooldridge` (when feasible; residual serial correlation)
- `diag/panel_fixed_effects/cross_sectional_dep/pesaran_cd` (when relevant/feasible)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls / adjustment set: `specification_tree/modules/robustness/controls.md`
- Sample rules (balanced panels, trimming): `specification_tree/modules/robustness/sample.md`
- Fixed effects (add/drop FE dimensions): `specification_tree/modules/robustness/fixed_effects.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`
- Data construction (panel building/aggregation): `specification_tree/modules/robustness/data_construction.md`
- Functional form: `specification_tree/modules/robustness/functional_form.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- SE and clustering (including Driscoll–Kraay, Newey–West when appropriate): `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- General regression diagnostics: `specification_tree/modules/diagnostics/regression_diagnostics.md`
- Design diagnostics (panel section): `specification_tree/modules/diagnostics/design_diagnostics.md`
