# Design: Structural VAR (SVAR) / Impulse Responses

This design file enumerates **within-design implementation choices** for VAR/SVAR analyses that produce impulse responses (IRFs).

SVAR outputs are typically **vectors/paths** (IRFs across horizons). The output contract requires:

1) a declared scalar focal summary for `coefficient/std_error/p_value`, and
2) storage of the full IRF path in `coefficient_vector_json`.

See `specification_tree/CONTRACT.md`.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical SVAR object:
  - variable set,
  - lag length,
  - identification scheme,
  - sample period,
  - shock normalization,
  - and focal response variable/horizon.

## Design implementation variants (`design/structural_var/*`)

Spec ID format:

- `design/structural_var/{axis}/{variant}`

### A) Identification scheme (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/structural_var/id/cholesky` | Recursive (Cholesky) identification |
| `design/structural_var/id/sign_restrictions` | Sign restrictions |
| `design/structural_var/id/zero_restrictions` | Short- or long-run zero restrictions |
| `design/structural_var/id/sign_and_zero` | Combined sign + zero restrictions |
| `design/structural_var/id/narrative` | Narrative identification (shock series) |
| `design/structural_var/id/external_iv` | Proxy SVAR / external instrument |

### B) Variable ordering (recursive settings)

| spec_id | Description |
|---|---|
| `design/structural_var/order/baseline` | Paper’s ordering |
| `design/structural_var/order/reversed` | Reversed ordering |
| `design/structural_var/order/alternative` | Alternative economically motivated ordering |

### C) Lag length

| spec_id | Description |
|---|---|
| `design/structural_var/lags/p1` | 1 lag |
| `design/structural_var/lags/p2` | 2 lags |
| `design/structural_var/lags/p3` | 3 lags |
| `design/structural_var/lags/p4` | 4 lags |
| `design/structural_var/lags/aic_optimal` | AIC-selected lag length |
| `design/structural_var/lags/bic_optimal` | BIC-selected lag length |

### D) Sample window (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/structural_var/sample/full` | Full sample |
| `design/structural_var/sample/pre_crisis` | Pre-crisis subsample (paper-defined) |
| `design/structural_var/sample/post_crisis` | Post-crisis subsample (paper-defined) |
| `design/structural_var/sample/rolling` | Rolling window estimation (paper-defined) |

### E) Bayesian vs frequentist estimation (if both are feasible in the package)

| spec_id | Description |
|---|---|
| `design/structural_var/estimation/frequentist` | Standard VAR/SVAR estimation |
| `design/structural_var/estimation/bayesian_minnesota` | Bayesian VAR with Minnesota prior |

## Standard SVAR diagnostics (record when applicable)

These are **diagnostics**, not estimates of the focal dynamic estimand. They are not part of the default core surface, but should be computed/recorded when the paper relies on them.

- `diag/structural_var/stability/roots_inside_unit_circle`
- `diag/structural_var/fit/residual_autocorr` (lag adequacy)
- `diag/structural_var/normalization/shock_scale_recorded`

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Pre-processing/coding (deflators, transformations): `specification_tree/modules/robustness/preprocessing.md`
- Data construction (vintage, aggregation): `specification_tree/modules/robustness/data_construction.md`

### Inference (`infer/*`)

- SE and resampling (credible intervals, bootstrap where applicable): `specification_tree/modules/inference/resampling.md`

### Exploration (`explore/*`)

- Alternative variable definitions / shock series: `specification_tree/modules/exploration/variable_definitions.md`
- Alternative estimands: `specification_tree/modules/exploration/alternative_estimands.md`
- Local-projection comparisons (different design family): `specification_tree/designs/local_projection.md`
