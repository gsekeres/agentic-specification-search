# Design: Local Projections (Impulse Responses / Dynamic Responses)

Local projections (Jordà 2005) estimate dynamic responses by running a separate regression at each horizon \(h\), producing an impulse-response path.

This design file enumerates **within-design implementation choices** for local projections. Because LPs produce a **vector**, the output contract requires:

1) a declared scalar focal summary for `coefficient/std_error/p_value`, and
2) storage of the full path in `coefficient_vector_json`.

See `specification_tree/CONTRACT.md`.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical LP object (shock definition, horizons, controls/dynamics, and inference).
- Record the full IRF path \(\{\beta_h\}_{h=0}^H\).
- Record design-defining metadata under `coefficient_vector_json.design.local_projection` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design implementation variants (`design/local_projection/*`)

Spec ID format:

- `design/local_projection/{axis}/{variant}`

### A) Horizon window / grid

| spec_id | Description |
|---|---|
| `design/local_projection/horizon/short` | Short window (paper-aligned; e.g., \(H=4\)) |
| `design/local_projection/horizon/medium` | Medium window (e.g., \(H=12\)) |
| `design/local_projection/horizon/long` | Long window (paper max, e.g., \(H=24\)) |

### B) Dynamic controls (LP specification)

These are design-specific “dynamics” choices, distinct from general covariate inclusion.

| spec_id | Description |
|---|---|
| `design/local_projection/dynamics/y_lags_only` | Include lags of \(Y\) only |
| `design/local_projection/dynamics/shock_lags_only` | Include lags of the shock only |
| `design/local_projection/dynamics/y_and_shock_lags` | Include both \(Y\) and shock lags |
| `design/local_projection/dynamics/paper_baseline` | Paper’s baseline dynamic controls |

### C) Local-projection-IV / proxy shocks (paper-relevant only)

If the paper’s LP uses an external instrument/proxy to identify the shock, treat the IV structure as part of the design implementation.

| spec_id | Description |
|---|---|
| `design/local_projection/shock_id/ols_shock` | Shock treated as observed/exogenous (paper-relevant only) |
| `design/local_projection/shock_id/proxy_iv` | Proxy/LP-IV implementation (external instrument) |

## Standard LP diagnostics (record when applicable)

These are **diagnostics**, not estimates of the focal dynamic estimand. They are not part of the default core surface, but should be computed/recorded when the paper relies on them.

- `diag/local_projection/irf/path_recorded` (sanity: full path stored; horizon grid matches)
- `diag/local_projection/inference/hac_bandwidth_rule` (recorded in JSON; when HAC is used)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls (non-dynamic covariates): `specification_tree/modules/robustness/controls.md`
- Sample rules: `specification_tree/modules/robustness/sample.md`
- Fixed effects (panel LP): `specification_tree/modules/robustness/fixed_effects.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`

### Inference (`infer/*`)

- HAC / Driscoll–Kraay / clustering: `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Exploration (`explore/*`)

- Alternative shock series / timing definitions: `specification_tree/modules/exploration/variable_definitions.md`
