# Design: Shift-Share / Bartik (Instrument Construction + IV)

Shift-share (Bartik) designs construct an instrument by interacting **exposure shares** with **aggregate shocks**, typically:

\[
Z_i = \sum_s \text{share}_{is} \cdot \text{shock}_s
\]

They are usually implemented as IV/2SLS with a **constructed instrument** and share/shock definitions that can materially affect identification, interpretation, and finite-sample behavior.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical shift-share construction and IV estimate for the claim object:
  - share definition (levels, base year, normalization, unit of aggregation),
  - shock definition (source series, base year, de-meaning/standardization),
  - leave-one-out logic if used,
  - the IV estimating equation (controls, FE, weights, sample).
- Record design-defining metadata under `coefficient_vector_json.design.shift_share` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design implementation variants (`design/shift_share/*`)

Spec ID format:

- `design/shift_share/{axis}/{variant}`

These are **within-design implementation choices**. Some may change the effective estimand if they materially change the instrument’s meaning; verification should be conservative about core eligibility when interpretation shifts.

### A) Share definition (exposure construction)

| spec_id | Description |
|---|---|
| `design/shift_share/shares/baseline` | Paper’s baseline share definition |
| `design/shift_share/shares/base_year_alt` | Alternative plausible base-year for shares (paper-relevant) |
| `design/shift_share/shares/normalize_to_one` | Normalize shares to sum to 1 within unit (when appropriate) |
| `design/shift_share/shares/topcode_concentration` | Top-code extreme shares (guardrail; record rule) |

### B) Shock definition (aggregate shifters)

| spec_id | Description |
|---|---|
| `design/shift_share/shocks/baseline` | Paper’s baseline shock series |
| `design/shift_share/shocks/demean_time` | Demean shock over time (if panel) |
| `design/shift_share/shocks/standardize` | Standardize shock (units-only; should not change sign conclusions) |
| `design/shift_share/shocks/source_alt` | Alternative plausible shock source series (paper-relevant; may change interpretation) |

### C) Leave-one-out / mechanical-correlation guardrails

| spec_id | Description |
|---|---|
| `design/shift_share/loo/off` | No leave-one-out adjustment (baseline if paper does not use) |
| `design/shift_share/loo/leave_unit_out` | Leave-one-out at the unit level (when shocks use unit outcomes) |
| `design/shift_share/loo/leave_region_out` | Leave-one-out at a higher aggregation (state/region) |

### D) Aggregation / construction mechanics

| spec_id | Description |
|---|---|
| `design/shift_share/construction/levels` | Construct \(Z\) in levels (paper baseline) |
| `design/shift_share/construction/log` | Construct using log shocks or log exposures (paper-relevant) |
| `design/shift_share/construction/per_capita` | Per-capita normalization when exposure is a total |

## Standard shift-share diagnostics (record when applicable)

These are **diagnostics**, not new estimates. They are not part of the default core surface, but they are standard for credible shift-share reporting and help catch fragile constructions.

- `diag/shift_share/weights/rotemberg` (Rotemberg weights / effective shocks)
- `diag/shift_share/concentration/share_hhi` (share concentration; dominance by a few sectors/shocks)
- `diag/shift_share/sanity/shares_sum_to_one` (when normalization is intended)

Also compute standard IV diagnostics under `diag/instrumental_variables/*` when feasible (first-stage strength, overid tests when applicable).

## Shift-share audit requirements (write to `coefficient_vector_json`)

For baseline and `design/shift_share/*` rows, record the instrument construction under `coefficient_vector_json.design.shift_share`:

```json
{
  "design": {
    "shift_share": {
      "share_unit": "county",
      "share_base_year": 1990,
      "share_vars": ["share_industry1", "share_industry2"],
      "shock_series": "national_industry_growth",
      "shock_window": "1990-2000",
      "leave_one_out": "leave_unit_out",
      "normalization": "shares_sum_to_one",
      "formula": "Z_i = sum_s share_{is} * shock_s",
      "notes": "Records the constructed instrument definition; changes here may change interpretation."
    }
  }
}
```

If implemented as IV, also record a `bundle` block in `coefficient_vector_json` (as for generic IV) indicating whether adjustment sets are linked across stages. See `specification_tree/REVEALED_SEARCH_SPACE.md`.

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

- Unified design diagnostics: `specification_tree/modules/diagnostics/design_diagnostics.md`

### Sensitivity (`sens/*`)

- IV assumption sensitivity: `specification_tree/modules/sensitivity/assumptions/instrumental_variables.md`

### Exploration (`explore/*`)

- Alternative exposure mappings / instrument constructions (often change LATE): `specification_tree/modules/exploration/variable_definitions.md`
