# Joint Robustness Variants (Multi-Axis RC)

## Spec ID format

Use:

- `rc/joint/{family}/{variant}`

Examples:

- `rc/joint/sample_preprocess/window_1980_2000__hp1600`
- `rc/joint/controls_fe/baseline_controls__region_time_fe`
- `rc/joint/sampled/draw_017`

## Purpose

The project default is **one-axis-at-a-time** robustness (`rc/controls/*` OR `rc/sample/*` OR `rc/form/*`, etc.) because it is easier to interpret.

Use `rc/joint/*` only when:

- the manuscript reveals **linked axes** that must move together (e.g., a specific sample window paired with a specific detrending rule), or
- the design is inherently multi-dimensional and one-axis variation would produce incoherent specs.

If the joint change materially alters the claim object (estimand/population/concept), it should be recorded as `explore/*` instead of `rc/*`.

## Required audit fields (`coefficient_vector_json`)

Every `rc/joint/*` row must include a `joint` block:

```json
{
  "joint": {
    "spec_id": "rc/joint/sample_preprocess/window_1980_2000__hp1600",
    "axes_changed": ["sample", "preprocess"],
    "details": {
      "sample_window": "1980-2000",
      "hp_lambda": 1600
    }
  }
}
```

Rules:

- `joint.spec_id` must equal the row `spec_id`
- `axes_changed` must be a non-empty list of axis labels
- `details` must be a JSON object (can be empty, but prefer explicit parameterization)

## Notes on budgeted sampling

When a paper’s revealed surface implies a large cross-product over multiple linked axes, the surface may budget the joint universe and include sampled joint draws such as `rc/joint/sampled/draw_{k}`.

In that case:

- record the sampler + seed + draw index in `coefficient_vector_json.sampling`,
- and record the realized joint choices in `joint.details`.

