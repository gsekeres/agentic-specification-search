# Fixed Effects & Absorption Structure (Robustness Checks)

## Spec ID format

Use:

- `rc/fe/{family}/{variant}`

Examples:

- `rc/fe/add/time`
- `rc/fe/add/unit`
- `rc/fe/drop/unit`
- `rc/fe/add/region_time`

## Purpose

Fixed effects choices can be among the most consequential “specification” degrees of freedom:

- absorbing unit heterogeneity (unit FE),
- absorbing aggregate shocks (time FE),
- absorbing differential trends (region×time, industry×time),
- changing identifying variation (within vs between).

This module defines **estimand-preserving** FE robustness checks when they plausibly preserve the baseline claim object.

## Core principle: RC vs estimand change

Some FE changes can change the estimand concept (e.g., pooled OLS vs unit FE).
Default rule:

- If the baseline claim is already defined in a within-unit framework, adding/removing *additional* FE is often RC.
- If an FE change materially changes the identifying variation (between → within), treat it as exploration unless the paper clearly treats it as an implementation choice.

When in doubt, run the spec but label it as exploration in verification (and record why).

## A) Additive FE variations (relative to baseline)

| spec_id | Description |
|---|---|
| `rc/fe/add/time` | Add time FE |
| `rc/fe/add/unit` | Add unit FE |
| `rc/fe/add/region` | Add region FE |
| `rc/fe/add/industry` | Add industry FE |
| `rc/fe/add/region_time` | Add region×time FE |
| `rc/fe/add/industry_time` | Add industry×time FE |

## B) Dropping FE (relative to baseline)

Use with care: dropping FE can change identification and is often closer to “method change” than “robustness”.

| spec_id | Description |
|---|---|
| `rc/fe/drop/time` | Drop time FE (if baseline includes) |
| `rc/fe/drop/unit` | Drop unit FE (if baseline includes) |

## Required audit fields (`coefficient_vector_json`)

Every `rc/fe/*` row should include:

```json
{
  "fixed_effects": {
    "spec_id": "rc/fe/add/region_time",
    "family": "add",
    "added": ["region:time"],
    "dropped": [],
    "baseline_fe": ["unit", "time"],
    "new_fe": ["unit", "time", "region:time"]
  }
}
```
