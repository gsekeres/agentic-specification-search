# Difference-in-Differences / Event-Study Assumption Sensitivity (Parallel Trends, Anticipation, Composition)

## Spec ID format

Use:

- `sens/assumption/difference_in_differences/{assumption}/{variant}`

This file records **design-specific sensitivity analyses** for DiD/event-study designs that go beyond routine control/sample/SE robustness.

## Key DiD assumptions

1) **Parallel trends** (in the relevant counterfactual): absent treatment, treated and control would follow comparable trends (possibly conditional on covariates).
2) **No anticipation**: outcomes do not respond before treatment takes effect (or anticipation is explicitly modeled).
3) **Stable composition / no selective attrition**: the composition of units does not change differentially in a way that mimics treatment effects.
4) **No spillovers / interference**: one unit’s treatment does not affect another unit’s outcome (often violated in spatial/policy settings).

## A) Parallel-trends sensitivity (structured deviations)

These methods quantify how large a violation must be for conclusions to change, or produce “honest” intervals under bounded deviations.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/difference_in_differences/parallel_trends/rambachan_roth` | Honest DiD sensitivity under smoothness/bounded second differences (when implementable) |
| `sens/assumption/difference_in_differences/parallel_trends/rambachan_roth_grid` | Grid over violation parameters; report breakdown threshold |
| `sens/assumption/difference_in_differences/parallel_trends/linear_drift_bound` | Allow linear differential drift of bounded magnitude; report effect bounds |
| `sens/assumption/difference_in_differences/parallel_trends/pretrend_extrapolation_bound` | Use estimated pre-trends to extrapolate worst-case bias bounds |
| `sens/assumption/difference_in_differences/parallel_trends/placebo_breakdown` | Smallest placebo effect (in pre-period) that would rationalize the estimated post effect |

## B) Anticipation sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/difference_in_differences/no_anticipation/allow_k_leads` | Allow k pre-treatment leads to be nonzero; report sensitivity of post estimates |
| `sens/assumption/difference_in_differences/no_anticipation/recode_treatment_start` | Alternative plausible treatment-start timing (when ambiguity exists) |

Note: if changing treatment timing changes the treatment concept, reclassify as `explore/definition/treatment/*`.

## C) Composition / attrition sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/difference_in_differences/composition/ipw_attrition` | Inverse-probability weighting for differential attrition (under MAR-type assumptions) |
| `sens/assumption/difference_in_differences/composition/lee_bounds` | Lee bounds for attrition under monotone selection (when applicable) |
| `sens/assumption/difference_in_differences/composition/reweight_to_pre` | Reweight post-period composition to match pre-period covariates |

## D) Spillovers / interference sensitivity (often exploration-adjacent)

Spillover checks frequently change the estimand (e.g., redefining treatment exposure using distance). Default to exploration unless the baseline claim explicitly defines exposure via a mapping.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/difference_in_differences/interference/drop_neighbors` | Exclude units plausibly exposed to spillovers (buffer) |
| `sens/assumption/difference_in_differences/interference/exposure_buffer_grid` | Grid over buffer radii; report sensitivity |

## Related diagnostics / inference (to be typed elsewhere)

Related menus:

- Diagnostics: `specification_tree/modules/diagnostics/design_diagnostics.md` (pretrend tests, placebos, support diagnostics)
- Inference: `specification_tree/modules/inference/standard_errors.md` and `specification_tree/modules/inference/resampling.md`

This file is for **assumption-relaxation / sensitivity** objects only.
