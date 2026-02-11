# RD Assumption Sensitivity (Continuity, Manipulation, Functional Form)

## Spec ID format

Use:

- `sens/assumption/regression_discontinuity/{assumption}/{variant}`

RD designs rely on local continuity assumptions and are sensitive to bandwidth/functional-form choices. This file enumerates **assumption-focused sensitivity analyses** beyond routine robustness.

## Key RD assumptions

1) **Continuity of potential outcomes** at the cutoff in the absence of treatment.
2) **No manipulation / sorting** of the running variable around the cutoff.
3) **Local comparability**: covariates and composition evolve smoothly through the cutoff.
4) **Correct local approximation**: functional form and bandwidth capture the local limit.

## A) Continuity / functional-form sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/regression_discontinuity/functional_form/bw_selector_alt` | Alternative bandwidth selectors (when implementable) |
| `sens/assumption/regression_discontinuity/functional_form/kernel_alt` | Alternative kernels (triangular vs uniform vs Epanechnikov) |
| `sens/assumption/regression_discontinuity/functional_form/poly_order_grid` | Grid over polynomial order; report stability |
| `sens/assumption/regression_discontinuity/functional_form/rbc_vs_conventional` | Compare robust-bias-corrected vs conventional inference |

## B) Manipulation / sorting sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/regression_discontinuity/manipulation/donut_grid` | Donut RD grid: exclude a window around cutoff; report effect vs donut size |
| `sens/assumption/regression_discontinuity/manipulation/cutoff_placebo_grid` | Placebo cutoffs at nearby thresholds; report distribution |

## C) Local randomization sensitivity

When RD is treated as local randomization in a narrow window, results depend on window width and assignment-model choices.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/regression_discontinuity/local_randomization/window_grid` | Grid over local-randomization windows; report inference stability |
| `sens/assumption/regression_discontinuity/local_randomization/rank_inference` | Randomization inference within the chosen window |

## Related diagnostics (to be typed elsewhere)

Related menus:

- Diagnostics: `specification_tree/modules/diagnostics/design_diagnostics.md` (RD manipulation/balance/plots)
- Inference: `specification_tree/modules/inference/standard_errors.md` and `specification_tree/modules/inference/resampling.md`

- McCrary density test / manipulation tests
- covariate balance around cutoff
- continuity checks for predetermined outcomes
