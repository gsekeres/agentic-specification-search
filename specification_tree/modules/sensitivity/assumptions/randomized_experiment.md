# Randomized-Experiment Assumption Sensitivity (Attrition, Noncompliance, Interference)

## Spec ID format

Use:

- `sens/assumption/randomized_experiment/{assumption}/{variant}`

Randomized experiments are strong on identification but remain vulnerable to practical threats (attrition, noncompliance, spillovers). This file records sensitivity analyses for those threats.

## Key randomized-experiment assumptions (practical)

1) **Random assignment** (often conditional on strata/blocks).
2) **No differential attrition / missing outcomes** (or missingness is ignorable).
3) **No interference / spillovers** (SUTVA).
4) **Compliance**: ITT is identified; TOT/LATE interpretation requires additional assumptions.

## A) Attrition / missing outcomes sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/randomized_experiment/attrition/lee_bounds` | Lee bounds under monotone attrition |
| `sens/assumption/randomized_experiment/attrition/manski_worst_case` | Worst-case (Manski) bounds under bounded outcomes |
| `sens/assumption/randomized_experiment/attrition/ipw_mcar_mar` | Inverse-probability weighting under MAR-style models |
| `sens/assumption/randomized_experiment/attrition/tipping_point` | Tipping-point analysis: how bad must missing outcomes be to flip conclusion |

## B) Noncompliance sensitivity

Noncompliance changes which estimand is being estimated (ITT vs TOT/LATE). Treat TOT as exploration unless the baseline claim is TOT.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/randomized_experiment/compliance/as_treated` | As-treated estimate (biased; diagnostic) |
| `sens/assumption/randomized_experiment/compliance/per_protocol` | Per-protocol estimate (biased; diagnostic) |
| `sens/assumption/randomized_experiment/compliance/iv_tot` | IV/TOT estimate using assignment as instrument (estimand change if baseline is ITT) |
| `sens/assumption/randomized_experiment/compliance/bounds` | Bounds under partial compliance (when implementable) |

## C) Interference / spillovers sensitivity

These checks often change the exposure mapping; default to exploration unless the baseline claim defines exposure explicitly.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/randomized_experiment/interference/drop_close_units` | Exclude close neighbors / buffer to reduce spillovers |
| `sens/assumption/randomized_experiment/interference/exposure_mapping_alt` | Alternative exposure mappings (e.g., share treated neighbors) |

## Related diagnostics / inference (references)

Record these under `diag/*` and `infer/*` (not under `sens/*`):

- baseline balance checks (covariate balance, randomization tests): `specification_tree/modules/diagnostics/design_diagnostics.md`
- randomization inference / permutation tests: `specification_tree/modules/inference/resampling.md`
- clustered SEs / design-based variance estimators (CR2, etc.): `specification_tree/modules/inference/standard_errors.md`
