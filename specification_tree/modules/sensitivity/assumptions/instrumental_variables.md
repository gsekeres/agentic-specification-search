# IV Assumption Sensitivity (Exclusion, Exogeneity, Monotonicity)

## Spec ID format

Use:

- `sens/assumption/instrumental_variables/{assumption}/{variant}`

This file records **design-specific sensitivity analyses** for instrumental variables beyond standard “controls/sample/SE” robustness. Many of these objects are partial-identification or bound-based.

## Scope and typing

- These are **sensitivity** objects (`sens/*`), not new estimands by default.
- Some procedures may change the estimand (e.g., switching instruments can change the LATE); those belong in `explore/*`.
- Diagnostics and inference-only procedures should be typed under `diag/*` and `infer/*`. This file lists closely related diagnostics/inference references for convenience.

## Key IV assumptions

1) **Relevance**: instrument shifts treatment.
2) **Exogeneity / independence**: instrument is as-good-as-random (conditional on controls, if applicable).
3) **Exclusion restriction**: instrument affects outcome only through treatment.
4) **Monotonicity** (for LATE interpretation): no defiers.

## A) Exclusion restriction sensitivity (plausibly exogenous IV)

These methods allow a nonzero “direct effect” of the instrument on the outcome and report bounds or robust intervals for the effect of interest.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/instrumental_variables/exclusion/conley_bound_small` | Conley-style bounds under a small direct-effect bound |
| `sens/assumption/instrumental_variables/exclusion/conley_bound_grid` | Grid over direct-effect bounds; report “breakdown” bound where sign/significance flips |
| `sens/assumption/instrumental_variables/exclusion/conley_prior_normal` | Conley-style sensitivity using a normal prior on direct effect |
| `sens/assumption/instrumental_variables/exclusion/break_even_direct_effect` | Compute direct-effect magnitude needed to move estimate to 0 |

Required audit fields (`coefficient_vector_json`):

- instrument(s),
- direct-effect parameterization (bound/prior),
- resulting identified set / interval,
- the “breakdown” value (if computed).

## B) Imperfect-IV / partial exogeneity sensitivity

These methods allow correlation between the instrument and the structural error.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/instrumental_variables/exogeneity/nevo_rosen_bounds` | Nevo–Rosen imperfect-IV bounds (where applicable) |
| `sens/assumption/instrumental_variables/exogeneity/c_dependence_grid` | Bounded dependence (“c-dependence”) grid sensitivity; report identified set vs c |
| `sens/assumption/instrumental_variables/exogeneity/breakdown_c` | Smallest c that makes 0 belong to identified set |

## C) Monotonicity sensitivity / bounds without monotonicity

Monotonicity is often invoked for LATE interpretation but is not directly testable. A principled workflow should be able to report what is identified without it.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/instrumental_variables/monotonicity/bounds_no_monotonicity` | Bounds that do not assume monotonicity (if available for the setting) |
| `sens/assumption/instrumental_variables/monotonicity/sign_of_first_stage` | Sensitivity note: how interpretation changes if first stage varies across subgroups |

## D) Exclusion via “auxiliary outcomes” (negative controls)

When plausible negative-control outcomes or exposures exist, they provide evidence about exclusion/exogeneity but are not direct tests.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/instrumental_variables/exclusion/neg_control_outcome` | Run IV with a negative-control outcome as placebo (conceptually diagnostic) |
| `sens/assumption/instrumental_variables/exogeneity/neg_control_exposure` | Include negative-control exposure to probe confounding of Z |

## Related diagnostics / inference (references)

Related menus:

- Diagnostics: `specification_tree/modules/diagnostics/design_diagnostics.md` (weak-IV strength, overid, balance-type checks)
- Inference: `specification_tree/modules/inference/standard_errors.md` and `specification_tree/modules/inference/resampling.md`

This file is for **assumption-relaxation / sensitivity** objects only.
