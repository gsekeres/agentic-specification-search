# Synthetic-Control / SDID Assumption Sensitivity (Fit, Donor Pool, Spillovers)

## Spec ID format

Use:

- `sens/assumption/synthetic_control/{assumption}/{variant}`

Comparative case-study methods depend on pre-treatment fit and donor pool credibility. This file records sensitivity analyses targeting those assumptions.

## Key assumptions (informal)

1) **Pre-treatment fit / latent factor structure**: treated unit’s counterfactual can be approximated by a combination of donors.
2) **No spillovers / interference**: donors are not affected by treatment.
3) **Stable outcome measurement and timing**: no confounding structural breaks unique to treated unit.
4) **Donor pool validity**: donors are comparable and not contaminated.

## A) Donor pool sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/synthetic_control/donor_pool/leave_one_out` | Leave-one-donor-out sensitivity |
| `sens/assumption/synthetic_control/donor_pool/drop_suspect_donors` | Drop donors likely contaminated by spillovers |
| `sens/assumption/synthetic_control/donor_pool/expand_pool` | Expand donor pool (when reasonable) |
| `sens/assumption/synthetic_control/donor_pool/restrict_pool` | Restrict donor pool to closer comparators |

## B) Pre-treatment window / fit sensitivity

| spec_id | Description |
|--------|-------------|
| `sens/assumption/synthetic_control/fit/pre_window_alt` | Alternative pre-treatment fit windows |
| `sens/assumption/synthetic_control/fit/predictor_set_alt` | Alternative predictor sets used for fitting |
| `sens/assumption/synthetic_control/fit/regularization_grid` | Regularization / penalty grid (ridge / elastic-net variants) |

## C) Placebo-based sensitivity (distributional evidence)

These are often diagnostics, but they can be treated as sensitivity objects when they feed into uncertainty quantification.

| spec_id | Description |
|--------|-------------|
| `sens/assumption/synthetic_control/placebo/in_space` | In-space placebo distribution |
| `sens/assumption/synthetic_control/placebo/in_time` | In-time placebo (fake treatment dates) |
| `sens/assumption/synthetic_control/placebo/rmspe_ratio` | RMSPE ratio sensitivity for significance thresholds |

## Related diagnostics / inference (references)

Record these under `diag/*` and `infer/*` (not under `sens/*`):

- pre-treatment fit metrics (RMSPE, predictor balance): `specification_tree/modules/diagnostics/design_diagnostics.md`
- permutation inference from placebo distributions: `specification_tree/modules/inference/resampling.md`
