# Unobserved Confounding & Sensitivity Analysis (Sensitivity / Partial-ID)

## Spec ID format

This module defines **typed sensitivity analyses** that assess robustness to violations of unconfoundedness / selection-on-observables or related assumptions.

These are **not in the verified core by default** (see `specification_tree/ARCHITECTURE.md`): they are different statistical objects (bounds/breakdown parameters), not ordinary estimand-preserving re-estimates.

Use:

- `sens/unobs/{family}/{variant}`

Examples:

- `sens/unobs/oster/delta`
- `sens/unobs/rosenbaum/gamma_grid`

## Purpose

Controls/sample/functional-form robustness only probes a small slice of the identification risk. A comprehensive robustness workflow should include *structured sensitivity analyses* for **unobserved confounding** or **imperfect identification assumptions**, especially for:

- cross-sectional OLS with rich controls,
- panel FE with time-varying confounding risk,
- DiD designs with plausible violations of parallel trends / composition,
- IV designs with imperfect exclusion/plausible exogeneity.

These are not “new regressions” in the usual sense; they are **sensitivity objects** derived from baseline and control-progression estimates.

## Oster / Altonji–Elder–Taber style selection-on-unobservables (regression-based)

| spec_id | Description |
|--------|-------------|
| `sens/unobs/oster/delta` | Oster δ implied by coefficient/R² movement from baseline → full controls |
| `sens/unobs/oster/bounds` | Oster bounds on β under R²_max assumption |
| `sens/unobs/aet/ratio` | Altonji–Elder–Taber selection ratio (when implementable) |

Required inputs:

- a defined “baseline controls” model and a “full controls” model,
- R² for both models,
- coefficient movement on the treatment.

## Rosenbaum bounds (matching / assignment sensitivity)

| spec_id | Description |
|--------|-------------|
| `sens/unobs/rosenbaum/gamma_1_2` | Rosenbaum Γ = 1.2 bound |
| `sens/unobs/rosenbaum/gamma_1_5` | Rosenbaum Γ = 1.5 bound |
| `sens/unobs/rosenbaum/gamma_grid` | Grid over plausible Γ values with “breakpoint” summary |

## “E-values” / robustness values (effect-size sensitivity)

| spec_id | Description |
|--------|-------------|
| `sens/unobs/evalue/estimate` | E-value for point estimate |
| `sens/unobs/evalue/ci` | E-value for confidence interval limit |

## Design-specific assumption sensitivity (design-specific)

For sensitivity to **design-specific causal assumptions** (exclusion restriction, parallel trends, RD manipulation, attrition, donor-pool validity), use:

- `specification_tree/modules/sensitivity/assumptions/instrumental_variables.md`
- `specification_tree/modules/sensitivity/assumptions/difference_in_differences.md`
- `specification_tree/modules/sensitivity/assumptions/regression_discontinuity.md`
- `specification_tree/modules/sensitivity/assumptions/randomized_experiment.md`
- `specification_tree/modules/sensitivity/assumptions/synthetic_control.md`

## Output contract (`coefficient_vector_json`)

Sensitivity objects must include a `sensitivity` block and may omit `(coef, se)` if not meaningful. If included in the regression-results table, set missing numeric fields to empty/NaN and record the sensitivity outputs in JSON.

Example:

```json
{
  "sensitivity": {
    "spec_id": "sens/unobs/oster/delta",
    "method": "oster",
    "inputs": {
      "beta_tilde": 0.12,
      "beta_full": 0.08,
      "r2_tilde": 0.10,
      "r2_full": 0.25,
      "r2_max": 1.0
    },
    "outputs": {
      "delta": 2.3,
      "beta_bound_low": 0.03,
      "beta_bound_high": 0.10
    },
    "interpretation": "Would require selection on unobservables >2x observables to explain away."
  }
}
```
