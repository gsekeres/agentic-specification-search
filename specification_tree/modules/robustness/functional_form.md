# Functional Form & Transformations (Robustness Checks)

## Spec ID format

Use:

- `rc/form/{target}/{variant}`

Examples:

- `rc/form/outcome/log1p`
- `rc/form/outcome/asinh`
- `rc/form/treatment/log1p`
- `rc/form/model/quadratic_treatment`

## Purpose

Functional-form choices are a common source of fragility:

- outliers and heavy tails motivate logs/asinh,
- nonlinear dose-response motivates polynomials/splines,
- “levels vs percent” interpretations can materially change conclusions.

This module defines **estimand-preserving** functional-form robustness checks when they plausibly preserve the baseline claim object.

## Core principle: RC vs estimand change

Many transformations change the *interpretation* of the coefficient (semi-elasticity vs level effect).
Default rule:

- If the paper’s claim is inherently about percent changes/elasticities (or is written in logs), these are RC.
- Otherwise, treat strong re-interpretations as `explore/estimand/*` (see `specification_tree/modules/exploration/alternative_estimands.md`).

When in doubt, record the run but label as exploration.

## A) Outcome transformations

| spec_id | Description |
|---|---|
| `rc/form/outcome/level` | Outcome in levels (baseline if baseline is levels) |
| `rc/form/outcome/log` | Log(y) when y>0 and claim supports elasticity/semi-elasticity interpretation |
| `rc/form/outcome/log1p` | Log(1+y) when zeros exist |
| `rc/form/outcome/asinh` | asinh(y) (handles zeros; approx log for large y) |
| `rc/form/outcome/rank` | Rank transform (robust to outliers; interpret cautiously) |

## B) Treatment transformations

| spec_id | Description |
|---|---|
| `rc/form/treatment/level` | Treatment in levels (baseline if baseline is levels) |
| `rc/form/treatment/log` | Log(x) when positive |
| `rc/form/treatment/log1p` | Log(1+x) when zeros exist |
| `rc/form/treatment/standardize_z` | Standardize treatment (units only; often RC) |

Thresholding / binning a continuous treatment is typically an **estimand change** and should be recorded as:

- `explore/definition/treatment/*` (see `specification_tree/modules/exploration/variable_definitions.md`).

## C) Nonlinear dose-response (within the same concept)

These specs test whether linearity is driving the baseline result.

| spec_id | Description |
|---|---|
| `rc/form/model/quadratic_treatment` | Add \(D^2\) term |
| `rc/form/model/cubic_treatment` | Add \(D^3\) term |
| `rc/form/model/spline_3_knots` | Spline in treatment (3 knots) |
| `rc/form/model/spline_5_knots` | Spline in treatment (5 knots) |

Interpretation note: once nonlinear terms are included, a single “treatment coefficient” is not sufficient. Record marginal effects at a reference point (mean/median) in JSON.

## D) Interactions as functional-form robustness (not heterogeneity claims)

Sometimes papers include interactions to soak up flexible trends, not to claim subgroup heterogeneity (e.g., treatment × time trends).
If the interaction is part of the identifying functional form rather than an estimand change, it can be RC:

| spec_id | Description |
|---|---|
| `rc/form/model/treatment_x_time_trend` | Add treatment × time trend |
| `rc/form/model/controls_quadratic` | Add squared terms for key controls |

If interactions are interpreted as heterogeneous effects, use `explore/heterogeneity/*`.

## Required audit fields (`coefficient_vector_json`)

Include a `functional_form` block:

```json
{
  "functional_form": {
    "spec_id": "rc/form/outcome/asinh",
    "outcome_transform": "asinh",
    "treatment_transform": "level",
    "model_terms": ["treat", "I(treat**2)"],
    "marginal_effects": {"at": "mean_treat", "dy_dx": 0.07},
    "interpretation": "Approx semi-elasticity for large y; preserves outcome concept as 'level' up to monotone transform."
  }
}
```
