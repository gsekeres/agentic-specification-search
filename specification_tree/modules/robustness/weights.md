# Weights (Robustness Checks)

## Spec ID format

Use:

- `rc/weights/{family}/{variant}`

Examples:

- `rc/weights/main/unweighted`
- `rc/weights/main/paper_weights`
- `rc/weights/trim/p99`

## Purpose

Weights are common and high-leverage in applied work:

- survey/sampling weights (targeting a population estimand),
- analytic weights (precision),
- IPW/attrition weights (selection adjustment),
- trimming/stabilization guardrails.

This module defines **estimand-preserving** weight robustness checks.

## Core principle: RC vs target-population change

Whether changing weights is “estimand-preserving” depends on the baseline claim object.

Default rules:

- If the paper’s baseline estimand is explicitly weighted (survey representation), alternative reasonable weight-handling is `rc/weights/*`.
- If a weight choice implies a different target population (e.g., reweighting to a different population), treat it as `explore/*` unless the baseline claim is about that target.

When in doubt, run the spec but label it as exploration and explain.

## A) Main weight choices

| spec_id | Description |
|---|---|
| `rc/weights/main/unweighted` | Unweighted analysis |
| `rc/weights/main/paper_weights` | Use the paper’s baseline weight variable |
| `rc/weights/main/alternative_weights` | Use a plausible alternative weight variable present in the package |

## B) Trimming and stabilization (guardrails)

| spec_id | Description |
|---|---|
| `rc/weights/trim/p99` | Trim weights at 99th percentile |
| `rc/weights/trim/p95` | Trim weights at 95th percentile |
| `rc/weights/stabilize` | Stabilize IPW (divide by mean; document) |

## C) Attrition / selection weights (when used as RC)

If the paper uses IPW/attrition weights as part of the identification story, varying reasonable versions can be RC.
If IPW changes the estimand or relies on strong modeling assumptions beyond the baseline claim, treat as `sens/*` or `explore/*`.

| spec_id | Description |
|---|---|
| `rc/weights/ipw/baseline` | Baseline IPW model as in the paper |
| `rc/weights/ipw/trim_p99` | IPW with weight trimming at 99th percentile |
| `rc/weights/ipw/covariate_set_alt` | IPW model with alternative covariate set (paper-justified) |

## Required audit fields (`coefficient_vector_json`)

Every `rc/weights/*` row should include a `weights` block:

```json
{
  "weights": {
    "spec_id": "rc/weights/trim/p99",
    "weight_var": "survey_w",
    "family": "trim",
    "trim_rule": {"upper_q": 0.99},
    "effective_n": 7421.3,
    "notes": "Trimming extreme weights; target population intended unchanged."
  }
}
```
