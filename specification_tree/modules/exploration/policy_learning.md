# Policy Learning & Optimal Treatment Rules (Exploration)

## Why this is a separate module

Policy learning is conceptually distinct from heterogeneity and from CATE estimation:

- it targets a **decision rule** (treat/do-not-treat) rather than an effect parameter,
- it typically requires a welfare/utility criterion and additional assumptions,
- it raises overfitting and evaluation concerns (needs honest evaluation / cross-validation).

This module standardizes how to record policy-learning objects as explicit exploration.

## Spec ID format

Use:

- `explore/policy/{family}/{variant}`

Examples:

- `explore/policy/optimal_treatment_rule/threshold`
- `explore/policy/optimal_treatment_rule/cate_based`
- `explore/policy/welfare/uplift`

## A) Simple policy rules

| spec_id | Description |
|---|---|
| `explore/policy/optimal_treatment_rule/threshold` | Treat if a covariate/score exceeds a threshold |
| `explore/policy/optimal_treatment_rule/rule_list` | Interpretable rule list (if implementable) |

## B) CATE-based rules

| spec_id | Description |
|---|---|
| `explore/policy/optimal_treatment_rule/cate_based` | Treat if \(\hat{\tau}(x) > 0\) (or > cost) |
| `explore/policy/optimal_treatment_rule/cate_quantile` | Treat top-q quantile by predicted CATE |

This should reference the CATE method used (e.g., from `explore/cate/*`) in the JSON.

## C) Welfare / uplift summaries

| spec_id | Description |
|---|---|
| `explore/policy/welfare/uplift` | Expected gain from deploying a rule vs baseline policy |
| `explore/policy/welfare/cost_sensitive` | Welfare with treatment cost / capacity constraint |

## Output contract (`exploration_results.csv`)

Write `explore/*` objects to `exploration_results.csv` (see `specification_tree/CONTRACT.md`) and store outputs in `exploration_json` with an `exploration` block.

Policy objects must include:

- the rule definition,
- evaluation protocol (train/test split or cross-fitting),
- the welfare criterion and any costs/constraints,
- an “honest” estimate of performance when possible.

Example:

```json
{
  "exploration": {
    "spec_id": "explore/policy/optimal_treatment_rule/cate_based",
    "object": "policy_rule",
    "rule": "treat_if_cate_positive",
    "cate_source": "explore/cate/grf/causal_forest",
    "evaluation": {"scheme": "crossfit", "K": 5, "seed": 42},
    "welfare": {
      "criterion": "expected_outcome_gain",
      "cost_per_treat": 0.0,
      "uplift": 0.03
    }
  }
}
```

If you output a scalar `coefficient`, it must be clearly a *policy performance summary* (not a treatment effect).
