# CATE Estimation (Exploration)

## Why this is a separate module

CATE estimation targets a **function-valued estimand** \(x \mapsto \tau(x)\), not a single scalar effect.
It introduces additional modeling choices (features, learners, cross-fitting, tuning) and is typically exploratory unless the paper’s baseline claim is explicitly about treatment-effect heterogeneity.

## Spec ID format

Use:

- `explore/cate/{family}/{variant}`

Examples:

- `explore/cate/grf/causal_forest`
- `explore/cate/best_linear/default`
- `explore/cate/x_learner/default`

## A) Tree/forest-based CATE

| spec_id | Description |
|---|---|
| `explore/cate/grf/causal_forest` | Causal forest / generalized random forest |
| `explore/cate/grf/honest_forest` | “Honest” forest variant (if available) |

## B) Meta-learners (binary treatment)

| spec_id | Description |
|---|---|
| `explore/cate/meta/s_learner` | S-learner |
| `explore/cate/meta/t_learner` | T-learner |
| `explore/cate/meta/x_learner` | X-learner |
| `explore/cate/meta/r_learner` | R-learner (if available) |

## C) Low-dimensional summaries of heterogeneity

| spec_id | Description |
|---|---|
| `explore/cate/best_linear/default` | Best linear predictor / low-dimensional CATE summary |
| `explore/cate/sorted_effects/default` | Sorted effects / GATES-style summaries (if implementable) |

## Output contract (`exploration_results.csv`)

Write `explore/*` objects to `exploration_results.csv` (see `specification_tree/CONTRACT.md`) and store outputs in `exploration_json` with an `exploration` block.

Since CATE is not a scalar object, store:

- the feature set used,
- learner + hyperparameters,
- cross-fitting configuration,
- summary statistics of the CATE distribution,
- and (optionally) group-average effects by quantiles of predicted CATE.

Example:

```json
{
  "exploration": {
    "spec_id": "explore/cate/grf/causal_forest",
    "object": "cate_function",
    "features": ["x1", "x2", "x3"],
    "crossfit": {"K": 5, "seed": 42},
    "learner": {"name": "causal_forest", "params": {"n_trees": 2000}},
    "summary": {
      "cate_mean": 0.04,
      "cate_sd": 0.12,
      "share_positive": 0.61
    },
    "gates": [
      {"q": "0-20", "ate": -0.02},
      {"q": "80-100", "ate": 0.19}
    ]
  }
}
```

If you report a scalar `coefficient` for CATE, it must be explicitly labeled as a summary (e.g., mean CATE) and not conflated with the baseline ATE.
