# Heterogeneity & Subgroup Effects (Exploration)

## Why this is a separate module

Heterogeneity analysis targets **different estimands** than an average-effect baseline:

- subgroup effects change the **target population**,
- interactions test **effect modification** parameters,
- subgroup selection choices create additional researcher degrees of freedom.

These are scientifically valuable but are **not core by default** unless the paper’s baseline claim object is itself subgroup-specific.

## Spec ID format

Use:

- `explore/heterogeneity/{family}/{variant}`

Examples:

- `explore/heterogeneity/subgroup/by_gender`
- `explore/heterogeneity/subgroup/by_income_bins`
- `explore/heterogeneity/interaction/gender`

## A) Subgroup splits (population changes)

| spec_id | Description |
|---|---|
| `explore/heterogeneity/subgroup/by_gender` | Estimate separately by gender |
| `explore/heterogeneity/subgroup/by_age_bins` | Estimate by age bins (paper-defined or terciles) |
| `explore/heterogeneity/subgroup/by_income_bins` | Estimate by income bins |
| `explore/heterogeneity/subgroup/by_region` | Estimate by region groups |
| `explore/heterogeneity/subgroup/by_baseline_y_bins` | Estimate by baseline-outcome bins (if baseline outcome exists) |

If a subgroup is a headline claim, verification should define it as a **baseline group** and then run `rc/*` within it.
Otherwise, keep it as exploration.

## B) Interaction specifications (effect modification tests)

| spec_id | Description |
|---|---|
| `explore/heterogeneity/interaction/gender` | Treatment × gender interaction |
| `explore/heterogeneity/interaction/age` | Treatment × age interaction |
| `explore/heterogeneity/interaction/income` | Treatment × income interaction |
| `explore/heterogeneity/interaction/baseline_y` | Treatment × baseline outcome interaction |

## C) Multiple testing and disclosure

Heterogeneity creates many hypotheses. Treat multiplicity explicitly:

- Use `post/mht/family_heterogeneity/*` (see `specification_tree/modules/postprocess/multiple_testing.md`).
- Record the full set of subgroup tests executed, including how bins were formed.

## Output contract (`coefficient_vector_json`)

Every heterogeneity row must explicitly declare what concept changed.

Example for a subgroup split:

```json
{
  "exploration": {
    "spec_id": "explore/heterogeneity/subgroup/by_gender",
    "changed": ["population"],
    "baseline_population": "all",
    "subgroups": ["male", "female"],
    "notes": "Explores effect heterogeneity; not part of baseline claim."
  }
}
```

Example for an interaction:

```json
{
  "exploration": {
    "spec_id": "explore/heterogeneity/interaction/gender",
    "changed": ["estimand"],
    "baseline_estimand": "ATE",
    "new_estimand": "interaction_effect",
    "moderator": "female"
  }
}
```
