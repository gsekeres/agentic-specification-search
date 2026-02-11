# Alternative Estimands (Exploration)

## Spec ID format

Use:

- `explore/estimand/{family}/{variant}`

Examples:

- `explore/estimand/randomized_experiment/tot_instead_of_itt`
- `explore/estimand/distribution/quantile_50`
- `explore/estimand/instrumental_variables/mte`

## Purpose

Many “reasonable” extensions change the targeted parameter even when the data and treatment/outcome variables are unchanged.
This module enumerates common estimand changes so they are recorded explicitly as `explore/*` by default.

If the manuscript’s headline claim is itself one of these estimands, then that estimand should define the baseline claim object instead.

## A) Randomized experiment: ITT vs TOT / LATE

| spec_id | Description |
|---|---|
| `explore/estimand/randomized_experiment/tot_instead_of_itt` | Switch from ITT to TOT (requires exclusion + monotonicity) |
| `explore/estimand/randomized_experiment/itt_instead_of_tot` | Switch from TOT to ITT |

## B) Instrumental variables: LATE variants and MTE

Instrument changes often change the complier population; treat as exploration unless instruments are explicitly equivalent.

| spec_id | Description |
|---|---|
| `explore/estimand/instrumental_variables/late` | Explicitly label the IV estimand as LATE |
| `explore/estimand/instrumental_variables/mte` | Marginal treatment effects (requires additional structure) |

## C) Distributional and quantile effects

| spec_id | Description |
|---|---|
| `explore/estimand/distribution/quantile_25` | QTE at 25th percentile |
| `explore/estimand/distribution/quantile_50` | QTE at median |
| `explore/estimand/distribution/quantile_75` | QTE at 75th percentile |
| `explore/estimand/distribution/cdf_shift` | Distributional shift summaries (e.g., stochastic dominance checks) |

## D) Nonlinear average effects

| spec_id | Description |
|---|---|
| `explore/estimand/nonlinear/semielasticity` | Semi-elasticity interpretation (log outcome) |
| `explore/estimand/nonlinear/elasticity` | Elasticity interpretation (log-log) |

Whether log transforms are RC or exploration depends on the claim object:

- If the paper’s claim is inherently about percent changes/elasticities, log forms are baseline/RC.
- If the claim is about level changes, log forms are exploration unless explicitly treated as equivalent in the paper.

## E) Welfare / policy estimands

| spec_id | Description |
|---|---|
| `explore/estimand/welfare/cs_ps` | Consumer/producer surplus welfare measure |
| `explore/estimand/welfare/uplift` | Welfare under a treatment rule |

## Output contract (`coefficient_vector_json`)

Every `explore/estimand/*` row must include:

- `baseline_estimand`,
- `new_estimand`,
- a short note on extra assumptions required (if any).

Example:

```json
{
  "exploration": {
    "spec_id": "explore/estimand/randomized_experiment/tot_instead_of_itt",
    "changed": ["estimand"],
    "baseline_estimand": "ITT",
    "new_estimand": "TOT",
    "extra_assumptions": ["exclusion_restriction", "monotonicity"],
    "notes": "Uses assignment as instrument for take-up."
  }
}
```
