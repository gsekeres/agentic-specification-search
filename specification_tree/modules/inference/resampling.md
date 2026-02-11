# Resampling, Randomization, and Permutation Inference (Inference)

## Spec ID format

Use:

- `infer/resampling/{family}/{variant}`

Examples:

- `infer/resampling/bootstrap/pairs`
- `infer/resampling/bootstrap/cluster`
- `infer/resampling/wild_cluster/default`
- `infer/resampling/permutation/treatment`
- `infer/resampling/randomization/fisher_exact`

## Purpose

Resampling-based inference is valuable when:

- the number of clusters is small,
- assignment is randomized (RCT),
- asymptotic approximations are questionable,
- you want a robustness check on p-values for key coefficients.

These specs should preserve the estimating equation and only change the inference procedure.

## A) Bootstrap

| spec_id | Description |
|---|---|
| `infer/resampling/bootstrap/pairs` | Pairs bootstrap (resample rows) |
| `infer/resampling/bootstrap/cluster` | Cluster bootstrap (resample clusters) |
| `infer/resampling/bootstrap/block_time` | Block/bootstrap for time series (block length documented) |

## B) Wild cluster bootstrap

| spec_id | Description |
|---|---|
| `infer/resampling/wild_cluster/default` | Wild cluster bootstrap (recommended for few clusters) |
| `infer/resampling/wild_cluster/rademacher` | Rademacher weights |
| `infer/resampling/wild_cluster/webb` | Webb six-point weights (few clusters) |

## C) Permutation / randomization inference

Applicable when treatment assignment is exchangeable under the null (often RCT; sometimes placebo/permutation checks).

| spec_id | Description |
|---|---|
| `infer/resampling/permutation/treatment` | Permute treatment labels (unit or cluster level) |
| `infer/resampling/permutation/clustered` | Permute at cluster level |
| `infer/resampling/randomization/fisher_exact` | Fisher randomization inference (RCT) |

## Required audit fields (`coefficient_vector_json`)

Include an `inference` block:

```json
{
  "inference": {
    "spec_id": "infer/resampling/wild_cluster/default",
    "method": "wild_cluster_bootstrap",
    "cluster_var": "state",
    "reps": 999,
    "seed": 42,
    "p_value_resampling": 0.018
  }
}
```

If you include both asymptotic and resampling p-values, store both in JSON (and keep the scalar `p_value` consistent with the `spec_id`).
