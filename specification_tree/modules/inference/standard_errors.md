# Standard Errors & Clustering (Inference)

## Spec ID format

Use:

- `infer/se/{family}/{variant}`

Examples:

- `infer/se/hc/hc1`
- `infer/se/cluster/unit`
- `infer/se/cluster/unit_time`
- `infer/se/spatial/conley_100km`

## Purpose

Inference is a different statistical object than point estimation.
These specs typically **do not change the point estimate** (holding the estimating equation fixed) but do change uncertainty quantification.

## A) Heteroskedasticity-robust SE (no clustering)

| spec_id | Description |
|---|---|
| `infer/se/hc/classical` | Classical (homoskedastic) |
| `infer/se/hc/hc1` | HC1 (default robust) |
| `infer/se/hc/hc2` | HC2 |
| `infer/se/hc/hc3` | HC3 (small-sample leverage) |

## B) Single-level clustering

Use the paper’s natural clustering unit as baseline. Common levels:

| spec_id | Description |
|---|---|
| `infer/se/cluster/unit` | Cluster by unit (person/firm/county/school) |
| `infer/se/cluster/time` | Cluster by time |
| `infer/se/cluster/region` | Cluster by region |
| `infer/se/cluster/industry` | Cluster by industry |
| `infer/se/cluster/treatment_group` | Cluster by treatment cohort/group (DiD) |

## C) Multi-way clustering

| spec_id | Description |
|---|---|
| `infer/se/cluster/unit_time` | Two-way: unit × time |
| `infer/se/cluster/unit_region` | Two-way: unit × region |
| `infer/se/cluster/region_time` | Two-way: region × time |

## D) Spatial and panel dependence corrections

| spec_id | Description |
|---|---|
| `infer/se/spatial/conley_50km` | Conley spatial HAC (50km) |
| `infer/se/spatial/conley_100km` | Conley spatial HAC (100km) |
| `infer/se/spatial/conley_200km` | Conley spatial HAC (200km) |
| `infer/se/panel/driscoll_kraay` | Driscoll–Kraay (cross-sectional dependence) |
| `infer/se/time/newey_west` | Newey–West HAC (time series) |

## E) Few-cluster guardrails

When the number of clusters is small (rule-of-thumb: < 30), asymptotic cluster SE can be unreliable.
Prefer resampling-based inference:

- `specification_tree/modules/inference/resampling.md` (`infer/resampling/*`)

## Required audit fields (`coefficient_vector_json`)

Include an `inference` block for all successful rows:

- estimate rows (`baseline`, `design/*`, `rc/*`): record the **canonical** inference choice used for the row’s scalar `std_error`/`p_value`.
- inference-only rows (`infer/se/*`): record the variant choice (and any realized cluster counts, etc.).

```json
{
  "inference": {
    "spec_id": "infer/se/cluster/unit_time",
    "method": "cluster",
    "cluster_vars": ["unit", "time"],
    "n_clusters": {"unit": 120, "time": 15}
  }
}
```
