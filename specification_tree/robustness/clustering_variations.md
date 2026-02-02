# Clustering Variations Robustness Checks

## Spec ID Format: `robust/cluster/{cluster_level}`

## Purpose

Test sensitivity of standard errors and inference to different clustering assumptions. The point estimates remain unchanged, but confidence intervals and p-values may vary substantially.

---

## Standard Clustering Specifications

### Single-Level Clustering

| spec_id | Description |
|---------|-------------|
| `robust/cluster/none` | No clustering (robust SE only) |
| `robust/cluster/unit` | Cluster by unit (individual/firm) |
| `robust/cluster/time` | Cluster by time period |
| `robust/cluster/region` | Cluster by geographic region |
| `robust/cluster/industry` | Cluster by industry |
| `robust/cluster/treatment_group` | Cluster by treatment status |

### Two-Way Clustering

| spec_id | Description |
|---------|-------------|
| `robust/cluster/unit_time` | Two-way: unit × time |
| `robust/cluster/unit_region` | Two-way: unit × region |
| `robust/cluster/region_time` | Two-way: region × time |

### Higher-Level Clustering

| spec_id | Description |
|---------|-------------|
| `robust/cluster/state` | State level (if available) |
| `robust/cluster/country` | Country level (if available) |
| `robust/cluster/cohort` | Treatment cohort |

### Alternative SE Methods

| spec_id | Description |
|---------|-------------|
| `robust/se/hc1` | HC1 robust SE |
| `robust/se/hc2` | HC2 robust SE |
| `robust/se/hc3` | HC3 robust SE (small sample) |
| `robust/se/driscoll_kraay` | Driscoll-Kraay (cross-sectional dependence) |
| `robust/se/newey_west` | Newey-West HAC |
| `robust/se/bootstrap_cluster` | Wild cluster bootstrap |
| `robust/se/bootstrap_pairs` | Pairs bootstrap |

### Spatial Clustering

| spec_id | Description |
|---------|-------------|
| `robust/cluster/conley_50km` | Conley spatial SE (50km bandwidth) |
| `robust/cluster/conley_100km` | Conley spatial SE (100km bandwidth) |
| `robust/cluster/conley_200km` | Conley spatial SE (200km bandwidth) |
| `robust/cluster/conley_optimal` | Conley SE with optimal bandwidth |

### Few Clusters Adjustments

| spec_id | Description |
|---------|-------------|
| `robust/cluster/cameron_miller` | Cameron-Miller few clusters adjustment |
| `robust/cluster/fama_macbeth` | Fama-MacBeth standard errors |
| `robust/cluster/cgm` | Cameron-Gelbach-Miller adjustment |
| `robust/cluster/wild_bootstrap` | Wild cluster bootstrap (few clusters) |

### Multi-Dimensional Clustering

| spec_id | Description |
|---------|-------------|
| `robust/cluster/treatment_x_time` | Cluster by treatment cohort x time |
| `robust/cluster/three_way` | Three-way clustering |
| `robust/cluster/nested` | Nested clustering (e.g., county within state) |

---

## Implementation Notes

```python
import pyfixest as pf

# No clustering (robust)
model = pf.feols("y ~ treat | fe", data=df, vcov='hetero')

# Cluster by unit
model = pf.feols("y ~ treat | fe", data=df, vcov={'CRV1': 'unit'})

# Two-way clustering
model = pf.feols("y ~ treat | fe", data=df, vcov={'CRV1': ['unit', 'time']})

# Using statsmodels
import statsmodels.api as sm
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster_var']})

# Wild cluster bootstrap (using wildboottest)
from wildboottest import wildboottest
boot_result = wildboottest(model, param='treat', cluster='unit', reps=999)
```

---

## Output Format

```json
{
  "spec_id": "robust/cluster/unit_time",
  "spec_tree_path": "robustness/clustering_variations.md",
  "cluster_vars": ["unit", "time"],
  "n_clusters": {
    "unit": 100,
    "time": 20
  },
  "treatment": {
    "coef": 0.052,
    "se_unclustered": 0.012,
    "se_clustered": 0.021,
    "se_ratio": 1.75,
    "pval_unclustered": 0.000,
    "pval_clustered": 0.013,
    "ci_lower": 0.011,
    "ci_upper": 0.093
  },
  "significance_comparison": {
    "sig_at_05_unclustered": true,
    "sig_at_05_clustered": true,
    "sig_at_01_unclustered": true,
    "sig_at_01_clustered": false
  }
}
```

---

## Interpretation Guidelines

### SE Inflation

| SE Ratio (clustered/unclustered) | Interpretation |
|----------------------------------|----------------|
| 1.0 - 1.5 | Modest within-cluster correlation |
| 1.5 - 2.5 | Substantial clustering |
| > 2.5 | Strong clustering, be cautious |

### Cluster Count Rules of Thumb

| Number of Clusters | Concern |
|-------------------|---------|
| < 20 | Seriously consider bootstrap |
| 20 - 50 | Use caution, consider robust methods |
| > 50 | Standard cluster SE generally OK |

### When Different Levels Disagree

| Pattern | Recommended Action |
|---------|--------------------|
| Unit clustering insignificant, region significant | Report both, discuss |
| Two-way tightens SE | Report two-way as conservative |
| Bootstrap differs substantially | Report bootstrap |

---

## Best Practices

1. **Always report at least two clustering choices**
2. **If few clusters (<20), use wild cluster bootstrap**
3. **For panel data, default to clustering by unit**
4. **For DiD, consider clustering at treatment level**
5. **Report cluster count alongside SE**

---

## Checklist

- [ ] Ran with robust (unclustered) SE as baseline
- [ ] Ran with paper's main clustering choice
- [ ] Ran with at least one alternative clustering level
- [ ] Computed SE inflation ratios
- [ ] Noted if significance changes at 5% level
- [ ] Noted if significance changes at 1% level
- [ ] If few clusters, ran wild cluster bootstrap
- [ ] Reported cluster counts for each specification
