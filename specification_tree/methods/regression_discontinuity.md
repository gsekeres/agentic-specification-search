# Regression Discontinuity Specifications

## Spec ID Format: `rd/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main RD result
- Record: running variable, cutoff, bandwidth, polynomial order

---

## Core Variations

### Design Type

| spec_id | Description |
|---------|-------------|
| `rd/design/sharp` | Sharp RD (deterministic treatment) |
| `rd/design/fuzzy` | Fuzzy RD (probabilistic treatment) |
| `rd/design/kink` | Regression kink design |

### Bandwidth Selection

| spec_id | Description |
|---------|-------------|
| `rd/bandwidth/optimal_ik` | Imbens-Kalyanaraman optimal |
| `rd/bandwidth/optimal_ccft` | Calonico-Cattaneo-Farrell-Titiunik optimal |
| `rd/bandwidth/half_optimal` | Half optimal bandwidth |
| `rd/bandwidth/double_optimal` | Double optimal bandwidth |
| `rd/bandwidth/fixed_small` | Fixed small bandwidth |
| `rd/bandwidth/fixed_large` | Fixed large bandwidth |

### Polynomial Order

| spec_id | Description |
|---------|-------------|
| `rd/poly/local_linear` | Local linear (order 1) |
| `rd/poly/local_quadratic` | Local quadratic (order 2) |
| `rd/poly/local_cubic` | Local cubic (order 3) |
| `rd/poly/global_linear` | Global linear |
| `rd/poly/global_quadratic` | Global quadratic |

### Kernel Function

| spec_id | Description |
|---------|-------------|
| `rd/kernel/triangular` | Triangular kernel (default) |
| `rd/kernel/uniform` | Uniform (rectangular) kernel |
| `rd/kernel/epanechnikov` | Epanechnikov kernel |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `rd/controls/none` | No controls |
| `rd/controls/baseline` | Paper's baseline controls |
| `rd/controls/full` | All available controls |
| `rd/controls/predetermined` | Pre-determined covariates only |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `rd/sample/full` | Full sample near cutoff |
| `rd/sample/symmetric` | Symmetric sample around cutoff |
| `rd/sample/donut` | Donut hole (exclude obs at cutoff) |
| `rd/sample/trim_tails` | Exclude extreme running variable |

### Validation Tests

| spec_id | Description |
|---------|-------------|
| `rd/validity/density` | McCrary density test |
| `rd/validity/covariate_balance` | Balance on covariates at cutoff |
| `rd/validity/placebo_cutoff` | Placebo cutoffs test |
| `rd/validity/sensitivity` | Sensitivity to bandwidth |

### Donut Hole Specifications

| spec_id | Description |
|---------|-------------|
| `rd/donut/exclude_1pct` | Exclude 1% nearest to cutoff |
| `rd/donut/exclude_5pct` | Exclude 5% nearest to cutoff |
| `rd/donut/exclude_10pct` | Exclude 10% nearest to cutoff |
| `rd/donut/exclude_heaping` | Exclude heaping points at cutoff |

### Placebo Cutoff Tests

| spec_id | Description |
|---------|-------------|
| `rd/placebo/cutoff_median_left` | Placebo at median of left side |
| `rd/placebo/cutoff_median_right` | Placebo at median of right side |
| `rd/placebo/cutoff_quartile` | Placebo at quartiles |
| `rd/placebo/cutoff_above` | Placebo at cutoff + bandwidth |
| `rd/placebo/cutoff_below` | Placebo at cutoff - bandwidth |

### Inference Variations

| spec_id | Description |
|---------|-------------|
| `rd/inference/conventional` | Conventional standard errors |
| `rd/inference/robust` | Robust bias-corrected SE |
| `rd/inference/cluster` | Clustered standard errors |
| `rd/inference/randomization` | Randomization inference |

### Alternative Estimators

| spec_id | Description |
|---------|-------------|
| `rd/estimator/local_poly` | Local polynomial (default) |
| `rd/estimator/honest` | Honest confidence intervals |
| `rd/estimator/coverage_error` | Coverage-error optimal |
| `rd/estimator/mse_optimal` | MSE-optimal |

### Heterogeneity at Discontinuity

| spec_id | Description |
|---------|-------------|
| `rd/het/by_covariate` | Effect heterogeneity by covariate |
| `rd/het/interaction` | Running variable x covariate interaction |
| `rd/het/subgroup` | Subgroup-specific RD estimates |

---

## Python Implementation Notes

```python
from rdrobust import rdrobust, rdbwselect, rdplot

# Basic RD with optimal bandwidth
result = rdrobust(y, x, c=cutoff)

# Bandwidth selection
bw = rdbwselect(y, x, c=cutoff)

# Fuzzy RD
result = rdrobust(y, x, c=cutoff, fuzzy=treatment)

# McCrary density test
from rddensity import rddensity
density_test = rddensity(x, c=cutoff)
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "above_cutoff",
    "coef": 0.15,
    "se": 0.05,
    "pval": 0.003,
    "ci_lower": 0.05,
    "ci_upper": 0.25,
    "ci_type": "robust_bias_corrected"
  },
  "running_variable": {
    "var": "score",
    "cutoff": 50,
    "bandwidth_left": 10.5,
    "bandwidth_right": 10.5
  },
  "controls": [
    {"var": "age", "coef": 0.1, "se": 0.05, "pval": 0.04}
  ],
  "diagnostics": {
    "polynomial_order": 1,
    "kernel": "triangular",
    "n_left": 500,
    "n_right": 450,
    "mccrary_pval": 0.45,
    "covariate_balance_pval": 0.78,
    "first_stage_F": 45.2
  },
  "n_obs": 950,
  "n_effective": 680,
  "r_squared": null
}
```

---

## Checklist

Before completing an RD analysis, verify you have run:

- [ ] Baseline replication
- [ ] At least 3 bandwidth choices
- [ ] Local linear and at least one other polynomial
- [ ] McCrary density test
- [ ] Covariate balance test
- [ ] At least 1 placebo cutoff test
- [ ] Donut hole sensitivity
- [ ] Report bandwidth and effective N in coefficient vector
