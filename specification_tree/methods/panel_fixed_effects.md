# Panel Fixed Effects Specifications

## Spec ID Format: `panel/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main panel FE result
- Record: FE structure, clustering, panel dimensions

---

## Core Variations

### Fixed Effects Structure

| spec_id | Description |
|---------|-------------|
| `panel/fe/none` | Pooled OLS (no FE) |
| `panel/fe/unit` | Unit (individual/firm/country) FE |
| `panel/fe/time` | Time (year/quarter/month) FE |
| `panel/fe/twoway` | Unit + Time FE |
| `panel/fe/nested` | Nested FE (e.g., firm within industry) |
| `panel/fe/high_dimensional` | High-dimensional FE (3+ sets) |
| `panel/fe/random_effects` | Random effects (for comparison) |

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `panel/method/within` | Within estimator (standard FE) |
| `panel/method/first_diff` | First differences |
| `panel/method/between` | Between estimator |
| `panel/method/random` | Random effects GLS |
| `panel/method/correlated_re` | Correlated random effects |
| `panel/method/mundlak` | Mundlak correction |

### Standard Errors

| spec_id | Description |
|---------|-------------|
| `panel/se/robust` | Heteroskedasticity-robust |
| `panel/se/cluster_unit` | Clustered by unit |
| `panel/se/cluster_time` | Clustered by time |
| `panel/se/cluster_twoway` | Two-way clustering |
| `panel/se/driscoll_kraay` | Driscoll-Kraay (time series correlation) |
| `panel/se/newey_west` | Newey-West HAC |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `panel/controls/none` | No controls (treatment + FE only) |
| `panel/controls/time_invariant` | Time-invariant controls |
| `panel/controls/time_varying` | Time-varying controls |
| `panel/controls/baseline` | Paper's baseline controls |
| `panel/controls/full` | All available controls |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `panel/sample/full` | Full unbalanced panel |
| `panel/sample/balanced` | Balanced panel only |
| `panel/sample/early` | First half of time period |
| `panel/sample/late` | Second half of time period |
| `panel/sample/continuous` | Continuously observed units |
| `panel/sample/trim_singletons` | Drop singleton observations |

### Panel Diagnostics

| spec_id | Description |
|---------|-------------|
| `panel/diagnostic/hausman` | Hausman test (FE vs RE) |
| `panel/diagnostic/breusch_pagan` | Breusch-Pagan test for RE |
| `panel/diagnostic/serial_corr` | Test for serial correlation |
| `panel/diagnostic/cross_sectional_dep` | Test for cross-sectional dependence |

---

## Python Implementation Notes

```python
import pyfixest as pf
from linearmodels.panel import PanelOLS, RandomEffects

# Using pyfixest (recommended)
model = pf.feols("y ~ x + controls | unit + time", data=df)

# Using linearmodels
df = df.set_index(['unit', 'time'])
model = PanelOLS.from_formula("y ~ x + controls + EntityEffects + TimeEffects",
                               data=df)
result = model.fit(cov_type='clustered', cluster_entity=True)

# Hausman test
from linearmodels.panel import compare
fe_result = PanelOLS(...).fit()
re_result = RandomEffects(...).fit()
hausman = compare({'FE': fe_result, 'RE': re_result})
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "policy",
    "coef": 0.12,
    "se": 0.04,
    "pval": 0.003,
    "ci_lower": 0.04,
    "ci_upper": 0.20
  },
  "controls": [
    {"var": "gdp_growth", "coef": 0.05, "se": 0.02, "pval": 0.01},
    {"var": "population", "coef": 0.001, "se": 0.0005, "pval": 0.05}
  ],
  "fixed_effects_absorbed": ["country", "year"],
  "diagnostics": {
    "hausman_stat": 15.4,
    "hausman_pval": 0.02,
    "serial_corr_stat": 2.1,
    "serial_corr_pval": 0.15,
    "n_units": 50,
    "n_periods": 20,
    "avg_obs_per_unit": 18.5
  },
  "n_obs": 925,
  "n_clusters": 50,
  "r_squared": 0.65,
  "r_squared_within": 0.32,
  "r_squared_between": 0.78
}
```

---

## Checklist

Before completing a panel FE analysis, verify you have run:

- [ ] Baseline replication
- [ ] Multiple FE structures (unit only, time only, two-way)
- [ ] Pooled OLS for comparison
- [ ] Random effects for comparison
- [ ] Hausman test (if RE is plausible)
- [ ] At least 2 clustering choices
- [ ] Balanced vs unbalanced panel comparison
- [ ] Report panel dimensions in coefficient vector
