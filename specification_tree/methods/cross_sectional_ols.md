# Cross-Sectional OLS Specifications

## Spec ID Format: `ols/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main cross-sectional result
- Record: sample, controls, standard errors

---

## Core Variations

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `ols/method/ols` | Standard OLS |
| `ols/method/wls` | Weighted least squares |
| `ols/method/robust` | Robust regression (M-estimation) |
| `ols/method/quantile_median` | Median regression (LAD) |
| `ols/method/quantile_25` | 25th percentile regression |
| `ols/method/quantile_75` | 75th percentile regression |

### Standard Errors

| spec_id | Description |
|---------|-------------|
| `ols/se/classical` | Classical (homoskedastic) SE |
| `ols/se/robust` | Heteroskedasticity-robust (HC1) |
| `ols/se/hc2` | HC2 robust SE |
| `ols/se/hc3` | HC3 robust SE |
| `ols/se/clustered` | Clustered by natural grouping |
| `ols/se/bootstrap` | Bootstrap SE |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `ols/controls/none` | No controls (bivariate) |
| `ols/controls/demographics` | Demographic controls only |
| `ols/controls/baseline` | Paper's baseline controls |
| `ols/controls/full` | All available controls |
| `ols/controls/kitchen_sink` | Maximum control set |

### Functional Form

| spec_id | Description |
|---------|-------------|
| `ols/form/linear` | Linear specification |
| `ols/form/log_dep` | Log dependent variable |
| `ols/form/log_indep` | Log independent variable |
| `ols/form/log_log` | Log-log (elasticity) |
| `ols/form/quadratic` | Quadratic in treatment |
| `ols/form/polynomial` | Higher-order polynomial |
| `ols/form/spline` | Spline specification |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `ols/sample/full` | Full sample |
| `ols/sample/trimmed` | Trim outliers (1%/99%) |
| `ols/sample/winsorized` | Winsorize outliers |
| `ols/sample/complete_cases` | Complete cases only |
| `ols/sample/subgroup_male` | Male subsample |
| `ols/sample/subgroup_female` | Female subsample |
| `ols/sample/subgroup_high` | High-value subsample |
| `ols/sample/subgroup_low` | Low-value subsample |

### Interaction Effects

| spec_id | Description |
|---------|-------------|
| `ols/interact/none` | No interactions |
| `ols/interact/gender` | Gender × treatment |
| `ols/interact/age` | Age × treatment |
| `ols/interact/region` | Region × treatment |
| `ols/interact/fully_saturated` | All two-way interactions |

### Fixed Effects (if groups exist)

| spec_id | Description |
|---------|-------------|
| `ols/fe/none` | No fixed effects |
| `ols/fe/region` | Region fixed effects |
| `ols/fe/industry` | Industry fixed effects |
| `ols/fe/occupation` | Occupation fixed effects |

---

## Python Implementation Notes

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pyfixest as pf

# Standard OLS
model = smf.ols("y ~ x + controls", data=df).fit(cov_type='HC1')

# With fixed effects
model = pf.feols("y ~ x + controls | region", data=df)

# Quantile regression
from statsmodels.regression.quantile_regression import QuantReg
model = QuantReg(df['y'], df[['const', 'x', 'controls']]).fit(q=0.5)

# WLS
model = smf.wls("y ~ x + controls", data=df, weights=df['weight']).fit()
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "education_years",
    "coef": 0.08,
    "se": 0.01,
    "pval": 0.000,
    "ci_lower": 0.06,
    "ci_upper": 0.10
  },
  "controls": [
    {"var": "age", "coef": 0.02, "se": 0.005, "pval": 0.000},
    {"var": "age_sq", "coef": -0.0003, "se": 0.0001, "pval": 0.001},
    {"var": "female", "coef": -0.15, "se": 0.02, "pval": 0.000},
    {"var": "urban", "coef": 0.05, "se": 0.02, "pval": 0.01}
  ],
  "fixed_effects_absorbed": ["region"],
  "diagnostics": {
    "ramsey_reset_pval": 0.34,
    "breusch_pagan_pval": 0.02,
    "vif_max": 3.2,
    "condition_number": 45
  },
  "n_obs": 10000,
  "r_squared": 0.35,
  "adj_r_squared": 0.34,
  "f_stat": 125.4,
  "f_pval": 0.000
}
```

---

## Checklist

Before completing a cross-sectional OLS analysis, verify you have run:

- [ ] Baseline replication
- [ ] Multiple control sets (none, minimal, full)
- [ ] Robust standard errors
- [ ] At least 2 functional form variations
- [ ] Outlier sensitivity (trim/winsorize)
- [ ] Key subgroup analyses
- [ ] Report F-stat and R-squared in coefficient vector
