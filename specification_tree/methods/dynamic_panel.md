# Dynamic Panel Specifications

## Spec ID Format: `dynpanel/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main dynamic panel result
- Record: lag structure, GMM instruments, AR tests

---

## Core Variations

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `dynpanel/method/ols_fe` | OLS with FE (biased, for comparison) |
| `dynpanel/method/anderson_hsiao` | Anderson-Hsiao IV |
| `dynpanel/method/diff_gmm` | Difference GMM (Arellano-Bond) |
| `dynpanel/method/sys_gmm` | System GMM (Blundell-Bond) |
| `dynpanel/method/bias_corrected` | Bias-corrected LSDV |
| `dynpanel/method/pmg` | Pooled mean group estimator |
| `dynpanel/method/mg` | Mean group estimator |
| `dynpanel/method/dfe` | Dynamic fixed effects |

### Lag Structure

| spec_id | Description |
|---------|-------------|
| `dynpanel/lags/one` | Single lag of dependent variable |
| `dynpanel/lags/two` | Two lags of dependent variable |
| `dynpanel/lags/aic_optimal` | AIC-optimal lag length |
| `dynpanel/lags/bic_optimal` | BIC-optimal lag length |

### Instrument Matrix

| spec_id | Description |
|---------|-------------|
| `dynpanel/inst/all` | All available lags as instruments |
| `dynpanel/inst/collapsed` | Collapsed instrument matrix |
| `dynpanel/inst/lag2_4` | Lags 2-4 only |
| `dynpanel/inst/lag2_6` | Lags 2-6 only |
| `dynpanel/inst/limited` | Limited instruments (small T) |

### Weighting

| spec_id | Description |
|---------|-------------|
| `dynpanel/weight/one_step` | One-step GMM |
| `dynpanel/weight/two_step` | Two-step GMM |
| `dynpanel/weight/iterated` | Iterated GMM |
| `dynpanel/weight/windmeijer` | Windmeijer-corrected SE |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `dynpanel/controls/none` | No controls beyond lagged Y |
| `dynpanel/controls/exogenous` | Strictly exogenous controls |
| `dynpanel/controls/predetermined` | Predetermined controls |
| `dynpanel/controls/endogenous` | Endogenous controls (instrumented) |
| `dynpanel/controls/full` | Full control set |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `dynpanel/sample/full` | Full sample |
| `dynpanel/sample/balanced` | Balanced panel only |
| `dynpanel/sample/large_T` | Subsample with longer time series |
| `dynpanel/sample/trim_outliers` | Trim extreme values |

### Diagnostics

| spec_id | Description |
|---------|-------------|
| `dynpanel/diagnostic/ar1` | AR(1) test in residuals |
| `dynpanel/diagnostic/ar2` | AR(2) test in residuals |
| `dynpanel/diagnostic/sargan` | Sargan test of overidentification |
| `dynpanel/diagnostic/hansen` | Hansen J test |
| `dynpanel/diagnostic/diff_sargan` | Difference-in-Sargan test |

---

## Python Implementation Notes

```python
# Dynamic panel GMM is best done in Stata or R
# Python options are limited

# For simpler approaches:
import pyfixest as pf

# Biased FE with lagged Y (for comparison)
model = pf.feols("y ~ l(y, 1) + x | unit + time", data=df)

# For proper GMM, use R's plm package or Stata's xtabond2
# R example (via rpy2):
"""
library(plm)
pgmm(y ~ lag(y, 1) + x | lag(y, 2:99),
     data = pdata,
     effect = "twoways",
     model = "twosteps",
     transformation = "d")
"""

# Alternative: use linearmodels IV with lagged instruments
from linearmodels.iv import IV2SLS

# Manual first-differencing + IV
df_diff = df.groupby('unit').diff()
# Then IV with lagged levels
```

---

## Coefficient Vector Format

```json
{
  "lagged_dependent": {
    "var": "y_lag1",
    "coef": 0.65,
    "se": 0.08,
    "pval": 0.000,
    "persistence": "moderate"
  },
  "treatment": {
    "var": "policy",
    "coef": 0.12,
    "se": 0.04,
    "pval": 0.003,
    "long_run_effect": 0.34
  },
  "controls": [
    {"var": "gdp", "coef": 0.05, "se": 0.02, "pval": 0.01}
  ],
  "diagnostics": {
    "ar1_stat": -2.45,
    "ar1_pval": 0.01,
    "ar2_stat": 0.87,
    "ar2_pval": 0.38,
    "sargan_stat": 45.2,
    "sargan_pval": 0.32,
    "hansen_stat": 42.1,
    "hansen_pval": 0.41,
    "n_instruments": 48,
    "n_groups": 100,
    "n_obs": 1500,
    "instrument_ratio": 0.48
  },
  "n_obs": 1500,
  "n_groups": 100,
  "avg_T": 15
}
```

---

## Long-Run Effect Calculation

For a model: $y_{it} = \alpha y_{i,t-1} + \beta x_{it} + \epsilon_{it}$

Long-run effect of $x$: $\frac{\beta}{1-\alpha}$

```json
{
  "short_run_effect": 0.12,
  "persistence_parameter": 0.65,
  "long_run_effect": 0.34,
  "long_run_se": 0.15
}
```

---

## Checklist

Before completing a dynamic panel analysis, verify you have run:

- [ ] Baseline replication
- [ ] Biased OLS-FE for comparison
- [ ] Difference GMM (Arellano-Bond)
- [ ] System GMM (Blundell-Bond)
- [ ] AR(1) and AR(2) tests reported
- [ ] Sargan/Hansen overidentification test
- [ ] Multiple lag lengths
- [ ] Collapsed vs full instrument matrix
- [ ] Report long-run effects if applicable
- [ ] Check instrument count vs groups (rule: inst < groups)
