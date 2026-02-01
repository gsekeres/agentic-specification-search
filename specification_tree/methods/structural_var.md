# Structural VAR Specifications

## Spec ID Format: `svar/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main SVAR result
- Record: variables, lags, identification scheme, sample period

---

## Core Variations

### Identification Scheme

| spec_id | Description |
|---------|-------------|
| `svar/id/cholesky` | Recursive (Cholesky) identification |
| `svar/id/sign_restrict` | Sign restrictions only |
| `svar/id/zero_restrict` | Zero restrictions (short or long-run) |
| `svar/id/sign_zero` | Combined sign and zero restrictions |
| `svar/id/narrative` | Narrative identification |
| `svar/id/external_iv` | External instrument (proxy SVAR) |
| `svar/id/heteroskedasticity` | Identification via heteroskedasticity |

### Variable Ordering (for Cholesky)

| spec_id | Description |
|---------|-------------|
| `svar/order/baseline` | Paper's ordering |
| `svar/order/reversed` | Reversed ordering |
| `svar/order/alternative` | Alternative economically-motivated ordering |

### Lag Length

| spec_id | Description |
|---------|-------------|
| `svar/lags/p1` | 1 lag |
| `svar/lags/p2` | 2 lags |
| `svar/lags/p3` | 3 lags |
| `svar/lags/p4` | 4 lags |
| `svar/lags/aic_optimal` | AIC-optimal lag length |
| `svar/lags/bic_optimal` | BIC-optimal lag length |
| `svar/lags/hq_optimal` | Hannan-Quinn optimal |

### Sample Period

| spec_id | Description |
|---------|-------------|
| `svar/sample/full` | Full available sample |
| `svar/sample/pre_crisis` | Excluding financial crisis |
| `svar/sample/post_volcker` | Post-1983 (Great Moderation) |
| `svar/sample/pre_volcker` | Pre-1983 |
| `svar/sample/rolling` | Rolling window estimation |
| `svar/sample/expanding` | Expanding window estimation |

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `svar/method/ols` | OLS/MLE |
| `svar/method/bayesian_flat` | Bayesian with flat priors |
| `svar/method/bayesian_minnesota` | Minnesota prior |
| `svar/method/bayesian_glp` | Giannone-Lenza-Primiceri hierarchical prior |
| `svar/method/bayesian_ssvs` | Stochastic search variable selection |

### Variable Measurement

| spec_id | Description |
|---------|-------------|
| `svar/vars/baseline` | Paper's variable definitions |
| `svar/vars/alternative` | Alternative measures |
| `svar/vars/more` | Additional variables (larger VAR) |
| `svar/vars/fewer` | Fewer variables (smaller VAR) |

### IRF Horizon

| spec_id | Description |
|---------|-------------|
| `svar/horizon/h0` | Impact (horizon 0) |
| `svar/horizon/h4` | 1 year (quarterly data) |
| `svar/horizon/h20` | 5 years |
| `svar/horizon/h40` | 10 years (long-run) |

---

## Sign and Zero Restriction Robustness

| spec_id | Description |
|---------|-------------|
| `svar/restrict/tighter` | More restrictive sign restrictions |
| `svar/restrict/looser` | Fewer sign restrictions |
| `svar/restrict/horizon_extend` | Extend sign restrictions to more horizons |
| `svar/restrict/no_long_run` | Remove long-run restrictions |
| `svar/restrict/alternative_zero` | Alternative zero restrictions |

---

## Local Projection Comparison

| spec_id | Description |
|---------|-------------|
| `svar/lp/comparison` | Local projection with same shock |
| `svar/lp/newey_west` | LP with Newey-West standard errors |
| `svar/lp/boot` | LP with bootstrap confidence intervals |

---

## Python Implementation Notes

```python
# SVAR estimation in Python is limited compared to MATLAB/Stata
# Options include:

# 1. statsmodels VAR (basic, no Bayesian)
from statsmodels.tsa.api import VAR
model = VAR(data)
results = model.fit(maxlags=4, ic='aic')
irf = results.irf(40)

# 2. For Bayesian VAR, use PyMC or consider R
# R packages: vars, svars, bvars

# 3. Local projections in Python
import pyfixest as pf
# For each horizon h:
# y_{t+h} - y_t = alpha + beta * shock_t + controls + epsilon

# 4. For sign restrictions: custom implementation needed
# Algorithms: Uhlig (2005), Rubio-Ramirez et al. (2010)
```

---

## Coefficient Vector Format

```json
{
  "shock": {
    "name": "demand_shock",
    "identification": "sign_restrictions"
  },
  "irf": {
    "response_var": "employment",
    "horizon": 40,
    "coef": 0.75,
    "se": 0.31,
    "pval": 0.028,
    "ci_95": [0.13, 1.41],
    "ci_68": [0.56, 0.97]
  },
  "estimation": {
    "method": "Bayesian_SVAR",
    "prior": "GLP_hierarchical",
    "n_draws": 2000,
    "prob_positive": 0.986
  },
  "model": {
    "n_vars": 4,
    "n_lags": 3,
    "variables": ["GDP", "Prices", "Employment", "Investment"],
    "sample": "1983Q1-2019Q4",
    "n_obs": 148
  },
  "diagnostics": {
    "forecast_error_variance_decomposition": {...},
    "historical_decomposition": {...}
  }
}
```

---

## Checklist

Before completing an SVAR analysis, verify you have run:

- [ ] Baseline replication
- [ ] Alternative lag lengths
- [ ] Alternative sample periods
- [ ] Alternative variable measurements
- [ ] Subsample stability checks
- [ ] Local projection comparison
- [ ] Multiple IRF horizons (impact, medium, long-run)
- [ ] Report FEVD (forecast error variance decomposition)
- [ ] Sensitivity to identification assumptions (if applicable)
- [ ] Bayesian vs frequentist comparison (if applicable)

---

## References

Key methodological papers:
- Sims (1980): Original VAR methodology
- Blanchard & Quah (1989): Long-run restrictions
- Uhlig (2005): Sign restrictions
- Arias, Rubio-Ramirez & Waggoner (2018): Sign/zero restriction algorithms
- Giannone, Lenza & Primiceri (2015): Bayesian VAR priors
- Jorda (2005): Local projections
