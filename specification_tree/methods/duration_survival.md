# Duration/Survival Analysis Specifications

## Spec ID Format: `duration/{category}/{variation}`

## Overview

Duration models (hazard/survival models) analyze time-to-event data. Common in labor economics (unemployment duration), health (time to recovery/death), and industrial organization (firm survival).

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main duration model result
- Record: Model type, clustering, sample restrictions

---

## Core Variations

### Model Type

| spec_id | Description |
|---------|-------------|
| `duration/model/cox` | Cox proportional hazards (semi-parametric) |
| `duration/model/weibull` | Weibull parametric model |
| `duration/model/exponential` | Exponential parametric model |
| `duration/model/log_normal` | Log-normal parametric model |
| `duration/model/log_logistic` | Log-logistic parametric model |
| `duration/model/gompertz` | Gompertz parametric model |
| `duration/model/discrete` | Discrete time hazard (logit/probit on person-period data) |
| `duration/model/competing_risks` | Competing risks model |

### Baseline Hazard Specification (for Cox)

| spec_id | Description |
|---------|-------------|
| `duration/baseline/breslow` | Breslow method for ties (default) |
| `duration/baseline/efron` | Efron method for ties |
| `duration/baseline/exact` | Exact partial likelihood |
| `duration/baseline/stratified` | Stratified baseline by group |

### Time-Varying Covariates

| spec_id | Description |
|---------|-------------|
| `duration/tvc/none` | No time-varying coefficients |
| `duration/tvc/treatment` | Treatment effect varies with duration |
| `duration/tvc/interaction` | Treatment × duration interaction |
| `duration/tvc/piece_wise` | Piece-wise constant hazard |

### Unobserved Heterogeneity (Frailty)

| spec_id | Description |
|---------|-------------|
| `duration/frailty/none` | No frailty |
| `duration/frailty/gamma` | Gamma frailty |
| `duration/frailty/gaussian` | Gaussian frailty |
| `duration/frailty/shared` | Shared frailty by group |

### Standard Errors

| spec_id | Description |
|---------|-------------|
| `duration/se/robust` | Robust (sandwich) standard errors |
| `duration/se/cluster_unit` | Clustered by individual |
| `duration/se/cluster_state` | Clustered by state |
| `duration/se/cluster_time` | Clustered by time period |
| `duration/se/cluster_twoway` | Two-way clustering |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `duration/sample/full` | Full sample |
| `duration/sample/young` | Young workers (e.g., 20-40) |
| `duration/sample/old` | Older workers (e.g., 41-60) |
| `duration/sample/high_assets` | High asset quartiles |
| `duration/sample/low_assets` | Low asset quartiles |
| `duration/sample/ui_recipients` | UI recipients only |

### Censoring Treatment

| spec_id | Description |
|---------|-------------|
| `duration/censor/right_50` | Right censor at 50 weeks |
| `duration/censor/right_26` | Right censor at 26 weeks |
| `duration/censor/right_99` | Right censor at 99 weeks |
| `duration/censor/uncensored` | No artificial censoring |

---

## Python Implementation Notes

```python
# Using lifelines
from lifelines import CoxPHFitter, WeibullFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

# Cox proportional hazards
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event',
        cluster_col='cluster_var', robust=True)

# With time-varying coefficients
cph.fit(df, duration_col='duration', event_col='event',
        formula="treatment + treatment:duration + controls")

# Parametric model
wf = WeibullAFTFitter()
wf.fit(df, duration_col='duration', event_col='event')

# Using statsmodels
from statsmodels.duration.hazard_regression import PHReg
model = PHReg(endog=df['duration'], exog=df[['treatment', 'controls']],
              status=df['event'])
result = model.fit()

# Using R via rpy2
from rpy2.robjects.packages import importr
survival = importr('survival')
```

---

## Coefficient Vector Format

For Cox models, coefficients represent log hazard ratios:

```json
{
  "treatment": {
    "var": "log_benefits",
    "coef": -0.36,
    "se": 0.11,
    "hazard_ratio": 0.70,
    "pval": 0.001,
    "ci_lower": -0.58,
    "ci_upper": -0.15
  },
  "time_varying_effect": {
    "var": "duration_x_log_benefits",
    "coef": -0.01,
    "se": 0.004,
    "pval": 0.008
  },
  "controls": [
    {"var": "age", "coef": -0.012, "se": 0.002, "pval": 0.000},
    {"var": "education", "coef": 0.020, "se": 0.007, "pval": 0.005}
  ],
  "diagnostics": {
    "concordance_index": 0.62,
    "log_likelihood": -25055.35,
    "aic": 50192.70,
    "n_subjects": 4529,
    "n_events": 3416,
    "n_censored": 1113,
    "time_at_risk": 82756,
    "proportional_hazards_test_pval": 0.15
  },
  "fixed_effects_absorbed": ["year", "state", "occupation", "industry"]
}
```

---

## Key Interpretation Notes

1. **Cox coefficients** are log hazard ratios:
   - Positive coef = higher hazard = shorter duration
   - Negative coef = lower hazard = longer duration

2. **For UI benefits on unemployment duration**:
   - Negative coefficient on log_benefits = higher benefits -> lower job finding hazard -> longer unemployment
   - The coefficient is the elasticity of the hazard with respect to benefits

3. **Time-varying coefficients** (treatment × duration):
   - Captures whether treatment effect changes over spell duration
   - Common in UI studies: moral hazard vs liquidity effects

---

## Diagnostics

| spec_id | Description |
|---------|-------------|
| `duration/diagnostic/ph_test` | Proportional hazards test (Schoenfeld residuals) |
| `duration/diagnostic/martingale` | Martingale residuals plot |
| `duration/diagnostic/deviance` | Deviance residuals |
| `duration/diagnostic/dfbeta` | Influential observations |
| `duration/diagnostic/loglog_plot` | Log-log plot for PH assumption |

---

## Checklist

Before completing a duration analysis, verify you have run:

- [ ] Baseline replication (exact match to paper)
- [ ] Cox vs at least one parametric alternative
- [ ] With and without time-varying coefficient on treatment
- [ ] Multiple clustering choices (unit, state, etc.)
- [ ] Age subgroup analysis (young vs old)
- [ ] Asset/wealth quartile analysis (if applicable)
- [ ] Proportional hazards test
- [ ] Report concordance index
- [ ] Report number of subjects, events, censored
