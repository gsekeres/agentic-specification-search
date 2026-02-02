# Instrumental Variables Specifications

## Spec ID Format: `iv/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main IV/2SLS result
- Record: endogenous variable, instrument(s), first-stage F-statistic

---

## Core Variations

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `iv/method/2sls` | Two-stage least squares |
| `iv/method/liml` | Limited information maximum likelihood |
| `iv/method/gmm_2step` | Two-step efficient GMM |
| `iv/method/gmm_cue` | Continuously updated GMM |
| `iv/method/control_function` | Control function approach |
| `iv/method/ols` | OLS (ignoring endogeneity, for comparison) |

### First Stage

| spec_id | Description |
|---------|-------------|
| `iv/first_stage/baseline` | Paper's first stage |
| `iv/first_stage/weak_robust` | Weak-IV robust inference |
| `iv/first_stage/ar_confidence` | Anderson-Rubin confidence set |
| `iv/first_stage/reduced_form` | Reduced form (direct effect of Z on Y) |

### Instrument Sets

| spec_id | Description |
|---------|-------------|
| `iv/instruments/all` | All instruments |
| `iv/instruments/single` | Single strongest instrument |
| `iv/instruments/subset` | Subset of instruments |
| `iv/instruments/alternative` | Alternative instrument (if available) |

### Fixed Effects

| spec_id | Description |
|---------|-------------|
| `iv/fe/none` | No fixed effects |
| `iv/fe/unit` | Unit fixed effects |
| `iv/fe/time` | Time fixed effects |
| `iv/fe/twoway` | Unit + Time fixed effects |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `iv/controls/none` | No controls |
| `iv/controls/minimal` | Minimal controls |
| `iv/controls/baseline` | Paper's baseline controls |
| `iv/controls/full` | All available controls |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `iv/sample/full` | Full sample |
| `iv/sample/compliers` | Complier subgroup (if identifiable) |
| `iv/sample/restricted` | Sample with valid instrument |
| `iv/sample/trimmed` | Trimmed extreme values |

### Diagnostics

| spec_id | Description |
|---------|-------------|
| `iv/diagnostic/weak_iv` | Weak instrument tests |
| `iv/diagnostic/overid` | Overidentification test (Sargan/Hansen) |
| `iv/diagnostic/endogeneity` | Endogeneity test (Hausman) |
| `iv/diagnostic/exclusion` | Exclusion restriction plausibility |

### Alternative IV Estimators

| spec_id | Description |
|---------|-------------|
| `iv/jackknife` | Jackknife IV estimator (JIVE) |
| `iv/split_sample` | Split-sample IV |
| `iv/leave_one_out` | Leave-one-out IV |
| `iv/many_instruments` | Many instruments robust (LIML/Fuller) |
| `iv/fuller` | Fuller's modified LIML |

### Bounds and Partial Identification

| spec_id | Description |
|---------|-------------|
| `iv/bounds/imbens_manski` | Imbens-Manski bounds |
| `iv/bounds/nevo_rosen` | Nevo-Rosen imperfect IV bounds |
| `iv/bounds/conley` | Conley et al. plausibly exogenous |
| `iv/bounds/armstrong_kolesar` | Armstrong-Kolesar sensitivity |

### Heterogeneous Effects

| spec_id | Description |
|---------|-------------|
| `iv/late` | Local average treatment effect |
| `iv/marginal_te` | Marginal treatment effects |
| `iv/complier_characteristics` | Complier characteristics |

### Placebos and Falsification

| spec_id | Description |
|---------|-------------|
| `iv/placebo/lagged_outcome` | IV on lagged outcome |
| `iv/placebo/predetermined` | IV on predetermined variable |
| `iv/balance/covariates` | Covariate balance on instrument |

---

## Python Implementation Notes

```python
import pyfixest as pf
from linearmodels.iv import IV2SLS

# Using pyfixest
model = pf.feols("y ~ 1 | unit + time | endog ~ instrument", data=df)

# Using linearmodels
from linearmodels.iv import IV2SLS
model = IV2SLS.from_formula("y ~ 1 + controls + [endog ~ instrument]", data=df)
result = model.fit(cov_type='clustered', clusters=df['cluster'])

# Weak IV robust inference
from linearmodels.iv import IVLIML
model = IVLIML.from_formula("y ~ 1 + [endog ~ instrument]", data=df)
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "endogenous_var",
    "coef": 0.25,
    "se": 0.08,
    "pval": 0.002,
    "ci_lower": 0.09,
    "ci_upper": 0.41
  },
  "first_stage": {
    "instrument": "z",
    "coef": 0.45,
    "se": 0.05,
    "pval": 0.000,
    "F_stat": 85.3,
    "partial_r2": 0.12
  },
  "controls": [
    {"var": "age", "coef": 0.1, "se": 0.05, "pval": 0.04}
  ],
  "fixed_effects_absorbed": ["unit", "time"],
  "diagnostics": {
    "first_stage_F": 85.3,
    "weak_iv_pval": null,
    "overid_stat": 1.23,
    "overid_pval": 0.54,
    "endogeneity_stat": 4.56,
    "endogeneity_pval": 0.03,
    "n_instruments": 2,
    "n_endogenous": 1
  },
  "n_obs": 5000,
  "n_clusters": 100
}
```

---

## Checklist

Before completing an IV analysis, verify you have run:

- [ ] Baseline replication
- [ ] First-stage F-statistic reported (>10 rule of thumb)
- [ ] 2SLS and at least one alternative estimator (LIML)
- [ ] Reduced form regression
- [ ] Overidentification test (if overidentified)
- [ ] Weak IV robust inference (if F < 20)
- [ ] OLS for comparison
- [ ] Report first stage coefficients in coefficient vector
