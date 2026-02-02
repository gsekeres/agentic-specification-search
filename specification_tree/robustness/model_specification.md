# Model Specification Robustness Checks

## Spec ID Format: `robust/model/{specification}`

## Purpose

Test sensitivity of results to fundamental modeling choices including estimator type, fixed effects structure, and distributional assumptions.

---

## Linear Model Alternatives

| spec_id | Description |
|---------|-------------|
| `robust/model/ols` | OLS without fixed effects |
| `robust/model/pooled_ols` | Pooled OLS ignoring panel structure |
| `robust/model/fe_unit` | Unit fixed effects only |
| `robust/model/fe_time` | Time fixed effects only |
| `robust/model/fe_twoway` | Two-way fixed effects |
| `robust/model/random_effects` | Random effects model |
| `robust/model/between` | Between estimator |
| `robust/model/first_diff` | First differences |
| `robust/model/long_diff` | Long differences |

---

## Fixed Effects vs Random Effects

| spec_id | Description |
|---------|-------------|
| `robust/model/hausman_test` | Hausman specification test |
| `robust/model/mundlak` | Correlated random effects (Mundlak) |
| `robust/model/chamberlain` | Chamberlain's approach |
| `robust/model/hybrid` | Hybrid model (within + between) |

---

## Binary/Discrete Outcome Models

| spec_id | Description |
|---------|-------------|
| `robust/model/lpm` | Linear probability model |
| `robust/model/logit` | Logit |
| `robust/model/probit` | Probit |
| `robust/model/logit_fe` | Conditional logit with fixed effects |
| `robust/model/cloglog` | Complementary log-log |
| `robust/model/logit_re` | Random effects logit |

---

## Count/Non-Negative Outcome Models

| spec_id | Description |
|---------|-------------|
| `robust/model/poisson` | Poisson regression |
| `robust/model/poisson_fe` | Poisson with fixed effects |
| `robust/model/ppml` | Poisson pseudo-maximum likelihood |
| `robust/model/negbin` | Negative binomial |
| `robust/model/zip` | Zero-inflated Poisson |
| `robust/model/zinb` | Zero-inflated negative binomial |
| `robust/model/tobit` | Tobit (censored) |
| `robust/model/truncreg` | Truncated regression |

---

## Quantile and Robust Estimators

| spec_id | Description |
|---------|-------------|
| `robust/model/quantile_25` | 25th percentile regression |
| `robust/model/quantile_50` | Median regression |
| `robust/model/quantile_75` | 75th percentile regression |
| `robust/model/quantile_90` | 90th percentile regression |
| `robust/model/robust_regression` | Robust regression (M-estimator) |
| `robust/model/lad` | Least absolute deviations |

---

## Selection and Sample Selection

| spec_id | Description |
|---------|-------------|
| `robust/model/heckman` | Heckman selection model |
| `robust/model/heckman_2step` | Heckman two-step |
| `robust/model/heckman_ml` | Heckman MLE |
| `robust/model/inverse_mills` | Include inverse Mills ratio |

---

## Dynamic Models

| spec_id | Description |
|---------|-------------|
| `robust/model/lagged_dv` | Lagged dependent variable |
| `robust/model/arellano_bond` | Arellano-Bond GMM |
| `robust/model/blundell_bond` | Blundell-Bond system GMM |
| `robust/model/anderson_hsiao` | Anderson-Hsiao |

---

## Matching Estimators

| spec_id | Description |
|---------|-------------|
| `robust/model/psm` | Propensity score matching |
| `robust/model/psm_kernel` | PSM with kernel matching |
| `robust/model/psm_nn` | PSM with nearest neighbor |
| `robust/model/cem` | Coarsened exact matching |
| `robust/model/mahalanobis` | Mahalanobis distance matching |
| `robust/model/entropy_balance` | Entropy balancing |

---

## Implementation Notes

```python
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS

# Fixed effects
fe_model = PanelOLS.from_formula('y ~ x + EntityEffects + TimeEffects', data=panel_df)
fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)

# Random effects
re_model = RandomEffects.from_formula('y ~ x', data=panel_df)
re_result = re_model.fit()

# Hausman test
from linearmodels.panel import compare
hausman = compare({'FE': fe_result, 'RE': re_result})

# First differences
fd_model = FirstDifferenceOLS.from_formula('y ~ x', data=panel_df)
fd_result = fd_model.fit()

# Logit with FE (conditional logit)
from statsmodels.discrete.conditional_models import ConditionalLogit
clogit = ConditionalLogit.from_formula('y ~ x', groups='unit', data=df)

# Poisson with FE
import pyfixest as pf
ppml = pf.fepois('y ~ x | unit + time', data=df)

# Propensity score matching
from causalinference import CausalModel
cm = CausalModel(Y, D, X)
cm.est_propensity_s()
cm.trim_s()
cm.stratify_s()
```

---

## Output Format

```json
{
  "spec_id": "robust/model/random_effects",
  "spec_tree_path": "robustness/model_specification.md",
  "model_type": "random_effects",
  "treatment": {
    "coef": 0.055,
    "se": 0.018,
    "pval": 0.002
  },
  "baseline_comparison": {
    "baseline_model": "fixed_effects",
    "baseline_coef": 0.052,
    "coef_change_pct": 5.8
  },
  "diagnostics": {
    "hausman_stat": 4.23,
    "hausman_pval": 0.12,
    "hausman_rejects_re": false
  }
}
```

---

## Interpretation Guidelines

| Pattern | Interpretation |
|---------|----------------|
| FE and RE similar | Unobserved heterogeneity uncorrelated with X |
| FE and OLS differ | Unit effects correlated with treatment |
| LPM and logit differ | Nonlinear effects matter |
| Quantiles vary | Heterogeneous effects across distribution |

### Model Selection Guidance

| Situation | Preferred Model |
|-----------|-----------------|
| Panel data, unit effects correlated | Fixed effects |
| Binary outcome, large N | Logit/Probit |
| Binary outcome, small N or FE | LPM |
| Count data with zeros | PPML |
| Selection into sample | Heckman |
| Dynamic panel | Arellano-Bond |

---

## Checklist

- [ ] Ran OLS and fixed effects comparison
- [ ] If panel: ran Hausman test (FE vs RE)
- [ ] If binary outcome: ran LPM and logit/probit
- [ ] Ran at least one alternative estimator
- [ ] If quantile regression: ran at least 3 quantiles
- [ ] Documented which model assumptions differ
- [ ] Flagged if conclusions change across models
