# Discrete Choice Specifications

## Spec ID Format: `discrete/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main discrete choice result
- Record: outcome categories, estimation method, marginal effects

---

## Core Variations

### Model Type (Binary Outcome)

| spec_id | Description |
|---------|-------------|
| `discrete/binary/logit` | Logistic regression |
| `discrete/binary/probit` | Probit regression |
| `discrete/binary/lpm` | Linear probability model |
| `discrete/binary/cloglog` | Complementary log-log |

### Model Type (Multinomial Outcome)

| spec_id | Description |
|---------|-------------|
| `discrete/multi/mlogit` | Multinomial logit |
| `discrete/multi/mprobit` | Multinomial probit |
| `discrete/multi/ordered_logit` | Ordered logit |
| `discrete/multi/ordered_probit` | Ordered probit |
| `discrete/multi/nested_logit` | Nested logit |
| `discrete/multi/mixed_logit` | Mixed (random coefficients) logit |

### Marginal Effects

| spec_id | Description |
|---------|-------------|
| `discrete/mfx/average` | Average marginal effects (AME) |
| `discrete/mfx/at_means` | Marginal effects at means (MEM) |
| `discrete/mfx/at_representative` | At representative values |
| `discrete/mfx/by_group` | Group-specific marginal effects |

### Fixed Effects

| spec_id | Description |
|---------|-------------|
| `discrete/fe/none` | No fixed effects |
| `discrete/fe/group` | Group fixed effects (conditional logit) |
| `discrete/fe/cre` | Correlated random effects (Mundlak) |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `discrete/controls/none` | No controls |
| `discrete/controls/baseline` | Paper's baseline controls |
| `discrete/controls/full` | All available controls |
| `discrete/controls/interactions` | Including key interactions |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `discrete/sample/full` | Full sample |
| `discrete/sample/rare_events` | Rare events correction |
| `discrete/sample/balanced_outcomes` | Balance by outcome |
| `discrete/sample/subsample` | Key subsample |

### Standard Errors

| spec_id | Description |
|---------|-------------|
| `discrete/se/classical` | MLE standard errors |
| `discrete/se/robust` | Robust (sandwich) SE |
| `discrete/se/clustered` | Clustered SE |
| `discrete/se/bootstrap` | Bootstrap SE |

---

## Python Implementation Notes

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Logit
model = smf.logit("y ~ x + controls", data=df).fit()

# Probit
model = smf.probit("y ~ x + controls", data=df).fit()

# Marginal effects
mfx = model.get_margeff(at='overall')  # AME
mfx = model.get_margeff(at='mean')     # MEM

# Multinomial logit
from statsmodels.discrete.discrete_model import MNLogit
model = MNLogit(y, X).fit()

# Ordered logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
model = OrderedModel(y, X, distr='logit').fit()

# Conditional logit with fixed effects (R preferred)
# Or use panel logit approximation
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "treatment",
    "coef": 0.45,
    "se": 0.12,
    "pval": 0.000,
    "odds_ratio": 1.57,
    "marginal_effect": 0.08,
    "mfx_se": 0.02
  },
  "controls": [
    {
      "var": "age",
      "coef": 0.02,
      "se": 0.005,
      "pval": 0.000,
      "odds_ratio": 1.02,
      "marginal_effect": 0.004
    },
    {
      "var": "income",
      "coef": 0.001,
      "se": 0.0003,
      "pval": 0.001,
      "odds_ratio": 1.001,
      "marginal_effect": 0.0002
    }
  ],
  "fixed_effects_absorbed": [],
  "diagnostics": {
    "pseudo_r2_mcfadden": 0.15,
    "pseudo_r2_count": 0.72,
    "ll_model": -5234,
    "ll_null": -6150,
    "lr_stat": 1832,
    "lr_pval": 0.000,
    "aic": 10490,
    "bic": 10560,
    "outcome_distribution": {
      "0": 7500,
      "1": 2500
    }
  },
  "n_obs": 10000,
  "marginal_effect_type": "average"
}
```

---

## Multinomial Coefficient Vector Format

```json
{
  "outcomes": ["choice_1", "choice_2", "choice_3"],
  "base_outcome": "choice_1",
  "coefficients_by_outcome": {
    "choice_2": {
      "treatment": {"coef": 0.3, "se": 0.1, "pval": 0.003, "rrr": 1.35},
      "age": {"coef": 0.01, "se": 0.005, "pval": 0.04, "rrr": 1.01}
    },
    "choice_3": {
      "treatment": {"coef": 0.5, "se": 0.15, "pval": 0.001, "rrr": 1.65},
      "age": {"coef": 0.02, "se": 0.006, "pval": 0.001, "rrr": 1.02}
    }
  },
  "marginal_effects": {
    "treatment": {
      "choice_1": {"mfx": -0.08, "se": 0.02},
      "choice_2": {"mfx": 0.03, "se": 0.01},
      "choice_3": {"mfx": 0.05, "se": 0.02}
    }
  },
  "diagnostics": {
    "iia_test_pval": 0.45
  },
  "n_obs": 10000
}
```

---

## Checklist

Before completing a discrete choice analysis, verify you have run:

- [ ] Baseline replication
- [ ] Logit AND probit (for binary)
- [ ] Linear probability model for comparison
- [ ] Average marginal effects reported
- [ ] Multiple control sets
- [ ] Robust or clustered standard errors
- [ ] Report pseudo R-squared and outcome distribution
- [ ] For multinomial: test IIA assumption
