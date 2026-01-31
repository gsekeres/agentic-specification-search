# Functional Form Robustness Checks

## Spec ID Format: `robust/form/{transformation}`

## Purpose

Test sensitivity of results to functional form assumptions including variable transformations, nonlinearities, and model specifications.

---

## Outcome Variable Transformations

| spec_id | Description |
|---------|-------------|
| `robust/form/y_level` | Outcome in levels (baseline) |
| `robust/form/y_log` | Log outcome (if positive) |
| `robust/form/y_asinh` | Inverse hyperbolic sine (handles zeros) |
| `robust/form/y_pct_change` | Percent change |
| `robust/form/y_growth_rate` | Growth rate |
| `robust/form/y_standardized` | Standardized (z-score) |
| `robust/form/y_rank` | Rank transformation |
| `robust/form/y_winsorized` | Winsorized outcome |

---

## Treatment Variable Transformations

| spec_id | Description |
|---------|-------------|
| `robust/form/x_level` | Treatment in levels (baseline) |
| `robust/form/x_log` | Log treatment |
| `robust/form/x_standardized` | Standardized treatment |
| `robust/form/x_binary` | Binary treatment (above/below threshold) |
| `robust/form/x_terciles` | Treatment in terciles |
| `robust/form/x_quintiles` | Treatment in quintiles |

---

## Nonlinear Specifications

| spec_id | Description |
|---------|-------------|
| `robust/form/quadratic` | Add treatment squared |
| `robust/form/cubic` | Add treatment cubed |
| `robust/form/spline_3` | Cubic spline (3 knots) |
| `robust/form/spline_5` | Cubic spline (5 knots) |
| `robust/form/polynomial_4` | Fourth-order polynomial |
| `robust/form/log_log` | Log-log specification (elasticity) |

---

## Control Variable Transformations

| spec_id | Description |
|---------|-------------|
| `robust/form/controls_log` | Log-transform continuous controls |
| `robust/form/controls_quadratic` | Add squared terms for key controls |
| `robust/form/controls_categorical` | Convert continuous to categorical |

---

## Interaction Terms

| spec_id | Description |
|---------|-------------|
| `robust/form/interact_time` | Treatment × time trend |
| `robust/form/interact_baseline_y` | Treatment × baseline outcome |
| `robust/form/interact_key_control` | Treatment × key control |
| `robust/form/fully_saturated` | All two-way interactions |

---

## Alternative Estimators

| spec_id | Description |
|---------|-------------|
| `robust/form/poisson` | Poisson regression (count/positive) |
| `robust/form/ppml` | Poisson pseudo-ML (robust to zeros) |
| `robust/form/tobit` | Tobit (censored outcome) |
| `robust/form/quantile_50` | Median regression |
| `robust/form/quantile_25` | 25th percentile regression |
| `robust/form/quantile_75` | 75th percentile regression |

---

## Implementation Notes

```python
import numpy as np
import pyfixest as pf
import statsmodels.api as sm

# Log transformation (handling zeros)
df['y_log'] = np.log(df['y'] + 1)
df['y_asinh'] = np.arcsinh(df['y'])

# Quadratic specification
model = pf.feols("y ~ x + I(x**2) | fe", data=df)

# Log-log (elasticity)
model = pf.feols("log(y) ~ log(x) | fe", data=df)

# Poisson (PPML)
from statsmodels.discrete.discrete_model import Poisson
model = Poisson.from_formula("y ~ x + controls", data=df).fit()

# Quantile regression
from statsmodels.regression.quantile_regression import QuantReg
model = QuantReg(df['y'], df[['const', 'x']]).fit(q=0.5)
```

---

## Output Format

```json
{
  "spec_id": "robust/form/y_log",
  "spec_tree_path": "robustness/functional_form.md",
  "transformation": {
    "outcome": "log(y + 1)",
    "treatment": "level",
    "interpretation": "semi-elasticity"
  },
  "treatment": {
    "coef": 0.045,
    "se": 0.015,
    "pval": 0.003,
    "interpretation": "4.5% increase in y per unit increase in x"
  },
  "comparison_to_baseline": {
    "baseline_form": "level",
    "baseline_coef": 3.2,
    "comparable_effect": "3.2 units ≈ 4.5% at mean(y)=71",
    "direction_consistent": true
  }
}
```

---

## Interpretation Across Transformations

### Level vs Log

| Baseline (Level) | Log Outcome | Interpretation |
|------------------|-------------|----------------|
| β = 5 | β = 0.07 | Compare at mean of y |

Conversion: If $\bar{y} = 71$, then $5/71 \approx 0.07$ (7%)

### Elasticities

| Specification | Coefficient Interpretation |
|---------------|---------------------------|
| Level-Level | $\Delta y / \Delta x$ |
| Log-Level | $\% \Delta y / \Delta x$ |
| Level-Log | $\Delta y / \% \Delta x$ |
| Log-Log | $\% \Delta y / \% \Delta x$ (elasticity) |

---

## Red Flags

| Pattern | Concern |
|---------|---------|
| Sign flips with log transform | Outliers driving results |
| Large coefficient change with quadratic | Nonlinear relationship |
| Poisson very different from OLS | Outcome distribution matters |
| Quantile varies widely | Heterogeneous effects |

---

## Checklist

- [ ] Ran baseline functional form
- [ ] Ran log outcome (if appropriate)
- [ ] Ran asinh if outcome has zeros
- [ ] Ran quadratic specification
- [ ] Ran at least one alternative estimator (Poisson/quantile)
- [ ] Documented interpretation of each specification
- [ ] Checked if direction of effect is consistent
- [ ] Flagged any large changes in magnitude
