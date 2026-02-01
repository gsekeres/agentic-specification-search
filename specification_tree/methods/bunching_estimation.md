# Bunching Estimation Specifications

## Spec ID Format: `bunching/{category}/{variation}`

## Overview

Bunching estimation is used to estimate behavioral elasticities from discontinuities in incentive schedules (kinks/notches). The method compares actual density distributions to counterfactual distributions around threshold points.

Key references:
- Saez (2010) - Kink bunching at tax brackets
- Chetty et al. (2011) - Bounds on elasticities
- Kleven and Waseem (2013) - Notch bunching

---

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main bunching estimate
- Record: threshold(s), bandwidth, polynomial order, excess mass, elasticity

---

## Core Variations

### Counterfactual Polynomial Order

| spec_id | Description |
|---------|-------------|
| `bunching/poly/order_3` | Third-order polynomial counterfactual |
| `bunching/poly/order_4` | Fourth-order polynomial counterfactual |
| `bunching/poly/order_5` | Fifth-order polynomial counterfactual |
| `bunching/poly/order_6` | Sixth-order polynomial counterfactual |
| `bunching/poly/order_7` | Seventh-order polynomial counterfactual |

### Bandwidth Selection

| spec_id | Description |
|---------|-------------|
| `bunching/bandwidth/narrow` | Narrow window around threshold |
| `bunching/bandwidth/baseline` | Paper's baseline bandwidth |
| `bunching/bandwidth/wide` | Wide window around threshold |
| `bunching/bandwidth/optimal` | Optimally selected bandwidth |
| `bunching/bandwidth/half` | Half baseline bandwidth |
| `bunching/bandwidth/double` | Double baseline bandwidth |

### Threshold-Specific Estimates

| spec_id | Description |
|---------|-------------|
| `bunching/threshold/single_{X}` | Single threshold estimate at X |
| `bunching/threshold/pooled` | Pooled estimate across thresholds |
| `bunching/threshold/average` | Average of threshold-specific estimates |
| `bunching/threshold/weighted_average` | N-weighted average |

### Excess Mass Calculation

| spec_id | Description |
|---------|-------------|
| `bunching/excess/point` | Point estimate at threshold |
| `bunching/excess/window` | Excess over window post-threshold |
| `bunching/excess/cumulative` | Cumulative excess mass |
| `bunching/excess/adjusted` | Adjusted for missing mass below |

### Smoothing/Adjustment

| spec_id | Description |
|---------|-------------|
| `bunching/smooth/none` | No smoothing |
| `bunching/smooth/seasonal` | Seasonal adjustment (integer dummies) |
| `bunching/smooth/moving_avg` | Moving average smoothing |
| `bunching/smooth/kernel` | Kernel smoothing |

### Standard Error Methods

| spec_id | Description |
|---------|-------------|
| `bunching/se/bootstrap` | Bootstrap standard errors |
| `bunching/se/parametric` | Parametric SEs |
| `bunching/se/block_bootstrap` | Block bootstrap |
| `bunching/se/delta_method` | Delta method |

---

## Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `bunching/sample/full` | Full sample |
| `bunching/sample/responders` | Responders only (binding constraint) |
| `bunching/sample/men` | Male subsample |
| `bunching/sample/women` | Female subsample |
| `bunching/sample/healthy` | Healthy subsample |
| `bunching/sample/by_industry` | Industry-specific estimates |
| `bunching/sample/early_period` | Early time period |
| `bunching/sample/late_period` | Late time period |

---

## Robustness Checks

| spec_id | Description |
|---------|-------------|
| `bunching/robust/donut` | Donut hole (exclude at threshold) |
| `bunching/robust/placebo_threshold` | Placebo threshold test |
| `bunching/robust/density_test` | Density continuity test |
| `bunching/robust/covariate_smooth` | Covariate smoothness test |
| `bunching/robust/round_number` | Control for round number bunching |

---

## With Controls (RD-style)

| spec_id | Description |
|---------|-------------|
| `bunching/controls/none` | No controls |
| `bunching/controls/demographics` | Demographic controls |
| `bunching/controls/full` | Full control set |
| `bunching/controls/residualized` | Residualized bunching |

---

## Python Implementation Notes

```python
import numpy as np
import pandas as pd
from scipy import stats

def estimate_bunching(df, running_var, threshold, bandwidth, poly_order=5):
    """
    Estimate excess mass at threshold using polynomial counterfactual.

    Parameters:
    -----------
    df : DataFrame with running variable
    running_var : column name of running variable
    threshold : location of kink/notch
    bandwidth : window around threshold (in units of running_var)
    poly_order : order of polynomial for counterfactual

    Returns:
    --------
    dict with excess_mass, elasticity, se (if bootstrap)
    """
    # Keep observations in bandwidth
    df_bw = df[(df[running_var] >= threshold - bandwidth) &
               (df[running_var] <= threshold + bandwidth)].copy()

    # Create bins and count
    counts = df_bw.groupby(running_var).size().reset_index(name='freq')

    # Create polynomial terms
    counts['v'] = counts[running_var] - threshold
    for i in range(1, poly_order + 1):
        counts[f'v{i}'] = counts['v'] ** i

    # Create left/right indicators
    counts['left'] = (counts[running_var] < threshold).astype(int)
    counts['right'] = (counts[running_var] > threshold).astype(int)

    # Create interaction terms
    for i in range(1, poly_order + 1):
        counts[f'v{i}_left'] = counts[f'v{i}'] * counts['left']
        counts[f'v{i}_right'] = counts[f'v{i}'] * counts['right']

    # Fit model excluding observations at threshold
    counts_fit = counts[counts[running_var] != threshold].copy()

    # Build formula
    poly_vars = [f'v{i}_left' for i in range(1, poly_order + 1)]
    poly_vars += [f'v{i}_right' for i in range(1, poly_order + 1)]

    X = counts_fit[poly_vars].values
    X = np.column_stack([np.ones(len(X)), X])
    y = counts_fit['freq'].values

    # OLS for counterfactual
    from numpy.linalg import lstsq
    coef, _, _, _ = lstsq(X, y, rcond=None)

    # Predict counterfactual at threshold
    X_threshold = np.zeros(len(poly_vars) + 1)
    X_threshold[0] = 1  # intercept

    counterfactual = X_threshold @ coef

    # Actual count at threshold
    actual = counts.loc[counts[running_var] == threshold, 'freq'].values[0]

    # Excess mass
    excess_mass = (actual - counterfactual) / counterfactual

    return {
        'excess_mass': excess_mass,
        'actual': actual,
        'counterfactual': counterfactual,
        'n_obs': len(df_bw)
    }

def bootstrap_bunching(df, running_var, threshold, bandwidth, poly_order=5, n_boot=1000):
    """Bootstrap standard errors for bunching estimate."""
    estimates = []
    for _ in range(n_boot):
        df_boot = df.sample(frac=1, replace=True)
        result = estimate_bunching(df_boot, running_var, threshold, bandwidth, poly_order)
        estimates.append(result['excess_mass'])

    se = np.std(estimates)
    ci_lower = np.percentile(estimates, 2.5)
    ci_upper = np.percentile(estimates, 97.5)

    return {'se': se, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "above_threshold",
    "excess_mass": 0.15,
    "elasticity": 0.45,
    "se": 0.08,
    "pval": 0.06,
    "ci_lower": -0.01,
    "ci_upper": 0.31
  },
  "threshold_info": {
    "threshold": 10,
    "threshold_unit": "years",
    "incentive_change": 0.33,
    "bandwidth": 3
  },
  "counterfactual": {
    "polynomial_order": 5,
    "actual_count": 450,
    "counterfactual_count": 391,
    "excess_mass_absolute": 59
  },
  "diagnostics": {
    "n_obs": 5000,
    "n_below_threshold": 2400,
    "n_above_threshold": 2600,
    "placebo_pval": 0.68,
    "density_test_pval": 0.45
  },
  "standard_errors": {
    "method": "bootstrap",
    "n_bootstrap": 1000
  }
}
```

---

## Checklist

Before completing a bunching analysis, verify you have run:

- [ ] Baseline replication
- [ ] At least 3 polynomial order choices
- [ ] At least 3 bandwidth choices
- [ ] Separate estimates by threshold (if multiple)
- [ ] Key subgroup analyses (gender, health, etc.)
- [ ] Placebo threshold test
- [ ] Reported bootstrap SEs or equivalent
- [ ] Documented excess mass and elasticity calculation

---

## Special Considerations

### Optimization Frictions

If frictions attenuate bunching:
- Report bounds (lower/upper)
- Consider friction-adjusted estimates

### Multiple Thresholds

If pooling across thresholds:
- Report threshold-specific estimates
- Document weighting scheme
- Test for heterogeneity across thresholds

### Diffuse Bunching

If bunching occurs over range (not point):
- Document bunching region
- Sum excess mass over region
- Consider adjustment methods
