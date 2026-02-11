# Data Pre-processing & Variable Coding (Robustness Checks)

## Spec ID format

This module defines **estimand-preserving** (core-eligible) robustness checks (RC) that operate through *data pre-processing* and *variable coding* choices.

Use:

- `rc/preprocess/{target}/{variant}`

Examples:

- `rc/preprocess/outcome/winsor_1_99`
- `rc/preprocess/treatment/standardize_z`
- `rc/preprocess/controls/collapse_categories`

## Purpose

In applied replication packages, a large share of researcher degrees of freedom live *before* the regression:

- how variables are constructed and coded,
- how continuous variables are discretized for controls,
- top-coding / winsorization / trimming rules,
- standardization and index construction,
- handling of missing values and special codes.

This module makes these choices explicit and auditable.

## Core principle (what is and is not “core”)

These specs are **core-eligible** when they preserve the baseline claim’s:

- outcome concept,
- treatment concept,
- estimand concept,
- target population.

If a coding change *redefines* treatment or outcome in a way that changes the estimand (e.g., converting a continuous treatment into a binary “above threshold” indicator when the baseline is continuous), it should be recorded as `explore/*` in the exploration modules instead.

## Outcome coding variations

| spec_id | Description |
|--------|-------------|
| `rc/preprocess/outcome/winsor_1_99` | Winsorize outcome at 1st/99th percentiles |
| `rc/preprocess/outcome/winsor_5_95` | Winsorize outcome at 5th/95th percentiles |
| `rc/preprocess/outcome/topcode_p99` | Top-code outcome at 99th percentile |
| `rc/preprocess/outcome/standardize_z` | Standardize outcome (z-score) using full sample |
| `rc/preprocess/outcome/standardize_within_fe` | Standardize within key FE groups (e.g., unit, region, cohort) |
| `rc/preprocess/outcome/standardize_baseline_period` | Standardize using baseline/pre period moments (panel settings) |
| `rc/preprocess/outcome/index_equal_weight` | If outcome is an index: equal-weighted components |
| `rc/preprocess/outcome/index_zscore_average` | Index = average of z-scored components |
| `rc/preprocess/outcome/index_pca_first` | Index = first principal component of components |

## Treatment coding variations

These are core-eligible when they preserve the treatment concept (e.g., “exposure intensity”) and only change units/scale or outlier handling.

| spec_id | Description |
|--------|-------------|
| `rc/preprocess/treatment/standardize_z` | Standardize treatment (z-score) to improve conditioning |
| `rc/preprocess/treatment/rescale_units` | Rescale treatment units (per 1,000; per capita; per SD) |
| `rc/preprocess/treatment/topcode_p99` | Top-code treatment at 99th percentile |
| `rc/preprocess/treatment/winsor_1_99` | Winsorize treatment at 1st/99th percentiles |
| `rc/preprocess/treatment/moving_average_3` | Smooth treatment via 3-period moving average (time series/panel) |
| `rc/preprocess/treatment/moving_average_5` | Smooth treatment via 5-period moving average (time series/panel) |
| `rc/preprocess/treatment/cumulative_exposure` | Cumulative exposure up to time t (panel) |

## Covariate coding (controls)

These variations help diagnose sensitivity to adjustment-set *coding* rather than adjustment-set *membership*.

| spec_id | Description |
|--------|-------------|
| `rc/preprocess/controls/collapse_categories` | Collapse sparse categories for a key categorical control |
| `rc/preprocess/controls/one_hot_encode` | One-hot encode categorical controls (vs numeric codes) |
| `rc/preprocess/controls/bin_continuous_terciles` | Replace a continuous control with tercile bins |
| `rc/preprocess/controls/bin_continuous_quartiles` | Replace a continuous control with quartile bins |
| `rc/preprocess/controls/standardize_continuous` | Standardize all continuous controls |
| `rc/preprocess/controls/missing_indicator` | Add missingness indicators for key controls |

## Missing values and special codes (core-eligible when concept preserved)

Missingness handling can change both sample and measurement; treat it as core-eligible only if it is clearly within the paper’s intended data definition.

| spec_id | Description |
|--------|-------------|
| `rc/preprocess/missing/listwise` | Listwise deletion (complete cases) |
| `rc/preprocess/missing/missing_indicator` | Missing indicators + simple imputation for controls |
| `rc/preprocess/missing/impute_mean` | Mean imputation (controls only) |
| `rc/preprocess/missing/impute_median` | Median imputation (controls only) |
| `rc/preprocess/missing/impute_mice` | Multiple imputation (MICE) where feasible |

## Required audit fields (write to `coefficient_vector_json`)

For every `rc/preprocess/*` spec, include a `preprocess` block in `coefficient_vector_json`:

```json
{
  "preprocess": {
    "spec_id": "rc/preprocess/outcome/winsor_1_99",
    "target": "outcome",
    "operation": "winsorize",
    "params": {"lower_q": 0.01, "upper_q": 0.99},
    "variables_affected": ["y"],
    "notes": "Winsorized within baseline sample before FE absorption."
  }
}
```

## Implementation notes (Python)

```python
import numpy as np
import pandas as pd

def winsorize(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lo, hi)

def standardize(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    return (s - mu) / sd if sd > 0 else s * np.nan

df["y_winsor"] = winsorize(df["y"], 0.01, 0.99)
df["t_z"] = standardize(df["treat"])
df["age_bin_q"] = pd.qcut(df["age"], 4, labels=False, duplicates="drop")
```
