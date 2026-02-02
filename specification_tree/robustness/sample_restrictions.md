# Sample Restrictions Robustness Checks

## Spec ID Format: `robust/sample/{restriction_type}`

## Purpose

Test sensitivity of results to sample composition by running the baseline specification on different subsamples.

---

## Standard Sample Restrictions

### Time-Based Restrictions

| spec_id | Description |
|---------|-------------|
| `robust/sample/early_period` | First half of time period |
| `robust/sample/late_period` | Second half of time period |
| `robust/sample/pre_crisis` | Before major structural break |
| `robust/sample/post_crisis` | After major structural break |
| `robust/sample/exclude_first_year` | Drop first year of data |
| `robust/sample/exclude_last_year` | Drop last year of data |

### Outlier Handling

| spec_id | Description |
|---------|-------------|
| `robust/sample/trim_1pct` | Drop top and bottom 1% |
| `robust/sample/trim_5pct` | Drop top and bottom 5% |
| `robust/sample/winsor_1pct` | Winsorize at 1%/99% |
| `robust/sample/winsor_5pct` | Winsorize at 5%/95% |
| `robust/sample/drop_extreme_y` | Drop extreme outcome values |
| `robust/sample/drop_extreme_x` | Drop extreme treatment values |

### Panel-Specific

| spec_id | Description |
|---------|-------------|
| `robust/sample/balanced` | Balanced panel only |
| `robust/sample/continuous` | Continuously observed units |
| `robust/sample/min_obs_2` | Units with ≥2 observations |
| `robust/sample/min_obs_5` | Units with ≥5 observations |
| `robust/sample/drop_singletons` | Drop singleton observations |

### Geographic/Unit Restrictions

| spec_id | Description |
|---------|-------------|
| `robust/sample/drop_largest` | Exclude largest unit |
| `robust/sample/drop_smallest` | Exclude smallest unit |
| `robust/sample/urban_only` | Urban areas only |
| `robust/sample/rural_only` | Rural areas only |
| `robust/sample/exclude_{region}` | Exclude specific region |

### Demographic Subgroups

| spec_id | Description |
|---------|-------------|
| `robust/sample/male_only` | Male subsample |
| `robust/sample/female_only` | Female subsample |
| `robust/sample/young` | Young age group |
| `robust/sample/old` | Old age group |
| `robust/sample/high_income` | High income subsample |
| `robust/sample/low_income` | Low income subsample |

### Data Quality

| spec_id | Description |
|---------|-------------|
| `robust/sample/complete_cases` | No missing values |
| `robust/sample/imputed_excluded` | Exclude imputed values |
| `robust/sample/high_quality` | High data quality flag |

### Influential Observations

| spec_id | Description |
|---------|-------------|
| `robust/sample/drop_influential` | Drop high-leverage observations (Cook's D > 4/n) |
| `robust/sample/drop_leverage` | Drop high leverage points |
| `robust/sample/drop_residual_outliers` | Drop studentized residual outliers |
| `robust/sample/jackknife_units` | Leave-one-unit-out sensitivity |

### Survey and Measurement

| spec_id | Description |
|---------|-------------|
| `robust/sample/survey_wave_1` | First survey wave only |
| `robust/sample/survey_wave_last` | Last survey wave only |
| `robust/sample/cross_section_t0` | Cross-section at baseline only |
| `robust/sample/drop_missing_key_var` | Drop if missing on key variable |

### Treatment-Based Restrictions

| spec_id | Description |
|---------|-------------|
| `robust/sample/treated_only` | Treated units only |
| `robust/sample/control_only` | Control units only |
| `robust/sample/on_support` | Common support (propensity score) |
| `robust/sample/trimmed_propensity` | Trim extreme propensity scores |

---

## Implementation Notes

```python
# Pseudocode for sample restrictions
def run_sample_restriction(df, restriction_type):
    if restriction_type == 'early_period':
        df_sub = df[df['year'] <= df['year'].median()]
    elif restriction_type == 'trim_1pct':
        df_sub = df[(df['y'] > df['y'].quantile(0.01)) &
                    (df['y'] < df['y'].quantile(0.99))]
    elif restriction_type == 'balanced':
        # Keep only units observed in all periods
        full_obs = df.groupby('unit').size()
        balanced_units = full_obs[full_obs == full_obs.max()].index
        df_sub = df[df['unit'].isin(balanced_units)]
    # ... etc

    return run_regression(df_sub)
```

---

## Output Format

```json
{
  "spec_id": "robust/sample/trim_1pct",
  "spec_tree_path": "robustness/sample_restrictions.md",
  "restriction_type": "trim_1pct",
  "restriction_description": "Dropped top and bottom 1% of outcome",
  "sample_comparison": {
    "n_baseline": 10000,
    "n_restricted": 9800,
    "pct_retained": 98.0,
    "dropped_reason": "outlier trimming"
  },
  "treatment": {
    "coef": 0.047,
    "se": 0.020,
    "pval": 0.019
  },
  "baseline_comparison": {
    "baseline_coef": 0.052,
    "coef_change_pct": -9.6,
    "significance_changed": false
  }
}
```

---

## Key Diagnostics

For each sample restriction, report:

1. **Sample size change**: How many observations dropped
2. **Composition change**: Any systematic differences in who is dropped
3. **Coefficient change**: Magnitude and direction
4. **Significance change**: Whether inference conclusions change

---

## Red Flags

| Pattern | Concern |
|---------|---------|
| Large N drop, no coef change | Dropped obs uninformative |
| Small N drop, large coef change | Results driven by outliers |
| Subgroup effects flip signs | Heterogeneous effects |
| Loses significance with trimming | Results driven by extremes |

---

## Checklist

- [ ] Ran at least 2 time-based restrictions
- [ ] Ran at least 1 outlier handling specification
- [ ] Ran balanced panel (if panel data)
- [ ] Ran at least 2 subgroup analyses
- [ ] Documented sample sizes for all restrictions
- [ ] Flagged any large coefficient changes
- [ ] Noted if significance changes with any restriction
