# Control Variable Progression

## Spec ID Format: `robust/build/{control_set}`

## Purpose

Systematically build up control sets from bivariate to fully saturated specifications. This reveals how the treatment coefficient evolves as controls are added and helps identify potential confounders.

---

## Standard Build-Up Sequence

| spec_id | Description |
|---------|-------------|
| `robust/build/bivariate` | Treatment only (no controls) |
| `robust/build/demographics` | + demographic controls |
| `robust/build/socioeconomic` | + socioeconomic controls |
| `robust/build/geographic` | + geographic controls |
| `robust/build/temporal` | + temporal controls |
| `robust/build/baseline` | Paper's baseline control set |
| `robust/build/extended` | Baseline + additional controls |
| `robust/build/full` | All reasonable controls |
| `robust/build/kitchen_sink` | All available controls |

---

## Category-Specific Additions

### Demographics

| spec_id | Description |
|---------|-------------|
| `robust/build/add_age` | Add age |
| `robust/build/add_gender` | Add gender |
| `robust/build/add_education` | Add education |
| `robust/build/add_race` | Add race/ethnicity |
| `robust/build/add_marital` | Add marital status |
| `robust/build/add_household` | Add household composition |

### Socioeconomic

| spec_id | Description |
|---------|-------------|
| `robust/build/add_income` | Add income |
| `robust/build/add_wealth` | Add wealth |
| `robust/build/add_employment` | Add employment status |
| `robust/build/add_occupation` | Add occupation |
| `robust/build/add_industry` | Add industry |

### Geographic

| spec_id | Description |
|---------|-------------|
| `robust/build/add_region` | Add region indicators |
| `robust/build/add_urban` | Add urban/rural |
| `robust/build/add_state` | Add state fixed effects |
| `robust/build/add_local` | Add local area controls |

### Temporal

| spec_id | Description |
|---------|-------------|
| `robust/build/add_year` | Add year fixed effects |
| `robust/build/add_quarter` | Add quarter fixed effects |
| `robust/build/add_month` | Add month fixed effects |
| `robust/build/add_trend` | Add time trend |
| `robust/build/add_trend_sq` | Add quadratic time trend |

---

## Fixed Effects Progression

| spec_id | Description |
|---------|-------------|
| `robust/build/fe_none` | No fixed effects |
| `robust/build/fe_unit` | Add unit FE |
| `robust/build/fe_time` | Add time FE |
| `robust/build/fe_twoway` | Add unit + time FE |
| `robust/build/fe_region_time` | Add region x time FE |
| `robust/build/fe_unit_trend` | Add unit-specific trends |

---

## Saturated Specifications

| spec_id | Description |
|---------|-------------|
| `robust/build/saturated_controls` | All controls + interactions |
| `robust/build/saturated_fe` | Highest-dimensional FE |
| `robust/build/saturated_full` | Kitchen sink + saturated FE |

---

## Oster Bounds and Selection

| spec_id | Description |
|---------|-------------|
| `robust/build/oster_delta` | Oster delta (selection on unobservables) |
| `robust/build/oster_bounds` | Oster bounds on treatment effect |
| `robust/build/r2_max` | R-squared at max controls |

---

## Implementation Notes

```python
import pandas as pd
import numpy as np

# Define control sets
control_sets = {
    'bivariate': [],
    'demographics': ['age', 'female', 'education'],
    'socioeconomic': ['age', 'female', 'education', 'income', 'employed'],
    'geographic': ['age', 'female', 'education', 'income', 'employed', 'urban', 'region'],
    'full': ['age', 'female', 'education', 'income', 'employed', 'urban', 'region',
             'married', 'children', 'homeowner']
}

# Run progression
results = []
for name, controls in control_sets.items():
    spec_id = f'robust/build/{name}'
    result = run_regression(df, controls=controls)
    result['spec_id'] = spec_id
    result['n_controls'] = len(controls)
    results.append(result)

# Oster bounds (using Python implementation)
def oster_delta(r2_tilde, r2_full, beta_tilde, beta_full, r2_max=1.0):
    """Calculate Oster's delta for proportional selection assumption."""
    numerator = (beta_full - 0) * (r2_max - r2_full)
    denominator = (beta_tilde - beta_full) * (r2_full - r2_tilde)
    if denominator == 0:
        return np.inf
    return numerator / denominator
```

---

## Output Format

```json
{
  "spec_id": "robust/build/demographics",
  "spec_tree_path": "robustness/control_progression.md",
  "control_set": "demographics",
  "controls_included": ["age", "female", "education"],
  "n_controls": 3,
  "treatment": {
    "coef": 0.058,
    "se": 0.023,
    "pval": 0.012
  },
  "progression_comparison": {
    "bivariate_coef": 0.082,
    "change_from_bivariate": -0.024,
    "change_pct": -29.3,
    "r_squared": 0.15,
    "r_squared_bivariate": 0.05
  }
}
```

---

## Progression Analysis Summary

After running all build-up specifications, compute:

```json
{
  "spec_id": "robust/build/summary",
  "progression_summary": {
    "bivariate_coef": 0.082,
    "baseline_coef": 0.052,
    "full_coef": 0.048,
    "kitchen_sink_coef": 0.045,
    "total_change_pct": -45.1,
    "largest_single_change": {
      "control": "income",
      "change_pct": -18.2
    },
    "stable_after": "socioeconomic",
    "oster_delta": 2.3,
    "oster_interpretation": "Would need unobservables 2.3x as important as observables to explain away effect"
  }
}
```

---

## Interpretation Guidelines

| Pattern | Interpretation |
|---------|----------------|
| Coefficient stable across sets | Robust to control selection |
| Large drop with certain control | That control is confounder |
| Coefficient increases with controls | Negative confounding (suppressor) |
| Oster delta > 1 | Robust to selection on unobservables |
| Oster delta < 1 | Vulnerable to unobserved confounding |

### Red Flags

| Pattern | Concern |
|---------|---------|
| Sign flip during progression | Serious confounding |
| Coefficient halves with controls | Strong selection bias |
| Oster delta < 0.5 | Very sensitive to unobservables |

---

## Checklist

- [ ] Ran bivariate (no controls) specification
- [ ] Ran paper's baseline control set
- [ ] Ran full control set
- [ ] Ran kitchen sink specification
- [ ] Tracked coefficient evolution across sets
- [ ] Identified controls with largest impact
- [ ] Computed Oster bounds (if applicable)
- [ ] Documented R-squared progression
- [ ] Flagged any large coefficient changes
