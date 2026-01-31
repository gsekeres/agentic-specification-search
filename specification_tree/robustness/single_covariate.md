# Single Covariate Robustness Checks

## Spec ID Format: `robust/single/{control_variable}`

## Purpose

Test the bivariate relationship between treatment and outcome, and how it changes when adding each control one at a time. This helps identify which controls are driving changes in the treatment effect.

---

## Methodology

1. Run bivariate regression: treatment only (no controls)
2. For each available control, run treatment + that single control
3. Compare treatment coefficient evolution

---

## Required Specifications

| spec_id | Description |
|---------|-------------|
| `robust/single/none` | Treatment only (bivariate) |
| `robust/single/{var1}` | Treatment + var1 only |
| `robust/single/{var2}` | Treatment + var2 only |
| `robust/single/{var3}` | Treatment + var3 only |
| ... | Continue for key controls |

### Example

If available controls are `[age, income, education, region]`:

| spec_id | Specification |
|---------|---------------|
| `robust/single/none` | y ~ treatment |
| `robust/single/age` | y ~ treatment + age |
| `robust/single/income` | y ~ treatment + income |
| `robust/single/education` | y ~ treatment + education |
| `robust/single/region` | y ~ treatment + region |

---

## Implementation Notes

```python
# Pseudocode for single covariate analysis
all_controls = ['age', 'income', 'education', 'region']
results = []

# Bivariate (no controls)
results.append(run_regression(controls=[], spec_id='robust/single/none'))

# Single covariate
for var in all_controls:
    spec_id = f'robust/single/{var}'
    results.append(run_regression(controls=[var], spec_id=spec_id))
```

---

## Output Format

```json
{
  "spec_id": "robust/single/income",
  "spec_tree_path": "robustness/single_covariate.md",
  "single_control": "income",
  "treatment": {
    "coef": 0.055,
    "se": 0.022,
    "pval": 0.012
  },
  "comparison": {
    "bivariate_coef": 0.082,
    "change_from_bivariate": -0.027,
    "change_pct": -32.9,
    "r_squared_bivariate": 0.05,
    "r_squared_with_control": 0.18,
    "r_squared_increase": 0.13
  }
}
```

---

## Interpretation

### Coefficient Changes from Bivariate

| Pattern | Interpretation |
|---------|----------------|
| Large decrease | Control is positive confounder (upward bias) |
| Large increase | Control is negative confounder (downward bias) |
| Little change | Control unrelated to treatment-outcome relationship |

### R-squared Changes

| R² Increase | Interpretation |
|-------------|----------------|
| Large increase | Control explains substantial outcome variation |
| Small increase | Control has limited explanatory power |

---

## Building Control Sequences

After single-covariate analysis, consider building up specifications:

```
robust/single/none         → bivariate
robust/single/age          → + age
robust/build/age_income    → + age + income
robust/build/age_inc_edu   → + age + income + education
baseline                   → full model
```

This shows how the treatment effect evolves as controls are added.

---

## Checklist

- [ ] Ran bivariate (treatment only) regression
- [ ] Ran single covariate for each available control
- [ ] Computed change from bivariate for each
- [ ] Identified controls with largest coefficient impact
- [ ] Noted controls that change significance
- [ ] Considered building control sequence if informative
