# Leave-One-Out Robustness Checks

## Spec ID Format: `robust/loo/{dropped_variable}`

## Purpose

Test sensitivity of results to individual control variables by systematically dropping each control one at a time from the baseline specification.

---

## Methodology

Starting from the paper's baseline specification with controls $X_1, X_2, ..., X_k$:

1. Run baseline with all controls
2. For each $X_j$, run specification dropping $X_j$ while keeping all other controls
3. Compare treatment coefficient across specifications

---

## Required Specifications

For each control variable in the baseline model:

| spec_id | Description |
|---------|-------------|
| `robust/loo/drop_{var1}` | Drop first control |
| `robust/loo/drop_{var2}` | Drop second control |
| `robust/loo/drop_{var3}` | Drop third control |
| ... | Continue for all controls |

### Example

If baseline controls are `[age, income, education, region]`:

| spec_id | Controls Included |
|---------|-------------------|
| `baseline` | age, income, education, region |
| `robust/loo/drop_age` | income, education, region |
| `robust/loo/drop_income` | age, education, region |
| `robust/loo/drop_education` | age, income, region |
| `robust/loo/drop_region` | age, income, education |

---

## Implementation Notes

```python
# Pseudocode for leave-one-out
baseline_controls = ['age', 'income', 'education', 'region']
results = []

# Baseline
results.append(run_regression(controls=baseline_controls, spec_id='baseline'))

# Leave-one-out
for var in baseline_controls:
    loo_controls = [c for c in baseline_controls if c != var]
    spec_id = f'robust/loo/drop_{var}'
    results.append(run_regression(controls=loo_controls, spec_id=spec_id))
```

---

## Output Format

Each leave-one-out specification should record:

```json
{
  "spec_id": "robust/loo/drop_age",
  "spec_tree_path": "robustness/leave_one_out.md",
  "dropped_variable": "age",
  "controls_included": ["income", "education", "region"],
  "treatment": {
    "coef": 0.052,
    "se": 0.021,
    "pval": 0.013
  },
  "baseline_comparison": {
    "baseline_coef": 0.048,
    "coef_change_pct": 8.3,
    "significance_changed": false
  }
}
```

---

## Interpretation Guidelines

| Change in Coefficient | Interpretation |
|----------------------|----------------|
| < 5% | Very robust to this control |
| 5-15% | Moderately sensitive |
| 15-30% | Sensitive to this control |
| > 30% | Highly sensitive - potential confounding |

| Significance Change | Flag |
|--------------------|------|
| Remains significant | Robust |
| Loses significance | Fragile |
| Gains significance | Suppressor variable |

---

## Checklist

- [ ] Identified all controls in baseline specification
- [ ] Ran leave-one-out for EACH control variable
- [ ] Computed coefficient change percentages
- [ ] Flagged any specifications where significance changes
- [ ] Summarized which controls matter most for results
