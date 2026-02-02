# Placebo Tests

## Spec ID Format: `robust/placebo/{test_type}`

## Purpose

Validate the causal interpretation of results by testing for effects where none should exist. A significant placebo effect suggests confounding, specification error, or spurious correlation.

---

## Temporal Placebos

| spec_id | Description |
|---------|-------------|
| `robust/placebo/fake_treatment_lead1` | Shift treatment 1 period earlier |
| `robust/placebo/fake_treatment_lead2` | Shift treatment 2 periods earlier |
| `robust/placebo/fake_treatment_lag1` | Shift treatment 1 period later |
| `robust/placebo/fake_treatment_lag2` | Shift treatment 2 periods later |
| `robust/placebo/pre_trend_test` | Test for pre-existing trends |
| `robust/placebo/pre_period_effect` | Effect in pre-treatment period only |

---

## Treatment Assignment Placebos

| spec_id | Description |
|---------|-------------|
| `robust/placebo/random_treatment` | Randomly reassign treatment status |
| `robust/placebo/permuted_treatment` | Permute treatment across units |
| `robust/placebo/reversed_treatment` | Flip treatment indicator |
| `robust/placebo/neighbor_treatment` | Assign treatment based on neighbor's status |

---

## Outcome Placebos

| spec_id | Description |
|---------|-------------|
| `robust/placebo/lagged_outcome` | Use lagged outcome as dependent variable |
| `robust/placebo/outcome_unaffected` | Run on outcome that shouldn't be affected |
| `robust/placebo/outcome_predetermined` | Use pre-determined outcome |
| `robust/placebo/outcome_components` | Test on unrelated outcome components |

---

## Unit Placebos

| spec_id | Description |
|---------|-------------|
| `robust/placebo/untreated_only` | Run on never-treated units only |
| `robust/placebo/always_treated_only` | Run on always-treated units only |
| `robust/placebo/control_group_fake` | Fake treatment within control group |
| `robust/placebo/wrong_geography` | Apply treatment to wrong geographic units |

---

## Threshold/Cutoff Placebos (for RD/Bunching)

| spec_id | Description |
|---------|-------------|
| `robust/placebo/fake_cutoff_above` | Test at cutoff above true cutoff |
| `robust/placebo/fake_cutoff_below` | Test at cutoff below true cutoff |
| `robust/placebo/median_cutoff` | Test at median of running variable |
| `robust/placebo/quartile_cutoffs` | Test at quartile cutoffs |

---

## Reverse Causation Tests

| spec_id | Description |
|---------|-------------|
| `robust/placebo/reverse_causation` | Swap treatment and outcome |
| `robust/placebo/future_on_past` | Regress future treatment on past outcome |
| `robust/placebo/granger_test` | Granger causality test |

---

## Implementation Notes

```python
import numpy as np
import pandas as pd

# Fake treatment timing
def shift_treatment(df, periods):
    df_shifted = df.copy()
    df_shifted['treatment'] = df.groupby('unit')['treatment'].shift(periods)
    return df_shifted

# Random treatment permutation
def permute_treatment(df, seed=42):
    np.random.seed(seed)
    df_perm = df.copy()
    # Permute at unit level
    units = df['unit'].unique()
    treatment_status = df.groupby('unit')['ever_treated'].first().values
    np.random.shuffle(treatment_status)
    treatment_map = dict(zip(units, treatment_status))
    df_perm['treatment_permuted'] = df_perm['unit'].map(treatment_map)
    return df_perm

# Pre-trend test
def pre_trend_regression(df):
    pre_df = df[df['post'] == 0]
    # Regress outcome on time trend interacted with treatment
    return run_regression(pre_df, formula='y ~ time * treatment_group')
```

---

## Output Format

```json
{
  "spec_id": "robust/placebo/fake_treatment_lead2",
  "spec_tree_path": "robustness/placebo_tests.md",
  "placebo_type": "temporal",
  "placebo_description": "Treatment shifted 2 periods earlier",
  "treatment": {
    "coef": 0.008,
    "se": 0.019,
    "pval": 0.67
  },
  "baseline_comparison": {
    "baseline_coef": 0.052,
    "placebo_as_pct_of_baseline": 15.4,
    "placebo_significant": false
  },
  "interpretation": "No pre-trend detected"
}
```

---

## Interpretation Guidelines

| Placebo Result | Interpretation |
|----------------|----------------|
| Insignificant, near zero | Passes placebo test, supports causality |
| Significant, same sign | Potential confounding or pre-trend |
| Significant, opposite sign | Unusual, investigate further |
| Significant at fake cutoff | RD design may be invalid |

### Red Flags

| Pattern | Concern |
|---------|---------|
| Significant pre-trend | Parallel trends violated |
| Random treatment significant | Spurious correlation |
| Effect on unaffected outcome | Omitted variable bias |
| Fake cutoff shows effect | Discontinuity not causal |

---

## Best Practices

1. **Run multiple placebo tests** - Single placebo insufficient
2. **Report all placebo results** - Even if they "fail"
3. **Use same specification** - Match baseline exactly except for placebo element
4. **Power considerations** - Placebo should have similar power to detect effects
5. **Multiple fake cutoffs for RD** - Test several alternative cutoffs

---

## Checklist

- [ ] Ran at least 2 temporal placebos (lead/lag)
- [ ] Ran random/permuted treatment placebo
- [ ] Ran placebo on unaffected outcome (if available)
- [ ] Ran pre-trend test (if panel data)
- [ ] For RD: ran fake cutoff tests
- [ ] Documented placebo effect sizes relative to baseline
- [ ] Flagged any significant placebo effects
