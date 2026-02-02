# Heterogeneity Analysis

## Spec ID Format: `robust/het/{dimension}`

## Purpose

Systematically test whether treatment effects vary across subgroups. Heterogeneity analysis is essential for understanding external validity, targeting policy, and detecting effect modification.

---

## Demographic Subgroups

| spec_id | Description |
|---------|-------------|
| `robust/het/by_gender` | Split by gender |
| `robust/het/by_gender_male` | Male subsample only |
| `robust/het/by_gender_female` | Female subsample only |
| `robust/het/by_age_young` | Young age tercile |
| `robust/het/by_age_middle` | Middle age tercile |
| `robust/het/by_age_old` | Old age tercile |
| `robust/het/by_age_median_split` | Above/below median age |
| `robust/het/by_education_low` | Low education |
| `robust/het/by_education_high` | High education |
| `robust/het/by_race` | Split by race/ethnicity |
| `robust/het/by_marital_status` | Split by marital status |

---

## Socioeconomic Subgroups

| spec_id | Description |
|---------|-------------|
| `robust/het/by_income_low` | Low income tercile |
| `robust/het/by_income_middle` | Middle income tercile |
| `robust/het/by_income_high` | High income tercile |
| `robust/het/by_income_median_split` | Above/below median income |
| `robust/het/by_wealth_low` | Low wealth |
| `robust/het/by_wealth_high` | High wealth |
| `robust/het/by_employment` | By employment status |
| `robust/het/by_occupation` | By occupation type |

---

## Geographic Subgroups

| spec_id | Description |
|---------|-------------|
| `robust/het/by_region` | Split by geographic region |
| `robust/het/by_urban_rural` | Urban vs rural |
| `robust/het/by_urban` | Urban only |
| `robust/het/by_rural` | Rural only |
| `robust/het/by_state` | By state/province |
| `robust/het/by_country` | By country (if multi-country) |
| `robust/het/by_developed` | Developed vs developing |

---

## Baseline Characteristics

| spec_id | Description |
|---------|-------------|
| `robust/het/by_baseline_outcome_low` | Low baseline outcome |
| `robust/het/by_baseline_outcome_high` | High baseline outcome |
| `robust/het/by_baseline_outcome_terciles` | Baseline outcome terciles |
| `robust/het/by_prior_treatment` | Prior treatment exposure |
| `robust/het/by_initial_conditions` | Initial conditions |

---

## Treatment-Related Heterogeneity

| spec_id | Description |
|---------|-------------|
| `robust/het/by_treatment_intensity` | By treatment intensity |
| `robust/het/by_treatment_timing` | By timing of treatment |
| `robust/het/by_treatment_duration` | By treatment duration |
| `robust/het/by_compliance` | Compliers vs non-compliers |
| `robust/het/by_dosage` | By treatment dosage |

---

## Time-Based Heterogeneity

| spec_id | Description |
|---------|-------------|
| `robust/het/by_cohort` | By treatment cohort |
| `robust/het/by_period` | By time period |
| `robust/het/by_early_late` | Early vs late adopters |
| `robust/het/by_year` | By calendar year |

---

## Interaction Specifications

| spec_id | Description |
|---------|-------------|
| `robust/het/interaction_gender` | Treatment x gender interaction |
| `robust/het/interaction_age` | Treatment x age interaction |
| `robust/het/interaction_income` | Treatment x income interaction |
| `robust/het/interaction_education` | Treatment x education interaction |
| `robust/het/interaction_baseline_y` | Treatment x baseline outcome |
| `robust/het/interaction_region` | Treatment x region interaction |
| `robust/het/interaction_time` | Treatment x time trend |
| `robust/het/triple_diff` | Triple difference (treatment x group x time) |

---

## Machine Learning Heterogeneity

| spec_id | Description |
|---------|-------------|
| `robust/het/causal_forest` | Causal forest CATE estimation |
| `robust/het/grf` | Generalized random forest |
| `robust/het/sorted_effects` | Sorted group average treatment effects |
| `robust/het/best_linear` | Best linear predictor of CATE |

---

## Implementation Notes

```python
import pandas as pd
import numpy as np

# Subgroup analysis
def run_subgroup(df, subgroup_var, subgroup_value):
    df_sub = df[df[subgroup_var] == subgroup_value]
    return run_regression(df_sub)

# Interaction specification
def run_interaction(df, moderator):
    formula = f'y ~ treatment * {moderator} + controls | fe'
    return run_regression(df, formula=formula)

# Tercile splits
def create_terciles(df, var):
    df[f'{var}_tercile'] = pd.qcut(df[var], 3, labels=['low', 'mid', 'high'])
    return df

# Causal forest (using econml)
from econml.dml import CausalForestDML
cf = CausalForestDML(model_y='auto', model_t='auto')
cf.fit(Y, T, X=X, W=W)
cate = cf.effect(X)
```

---

## Output Format

```json
{
  "spec_id": "robust/het/by_gender",
  "spec_tree_path": "robustness/heterogeneity.md",
  "heterogeneity_dimension": "gender",
  "subgroups": {
    "male": {
      "coef": 0.062,
      "se": 0.028,
      "pval": 0.027,
      "n_obs": 4500
    },
    "female": {
      "coef": 0.038,
      "se": 0.031,
      "pval": 0.221,
      "n_obs": 5500
    }
  },
  "difference_test": {
    "coef_diff": 0.024,
    "se_diff": 0.042,
    "pval_diff": 0.568,
    "significant_difference": false
  },
  "pooled_coef": 0.052
}
```

---

## Interpretation Guidelines

| Pattern | Interpretation |
|---------|----------------|
| Similar effects across groups | Homogeneous effect, good external validity |
| Significant heterogeneity | Effect modification, consider targeting |
| One group significant, other not | May be power issue or true heterogeneity |
| Opposite signs across groups | Strong effect modification |

### Formal Tests

| Test | Purpose |
|------|---------|
| Interaction p-value | Test if difference is significant |
| Chow test | Test equality of coefficients across groups |
| Oaxaca decomposition | Decompose differences |

---

## Multiple Testing Considerations

When testing many subgroups:
- Apply FDR or Bonferroni correction
- Pre-specify primary heterogeneity dimensions
- Distinguish confirmatory vs exploratory analyses
- Report all tests conducted

---

## Checklist

- [ ] Ran at least 2 demographic subgroup analyses
- [ ] Ran at least 1 socioeconomic subgroup analysis
- [ ] Ran at least 1 baseline characteristic heterogeneity
- [ ] Ran interaction specifications for key moderators
- [ ] Tested significance of subgroup differences
- [ ] Applied multiple testing correction if many subgroups
- [ ] Reported sample sizes for each subgroup
- [ ] Flagged any significant heterogeneity
