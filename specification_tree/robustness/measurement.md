# Measurement Variations Robustness Checks

## Spec ID Format: `robust/measure/{variation}`

## Purpose

Test sensitivity of results to how key variables are measured, coded, and constructed. Measurement choices can substantially affect estimates, as highlighted by many I4R replications finding coding errors or alternative reasonable measurements.

---

## Outcome Variable Alternatives

| spec_id | Description |
|---------|-------------|
| `robust/measure/outcome_alternate` | Alternative outcome definition |
| `robust/measure/outcome_broader` | Broader/more inclusive outcome |
| `robust/measure/outcome_narrower` | Narrower/stricter outcome |
| `robust/measure/outcome_components` | Disaggregated outcome components |
| `robust/measure/outcome_index` | Index/composite outcome |
| `robust/measure/outcome_pca` | PCA-constructed outcome |
| `robust/measure/outcome_standardized` | Standardized (within-sample z-score) |
| `robust/measure/outcome_raw` | Raw/unstandardized outcome |

---

## Treatment Variable Alternatives

| spec_id | Description |
|---------|-------------|
| `robust/measure/treatment_alternate` | Alternative treatment coding |
| `robust/measure/treatment_binary` | Binary treatment (if originally continuous) |
| `robust/measure/treatment_continuous` | Continuous treatment (if originally binary) |
| `robust/measure/treatment_intensity` | Treatment intensity measure |
| `robust/measure/treatment_duration` | Treatment duration/exposure |
| `robust/measure/treatment_dosage` | Treatment dosage/amount |
| `robust/measure/treatment_timing` | Alternative treatment timing definition |

---

## Missing Data Handling

| spec_id | Description |
|---------|-------------|
| `robust/measure/listwise_deletion` | Complete cases only (listwise deletion) |
| `robust/measure/imputation_mean` | Mean imputation |
| `robust/measure/imputation_median` | Median imputation |
| `robust/measure/imputation_multiple` | Multiple imputation (MICE) |
| `robust/measure/imputation_regression` | Regression imputation |
| `robust/measure/imputation_knn` | K-nearest neighbors imputation |
| `robust/measure/missing_indicator` | Include missing indicators as controls |
| `robust/measure/bounds_manski` | Manski bounds for missing data |

---

## Weighting Variations

| spec_id | Description |
|---------|-------------|
| `robust/measure/unweighted` | Unweighted regression |
| `robust/measure/survey_weights` | With survey/sampling weights |
| `robust/measure/population_weights` | Population weights |
| `robust/measure/inverse_prob_weights` | Inverse probability weights |
| `robust/measure/attrition_weights` | Attrition-adjusted weights |
| `robust/measure/entropy_weights` | Entropy balancing weights |
| `robust/measure/trimmed_weights` | Trimmed extreme weights |

---

## Measurement Error Corrections

| spec_id | Description |
|---------|-------------|
| `robust/measure/simex` | SIMEX correction for measurement error |
| `robust/measure/error_in_variables` | Errors-in-variables model |
| `robust/measure/reliability_adjusted` | Reliability-adjusted estimates |
| `robust/measure/proxy_variable` | Use proxy variable |

---

## Scale and Coding Variations

| spec_id | Description |
|---------|-------------|
| `robust/measure/scale_original` | Original scale |
| `robust/measure/scale_rescaled` | Rescaled (0-1 or 0-100) |
| `robust/measure/coding_reverse` | Reverse coding check |
| `robust/measure/coding_categorical` | Treat as categorical |
| `robust/measure/coding_ordinal` | Treat as ordinal |
| `robust/measure/coding_continuous` | Treat as continuous |

---

## Data Source Variations

| spec_id | Description |
|---------|-------------|
| `robust/measure/source_alternate` | Alternative data source |
| `robust/measure/source_harmonized` | Harmonized variable (cross-source) |
| `robust/measure/vintage_original` | Original data vintage |
| `robust/measure/vintage_revised` | Revised/updated data |

---

## Implementation Notes

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Multiple imputation
def multiple_imputation(df, n_imputations=5):
    imputer = IterativeImputer(max_iter=10, random_state=0)
    results = []
    for i in range(n_imputations):
        imputer.random_state = i
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns
        )
        results.append(run_regression(df_imputed))
    return combine_mi_results(results)  # Rubin's rules

# Inverse probability weighting for attrition
def ipw_attrition(df):
    # Model probability of remaining in sample
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(df[covariates], df['not_attrited'])
    df['ipw'] = 1 / model.predict_proba(df[covariates])[:, 1]
    df['ipw'] = df['ipw'].clip(upper=df['ipw'].quantile(0.99))
    return df

# SIMEX for measurement error
# Typically requires specialized package (simex in R)
```

---

## Output Format

```json
{
  "spec_id": "robust/measure/imputation_multiple",
  "spec_tree_path": "robustness/measurement.md",
  "measurement_variation": "multiple_imputation",
  "description": "MICE with 20 imputations",
  "sample_comparison": {
    "n_baseline_complete": 8500,
    "n_with_imputation": 10000,
    "pct_imputed": 15.0
  },
  "treatment": {
    "coef": 0.049,
    "se": 0.022,
    "pval": 0.026
  },
  "baseline_comparison": {
    "baseline_coef": 0.052,
    "coef_change_pct": -5.8,
    "inference_agrees": true
  }
}
```

---

## Interpretation Guidelines

| Pattern | Interpretation |
|---------|----------------|
| Robust to outcome definition | Core finding, not measurement artifact |
| Sensitive to treatment coding | Treatment definition matters |
| MI changes results substantially | Missing data non-random |
| Weights change results | Sample not representative |

### Red Flags from I4R Replications

| Issue | Example from I4R |
|-------|------------------|
| Incorrect variable coding | Treatment indicator miscoded |
| Missing data not handled | Observations silently dropped |
| Wrong scale used | Ordinal treated as continuous |
| Data entry errors | Manual transcription mistakes |

---

## Checklist

- [ ] Ran at least one alternative outcome measure
- [ ] Ran at least one alternative treatment measure
- [ ] Tested with and without survey weights
- [ ] If missing data: ran multiple imputation
- [ ] Checked variable coding matches description
- [ ] Documented measurement choices
- [ ] Flagged any large changes from measurement variation
