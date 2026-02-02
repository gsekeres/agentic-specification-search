# Inference Alternatives Robustness Checks

## Spec ID Format: `robust/inference/{method}`

## Purpose

Test sensitivity of statistical inference to alternative methods beyond standard asymptotic approximations. Particularly important when sample sizes are small, clusters are few, or multiple hypotheses are tested.

---

## Randomization and Permutation Inference

| spec_id | Description |
|---------|-------------|
| `robust/inference/randomization` | Randomization inference (Fisher exact test) |
| `robust/inference/permutation` | Permutation test on treatment assignment |
| `robust/inference/ri_clustered` | Randomization inference with cluster-level permutation |

---

## Bayesian Inference

| spec_id | Description |
|---------|-------------|
| `robust/inference/bayesian` | Bayesian posterior credible intervals |
| `robust/inference/bayesian_flat` | Bayesian with flat/uninformative prior |
| `robust/inference/bayesian_informative` | Bayesian with informative prior from literature |

---

## Multiple Testing Corrections

| spec_id | Description |
|---------|-------------|
| `robust/inference/bonferroni` | Bonferroni correction for multiple testing |
| `robust/inference/holm` | Holm-Bonferroni stepdown correction |
| `robust/inference/fdr` | False discovery rate (Benjamini-Hochberg) |
| `robust/inference/fdr_by` | Benjamini-Yekutieli FDR (arbitrary dependence) |
| `robust/inference/romano_wolf` | Romano-Wolf stepdown procedure |
| `robust/inference/westfall_young` | Westfall-Young adjusted p-values |
| `robust/inference/sidak` | Sidak correction |

---

## Spatial and Time-Series Corrections

| spec_id | Description |
|---------|-------------|
| `robust/inference/conley_se` | Conley spatial HAC standard errors |
| `robust/inference/conley_50km` | Conley SE with 50km bandwidth |
| `robust/inference/conley_100km` | Conley SE with 100km bandwidth |
| `robust/inference/conley_200km` | Conley SE with 200km bandwidth |
| `robust/inference/hac_nw` | Newey-West HAC (time series) |
| `robust/inference/hac_nw_auto` | Newey-West with automatic lag selection |

---

## Small Sample Corrections

| spec_id | Description |
|---------|-------------|
| `robust/inference/small_sample_df` | Degrees of freedom correction |
| `robust/inference/t_distribution` | Use t-distribution instead of normal |
| `robust/inference/bell_mccaffrey` | Bell-McCaffrey bias-reduced linearization |

---

## Implementation Notes

```python
# Randomization inference
from scipy.stats import permutation_test
def stat_func(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
result = permutation_test((treated, control), stat_func, n_resamples=10000)

# Benjamini-Hochberg FDR
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

# Conley standard errors (using Python)
# Requires spatial weights matrix
from spreg import OLS
model = OLS(y, X, w=spatial_weights, spat_diag=True)

# Romano-Wolf (typically in Stata: rwolf)
# In Python, implement via bootstrap
```

---

## Output Format

```json
{
  "spec_id": "robust/inference/randomization",
  "spec_tree_path": "robustness/inference_alternatives.md",
  "inference_method": "randomization",
  "treatment": {
    "coef": 0.052,
    "se_asymptotic": 0.021,
    "pval_asymptotic": 0.013,
    "pval_randomization": 0.018,
    "ci_lower_ri": 0.008,
    "ci_upper_ri": 0.096,
    "n_permutations": 10000
  },
  "comparison": {
    "pval_ratio": 1.38,
    "inference_agrees": true
  }
}
```

---

## When to Use Each Method

| Method | Use When |
|--------|----------|
| Randomization inference | RCT with small sample, exact test needed |
| Permutation test | Treatment assignment is exchangeable |
| Bonferroni | Conservative, few hypotheses |
| FDR | Many hypotheses, some false positives acceptable |
| Romano-Wolf | Family of hypotheses, control FWER |
| Conley SE | Spatial correlation in errors |
| Small sample corrections | N < 50 or few clusters |

---

## Interpretation Guidelines

| Pattern | Interpretation |
|---------|----------------|
| RI p-value >> asymptotic | Asymptotic approx. overstates significance |
| RI p-value << asymptotic | Asymptotic approx. conservative |
| Loses significance after MHT correction | Fragile to multiple testing |
| Conley SE much larger | Strong spatial correlation |

---

## Checklist

- [ ] Ran at least one non-asymptotic inference method
- [ ] If multiple outcomes, applied multiple testing correction
- [ ] If spatial data, ran Conley standard errors
- [ ] If few clusters (<30), ran small sample correction
- [ ] Noted if inference conclusions change
- [ ] Reported both asymptotic and alternative p-values
