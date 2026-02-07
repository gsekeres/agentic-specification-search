# Verification Report: 125821-V1

## Paper Summary

**Title**: Wisconsin School Referendum RDD Study
**Hypothesis**: Passing an operating referendum (vote share >= 50%) leads to increased school district expenditures per member
**Method**: Regression Discontinuity Design (RDD)
**Data**: Wisconsin school districts with operating referendum elections (1996-2014), N=7,340 elections

---

## Baseline Groups

### G1: Effect of referendum passage on total expenditures per member

- **Baseline spec_id**: `baseline`
- **Outcome**: `tot_exp_mem` (total expenditures per member)
- **Treatment**: `treatment` (referendum passage indicator, vote share >= 50%)
- **Expected sign**: Positive (+)
- **Baseline result**: Coefficient = 641.24, SE = 344.34, p = 0.063
- **Method**: Local linear RDD, bandwidth = 10 percentage points, clustered by district_code

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | 62 |
| **Baseline** | 1 |
| **Core tests (is_core_test=1)** | 43 |
| **Non-core (is_core_test=0)** | 19 |
| **Invalid** | 4 |
| **Unclear** | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| `core_method` | 16 | Bandwidth variations, polynomial order, sensitivity checks |
| `core_controls` | 9 | Control set variations (none, minimal, full, leave-one-out) |
| `core_sample` | 14 | Donut holes, time splits, trimming, subsamples |
| `core_inference` | 2 | Clustering variations (none, LEA ID) |
| `core_funcform` | 2 | Log and IHS transformations of outcome |
| `noncore_alt_outcome` | 8 | Different outcome variables (instructional exp, compensation, dropout, salary, etc.) |
| `noncore_placebo` | 4 | Placebo cutoffs at 40%, 45%, 55%, 60% |
| `noncore_heterogeneity` | 3 | Interaction terms (urban, large district, high poverty) |
| `invalid` | 4 | Exact duplicates of baseline with different labels (3 kernel variants + rd/controls/baseline) |

---

## Top 5 Most Suspicious Rows

### 1. rd/kernel/triangular, rd/kernel/uniform, rd/kernel/epanechnikov (3 specs)
**Issue**: All three kernel specifications produce coefficients, SEs, and p-values exactly identical to the baseline (coefficient = 641.238, SE = 344.341, p = 0.063). This means the kernel weighting was not actually implemented -- the analysis used the same rectangular/uniform kernel (or no kernel weighting) regardless of the label. These are marked as `invalid` because they provide no additional robustness information.

### 2. rd/controls/baseline
**Issue**: This spec has the label "baseline controls" under the control-set variations path, but produces results exactly identical to the `baseline` spec_id. It is a pure duplicate. Marked as `invalid`.

### 3. rd/bandwidth/bw10
**Issue**: This bandwidth=10 specification is an exact duplicate of the baseline (which also uses bandwidth=10). It is still marked as `core_method` since it at least correctly labels what it is, but it adds no new information. Less suspicious than the kernel duplicates since the duplication is self-consistent.

### 4. rd/poly/global_linear vs rd/bandwidth/full
**Issue**: These two specifications produce identical results (coefficient = 450.49, SE = 261.28, p = 0.085, N = 7340). A "global linear" on the full sample is equivalent to "full bandwidth local linear." The duplication is conceptually justified (they are the same thing from different perspectives), but inflates the apparent spec count.

### 5. rd/placebo/cutoff_45
**Issue**: This placebo test at 45% shows a statistically significant negative effect (coefficient = -514.35, p = 0.037). While this is a diagnostic/falsification test and would not normally be alarming, a significant placebo at a nearby cutoff could raise concerns about the validity of the RDD design. The SPECIFICATION_SEARCH.md reports this as not significant, which is incorrect -- the p-value is 0.037.

---

## Recommendations for Spec-Search Script

1. **Fix kernel implementations**: The kernel variation specs should actually implement different kernel weighting functions (triangular, Epanechnikov) using weighted least squares. Currently they appear to use OLS without weights, producing identical results regardless of kernel label. This is a bug.

2. **Remove or flag pure duplicates**: rd/bandwidth/bw10 and rd/controls/baseline are trivial duplicates of the baseline. The script should either skip these or flag them as confirmation checks rather than robustness variations.

3. **Clarify global_linear vs full**: These are the same specification. The script should either report one or explicitly note the equivalence.

4. **Reconsider alternative outcome classification**: The spec search labeled all alternative outcomes under robustness/measurement.md, but outcomes like dropout_rate, compensation, and log_el_avgsalary are fundamentally different claims than "total expenditures per member." Only tot_exp_inst_mem (instructional expenditures) and possibly tot_exp_ss_mem (support services) could be argued as sub-components of the same claim. The others should be separated.

5. **Note placebo test anomaly**: The placebo at 45% is significant (p=0.037), which warrants discussion. The SPECIFICATION_SEARCH.md incorrectly claims all placebo tests are insignificant.

---

## Verification Notes

- The baseline claim is well-defined: referendum passage -> increased total expenditures per member.
- The baseline result is only marginally significant (p=0.063), not significant at the 5% level.
- 43 of 62 specifications are marked as core tests (is_core_test=1). The 4 invalid specs are marked is_core_test=0.
- The 19 non-core specs include: 4 invalid duplicates, 8 alternative outcomes, 4 placebos, and 3 heterogeneity interactions.
- rd/bandwidth/bw10 is kept as core since the label is self-consistent, but adds no new info.
- The sign is consistently positive across core tests (with rare exceptions in extreme subsamples like small_districts and donut_5pct).
