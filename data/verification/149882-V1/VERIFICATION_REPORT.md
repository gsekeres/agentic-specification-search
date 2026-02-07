# Verification Report: 149882-V1

## Paper
**Title**: Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India  
**Journal**: AER  
**Paper ID**: 149882-V1

## Baseline Groups

### G1: Treatment effect on gender attitudes index
- **Claim**: The Breakthrough school-based gender-sensitization program improves adolescents' gender-egalitarian attitudes (standardized gender index) at Endline 1.
- **Expected sign**: Positive (+)
- **Baseline spec_ids**: baseline
- **Baseline outcome**: E_Sgender_index2_std (standardized gender attitudes index at endline)
- **Baseline treatment**: B_treat (randomized treatment indicator)
- **Baseline coefficient**: 0.014 SD (p = 0.42, not significant)
- **Method**: Cross-sectional OLS with baseline gender index, grade FE, district FE, clustered at school level

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 53 |
| Baseline | 1 |
| Core tests (including baseline) | 36 |
| Non-core tests | 13 |
| Invalid | 4 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 9 | Control set variations (add/drop controls, bivariate, control progression) |
| core_sample | 18 | Sample restrictions (gender, district, school leave-one-out, baseline splits) |
| core_fe | 3 | Fixed effects variations (only grade FE, only district FE, no FE) |
| core_inference | 3 | Clustering/SE variations (HC1 robust, district cluster, school cluster) |
| core_funcform | 3 | Functional form (unstandardized outcome, levels, quadratic baseline control) |
| noncore_heterogeneity | 8 | Heterogeneity interaction models (gender, grade, districts, baseline attitudes) |
| noncore_alt_outcome | 2 | Alternative outcomes (behavior index, aspirations index for girls) |
| noncore_placebo | 3 | Placebo/balance tests (baseline outcome, grade, gender) |
| invalid | 4 | Winsorized/trimmed specs with likely coding errors |

## Top 5 Most Suspicious Rows

### 1. robust/sample/winsorize_1pct (INVALID)
- **Issue**: Coefficient is 0.002 with SE 0.0003 and p ~ 1e-9, compared to baseline of 0.014 with SE 0.018 and p = 0.42
- **Diagnosis**: The winsorization appears to have changed the outcome scale by a factor of ~7x (coefficient ratio ~0.002/0.014), while the SE shrunk by ~60x. This is physically implausible for 1% winsorization. The winsorization code likely replaced values with ranks, quantiles, or otherwise transformed the data rather than simply capping extreme values.
- **Same issue affects**: winsorize_5pct, winsorize_10pct, trim_1pct (all marked invalid)

### 2. robust/sample/winsorize_5pct (INVALID)
- **Issue**: Nearly identical to winsorize_1pct (coef 0.00201, p ~ 1e-9). The fact that 1%, 5%, and 10% winsorization give nearly identical results further confirms a coding error: real winsorization at these very different thresholds should produce meaningfully different results.

### 3. robust/sample/grade7_only (SUSPICIOUS)
- **Issue**: The coefficient (0.01432) and SE (0.01771) are identical to only_baseline_outcome, suggesting that restricting to "Grade 7 only" removed zero observations. This implies all students in the sample are grade 7, making the grade FE in the baseline specification vacuous.
- **Implication**: The grade FE in the baseline is not actually doing anything, and specs like only_grade_fe produce the same coefficient as no-controls.

### 4. robust/control/add_B_Sgrade6 (SUSPICIOUS)
- **Issue**: The coefficient is identical to add_B_Sgender_index2 (the prior step in the control progression), confirming that the grade variable B_Sgrade6 is either constant in the sample or perfectly collinear with existing terms. The "control progression" did not actually add a new control.

### 5. robust/heterogeneity/gender (MISLEADING)
- **Issue**: This reports the main effect of B_treat (coef = 0.063, p = 0.02) from a model that includes a gender interaction (treat_girl). This is NOT the pooled treatment effect; it is the treatment effect for boys only (the omitted gender category). Presenting this as a heterogeneity test of the main claim is misleading because the reported coefficient changes meaning in the presence of the interaction. The boys-only subsample spec (robust/sample/boys_only) already captures this effect more transparently.

## Recommendations for the Specification Search Script

1. **Fix winsorization/trimming code**: The winsorize and trim specifications produce implausible results. The code should be reviewed to ensure it performs standard winsorization (capping values at percentiles) rather than some transformation that changes the variable scale. After fixing, coefficients should be on the same order of magnitude as the baseline.

2. **Remove redundant grade specifications**: Since all students appear to be in grade 7, the grade FE and grade-related specs are vacuous. The script should detect when a sample restriction removes zero observations and flag it, or when a control variable is constant/collinear.

3. **Clarify heterogeneity spec reporting**: For interaction models, consider reporting both the main effect AND the interaction term as separate rows, with clear labeling that the main effect represents the effect for the reference group (not the overall effect). Currently, the gender_interaction row captures the interaction term, but the gender row captures the boys-only main effect, which can be confused with a pooled estimate.

4. **Index construction caveat**: The SPECIFICATION_SEARCH.md notes that the index is a simplified mean rather than the paper's inverse-covariance-weighted Anderson index. This may explain why the baseline effect (0.014, p=0.42) differs from the published result. The spec search script should ideally replicate the original index construction method.

5. **Consider adding Endline 2 specifications**: The paper reports results for two endline waves. The current search focuses only on Endline 1. Adding Endline 2 would provide a more complete picture of robustness.
