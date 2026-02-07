# Verification Report: 114344-V1

## Paper
**Incomplete Disclosure: Evidence of Signaling and Countersignaling** (AER)

## Baseline Groups

### G1: A+ Disclosure Premium (Signaling)
- **Claim**: A+ restaurants (top-half quality within A grade) are significantly more likely to voluntarily disclose their hygiene grade compared to CD- restaurants (omitted category), consistent with signaling theory.
- **Baseline spec_ids**: `baseline`
- **Outcome**: `disclosure` (binary)
- **Treatment**: `half_grade_def1_d1` (A+ indicator)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.295 (p < 1e-37)

## Classification Summary

| Category | Count |
|----------|-------|
| **Total specifications** | **67** |
| Baseline | 1 |
| Core tests (excl. baseline) | 50 |
| Non-core | 16 |
| Invalid | 0 |
| Unclear | 0 |

### Core Test Breakdown

| Core Category | Count |
|---------------|-------|
| core_controls | 15 (includes baseline) |
| core_sample | 24 |
| core_inference | 5 |
| core_fe | 4 |
| core_method | 2 |
| core_funcform | 1 |

### Non-Core Breakdown

| Non-Core Category | Count |
|-------------------|-------|
| noncore_alt_treatment | 7 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 2 |
| noncore_alt_outcome | 1 |

## Top 5 Most Suspicious / Noteworthy Rows

1. **robust/sample/isA_pred1** (coef = -0.043, p = 5.2e-5): Classified as non-core because restricting to A-grade restaurants only changes the comparison group from CD- to A-, fundamentally altering the estimand. The negative coefficient (A+ discloses *less* than A-) is the opposite sign from the baseline, but this is expected given the different comparison. The specification search report acknowledges this.

2. **robust/sample/isB_pred1** (treatment = half_grade_def1_d3, coef = 0.017, p = 0.25): Uses B+ as treatment within B-grade sample. This is an entirely different estimand (B+ vs B-) with a different treatment variable. Classified as non-core alternative treatment.

3. **robust/het/interaction_popular** (treatment = half_grade_def1_d1_pop, coef = -0.148, p = 3.3e-4): Reports the interaction coefficient (A+ x popularity), not the main A+ effect. Tests the countersignaling hypothesis specifically. Classified as non-core heterogeneity because the reported coefficient is the interaction term, not a robustness check of the main A+ effect.

4. **robust/funcform/quadratic_numgrading** (treatment = num_grading, coef = -0.0004, p = 0.91): Uses a continuous num_grading variable with quadratic specification. The coefficient on the linear term is near zero and insignificant, but this reflects a fundamentally different parameterization (continuous grade score vs binary half-grade indicators). Classified as non-core alternative treatment.

5. **robust/treatment/binary_top_half** (treatment = top_half_A, coef = -0.046, p = 1.2e-5): Uses a different treatment definition (top_half_A, binary top-half of num_grading within grade). The negative coefficient suggests this captures a within-grade effect rather than the between-grade effect in the baseline. Classified as non-core alternative treatment because the treatment definition changes the estimand.

## Notes on Heterogeneity Specifications

Six heterogeneity specifications report interaction term coefficients (e.g., A+ x Yelp presence, A+ x popularity, A+ x good reviews, A+ x pricey, A+ x chain, A+ x firstbatch). These are classified as non-core because:
- The reported coefficient is the interaction term, not the main A+ effect
- They test the countersignaling hypothesis (whether the A+ disclosure effect is moderated by alternative quality signals), which is a distinct claim from the main signaling result
- The main A+ effect in these models would be the core test, but the interaction coefficient is what was extracted

Two heterogeneity specs (high_variability, low_variability) are classified as core_sample because they report the main A+ coefficient on a subsample split by restaurant variability, preserving the same estimand.

## Notes on Functional Form Specifications

Three of the four functional form specs change the treatment variable from the binary half-grade indicator to continuous num_grading, which fundamentally changes what is being estimated. Only robust/funcform/log_mean_numgrading is core because it merely log-transforms a control variable while keeping the same treatment.

## Recommendations for Spec Search Script

1. **Heterogeneity interactions**: When running heterogeneity specifications with interaction terms, the script should extract both the main effect and the interaction term separately. Currently, only the interaction term coefficient is extracted, which represents a different estimand from the baseline.

2. **Within-grade samples**: The script should flag when a sample restriction changes the comparison group (e.g., A-only sample changes omitted category from CD- to A-). This could be done by checking whether the omitted category in the regression changes.

3. **Treatment redefinitions**: Specifications that change the treatment variable (num_grading vs half_grade_def1_d1, top_half_A vs half_grade_def1_d1) should be clearly tagged as alternative-treatment rather than robustness checks, since they change the estimand.

4. **The baseline claim seems correctly identified**: The spec search correctly identifies Table 4 Column 2 as the main specification, and the A+ indicator as the key treatment variable. No changes needed to the baseline definition.
