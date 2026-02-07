# Verification Report: 128143-V1

## Paper
**Title**: Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion
**Authors**: Thomas Douenne and Adrien Fabre
**Journal**: AEJ: Economic Policy
**Method**: Cross-sectional OLS with IV/2SLS using randomized information treatments

## Baseline Groups

### G1: Self-interest channel
- **Claim**: Pessimistic beliefs about personal gains from a carbon tax with dividend (believing one loses) reduce support for the policy.
- **Baseline spec_id**: baseline
- **Outcome**: tax_acceptance
- **Treatment**: believes_not_lose
- **Coefficient**: 0.363 (SE = 0.018, p < 0.001)
- **Expected sign**: Positive (believing one does not lose increases acceptance)

### G2: Environmental effectiveness channel
- **Claim**: Beliefs about the environmental effectiveness of the carbon tax increase policy approval.
- **Baseline spec_id**: baseline_ee
- **Outcome**: tax_approval
- **Treatment**: believes_effective
- **Coefficient**: 0.405 (SE ~ 0.02, p < 0.001)
- **Expected sign**: Positive (believing tax is effective increases approval)

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 77 |
| Baselines | 2 |
| Core tests | 53 |
| Non-core | 24 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 25 |
| core_sample | 20 |
| core_inference | 3 |
| core_fe | 1 |
| core_funcform | 3 |
| core_method | 1 |
| noncore_alt_outcome | 6 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 2 |
| noncore_diagnostic | 7 |
| invalid | 0 |
| unclear | 0 |

## Top 5 Most Suspicious Rows

1. **robust/control/drop_left, drop_right, drop_extreme_right**: These three specs produce coefficients and p-values identical to the baseline (coef = 0.3632, p = 5.50e-88). The baseline coefficient_vector_json confirms left, right, and extreme_right have coefficient = 0.0 and SE = 0.0, meaning they are collinear or absorbed. These leave-one-out specs provide no additional information.

2. **robust/outcome/targeted_acceptance**: Uses tax_cible_acceptance as outcome and believes_not_lose_cible as treatment -- both differ from the G1 baseline. Labeled as alternative outcome but actually changes both the outcome concept (acceptance of targeted dividend) and the treatment variable, making it a different estimand entirely.

3. **robust/outcome/feedback_acceptance**: Uses tax_feedback_acceptance and believes_not_lose_feedback -- also changes both outcome and treatment. The extremely large coefficient (0.714) compared to baseline (0.363) suggests a very different population or measurement, reinforcing non-comparability.

4. **iv/si/2sls_main**: This 2SLS spec uses tax_cible_acceptance and believes_not_lose_cible -- the targeted dividend variants. Tests a different estimand from the G1 baseline with a much larger coefficient (0.603 vs 0.363).

5. **robust/treatment/continuous_gain**: Uses simule_gain (continuous simulated gain in euros) as treatment. The coefficient (0.000217) is on a completely different scale from the baseline binary treatment, making magnitude comparisons meaningless. Barely significant (p = 0.044).

## Recommendations for the Specification Search Script

1. **Drop collinear controls from leave-one-out**: The left, right, and extreme_right variables have zero coefficients in the baseline model (collinear or absorbed). The leave-one-out exercise for these is uninformative and should be flagged or excluded.

2. **Separate alternative-outcome specs that change treatment**: Specs like robust/outcome/targeted_acceptance change both the outcome and treatment variable. These should be labeled as alternative estimand rather than alternative outcome, since they are not testing the same causal relationship with a different dependent variable.

3. **Consider adding control variations for G2**: The environmental effectiveness channel (G2) has only sample-restriction robustness checks and one IV spec. It would benefit from control-variation and inference-variation specs parallel to those done for G1.

4. **Scale awareness for continuous treatments**: The robust/treatment/continuous_gain spec uses a continuous euro-denominated treatment. The script should flag when treatment variables are on fundamentally different scales, as coefficient magnitudes become non-comparable.

5. **Clarify heterogeneity vs. functional form**: Some specs under robust/funcform/ (interact_yv, interact_income) report the main effect from a model with interactions, while the robust/heterogeneity/ specs report the interaction coefficient. This distinction is important but could be confusing. The script should consistently label what coefficient is being extracted.
