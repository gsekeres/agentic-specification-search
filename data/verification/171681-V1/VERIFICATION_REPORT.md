# Verification Report: 171681-V1

## Paper
**Title**: Evaluating Deliberative Competence: A Simple Method with an Application to Financial Choice  
**Journal**: AER: Insights (2022)  
**Method**: Randomized Controlled Trial  

## Baseline Groups

### G1: Full intervention improves financial competence (negAbsDiff)
- **Claim**: The Full educational intervention (Rule of 72 video) improves financial competence, measured as the negative absolute difference between complex-framed and simple-framed WTP (negAbsDiff). Higher values indicate better alignment.
- **Expected sign**: Positive (+)
- **Baseline spec_ids**: baseline_expA (Exp A, n=180), baseline_expB (Exp B, n=185)
- **Notes**: Exp A is the primary experiment with 4 treatment arms (Control, Rule72, Rhetoric, Full). Exp B is a replication with only Control vs Full. Both baselines are no-controls OLS regressions. Data is simulated (calibrated to paper).

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **69** |
| **Baselines** | **2** |
| **Core tests (is_core_test=1)** | **52** |
| **Non-core tests (is_core_test=0)** | **17** |
| **Invalid** | **0** |
| **Unclear** | **0** |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 21 |
| core_sample | 22 |
| core_inference | 3 |
| core_funcform | 6 |
| core_method | 0 |
| noncore_placebo | 4 |
| noncore_alt_outcome | 4 |
| noncore_alt_treatment | 2 |
| noncore_heterogeneity | 7 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

Note: The 2 baselines are included in the core_controls count (is_baseline=1 and is_core_test=1). Total core = 52. Total non-core = 17. Grand total = 69.

## Top 5 Most Suspicious Rows

1. **robust/control/none** (spec_id: robust/control/none): This specification is numerically identical to baseline_expA (same coefficient 3.819, same SE 1.474, same p-value 0.00956). It is a duplicate entry. Classified as core but flagged as redundant.

2. **robust/sample/expA_only** (spec_id: robust/sample/expA_only): This uses the full controls on Exp A and is numerically identical to robust/control/add_fl_high (the final control progression step). It duplicates an existing spec under a different category label.

3. **robust/outcome/discount_unframed** vs **robust/placebo/simple_frame**: These two specs have identical coefficients (-0.051), SEs (2.409), and p-values (0.983). They are the same regression entered twice: once as an alternative outcome and once as a placebo test. Both classified as non-core (placebo).

4. **robust/outcome/score_indexing** vs **robust/placebo/score_indexing**: Again identical regressions (coef=0.090, SE=0.159, p=0.573). Duplicated across alternative outcome and placebo categories. Both classified as non-core (placebo).

5. **robust/outcome/raw_diff** (spec_id: robust/outcome/raw_diff): The raw signed difference (diff = discount_framed - discount_unframed) is a fundamentally different estimand than the absolute difference. A non-significant treatment effect on the signed difference (p=0.68) does not contradict a significant effect on the absolute difference, since treatment could improve alignment without shifting the mean signed gap. Correctly classified as non-core, but might confuse naive aggregation.

## Recommendations for Spec-Search Script

1. **De-duplicate identical regressions**: robust/control/none duplicates baseline_expA; robust/sample/expA_only duplicates robust/control/add_fl_high; robust/outcome/discount_unframed duplicates robust/placebo/simple_frame; robust/outcome/score_indexing duplicates robust/placebo/score_indexing. The script should avoid recording the same regression under multiple spec_ids, or at minimum flag duplicates.

2. **Clarify functional form sign expectations**: The functional form specs (log, IHS, sqrt) transform absDiff (the unsigned version) rather than negAbsDiff. This means negative coefficients are expected and correct (treatment reduces absDiff). The spec-search summary notes this, but the specification_results.csv does not flag sign-flip expectations per row. Consider adding an expected_sign column.

3. **Separate heterogeneity from sample restrictions**: The script currently has both sample restriction specs (e.g., robust/sample/male, robust/sample/female) and heterogeneity interaction specs (e.g., robust/heterogeneity/male). These serve different analytical purposes: sample restrictions test the main effect in subgroups while heterogeneity tests differential effects via interactions. The current labeling is clear, which is good.

4. **Pooled treatment variable naming**: The robust/treatment/pooled spec uses treatment_var="Full_pooled" rather than "Full". While the coefficient concept is the same (effect of full treatment vs control), the name change could cause issues in automated matching. Consider standardizing.

5. **Consider flagging simulated data more prominently**: Since all results are based on simulated data calibrated to match paper effects (not actual replication data), this is a fundamental limitation. The baseline claim verification is approximate. Any downstream aggregation should weight these results accordingly.
