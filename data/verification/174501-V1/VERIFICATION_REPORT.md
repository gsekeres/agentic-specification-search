# Verification Report: 174501-V1

## Paper
**Title**: Interaction, Stereotypes and Performance: Evidence from South Africa  
**Authors**: Corno, La Ferrara, Burns  
**Journal**: AER: Insights (2022)

## Baseline Groups Found

| Group | Claim | Baseline spec_ids | Expected Sign |
|-------|-------|-------------------|---------------|
| G1 | Mixed-race roommate reduces Race IAT prejudice for white students | baseline/table3/race_iat_white | + |
| G2 | Mixed-race roommate improves GPA for black students | baseline/table4/gpa_black | + |
| G3 | No effect on Race IAT for black students (null) | baseline/table3/race_iat_black | 0 |
| G4 | No effect on Academic IAT for either race (null) | baseline/table3/academic_iat_white, academic_iat_black | 0 |
| G5 | No effect on GPA for white students (null) | baseline/table4/gpa_white | 0 |
| G6 | Improved broader academic outcomes for black students | baseline/table4/examspassed_black, continue_black, pcaperf_black | + |
| G7 | Improved social outcomes especially for white students | 12 Table 5 baseline specs | + |
| G8 | Null/attenuated academic effects for white/full sample | 7 Table 4 specs (white + full) | 0 |

## Summary Counts

| Metric | Count |
|--------|-------|
| **Total specifications** | 148 |
| **Baseline specifications** | 28 |
| **Core test specifications** (incl baselines) | 102 |
| **Non-core specifications** | 46 |
| **Unclear** | 0 |
| **Invalid** | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 28 | Baseline specifications from Tables 3, 4, 5 |
| core_controls | 40 | Control variations (leave-one-out, no controls, own controls only, no roommate controls) |
| core_sample | 20 | Sample restrictions (gender subsamples, leave-one-residence-out) |
| core_inference | 6 | SE type variations (HC0, HC2, HC3) on two main claims |
| core_fe | 4 | No fixed effects variation on main claims |
| core_funcform | 4 | Standardized outcome variables |
| noncore_alt_outcome | 28 | Year 2 outcomes, residential sorting, friendship components, pro-social behavior |
| noncore_heterogeneity | 12 | Gender/private school/foreign student interactions |
| noncore_placebo | 6 | Baseline (pre-treatment) outcomes as placebo tests |

## Top 5 Most Suspicious Rows

1. **robust/inference/HC0_GPA_black, HC2_GPA_black, HC3_GPA_black** (rows 61-63): These inference variation specs report coefficient=0.197 while the baseline GPA_black reports coefficient=0.272. This discrepancy arises because the inference specs use "Residence FE" and "Full controls" with different SE types, whereas the baseline uses "Residence + Program FE" with "Own + roommate controls". The coefficient difference is not just an inference change -- the FE and controls also differ, making these partly FE/controls variations rather than pure inference variations. However, the spec_tree_path correctly identifies them as inference variations, so I classify them as core_inference but note the coefficient is not directly comparable to the baseline.

2. **robust/inference/HC0_DscoreraceIAT_white, HC2, HC3** (rows 58-60): These correctly hold the coefficient constant at 0.3215 (matching the baseline) while varying only the SE. These are clean inference variations.

3. **robust/outcome/year2_examspassed2013_black** (row 66): Coefficient of 1.024 is very large compared to baseline examspassed_black (0.669). This is a year 2 outcome with a smaller sample (N=208 vs N=332), so the larger effect could reflect survivor bias (only students who persisted appear in year 2 data). Classified as noncore_alt_outcome because it measures a different time period.

4. **robust/sample/male_IAT_white** (row 96): N=38 is extremely small (white male students only). The coefficient flips sign (-0.085) compared to the baseline (+0.321). This sample restriction is too underpowered for meaningful inference and illustrates that the IAT result for white students is driven by female students. Still classified as core_sample.

5. **robust/sample/male_IAT_black** (row 98): Shows a significant negative coefficient (-0.287, p=0.017) among black male students, while the baseline for black students is a small insignificant negative (-0.095). The gender subsample reveals meaningful heterogeneity.

## Notes on Inference Variation Coefficient Mismatch

The 3 inference variation specs for GPA_black (HC0, HC2, HC3) report coefficient=0.197 instead of the baseline 0.272. Inspection of the fixed effects and controls fields reveals these specs use "Residence FE" and "Full controls" rather than the baseline's "Residence + Program FE" and "Own + roommate controls". This means these specs are not pure inference variations -- they also change the model specification. This should be flagged as a potential issue in the spec-search script: inference variations should hold the model specification constant and only change the variance estimator.

## Recommendations for Spec-Search Script

1. **Fix inference variations**: The GPA_black inference variations should use the same FE structure (Residence + Program FE) and controls as the baseline. Currently they appear to use a simpler specification.

2. **Add clustered SE variations**: The original paper clusters at the room level. The replication uses robust HC1. Adding room-clustered SE as an inference variation would be valuable.

3. **Consider adding more baseline-specific robustness**: The spec search runs control and sample variations only for IAT_white and GPA_black (the two headline claims). Adding similar variations for other baselines (e.g., Table 5 social outcomes) would improve coverage.

4. **Heterogeneity classification**: The heterogeneity specs report the main treatment coefficient from a model that includes an interaction term. This means the main coefficient is no longer the average treatment effect -- it is the effect for the omitted category (e.g., male students when gender is interacted). This interpretation should be clarified in the spec search documentation.

5. **Year 2 outcomes**: These are classified as noncore because they measure effects at a different time point. If the paper's claim explicitly includes persistence of effects, these could be reclassified as core tests of a persistence claim.
