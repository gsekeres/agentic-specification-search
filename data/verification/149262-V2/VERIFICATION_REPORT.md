# Verification Report: 149262-V2

## Paper Information

- **Paper**: "Student Performance, Peer Effects, and Friend Networks: Evidence from a Randomized Peer Intervention"
- **Authors**: Jia Wu, Junsen Zhang, and Chunchao Wang
- **Journal**: American Economic Journal: Economic Policy, Vol. 15, No. 1 (2023), pp. 510-42
- **Paper ID**: 149262-V2

## Baseline Groups

### G1: MSR Treatment Effect on Math Scores for Lower-Track Students

- **Baseline spec_id**: `baseline`
- **Claim**: The MSR treatment (mixed seating with financial rewards) increases endline math scores by approximately 0.24 SD for lower-track students in Chinese elementary schools.
- **Coefficient**: 0.24 (SE = 0.097, p = 0.018)
- **Expected sign**: Positive
- **Sample**: Lower-track students (N = 901)
- **Controls**: Baseline score, gender, age, height, health, hukou, minority, parental education, household assets
- **Fixed effects**: Grade FE
- **Clustering**: Class level

No additional baseline groups were identified. The paper has a single focal claim. The MS treatment arm (seating without incentives) consistently shows null effects and represents a separate treatment, not a variant of the main claim.

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **86** |
| Core tests (including baseline) | 25 |
| Non-core tests | 61 |
| Baselines | 1 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 8 | Baseline + control set variations (add/drop/swap controls, no controls, minimal controls) |
| core_sample | 8 | Sample restrictions (grade subsamples, compliers only, male/female, balanced pairs, combined track) |
| core_funcform | 5 | Functional form (class-standardized, raw scores, percentile rank, above-median binary, score gain) |
| core_inference | 3 | Clustering variations (robust HC1, school-clustered, grade-clustered) |
| core_method | 1 | 2SLS/TOT estimate (actual treatment instrumented by assignment) |
| noncore_alt_treatment | 34 | Different treatment variable: MS arm (10 specs), deskmate_baseline_score peer effects (16 specs), MS on other outcomes/samples (8 specs) |
| noncore_alt_outcome | 18 | Different outcome: average score (5), Chinese score (2), Big Five personality traits (10), avg on upper/combined (1) |
| noncore_heterogeneity | 7 | Heterogeneity analyses: income splits (2), baseline tercile splits (3), income interaction (1), upper-track MSR (1) |
| noncore_placebo | 2 | Placebo tests: midterm outcome, pre-treatment balance check |

## Core Test Analysis

The 25 core specifications all test the same fundamental claim: MSR treatment improves math outcomes for lower-track students. They vary along:

1. **Controls** (8 specs): Range from no controls to full controls plus distance-to-teacher. The baseline coefficient is robust across control specifications (range: 0.154 to 0.239).

2. **Sample** (8 specs): Grade-specific subsamples (3-5), compliers only, gender splits, balanced pairs, and pooled track sample. Coefficients range from 0.174 (combined track) to 0.28 (compliers only).

3. **Functional form** (5 specs): Alternative transformations of the math score (class-standardized, raw, percentile, binary, gain). All show positive MSR effects.

4. **Inference** (3 specs): Different clustering levels. Point estimate unchanged at 0.24; SEs range from 0.075 (robust HC1) to 0.12 (school-clustered).

5. **Estimation** (1 spec): 2SLS TOT estimate (0.28, SE=0.12), consistent with ITT baseline.

## Top 5 Most Suspicious Rows

1. **robust/placebo/midterm_outcome** (midterm exam): Labeled as placebo but coefficient is positive (0.12, p=0.18). A true placebo should show zero effect. However, this may reflect early treatment effects if treatment started before midterm. The classification as placebo is appropriate but the spec might actually be a partial-dosage test rather than a true placebo. Confidence: 0.8.

2. **robust/control/add_distance** vs **robust/control/add_distance_math**: The add_distance spec uses average_score_endline as the outcome while add_distance_math uses math_score_endline. There is no clear reason for this split in the specification search -- the average-score variant should have been linked to a separate baseline if one existed. As it stands, it is correctly classified as noncore_alt_outcome, but the spec search agent may have intended both as control variations of a single baseline.

3. **rct/lower/msr/math/with_controls**: Coefficient (0.224) differs from the baseline (0.24) despite both claiming "Full controls" on the same sample. This likely reflects a minor difference in the exact control set or a rounding/extraction issue from the i4r reproduction study. The spec is legitimate but the discrepancy should be noted.

4. **robust/het/interaction_income**: The treatment variable changes to MSR_x_high_income (an interaction term), which is a fundamentally different estimand from the main effect. Correctly classified as noncore_heterogeneity but the treatment_var field change makes it structurally distinct.

5. **robust/estimation/2sls_tot**: Treatment variable is "actual_treatment" (instrumented by random assignment). While this is correctly classified as core_method (TOT vs ITT), the treatment_var field differs from the baseline. The coefficient (0.28 > 0.24) is consistent with the expected LATE > ITT pattern.

## Recommendations

1. **Spec search script**: The script mixes multiple research questions (RCT effects, peer effects, personality outcomes) into a single specification_results.csv. Consider flagging or separating these at the search stage, as peer effects (deskmate_baseline_score as treatment) and Big Five outcomes represent fundamentally different analyses, not robustness checks.

2. **Baseline definition**: The baseline claim is well-defined and correctly identified. No changes needed.

3. **Control variation outcomes**: Some control-variation specs use average_score_endline instead of math_score_endline (e.g., add_distance, drop_health, income_not_edu). These should ideally all use math_score_endline to be comparable to the baseline. The spec search should be more consistent about keeping the outcome fixed when varying controls.

4. **Peer effects classification**: All 16 peer effect specifications (8 original + 8 difference) use deskmate_baseline_score as the treatment variable. These test a completely different causal mechanism and should not be compared to the MSR treatment effect baseline.

5. **Heterogeneity vs sample restriction**: The male/female splits are classified as core_sample (robustness to sample composition) while income/baseline splits are classified as noncore_heterogeneity. The distinction is that gender splits are pre-registered sample restrictions while income/baseline splits are exploratory heterogeneity analyses. This boundary is somewhat subjective.
