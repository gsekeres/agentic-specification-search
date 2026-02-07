# Verification Report: 183985-V2

## Paper
**Emoticons as Performance Feedback for College Students: A Large-Classroom Field Experiment**
Darshak Patel and Justin Roush, AER: P&P

## Baseline Groups

| Group | Claim | Baseline spec_id | Outcome | Coef | p-value |
|-------|-------|-------------------|---------|------|---------|
| G1 | Emoticon feedback improves exam scores (log) | baseline | ltest | 0.0226 | 0.084 |
| G2 | Emoticon feedback improves quiz scores | did/outcome/quiz_score | q_score_cond | 0.158 | <0.001 |
| G3 | Emoticon feedback improves attendance | did/outcome/attendance | attend | 0.031 | 0.015 |
| G4 | Emoticon feedback affects HW attempt rate | did/outcome/hw_attempted | hw_binary | -0.012 | 0.322 |

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **76** |
| Baselines | 4 |
| Core tests (incl baselines) | 33 |
| Non-core tests (excl invalid) | 42 |
| Invalid | 1 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 7 |
| core_fe | 8 |
| core_sample | 6 |
| core_funcform | 6 |
| core_inference | 3 |
| core_method | 3 |
| noncore_heterogeneity | 37 |
| noncore_placebo | 4 |
| noncore_alt_outcome | 1 |
| invalid | 1 |

## Core Specs by Baseline Group

| Baseline Group | Core specs (incl baseline) |
|----------------|---------------------------|
| G1 (exam scores) | 25 |
| G2 (quiz scores) | 2 |
| G3 (attendance) | 3 |
| G4 (HW attempt) | 3 |

Note: The exam score baseline (G1) has the most core robustness tests (FE variations, control variations, sample restrictions, functional form, inference, and method variations). The secondary outcomes (G2-G4) have only FE variations as core tests.

## Top 5 Most Suspicious Rows

1. **robust/cluster/treatment** (INVALID): Clustering standard errors on the treatment variable produces a degenerate result with SE approximately 4e-17 and t-statistic approximately 5.65e14. With only 2 clusters (treatment vs control), cluster-robust inference is completely unreliable. This specification should be excluded from any analysis.

2. **robust/placebo/pre_quiz_trend**: The pre-treatment placebo test for quiz scores is statistically significant (coef=0.125, p=0.009), which raises serious concerns about the parallel trends assumption for the quiz score outcome (G2). This finding undermines the causal interpretation of the quiz score result, which was the strongest result in the paper.

3. **robust/sample/drop_zeros**: This specification produces results identical to the baseline (same coefficient, SE, p-value, N=1188), suggesting there are no zero test scores in the sample when using log transformation. This is not a meaningful robustness check -- it adds no information.

4. **robust/se/hc1**: The HC1 specification produces results identical to the baseline, suggesting the baseline already uses HC1 robust standard errors. This is a redundant specification rather than a true alternative.

5. **robust/het/interaction_gender**: The gender interaction model reports the treat_post coefficient as the male-specific effect (0.0003, p=0.99) rather than the average treatment effect. This is structurally different from the baseline because the coefficient meaning changes when an interaction is included. The female-specific effect is captured by treat_post + treat_post_female.

## Recommendations for Spec-Search Script

1. **Remove degenerate cluster specifications**: The script should validate that cluster-robust standard errors are computed with a sufficient number of clusters (e.g., >10). Clustering on a binary treatment variable with 2 groups is never valid.

2. **Flag redundant specifications**: drop_zeros and hc1 produce identical results to baseline. The script should detect when a specification produces the exact same coefficient and flag it.

3. **Separate heterogeneity from robustness**: The 37 heterogeneity specifications (49% of total) are subgroup analyses that change the estimand. They should be clearly distinguished from core robustness checks that test the same estimand with different modeling choices.

4. **Add more core robustness for secondary outcomes**: Quiz scores (G2), attendance (G3), and HW attempt (G4) each have only FE variations as core tests. The script could add control variations, sample restrictions, and inference variations for these outcomes too.

5. **Flag failed placebos**: The pre-quiz placebo is significant at p=0.009, which is a substantive threat to the quiz result. The script should prominently flag when placebo tests fail.
