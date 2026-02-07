# Verification Report: 146241-V1

## Paper
**Title**: Ten isn't large\! Group size and coordination in a large-scale experiment
**Authors**: Arifovic, Hommes, Kopanyi-Peuker, and Salle
**Journal**: AEJ-Microeconomics

## Baseline Groups

### G1: Group size effect on withdrawal
- **Claim**: Larger groups coordinate worse than smaller groups, leading to higher withdrawal (bank run) rates.
- **Expected sign**: Positive (large groups withdraw more)
- **Baseline spec_id**: baseline
- **Baseline coefficient**: 0.507 (logit), p=0.010
- **Sample**: Period 8, all treatments except treatment 7, N=1178

## Counts

| Category | Count |
|----------|-------|
| Total specifications | 69 |
| Baseline | 1 |
| Core tests (incl. baseline) | 45 |
| Non-core tests | 24 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 14 |
| core_sample | 17 |
| core_inference | 5 |
| core_fe | 1 |
| core_funcform | 6 |
| core_method | 2 |
| noncore_placebo | 5 |
| noncore_heterogeneity | 13 |
| noncore_alt_treatment | 6 |
| noncore_alt_outcome | 0 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

## Top 5 Most Suspicious Rows

1. **paper/table4_col3_withdrawers**: Treatment var is large but this is a subgroup of period-7 withdrawers with a large:pastGroupRunning interaction. The coefficient on large is -2.09 (negative, opposite to all other specs), which is the conditional main effect when pastGroupRunning=0, not the average group-size effect. This is a heterogeneity analysis, not a core test.

2. **paper/table4_col5_period9**: Reports the conditional main effect of large when pastGroupRunning=0 (coef=-0.177, p=0.62). Includes the large:pastGroupRunning interaction, so the coefficient is not comparable to the baseline. Classified as noncore_heterogeneity.

3. **paper/table4_col2_waiters**: Subgroup of period-7 waiters with interaction term. Coefficient on large is 0.733 but p=0.42, reflecting a conditional effect. The interaction with pastGroupRunning means the large coefficient is only the effect at pastGroupRunning=0.

4. **robust/sample/late_periods**: N=54,457 is anomalously large compared to other single-period specs (about 1,178). This pools many periods (11+), dramatically changing the sample composition and effectively upweighting later-game dynamics. Coefficient=2.54 is much larger than baseline, suggesting strong period effects.

5. **robust/heterogeneity/by_pastgroup_main**: The main effect of large becomes 0.22 (p=0.73) when the large:pastGroupRunning interaction is included. This is a conditional effect at pastGroupRunning=0, not the average effect, making it not comparable to baseline. The interaction itself is also insignificant (p=0.64).

## Classification Rationale

### Core tests (45 specs)
All specifications that regress withdraw (or running_rate) on large (or log_groupSize / groupSize) with various control sets, sample restrictions, estimation methods, inference approaches, functional forms, and fixed effects are classified as core tests. These all test the same fundamental hypothesis: does group size causally affect coordination outcomes?

Key sub-categories:
- **Control variations (14)**: Stepwise additions and leave-one-out removals of controls. All maintain the same sample and treatment.
- **Sample restrictions (17)**: Period-by-period, interest rate subsets, location subsets, gender subsets, early/late pooling, and the first-period-only spec.
- **Inference variations (5)**: Different standard error approaches (MLE, group-clustered, session-clustered, treatment-clustered, HC1 robust) applied to LPM.
- **Estimation methods (2)**: LPM and probit as alternatives to the logit baseline.
- **Fixed effects (1)**: LPM with session fixed effects.
- **Functional form (6)**: Quadratic/log transforms of controls, log/continuous treatment, and group-level running_rate outcome.

### Non-core tests (24 specs)
- **Placebo tests (5)**: Training periods and permuted treatments. These deliberately destroy the treatment signal to validate the design. Not tests of the core hypothesis.
- **Heterogeneity specs (13)**: All specs with interaction terms (large x rho, large x rValue, large x female, large x pastDecision, large x pastGroupRunning, plus the Table 4 waiter/withdrawer/period9 specs with interactions). The main effects in these models are conditional on the interacted variable equaling zero, changing the estimand.
- **Alternative treatment (6)**: Treatment-by-treatment regressions using pastDecision as the treatment variable. These test within-treatment persistence of behavior, an entirely different hypothesis.

## Recommendations for Spec-Search Script

1. **Table 4 columns with interactions**: paper/table4_col2, col3, col5 include interaction terms but report the coefficient on large as the treatment effect. These report conditional main effects, not average effects. The script should either not extract these or compute average marginal effects.

2. **Treatment-by-treatment pastDecision regressions**: The six paper/treatment_X_pastDecision specs use pastDecision as the treatment variable, testing a completely different hypothesis. These should be excluded from the group-size specification search.

3. **Heterogeneity interaction specs**: The 10 heterogeneity specs (5 main effects + 5 interactions) report conditional quantities. Consider computing average marginal effects or dropping these from the core test set.

4. **Pooled period specs**: The early_periods and late_periods specs pool many subject-period observations, dramatically changing the effective sample. The late_periods spec (N=54,457) includes periods where the effect is strongest, mechanically inflating the coefficient.

5. **Standard errors**: The replication uses MLE standard errors rather than the paper's bootstrapped clustered SEs. The cluster variation specs partially address this but only for LPM, not logit.
