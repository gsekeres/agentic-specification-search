# Verification Report: 112870-V2

## Paper: Optimal Life Cycle Unemployment Insurance
**Authors**: Claudio Michelacci and Hernan Ruffo  
**Journal**: American Economic Review (AER)  
**Verified**: 2026-02-03  
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Pooled UI Benefit Effect on Unemployment
- **Baseline spec_id**: baseline_pooled
- **Claim**: Higher UI benefits (log weekly benefit amount) increase log unemployment rates in a state-year-age panel
- **Expected sign**: Negative (higher lnwba -> higher lnun, captured as negative coefficient in the regression)
- **Coefficient**: -0.476 (SE=0.074, p<0.001)
- **Model**: Panel FE with state+year+part FE, demographics+education controls, clustered at state

### G2: Age-Varying UI Benefit Effect
- **Baseline spec_id**: baseline
- **Claim**: The unemployment elasticity with respect to UI benefits varies systematically across age groups
- **Expected sign**: Unknown (multiple interaction coefficients)
- **Base coefficient**: 0.508 (for youngest age group)
- **Model**: Panel FE with age group interactions on treatment

---

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **52** |
| Baseline specs | 2 |
| Core test specs | 44 |
| Non-core specs | 5 |
| Invalid specs | 1 |
| Unclear specs | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_sample | 22 |
| core_inference | 2 |
| core_fe | 4 |
| core_funcform | 5 |
| core_method | 3 (incl. 2 baselines + first_difference) |
| noncore_placebo | 3 |
| noncore_heterogeneity | 2 |
| invalid | 1 |

---

## Classification Details

### Core Tests (44 specs)
The majority of specifications are core tests of the G1 (pooled) baseline:

- **Control variations (10)**: Leave-one-out dropping individual controls (married, spouse work, race, ed1-ed4), and progressive control addition (none, demographics only, education only). All use same outcome (lnun), treatment (lnwba), and basic FE structure.

- **Sample restrictions (22)**: Age-group subsamples (5 individual age groups + young/old/prime-age aggregates), time-period splits (early/late), leave-one-year-out (5 specs for 1984-1988), outcome trimming (1% and 5%), seasonal splits (part1/part2), and demographic subsamples (unmarried, married, white). All preserve the core outcome-treatment relationship.

- **Fixed effects variations (4)**: No FE, state-only, year-only, two-way (state+year). These test how the FE structure affects the pooled estimate.

- **Inference variations (2)**: Robust SE (no clustering) and year clustering. Same point estimates as baseline, different standard errors.

- **Functional form (5)**: Levels-levels, log-level, level-log, linear trend, quadratic trend. The trend specs replace year FE with state-specific trends. The levels/semi-elasticity specs change the scale but test the same underlying relationship.

- **Method (1 non-baseline)**: First difference estimator as alternative to FE. Uses differenced variables (lnun_diff, lnwba_diff) but tests the same relationship.

### Non-Core Specs (5)

- **Placebo tests (3)**: lag1_treatment, lead1_treatment, randomized_treatment. These use different treatment variables (lagged, led, or randomized benefits) and are falsification tests, not tests of the core claim.

- **Heterogeneity interactions (2)**: lnwba_x_m_married and lnwba_x_r_white. These estimate interaction effects (how the UI-unemployment relationship varies by marital status or race), changing the estimand from the main effect to a heterogeneity effect.

### Invalid Specs (1)

- **subsample_nonwhite**: Missing standard error, t-statistic, p-value, and confidence interval due to extremely small sample (N=41). The point estimate (-1.755) exists but cannot be reliably interpreted without inference statistics.

---

## Top 5 Most Suspicious Rows

1. **baseline (G2)**: The base coefficient (+0.508) has the opposite sign from baseline_pooled (-0.476). This sign flip suggests the age-interaction model coefficient may represent something different from the pooled effect, possibly due to how the interaction terms absorb variation. The age-group-specific coefficients in the coefficient_vector_json are all positive (0.09 to 0.84), which is inconsistent with the pooled negative effect. This likely reflects a specification issue where the base term in the interaction model captures a different estimand than the pooled model.

2. **robust/sample/old_workers**: Coefficient is +1.259, strongly positive and significant, while the pooled baseline is -0.476. This dramatic sign flip for older workers (50+) is notable and suggests the pooled effect masks substantial heterogeneity by age. However, this is consistent with the paper's thesis about age variation.

3. **robust/sample/age_group_2 through age_group_6**: All five individual age-group subsamples show positive coefficients (ranging from +0.485 to +0.919), opposite in sign to the pooled baseline (-0.476). This systematic sign reversal across all age subsamples is puzzling -- if every subgroup has a positive coefficient, the pooled effect should not be negative unless there is a compositional/Simpson's paradox effect driven by the control variables or FE structure.

4. **robust/placebo/lag1_treatment**: The lagged treatment has a significant positive coefficient (+0.071, p<0.001), which is unexpected for a placebo test. A significant lag could indicate persistence effects or serial correlation rather than a clean placebo failure, but it complicates the causal interpretation.

5. **robust/heterogeneity/subsample_nonwhite**: Marked invalid due to missing SE (N=41). The coefficient of -1.755 is the most extreme negative value in the dataset but cannot be evaluated without inference statistics.

---

## Recommendations for Spec-Search Script

1. **Investigate the sign discrepancy between pooled and age-group baselines**: The fact that the interaction model base coefficient (+0.508) and all individual age-group subsamples show positive effects while the pooled model shows a negative effect (-0.476) is a serious red flag. The spec-search script should verify whether this reflects a Simpson's paradox (control variables or FE structure reversing the sign), a coding error in the interaction model, or a genuine compositional effect. The age-group subsamples and the pooled full-sample model should be reconciled.

2. **Flag placebos that are significant**: The lag1 and lead1 placebo tests both yield significant coefficients, which undermines the causal interpretation. The SPECIFICATION_SEARCH.md stated "Placebo tests pass" but the lag and lead are both significant at conventional levels. The summary statistics should be corrected.

3. **Handle missing SE more gracefully**: The subsample_nonwhite spec should have been flagged during the search phase as having too few observations for reliable inference, rather than being included with NaN standard errors.

4. **Consider whether age-group subsamples are truly core tests**: The individual age-group subsamples (age_group_2 through age_group_6) estimate a within-age-group effect, which is conceptually different from the pooled cross-age effect. I classified them as core_sample because they use the same outcome/treatment, but the sign reversal suggests they may be testing a different estimand. The spec-search script should consider whether these belong in a separate baseline group.

5. **Clarify the Part FE**: The baseline uses "State + Year + Part" FE while many robustness specs use only "State + Year". The Part variable (presumably half-year) is important but its role in the identification is unclear. The script should note which specs include Part FE and which do not.
