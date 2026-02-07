# Verification Report: 120568-V1

## Paper
- **Title**: Declining Worker Turnover: The Role of Short-Duration Employment Spells
- **Authors**: Michael Pries and Richard Rogerson
- **Journal**: AEJ: Macroeconomics (2021)
- **Paper ID**: 120568-V1

## Baseline Groups

### G1: Declining one-quarter hazard rate
- **Baseline spec_id**: baseline
- **Claim**: The one-quarter hazard rate (fraction of new hires separating within one quarter) declined significantly over the 1999-2018 period in the US.
- **Outcome**: oneqhazrate
- **Treatment**: time_trend (linear time trend)
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0745, p < 1e-12, N=76
- **Notes**: This is a descriptive/accounting exercise documenting a secular trend, not a causal inference study.

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 74 |
| Core tests (is_core_test=1) | 18 |
| Non-core tests | 55 |
| Invalid | 1 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 1 | Baseline specification |
| core_sample | 10 | Time period restrictions and leave-one-year-out |
| core_funcform | 2 | Log transformation and quadratic trend |
| core_inference | 4 | HAC lag variations (0, 2, 8, 12) |
| core_controls | 1 | Adding hire rate as control |
| noncore_alt_outcome | 11 | Different turnover measures (sep rates, hazard rates, hire rates) |
| noncore_heterogeneity | 39 | Demographic (sex, age), industry, and state subsamples |
| noncore_placebo | 5 | Permutation tests with shuffled outcome |
| invalid | 1 | drop_year_2018 has identical coef/N to baseline |

## Core Tests Detail

The 18 core specifications include:
1. **Baseline** (1 spec): National oneqhazrate on time trend with HAC(4)
2. **Sample restrictions** (10 specs): Pre-recession, post-recession, excluding recession, early/late periods, and 5 leave-one-year-out specs (1999, 2000, 2008, 2009, 2017)
3. **Functional form** (2 specs): Log-transformed outcome; quadratic time trend
4. **Inference** (4 specs): HAC with 0, 2, 8, and 12 lags
5. **Controls** (1 spec): Adding hire rate as a control variable

## Top 5 Most Suspicious Rows

1. **robust/sample/drop_year_2018** (INVALID): Coefficient (-0.0745) and N (76) are identical to baseline. Dropping a year should reduce N to approximately 72 and change the coefficient. This is clearly a bug in the specification search script where the year filter was not applied.

2. **robust/heterogeneity/industry_information**: Shows a POSITIVE coefficient (+0.056, p<0.001), opposite to the baseline direction. While this is legitimately classified as non-core heterogeneity, it highlights that the Information sector trend runs opposite to the national trend -- potentially important for interpreting the results.

3. **robust/sample/post_recession**: Coefficient is +0.019 (wrong sign relative to baseline) and not significant (p=0.06). This is a valid core_sample spec but shows the decline has stopped/reversed post-2010.

4. **robust/sample/late_period**: Coefficient is +0.005, not significant (p=0.63). Combined with the post-recession result, this confirms the secular decline is concentrated pre-2010.

5. **robust/heterogeneity/industry_agriculture**: Not significant (p=0.38), suggesting agriculture did not experience the turnover decline.

## Recommendations for Spec Search Script

1. **Fix drop_year_2018 bug**: The leave-one-year-out implementation for 2018 did not actually filter out the year. Review the filtering logic in spec_search_120568.py. The data likely does not extend to 2018 Q4 (hence N=76 rather than 80), meaning there may be no 2018 data to drop.

2. **Consider reclassifying alternative outcomes**: The 11 alternative outcome specs could arguably be considered core tests if the paper frames declining turnover broadly (not just oneqhazrate). However, the paper title and abstract specifically emphasize short-duration employment spells, so the current classification as noncore_alt_outcome is conservative and appropriate.

3. **Heterogeneity is extensive but non-core**: 39 of 74 specs (53%) are heterogeneity analyses. While these are informative, they do not test the same aggregate national claim. Future spec searches might want a more balanced allocation between core robustness and heterogeneity.

4. **Missing robustness dimensions**: The search could benefit from additional core specs such as: alternative detrending methods (HP filter, first differences), different sample start/end dates, or state-level panel regressions with state fixed effects.
