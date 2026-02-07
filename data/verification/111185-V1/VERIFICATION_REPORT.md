# Verification Report: 111185-V1

## Paper
**Title**: Optimal Climate Policy When Damages Are Unknown
**Journal**: AEJ: Policy
**Paper ID**: 111185-V1

## Baseline Groups

### G1: Damage exponent meta-regression
- **Claim**: Climate damages follow a power law in temperature with exponent d2 approximately 2, estimated via log-log OLS meta-regression of damage estimates on temperature from the Howard and Sterner (2017) meta-analysis dataset.
- **Baseline spec_ids**: baseline
- **Outcome**: log_correct (log of corrected damage estimate)
- **Treatment**: logt (log of temperature increase in degrees Celsius)
- **Baseline coefficient**: 1.882 (SE = 0.451, p < 0.001, N = 43)
- **Expected sign**: Positive

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **78** |
| Core test specs | 64 |
| Non-core specs | 13 |
| Invalid specs | 1 |
| Unclear specs | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_sample | 31 | Sample restrictions + leave-one-out (15 LOO + 16 sample splits) |
| core_funcform | 10 | Functional form variations (level-level, semi-log, quadratic, cubic, asinh, alt outcomes) |
| core_controls | 9 | Control variable additions (year, grey, market, method indicators, kitchen sink) |
| core_method | 7 | Method changes (WLS variants, quantile regressions) |
| core_inference | 6 | Inference variations (HC1/HC2/HC3, cluster by method/author/model) |
| core_fe | 1 | Method fixed effects |
| noncore_heterogeneity | 10 | Interaction models (7) + year tercile splits (3) |
| noncore_placebo | 3 | Shuffled temperature placebos (2) + year-only regression (1) |
| invalid | 1 | Bootstrap with p > 1 |

## Top 5 Most Suspicious Rows

1. **robust/se/bootstrap** (INVALID): The p-value is 1.012, which exceeds 1.0 and is therefore mathematically invalid. The coefficient (1.793) also differs slightly from the baseline (1.882), suggesting the bootstrap may have been computed on resampled data rather than just resampling residuals. This spec should be re-implemented or dropped.

2. **robust/het/high_temp_interaction**: The reported coefficient on logt is -0.27, which is the main effect conditional on high_temp=0 (i.e., the effect for low-temperature observations only). This is not comparable to the unconditional baseline estimate of 1.88. The interaction term is 4.89 (p < 0.001), meaning the total effect for high-temp observations is approximately 4.62. The coefficient as recorded does not represent the average damage exponent.

3. **robust/form/quadratic_logt**: The linear coefficient on logt is -1.66 when a quadratic term (logt_sq) is included. This sign reversal is expected when including a squared term (the linear term captures the slope at logt=0), but it means this coefficient is not directly comparable to the baseline in terms of economic interpretation.

4. **robust/sample/market_only**: Coefficient is 0.010 (p = 0.99, N = 17), essentially zero. While correctly implemented as a sample restriction, this reveals that the pooled baseline result is driven entirely by non-market damage estimates. This extreme sensitivity (d2 going from 1.88 to 0.01) is noteworthy for understanding the robustness of the paper's claim.

5. **robust/het/statistical_interaction**: The reported coefficient on logt is 3.87, which is the main effect conditional on is_statistical=0. The interaction term is -3.62, meaning statistical-method studies show essentially zero temperature sensitivity (3.87 - 3.62 = 0.25). Like other interaction specs, this conditional main effect is not directly comparable to the unconditional baseline.

## Classification Decisions and Rationale

### Heterogeneity specs (classified as non-core)
All 10 heterogeneity specifications were classified as non-core because they either:
- Report a conditional main effect from an interaction model (7 specs), which changes the estimand relative to the baseline unconditional average effect.
- Represent year-tercile subsample splits (3 specs), which are heterogeneity analyses designed to show variation rather than test the pooled relationship.

Note: The year-tercile and subsample splits are borderline. They use the same specification as baseline but on a subsample, which could be argued as core_sample. However, since they are explicitly filed under robustness/heterogeneity.md and their purpose is to show heterogeneity (not to test the overall claim), they are classified as non-core. This is a conservative choice.

### Placebo specs (classified as non-core)
All 3 placebos appropriately use fake or irrelevant treatments. The shuffled temperature tests use randomly permuted temperature values. The year-only regression uses publication year as the treatment. All return insignificant results as expected, confirming the baseline relationship is not spurious. These are diagnostic tests, not tests of the core claim.

### Functional form specs
These are classified as core tests despite changing outcome and/or treatment variable scales, because the underlying claim (damages are a power function of temperature) can be tested in level-level, log-level, level-log, or log-log forms. The polynomial extensions (quadratic, cubic) change the interpretation of the linear coefficient, so they are lower confidence (0.7).

### Leave-one-out specs
All 15 LOO specs drop individual studies and re-estimate. These are standard robustness checks classified as core_sample. Results are highly stable (range: 1.86 to 2.26).

## Recommendations for Spec-Search Script

1. **Fix bootstrap implementation**: The bootstrap specification (robust/se/bootstrap) produces an invalid p-value > 1. The script should use scipy bootstrap confidence intervals or statsmodels bootstrap and correctly compute the p-value from the bootstrap distribution.

2. **Heterogeneity interaction coefficient reporting**: When interaction models are estimated, the script reports the main effect of logt, which is the conditional effect (logt when the interacted variable = 0). Consider also computing and reporting the average marginal effect of logt across the sample to make it more comparable to the baseline.

3. **Consider adding weighted meta-regression**: The specification search includes ad-hoc WLS weights (1/t, year, 1/year) but not precision-weighted meta-regression (e.g., weighting by inverse variance of original estimates), which is the standard approach in meta-analysis.

4. **Baseline claim is well-identified**: The paper Table 1 regression is correctly captured by the baseline specification. No changes needed to the baseline definition.
