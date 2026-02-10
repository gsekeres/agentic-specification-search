# Verification Report: 194841-V1

## Paper Information
- **Title**: Rational Inattention and the Business Cycle Effects of Productivity and News Shocks
- **Authors**: Bartosz Mackowiak and Mirko Wiederholt
- **Journal**: American Economic Review
- **Total Specifications**: 70

## Baseline Groups

### G1: Coibion-Gorodnichenko Regression (Under-reaction)
- **Claim**: Forecasters exhibit under-reaction to news -- forecast revisions predict forecast errors (beta > 0).
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.757 (SE: 0.299, p = 0.012)
- **Outcome**: `log_forecast_error`
- **Treatment**: `forecast_revision`
- **N**: 157 quarterly observations (SPF GDP forecasts, 1969-2019, outliers removed)

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **39** | |
| core_controls | 8 | Baseline + time trends, lagged variables, recession indicator, full model |
| core_sample | 17 | Outlier treatments (winsorize, trim), time splits, exclude recessions, drop influential |
| core_inference | 6 | HC1-HC3, Newey-West HAC (4,8 lags), bootstrap |
| core_funcform | 7 | Levels, growth rates, quadratic, standardized, quantile regressions (25th/50th/75th) |
| core_method | 4 | WLS, M-estimation, first differences, GLS AR(1) |
| **Non-core tests** | **31** | |
| noncore_heterogeneity | 15 | Time quartiles (4), rolling windows (9), recession/revision/moderation interactions (6 minus 4 counted in quartiles) |
| noncore_placebo | 4 | Permuted FR, lagged 4q FR, lead 1q FR, lead 4q FR |
| noncore_alt_outcome | 5 | Absolute/squared FE, sign indicators, large error binary/probit |
| **Duplicates noted** | **2** | subsample1_early = early_period; subsample2_late = late_period |
| **Total** | **70** | |

## Detailed Classification Notes

### Core Tests (39 specs including 1 baseline)

**Baseline (1 spec)**: The primary CG regression using SPF GDP forecasts. Bivariate OLS of log forecast error on forecast revision, with 99th percentile outlier removal.

**Control variations (7 non-baseline core_controls specs)**: These add controls to the baseline bivariate regression. Key finding: adding lagged forecast error flips the sign to -0.45 (p=0.052), indicating that the baseline relationship is heavily influenced by serial correlation in forecast errors. The full model (all controls) also shows a negative, insignificant coefficient.

**Sample restrictions (17 specs)**: Systematic exploration of outlier treatment and time period sensitivity. The coefficient remains positive across most sample restrictions (early/late periods, outlier treatments, influential observation removal). The recession-only subsample yields a very large coefficient (4.19), while excluding recessions yields an insignificant 0.23.

**Inference variations (6 specs)**: All maintain the same point estimate (0.757) with slightly different SEs. All remain significant at 5%.

**Functional form (7 specs)**: The growth rate specification reverses sign (-0.394, p<0.001), which is a notable sensitivity. Levels specification (0.33, p=0.04) and quantile regressions (0.58-0.67) all positive and significant.

**Method variations (4 specs)**: WLS and M-estimation maintain positive significant results. First differences (-0.31, p=0.10) and GLS AR(1) (-0.19, p=0.34) both reverse sign, indicating serial correlation concerns.

### Non-Core Tests (31 specs)

**Heterogeneity (15 specs)**: Rolling windows and time quartile splits show substantial time variation. The recession interaction reveals that nearly all the effect comes from recession periods (interaction coef = 3.96, p<0.001).

**Placebo (4 specs)**: Permuted FR and 4q lagged FR both produce near-zero coefficients (good placebos). Lead FRs are mechanically significant.

**Alternative outcomes (5 specs)**: Testing absolute, squared, sign, and binary forecast error measures. These address a different question (forecast accuracy vs. directional under-reaction).

## Robustness Assessment

The main finding -- that beta > 0 indicating under-reaction -- shows **moderate** robustness:

- **Robust to**: Inference method, outlier treatment, quantile regression, M-estimation, WLS
- **Sensitive to**: Dynamic specifications (lagged FE control, first differences, GLS AR1 all flip sign), growth rate transformation, time period (early sample insignificant), economic conditions (driven by recessions)

The sensitivity to lagged forecast errors is the most concerning finding, suggesting the baseline OLS may be misspecified due to serial correlation in forecast errors.
