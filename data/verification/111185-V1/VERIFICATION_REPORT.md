# Verification Report: 111185-V1

## Paper Information
- **Title**: Optimal Climate Policy When Damages Are Unknown
- **Journal**: AEJ: Policy
- **Total Specifications**: 78

## Baseline Groups

### G1: Damage Exponent (d2)
- **Claim**: Climate damages follow a power law in temperature, with exponent d2 approximately equal to 2, estimated via meta-regression of log(damages) on log(temperature).
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 1.882 (SE: 0.451, p < 0.001)
- **Outcome**: `log_correct`
- **Treatment**: `logt`
- **Table 1**

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **48** | |
| core_controls | 11 | 1 baseline + year, grey, market, year+grey, full, method indicators, kitchen sink, year trend, year quadratic controls |
| core_fe | 1 | Method fixed effects |
| core_sample | 24 | Trim/winsorize (4), time period splits (4), exclude grey, drop influential, leave-one-out (15), WLS weights (3) |
| core_inference | 8 | HC1/HC2/HC3 robust SE, cluster by method/author/model, bootstrap |
| core_funcform | 7 | Level-level, log-level, level-log, quadratic logt, quadratic t, asinh outcome, cubic |
| core_method | 3 | Quantile regressions (25th, 50th, 75th percentile) |
| **Non-core tests** | **30** | |
| noncore_heterogeneity | 16 | Market/nonmarket/method/temp subsamples (6), interaction terms (7), year tercile subsamples (3) |
| noncore_alt_outcome | 3 | D_new level, log D_new, original D measure |
| noncore_placebo | 3 | Shuffled temperature (x2), year as predictor |
| **Total** | **78** | |

## Detailed Classification Notes

### Core Tests (48 specs including baseline)

**Baseline (1 spec)**: The bivariate log-log OLS regression of corrected damage estimates on temperature from the Howard and Sterner (2017) meta-analysis dataset (n=43).

**Control variations (10 non-baseline core_controls specs)**: These progressively add study-level controls:
- Year of study, grey literature indicator, market/nonmarket indicator: individually and in combinations
- Method indicator dummies (enumerative, statistical, survey): captures estimation approach heterogeneity
- Kitchen sink with all controls
- Year trend (centered) and quadratic year trend

**Fixed effects (1 spec)**: Method FE absorbs unobserved differences across estimation approaches. Produces identical coefficient to method indicators (1.309).

**Sample restrictions (24 specs)**: The largest core category:
- Outlier treatment: 1% and 5% trimming, 1% and 5% winsorization
- Time period splits: pre-2005, pre-2008, post-2008, post-2010 -- tests stability over evolving literature
- Exclude grey literature: restricts to peer-reviewed only (n=34)
- Drop influential observations: Cook's distance criterion (n=39)
- Leave-one-out (15 specs): Drop each individual study to check no single study drives results. Coefficients range 1.86-2.26, confirming no single influential study.
- Weighted least squares (3 specs): inverse temperature weight, year weight, inverse year weight

**Inference variations (8 specs)**: All maintain same point estimate (1.882), varying only SE computation:
- HC1/HC2/HC3 heteroskedasticity-consistent SEs: SE increases from 0.45 (homoskedastic) to 0.92 (HC1), 1.01 (HC2), 1.14 (HC3)
- Clustered by method, author, model: SE ranges 1.10-1.40
- Bootstrap (1000 reps): SE = 0.87 (note: anomalous p-value of 1.012 suggests implementation issue)

**Functional form (7 specs)**: Systematic exploration of the functional form assumption:
- Level-level, log-level, level-log: alternative transformations
- Quadratic in logt and t: tests curvature
- Asinh outcome transformation: robust to zero/negative values
- Cubic polynomial: tests higher-order nonlinearity

**Method (3 specs)**: Quantile regressions at 25th, 50th, and 75th percentiles test whether the central tendency is representative of the conditional distribution.

### Non-Core Tests (30 specs)

**Heterogeneity (16 specs)**: These decompose the effect by study characteristics rather than providing alternative implementations:
- Subsample splits by market/nonmarket, method type, temperature range (6 specs): reveal strong heterogeneity (market-only d2 ~ 0.01 vs nonmarket d2 ~ 3.88)
- Interaction terms with grey, market, year, method, pre-2008, high-temp indicators (7 specs)
- Year tercile subsamples (3 specs): early/mid/late studies

**Alternative outcomes (3 specs)**: Different damage variable definitions (D_new in levels, log D_new, original D).

**Placebo tests (3 specs)**: Two with randomly shuffled temperature (should yield null), one with year-only as predictor. All yield insignificant coefficients, validating the temperature-damage relationship.

## Robustness Assessment

The damage exponent estimate of approximately 2 is **moderately robust**:

- **Stable across controls**: Adding year, grey, market, method controls changes coefficient from 1.88 to 1.31-1.88 range.
- **Leave-one-out stable**: No single study drives the result (range 1.86-2.26).
- **Sensitive to inference**: Robust/clustered SEs approximately double, making significance borderline (p = 0.04-0.18 depending on method).
- **Sensitive to sample composition**: Market-only estimates yield d2 ~ 0 (insignificant), while non-market estimates yield d2 ~ 3.9 (highly significant). This stark heterogeneity is the key finding.
- **Sensitive to outliers**: 5% trimming reduces coefficient to 0.37 (insignificant), suggesting a few extreme damage estimates drive much of the result.
