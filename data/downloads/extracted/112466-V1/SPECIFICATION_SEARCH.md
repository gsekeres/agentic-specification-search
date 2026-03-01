# Specification Search Report: 112466-V1

**Paper:** Xiong & Yu (2011), "The Chinese Warrants Bubble", AER 101(6)

## Overview

This paper studies the bubble in Chinese put warrants (2005-2008), testing whether
warrant prices covary with turnover, volatility, and volume as predicted by resale
option theory. Two baseline groups are analyzed:

- **G1**: Maturity effect regressions (cross-sectional averages by days remaining, N=60)
- **G2**: Daily time-series regressions (warrant 38004, N=468)

## G1 Baseline Specification

- **Design:** Cross-sectional OLS
- **Outcome:** Price (average warrant closing price by days to maturity)
- **Treatment:** Turnover (daily trading volume / shares outstanding)
- **Controls:** Volatility, Volume, Days_remaining
- **SE:** Newey-West HAC (5 lags)

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.114642 |
| Std. Error | 0.010869 |
| p-value | 0.000000 |
| 95% CI | [-0.135944, -0.093340] |
| N | 60 |
| R-squared | 0.9645 |

## G2 Baseline Specification

- **Design:** Time-series OLS
- **Outcome:** Warrant_price (daily closing price)
- **Treatment:** Turnover
- **Controls:** Volatility, Volume, days_to_exp
- **SE:** Newey-West HAC (5 lags)

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.382574 |
| Std. Error | 0.068763 |
| p-value | 0.000000 |
| 95% CI | [-0.517346, -0.247802] |
| N | 468 |
| R-squared | 0.6929 |

## Specification Counts

- Total specifications: 61
- Successful: 61
- Failed: 0
- Inference variants: 5

## G1 Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 3 | 3/3 | [-0.1146, -0.1146] |
| Controls LOO | 3 | 3/3 | [-0.2184, -0.0729] |
| Controls Sets | 3 | 3/3 | [-0.2400, -0.0661] |
| Controls Progression | 4 | 4/4 | [-0.1336, -0.0661] |
| Controls Subset | 10 | 10/10 | [-0.2400, -0.0729] |
| Sample Trimming/Restriction | 5 | 5/5 | [-0.1304, -0.1077] |
| Functional Form | 2 | 2/2 | [-0.1493, -0.1370] |
| Alt Treatment | 2 | 2/2 | [0.0001, 0.0330] |
| SE Variants | 2 | 2/2 | [-0.1146, -0.1146] |

## G2 Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 3 | 3/3 | [-0.3826, -0.3826] |
| Controls LOO | 3 | 2/3 | [-0.3785, -0.0264] |
| Controls Sets | 4 | 1/4 | [-0.1773, 0.1484] |
| Controls Progression | 5 | 2/5 | [-0.3826, 0.1140] |
| Controls Subset | 5 | 3/5 | [-0.3388, 0.2092] |
| Sample Trimming/Restriction | 5 | 5/5 | [-0.5359, -0.0190] |
| Functional Form | 2 | 1/2 | [-0.2807, 0.3151] |
| Alt Treatment | 2 | 1/2 | [0.0085, 0.7649] |
| SE Variants | 2 | 2/2 | [-0.3826, -0.3826] |

## Inference Variants

| Group | Spec ID | SE | p-value | 95% CI |
|-------|---------|-----|---------|--------|
| G1 | infer/se/hac/nw10 | 0.009436 | 0.000000 | [-0.133135, -0.096149] |
| G1 | infer/se/hc/hc1 | 0.013446 | 0.000000 | [-0.140996, -0.088288] |
| G2 | infer/se/hac/nw10 | 0.074536 | 0.000000 | [-0.528662, -0.236486] |
| G2 | infer/se/hac/nw20 | 0.079484 | 0.000001 | [-0.538359, -0.226789] |
| G2 | infer/se/hc/hc1 | 0.059226 | 0.000000 | [-0.498655, -0.266493] |

## Overall Assessment

### G1
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 32/32 (100.0%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.114642)
- **Robustness assessment:** WEAK

### G2
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 18/29 (62.1%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.177252)
- **Robustness assessment:** WEAK

Surface hash: `sha256:265e3c4415e530140cdd6f08736996853bf4518d94b962e888d664c62f507c49`
