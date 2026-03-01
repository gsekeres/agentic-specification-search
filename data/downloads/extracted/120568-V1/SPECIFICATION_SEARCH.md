# Specification Search Report: 120568-V1

**Paper:** Pries & Rogerson (2020), "Declining Worker Turnover: the Role of Short Duration Employment Spells", AEJ: Macroeconomics 12(1)

## Baseline Specification

- **Design:** Panel time-series OLS
- **Outcome:** oneqsepsrate (one-quarter separation rate)
- **Treatment:** time_trend (linear quarterly trend, 1999Q1-2017Q4)
- **Controls:** None (linear trend only)
- **Fixed effects:** state (30 US states)
- **Clustering:** state
- **Data:** QWI quarterly data for 30 states, private sector

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.00043708 |
| Std. Error | 0.00003129 |
| p-value | 0.000000 |
| 95% CI | [-0.00050107, -0.00037308] |
| N | 2276 |
| R-squared | 0.5897 |

## Specification Counts

- Total specifications: 69
- Successful: 69
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [-0.000437, -0.000437] |
| Alt. Outcomes | 18 | 18/18 | [-0.002088, -0.000090] |
| Controls/FE | 5 | 5/5 | [-0.001111, -0.000437] |
| Sample Restrictions | 37 | 37/37 | [-0.000736, 0.000286] |
| Panel Disaggregation | 5 | 5/5 | [-0.000672, -0.000374] |
| Functional Form | 2 | 1/2 | [-0.007121, 0.000005] |
| Weighting | 1 | 1/1 | [-0.000442, -0.000442] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.00001165 | 0.000000 | [-0.00045992, -0.00041423] |
| infer/se/twoway/state_quarter | 0.00003904 | 0.001528 | [-0.00056132, -0.00031283] |
| infer/se/iid | 0.00001094 | 0.000000 | [-0.00045854, -0.00041561] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 68/69 (98.6%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.00043866)
- **Robustness assessment:** WEAK

Surface hash: `sha256:26d94f87a965799c5af6df011f9ce803fc1129b36df009eb1159108442175b85`
