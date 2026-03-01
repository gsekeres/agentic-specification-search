# Specification Search Report: 112853-V1

**Paper:** Manuelli & Seshadri (2014), "Human Capital and the Wealth of Nations", AER 104(11)

## Baseline Specification

- **Design:** Cross-sectional OLS (cross-country development accounting)
- **Outcome:** log(output per worker relative to US) [log_av_rel_y]
- **Treatment:** Average years of schooling (Barro-Lee) [av_sch]
- **Controls:** 5 controls (life expectancy, TFR, education expenditure, relative capital price, infant mortality)
- **Fixed effects:** None
- **Standard errors:** HC1 robust

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.126167 |
| Std. Error | 0.040784 |
| p-value | 0.002869 |
| 95% CI | [0.044784, 0.207550] |
| N | 75 |
| R-squared | 0.8649 |

## Specification Counts

- Total specifications: 52
- Successful: 52
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [0.1262, 0.1262] |
| Controls LOO | 5 | 5/5 | [0.1160, 0.1517] |
| Controls Sets | 4 | 4/4 | [0.1262, 0.3344] |
| Controls Progression | 5 | 5/5 | [0.1262, 0.3344] |
| Controls Subset | 15 | 15/15 | [0.1149, 0.3330] |
| Sample Trimming | 3 | 3/3 | [0.0733, 0.1122] |
| Subgroup Analysis | 10 | 6/10 | [0.0451, 0.2116] |
| Outcome Variants | 3 | 3/3 | [0.0348, 0.1453] |
| Treatment Variants | 3 | 0/3 | [-0.0014, 0.4561] |
| Estimator Variants | 3 | 3/3 | [0.0979, 0.1273] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc3 | 0.050786 | 0.034746 | [0.011281, 0.241053] |
| infer/se/ols | 0.037067 | 0.001118 | [0.052200, 0.200133] |

## Overall Assessment

- **Same-outcome specifications:** 48 (log_av_rel_y ~ av_sch)
- **Sign consistency:** 47 positive, 1 negative  (all significant specs same sign)
- **Significance stability:** 42/48 (87.5%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.126167)
- **Robustness assessment:** STRONG

Surface hash: `sha256:2e9e6eb2468e87be77943493008e928fcfe803c3464dfc660bd986778a62b600`
