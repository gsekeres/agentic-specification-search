# Specification Search Report: 116224-V1

**Paper:** Kuruscu (2006), "Training and Lifetime Income", AER 96(1)

## Baseline Specification

- **Design:** Cross-sectional OLS (bivariate)
- **Outcome:** Later-career log-wage growth (slope of log real wage on age over 5-year window)
- **Treatment:** Earlier-career log-wage growth (slope of log real wage on age over 5-year window)
- **Controls:** None (bivariate regression)
- **Fixed effects:** None
- **SE type:** HC1 (robust)
- **Data:** PSID panel, individual-level wage growth computed from within-person regressions

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.064977 |
| Std. Error | 0.052416 |
| p-value | 0.215110 |
| 95% CI | [-0.167711, 0.037757] |
| N | 1072 |
| R-squared | 0.0026 |

## Specification Counts

- Total specifications: 56
- Successful: 56
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 2 | 0/2 | [-0.0650, 0.0261] |
| Yearahead Variations | 14 | 0/14 | [-0.0936, 0.1460] |
| Window Size (nobs) | 6 | 0/6 | [-0.0510, -0.0072] |
| Initial Age/Exper | 11 | 1/11 | [-0.1110, 0.0116] |
| Outlier Trim | 4 | 1/4 | [-0.0865, -0.0473] |
| Req. Observations | 2 | 0/2 | [-0.0464, -0.0290] |
| Schooling Restriction | 2 | 0/2 | [-0.2209, -0.0806] |
| Functional Form | 2 | 0/2 | [-0.0041, 0.0120] |
| Controls | 2 | 0/2 | [-0.0658, 0.0219] |
| Grid Combinations | 11 | 0/11 | [-0.0733, 0.0905] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc3 | 0.053067 | 0.220788 | [-0.168986, 0.039032] |
| infer/se/hc0 | 0.052367 | 0.214682 | [-0.167615, 0.037661] |
| infer/se/ols_classical | 0.039155 | 0.097310 | [-0.141806, 0.011852] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 2/56 (3.6%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.029007)
- **Robustness assessment:** FRAGILE

Surface hash: `sha256:e6068f869a3d765bae7f473a2fd69274b11782f3a3ef95c4f688bca6bc01c569`
