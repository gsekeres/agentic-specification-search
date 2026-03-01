# Specification Search Report
**Paper:** 113812-V1 (Stange 2012, AEJ: Applied)
**Title:** An Empirical Investigation of the Option Value of College Enrollment

## Design
- **Method:** Cross-sectional OLS (Mincer-style lifetime income regression)
- **Data:** NLSY79 (National Longitudinal Survey of Youth 1979)
- **Outcome:** pvlifeincd05 (PV lifetime income, 5% discount, thousands of 1992$)
- **Treatment:** m_s16 (male 4-year college completion dummy)
- **Controls:** 34 gender-interacted regressors (schooling, demographics, ability)
- **Standard errors:** HC1 (robust)

## Baseline Result

| Statistic | Value |
|-----------|-------|
| Coefficient | 330.3590 |
| Std. Error | 31.7938 |
| p-value | 0.000000 |
| 95% CI | [268.0210, 392.6970] |
| N | 3266 |
| R-squared | 0.3500 |

## Specification Counts

- Total specifications: 51
- Successful: 51
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [330.3590, 330.3590] |
| Controls LOO | 9 | 9/9 | [319.5297, 359.8782] |
| Controls Progression | 4 | 4/4 | [330.3590, 470.9335] |
| Controls Sets | 3 | 3/3 | [181.9048, 330.3590] |
| Alt DVs | 4 | 3/4 | [-5.7149, 186.1196] |
| Heterog Returns | 4 | 0/4 | [-63.5089, 162.9753] |
| Sample Restrictions | 8 | 8/8 | [157.1005, 377.3977] |
| Control Subsets | 15 | 15/15 | [316.9388, 359.9195] |
| Alt Treatment | 3 | 2/3 | [24.9124, 127.2245] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/homoskedastic | 23.7371 | 0.000000 | [283.8176, 376.9004] |
| infer/HC1 | 31.7938 | 0.000000 | [268.0210, 392.6970] |

## Overall Assessment

- **Sign consistency (m_s16 specs):** Mixed signs across specifications
- **Significance stability:** 41/46 (89.1%) specifications significant at 5%
- **Direction:** Median coefficient is positive (330.2889)
- **Robustness assessment:** WEAK

Surface hash: `sha256:4cbe233efd9754a6e6c3fb2afeec67ded8bbebade566dc9e4381de9ad5a85d40`
