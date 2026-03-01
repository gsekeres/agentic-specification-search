# Specification Search Report: 112805-V1

**Paper:** Deming (2014), "Using School Choice Lotteries to Test Measures of School Effectiveness", AER P&P 104(5)

## Baseline Specification

- **Design:** IV with lottery FE (randomized experiment)
- **Outcome:** testz2003 (average of math and reading z-scores)
- **Treatment (endogenous):** VA (school value-added measure)
- **Instrument:** lott_VA (lottery-determined school VAM)
- **VAM Model:** Model 2 (gains/lagged scores), average residual, all pre-lottery years
- **Controls:** 8 lagged score controls (imputed, with polynomials)
- **Fixed effects:** lottery_FE
- **Clustering:** lottery_FE

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.931767 |
| Std. Error | 0.353011 |
| p-value | 0.009532 |
| 95% CI | [0.232039, 1.631496] |
| N | 2553 |
| R-squared | nan |

## Specification Counts

- Total specifications: 51
- Successful: 51
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 4 | 3/4 | [0.0105, 1.0530] |
| VAM Grid | 36 | 18/36 | [0.0105, 1.1590] |
| Design Variants | 2 | 1/2 | [0.1892, 0.9318] |
| Controls Variants | 3 | 1/3 | [0.1892, 0.9034] |
| Sample (Grade) | 2 | 1/2 | [0.9112, 1.1887] |
| Outcome Variants | 2 | 1/2 | [0.7903, 1.0814] |
| Counterfactual | 2 | 2/2 | [0.9318, 1.0266] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/cluster/lottery_FE | 0.353011 | 0.009532 | [0.232039, 1.631496] |
| infer/se/hc/hc1 | 0.365470 | 0.010848 | [0.215103, 1.648432] |
| infer/se/iid | 0.333969 | 0.005312 | [0.276875, 1.586659] |

## Overall Assessment

- **Coefficient direction:** Median coefficient is positive (0.4760)
- **Significance stability:** 27/51 (52.9%) specifications significant at 5%
- **VAM validity (coef near 1):** 25/51 (49.0%) specifications have coefficient in [0.5, 2.0]
- **Note:** The paper's claim is that the VAM coefficient equals 1 (unbiased), not just significance
- **Robustness assessment:** WEAK

Surface hash: `sha256:e2dd21820ecb81fe3c3259f320388f3bce86473ea432382bec71b809690eeb61`
