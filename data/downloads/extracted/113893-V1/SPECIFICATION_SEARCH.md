# Specification Search Report: 113893-V1

**Paper:** DellaVigna, Enikolopov, Mironova, Petrova, and Zhuravskaya (2014), "Cross-Border Media and Nationalism: Evidence from Serbian Radio in Croatia", AEJ: Applied 6(3)

## Baseline Specification (G1)

- **Design:** IV / reduced-form OLS (cross-sectional)
- **Outcome:** Nazi_share (HSP nationalist party vote share)
- **Treatment:** radio1 (binary Serbian radio availability)
- **Instrument:** s1_1 (signal strength, for 2SLS)
- **Controls:** 21 controls (census, geographic, war/manual, region dummies)
- **Weights:** people_listed (population)
- **Clustering:** Opsina2 (municipality)

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.025537 |
| Std. Error | 0.008268 |
| p-value | 0.003515 |
| 95% CI | [0.008864, 0.042210] |
| N | 139 |
| R-squared | 0.6005 |

## Baseline Specification (G2)

- **Design:** LPM (cross-sectional)
- **Outcome:** graffiti (anti-Serb graffiti, binary)
- **Treatment:** radio1 (binary Serbian radio availability)
- **Weights:** unweighted
- **Clustering:** Opsina2 (municipality)

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.305491 |
| Std. Error | 0.124164 |
| p-value | 0.017973 |
| 95% CI | [0.055089, 0.555892] |
| N | 139 |
| R-squared | 0.2783 |

## Specification Counts

- Total specifications: 71
- Successful: 71
- Failed: 0
- Inference variants: 2

## Category Breakdown (G1: Nazi_share)

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baselines | 5 | 5/5 | [0.0254, 3.6708] |
| Design (2SLS/IV) | 5 | 3/5 | [0.0176, 2.6782] |
| Controls LOO | 10 | 9/10 | [0.0205, 0.0379] |
| Controls Sets | 4 | 3/4 | [0.0254, 0.0280] |
| Controls Add | 3 | 2/3 | [0.0094, 0.0297] |
| Controls Subset | 15 | 11/15 | [0.0204, 0.0385] |
| Sample Restrictions | 3 | 3/3 | [0.0254, 0.0373] |
| Weights | 1 | 0/1 | [0.0139, 0.0139] |
| RF Variants | 6 | 4/6 | [-0.3367, 3.6708] |
| Alt. Outcomes | 8 | 3/8 | [-4.0994, 1.8843] |
| First Stage | 1 | 1/1 | [18.5188, 18.5188] |

## Category Breakdown (G2: Graffiti)

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| All G2 | 9 | 5/9 | [0.2742, 6.1535] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.012107 | 0.036999 | [0.001566, 0.049508] |
| infer/g2/se/hc/hc1 | 0.139622 | 0.030609 | [0.029048, 0.581934] |

## Overall Assessment

### G1: Nazi_share ~ radio1 (core robustness, N=43 specs)
- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 33/43 (76.7%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.025574)
- **Robustness assessment:** MODERATE

### G2: Graffiti ~ radio1
- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 5/6 (83.3%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.334807)
- **Robustness assessment:** STRONG

Surface hash: `sha256:873e9deab3ffa5f05415586239b7a98c293c84fb0f535854f02830d0018900a6`
