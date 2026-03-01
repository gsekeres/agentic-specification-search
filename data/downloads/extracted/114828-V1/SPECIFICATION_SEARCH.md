# Specification Search Report: 114828-V1

**Paper:** Grosfeld, Rodnyansky & Zhuravskaya (2013), "Persistent Antimarket Culture: A Legacy of the Pale of Settlement", AEJ: Economic Policy 5(3)

## Baseline Specification

- **Design:** Sharp geographic/spatial RD
- **Running variable:** Distance to Pale of Settlement boundary (km)
- **Cutoff:** 0 (the boundary itself)
- **Outcome:** prefer_market (binary: prefers market economy)
- **Treatment:** Inside Pale (distance < 0)
- **Method:** Weighted local linear regression (triangular kernel, BW=60km)
- **Clustering:** PSU (psu1)
- **Sample:** Urban respondents in Russia, Ukraine, Latvia (former Russian Empire)

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.179773 |
| Std. Error | 0.104238 |
| p-value | 0.098617 |
| 95% CI | [-0.036404, 0.395949] |
| N | 460 |
| R-squared | 0.0414 |

## Specification Counts

- Total specifications: 62
- Successful: 62
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline (nonparametric) | 5 | 3/5 | [-0.2501, 0.2989] |
| Design (BW/poly/kernel) | 4 | 2/4 | [0.0345, 0.2473] |
| Controls LOO (parametric) | 10 | 10/10 | [-0.2296, -0.1640] |
| Controls Sets (parametric) | 6 | 6/6 | [-0.2473, -0.1850] |
| Controls Progression | 4 | 4/4 | [-0.2176, -0.1920] |
| Controls Add | 4 | 4/4 | [-0.2126, -0.1579] |
| Sample/Bandwidth | 3 | 2/3 | [0.0702, 0.2454] |
| Sample/Restrict | 7 | 1/7 | [0.0371, 0.2712] |
| Sample/Donut | 4 | 4/4 | [0.2482, 0.9369] |
| Cross Outcome x Design | 12 | 6/12 | [-0.2736, 0.3090] |
| Parametric (other outcomes) | 3 | 2/3 | [-0.2711, 0.3145] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.085981 | 0.037094 | [0.010805, 0.348740] |
| infer/se/cluster/country | 0.088136 | 0.178205 | [-0.199447, 0.558992] |

## Overall Assessment

### prefer_market (primary outcome, 44 specifications)
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 34/44 (77.3%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.180743)

### All outcomes (62 specifications)
- **Significance stability:** 44/62 (71.0%) specifications significant at 5%
- **Robustness assessment:** WEAK

Surface hash: `sha256:41b566f2da4d528628708e617dc47f53eb963dbce24a7fb9ca7b1b10e8b93b68`
