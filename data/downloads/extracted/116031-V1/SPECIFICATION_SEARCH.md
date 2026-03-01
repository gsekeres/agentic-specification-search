# Specification Search Report: 116031-V1

**Paper:** Neary (2004), "Rationalising the Penn World Table: True Multilateral Indices for International Comparisons of Real Income", AER 94(5)

## Baseline Specification

- **Design:** Cross-sectional OLS (cross-country)
- **Outcome:** log(GAIA_QUAIDS) (utility-based real income index from QUAIDS demand system)
- **Treatment:** log(Geary) (traditional Geary real income index)
- **Controls:** log(population)
- **Fixed effects:** none
- **Standard errors:** HC1 (robust)
- **N:** 60 countries (1980 ICP benchmark)

**Note:** This is a structural calibration paper. The original GAUSS code estimates QUAIDS demand systems and computes GAIA indexes. Since GAUSS is unavailable, GAIA indexes are approximated using the reported correlation structure. The specification search tests the robustness of the Geary-GAIA relationship across index pairs, country samples, and functional forms.

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.731777 |
| Std. Error | 0.020879 |
| p-value | 0.000000 |
| 95% CI | [0.690854, 0.772699] |
| N | 60 |
| R-squared | 0.9736 |

## Specification Counts

- Total specifications: 68
- Successful: 68
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 4 | 4/4 | [-0.1788, 0.9698] |
| Outcome Swaps | 5 | 5/5 | [-0.1788, 1.2056] |
| Treatment Swaps | 5 | 5/5 | [-0.7182, 1.3305] |
| Controls LOO | 4 | 4/4 | [0.6884, 0.9958] |
| Controls Sets | 6 | 6/6 | [0.6884, 0.9959] |
| Controls Progression | 4 | 4/4 | [0.6884, 0.9959] |
| Controls Subset | 10 | 10/10 | [0.7108, 0.9959] |
| Sample Trimming | 2 | 2/2 | [0.7238, 0.7255] |
| Sample Subsets | 7 | 7/7 | [0.6911, 0.8727] |
| Functional Form | 4 | 4/4 | [-0.2162, 0.9913] |
| Fixed Effects | 1 | 1/1 | [0.6884, 0.6884] |
| Scaling | 1 | 1/1 | [0.9092, 0.9092] |
| Grid | 15 | 13/15 | [-0.7182, 1.0268] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc3 | 0.021986 | 0.000000 | [0.688686, 0.774867] |
| infer/se/ols | 0.015963 | 0.000000 | [0.699811, 0.763742] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 66/68 (97.1%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.744191)
- **Robustness assessment:** WEAK

Surface hash: `sha256:bcc8b1bf273864c0012d5a11af469c65b6c867f914ecad101c0d0af165cbf674`
