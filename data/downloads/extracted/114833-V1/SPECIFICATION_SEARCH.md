# Specification Search Report: 114833-V1

**Paper:** Allcott (2013), "The Welfare Effects of Misperceived Product Costs: Data and Calibrations from the Automobile Market", AEJ: Economic Policy 5(1)

**Note:** This analysis uses a synthetic dataset calibrated to the paper's summary statistics (Table 1) and regression coefficients (Table 3, Appendix). The TESS microdata is not included in the openICPSR replication package and must be obtained separately from TESS/OSF.

## Baseline Specification

- **Design:** Cross-sectional OLS (survey experiment)
- **Outcome:** phi (belief parameter: ratio of perceived to true fuel cost difference)
- **Treatment:** AvMPG (harmonic mean of own and alternative vehicle MPG)
- **Controls:** None in baseline (bivariate)
- **Fixed effects:** None
- **Clustering:** CaseID (individual respondent)
- **Weights:** Survey weights (analytical)

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.005465 |
| Std. Error | 0.001650 |
| p-value | 0.000940 |
| 95% CI | [0.002230, 0.008701] |
| N | 3719 |
| R-squared | 0.0032 |

## Specification Counts

- Total specifications: 53
- Successful: 53
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 2 | 2/2 | [0.0055, 0.0055] |
| Controls LOO | 10 | 10/10 | [0.0052, 0.0056] |
| Controls Sets | 6 | 6/6 | [0.0052, 0.0056] |
| Controls Progression | 5 | 5/5 | [0.0052, 0.0056] |
| Controls Subset | 10 | 10/10 | [0.0051, 0.0056] |
| Sample Restriction | 8 | 6/8 | [0.0027, 0.0065] |
| Functional Form | 7 | 2/7 | [-0.7820, 0.0055] |
| Appendix | 5 | 5/5 | [0.0051, 0.0066] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.001643 | 0.000887 | [0.002244, 0.008686] |
| infer/se/hc/hc3 | 0.001650 | 0.000940 | [0.002230, 0.008701] |

## Overall Assessment

- **Core claim (phi ~ AvMPG) specs:** 47
- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 44/47 (93.6%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.005228)
- **Robustness assessment:** STRONG

Surface hash: `sha256:16a518d9c06d569502e7bcd3bbdb45c97f844ef4d7f17144db28856da98b9f19`
