# Verification Report: 146041-V1

## Paper Information
- **Title**: Human Capital in the Presence of Skilled-Biased Technical Change
- **Authors**: (Authors from AEJ-Policy)
- **Journal**: AEJ-Policy
- **Total Specifications**: 67

## Baseline Groups

### G1: Relative skill efficiency (AQ) is positively associated with GDP per worker acro...
- **Expected sign**: +
- **Baseline spec(s)**: baseline
- **Outcome**: l_irAQ53_dum_skti_hrs_secall
- **Treatment**: l_y
- **Notes**: Coef=1.41, p=0.005. Cross-sectional OLS with 12 countries. Elasticity of log(AQ) wrt log(GDP).

### G2: Relative human capital quality (Q) is positively associated with GDP per worker ...
- **Expected sign**: +
- **Baseline spec(s)**: ols/Q/baseline_us
- **Outcome**: l_irQ53_dum
- **Treatment**: l_y
- **Notes**: Coef=0.105, p<0.001. Much larger sample using immigrant wage data.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 2 |
| Core tests (non-baseline) | 53 |
| Non-core tests | 12 |
| **Total** | **67** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 11 |
| core_funcform | 8 |
| core_inference | 2 |
| core_method | 3 |
| core_sample | 31 |
| noncore_alt_outcome | 2 |
| noncore_heterogeneity | 10 |

## Global Notes

Cross-sectional study of human capital and skill efficiency across countries. Two main analyses: AQ (12-country micro sample) and Q (US immigrant data). 67 specs. AQ analysis has small N=12 but Q analysis is well-powered.
