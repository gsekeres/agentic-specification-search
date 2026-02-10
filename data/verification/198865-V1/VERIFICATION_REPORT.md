# Verification Report: 198865-V1

## Paper Information
- **Title**: Estimating Models of Supply and Demand: Instruments and Covariance Restrictions
- **Authors**: MacKay and Miller
- **Journal**: AEJ-Micro
- **Total Specifications**: 57

## Classification Summary

| Category | Count |
|----------|-------|
| **Core tests (incl. baselines)** | **50** |
| core_sample | 18 |
| core_controls | 16 |
| core_funcform | 5 |
| core_inference | 4 |
| core_fe | 3 |
| core_method | 2 |
| **Non-core tests** | **9** |
| noncore_alt_treatment | 5 |
| noncore_heterogeneity | 4 |
| **Total** | **57** |

## Robustness Assessment

**MODERATE** support for the main hypothesis.

This is a methods paper with illustrative empirical applications. Cement: OLS estimate (-0.47) and 2SLS (-1.22) differ substantially, confirming endogeneity. IV estimates vary considerably by instrument choice (some instruments are weak). Airlines: results are more stable, with consistently negative price coefficients. The nested logit sigma estimate of 0.89 is robust.
