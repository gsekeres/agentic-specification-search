# Verification Report: 194886-V3

## Paper Information
- **Title**: Resisting Social Pressure in the Household Using Mobile Money
- **Authors**: Riley
- **Journal**: AEJ-Applied
- **Total Specifications**: 116

## Classification Summary

| Category | Count |
|----------|-------|
| **Core tests (incl. baselines)** | **68** |
| core_sample | 32 |
| core_controls | 18 |
| core_inference | 6 |
| core_fe | 4 |
| core_funcform | 4 |
| **Non-core tests** | **52** |
| noncore_heterogeneity | 36 |
| noncore_alt_outcome | 14 |
| noncore_placebo | 2 |
| **Total** | **116** |

## Robustness Assessment

**MODERATE** support for the main hypothesis.

The MD treatment effect on earn_business is robust across most control, sample, and inference variations. The MA treatment is consistently weaker and rarely significant. Results are sensitive to winsorization level and clustering. Heterogeneity by family pressure is consistent with the paper's mechanism. The effect is concentrated in married women and high-baseline-profit entrepreneurs.
