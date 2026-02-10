# Verification Report: 195301-V1

## Paper Information
- **Title**: Enemies of the People
- **Authors**: Toews and Vezina
- **Journal**: AEJ-Applied
- **Total Specifications**: 64

## Classification Summary

| Category | Count |
|----------|-------|
| **Core tests (incl. baselines)** | **43** |
| core_sample | 23 |
| core_controls | 9 |
| core_inference | 3 |
| core_fe | 2 |
| core_funcform | 2 |
| core_method | 1 |
| **Non-core tests** | **24** |
| noncore_heterogeneity | 14 |
| noncore_alt_outcome | 5 |
| noncore_alt_treatment | 4 |
| noncore_placebo | 1 |
| **Total** | **64** |

## Robustness Assessment

**MODERATE** support for the main hypothesis.

The main finding (enemy share increases wages) is robust across control variations, spatial radii, sample restrictions, and outlier treatments. Key concerns: (1) effect reverses without Oblast FE (coef = -0.75); (2) effect is insignificant when unweighted (coef = 0.04, p = 0.85), suggesting large firms drive results; (3) 1939 placebo treatment is significant. Sector heterogeneity is substantial (construction +2.0, wholesale near zero).
