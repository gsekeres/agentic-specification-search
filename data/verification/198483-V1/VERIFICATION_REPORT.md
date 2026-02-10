# Verification Report: 198483-V1

## Paper Information
- **Title**: National Solidarity Program Impact in Afghanistan
- **Authors**: Beath, Christia, Enikolopov
- **Journal**: AEJ-Applied
- **Total Specifications**: 71

## Classification Summary

| Category | Count |
|----------|-------|
| **Core tests (incl. baselines)** | **41** |
| core_sample | 17 |
| core_controls | 11 |
| core_fe | 4 |
| core_inference | 3 |
| core_funcform | 3 |
| **Non-core tests** | **33** |
| noncore_heterogeneity | 15 |
| noncore_alt_outcome | 10 |
| noncore_placebo | 5 |
| noncore_diagnostic | 3 |
| **Total** | **71** |

## Robustness Assessment

**WEAK** support for the main hypothesis.

Treatment effects on the economic index are generally small and often insignificant. The result is stable across FE structures, control sets, and clustering levels. Regional heterogeneity (East vs non-East) is the paper's key finding and is supported across specifications. Placebo tests using security incident data (SIGACTS) show no pre-treatment effects. Alternative index constructions (Katz, PCA) give similar patterns.
