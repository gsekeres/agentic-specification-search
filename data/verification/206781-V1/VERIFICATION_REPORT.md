# Verification Report: 206781-V1

## Paper Information
- **Title**: Who Should Get Social Insurance? A Machine Learning Approach
- **Authors**: Various
- **Journal**: AER
- **Total Specifications**: 71

## Classification Summary

| Category | Count |
|----------|-------|
| **Core tests (incl. baselines)** | **50** |
| core_controls | 26 |
| core_sample | 11 |
| core_inference | 6 |
| core_funcform | 4 |
| **Non-core tests** | **24** |
| noncore_heterogeneity | 12 |
| noncore_alt_outcome | 9 |
| noncore_placebo | 3 |
| **Total** | **71** |

## Robustness Assessment

**STRONG** support for the main hypothesis.

Very robust. All 71 specifications show positive treatment effects (100%). 85% are significant at 5%. Results are stable across control progressions, leave-one-out analyses, clustering variations, sample restrictions, and functional forms. Heterogeneity analyses show significant effects across all subgroups. Placebo tests support the RCT design.
