# Verification Report: 157781-V1

## Paper Information
- **Title**: Canal Closure and Rebellions in Imperial China
- **Total Specifications**: 65

## Baseline Groups

### G1: Canal Closure -> Rebellions (DiD)
- **Coefficient**: 0.031 (p<0.001)
- **Outcome**: ashonset_km2 (asinh rebellions per km2)
- **Treatment**: interaction1 (Along Canal x Post-1825)

## Classification Summary

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_fe | 5 |
| core_funcform | 4 |
| core_inference | 2 |
| core_sample | 27 |
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 1 |
| noncore_diagnostic | 1 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 7 |
| **Total** | **65** |

## Key Notes

1. Baseline effect robust to FE variations, control sets, year/province drops.
2. Winsorized outcomes yield zero coefficients -- result driven by extreme rebellion events (outliers).
3. Pre-treatment trend test marginally significant (p=0.012), raising parallel trends concern.
4. Yangtze placebo shows significant negative effect, suggesting spatial confounding.
5. All heterogeneity interactions are insignificant.
6. Time window sensitivity: effect strongest at 30-50yr windows, weak at 10-20yr.
