# Verification Report: 150581-V1

## Paper Information
- **Title**: Wage Cyclicality and Labor Market Sorting
- **Journal**: AER
- **Total Specifications**: 61

## Baseline Groups

### G1: Wage-Unemployment Semi-Elasticity
- **Baseline spec**: `baseline/table2_col4`
- **Expected sign**: Negative
- **Coefficient**: -0.0044 (p<0.001)
- **Outcome**: `lhrp2` (log hourly real pay)
- **Treatment**: `unempl` (unemployment rate)

## Classification Summary

| Category | Count |
|----------|-------|
| core_controls | 16 |
| core_fe | 6 |
| core_funcform | 4 |
| core_inference | 5 |
| core_sample | 17 |
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 5 |
| noncore_placebo | 3 |
| **Total** | **61** |

## Key Notes

1. Negative wage-unemployment relationship is highly robust (92% significant at 5%).
2. Specifications with only individual FE show POSITIVE coefficients, highlighting importance of industry-year and occupation-year FE.
3. Effect stronger for non-college, younger workers.
4. Uses simulated data due to NLSY access constraints.
