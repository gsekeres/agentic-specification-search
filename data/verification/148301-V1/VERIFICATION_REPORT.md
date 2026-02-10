# Verification Report: 148301-V1

## Paper Information
- **Title**: Multinationals' Sales and Profit Shifting in Tax Havens
- **Authors**: Laffitte, Toubal
- **Journal**: AEJ: Economic Policy (2022)
- **Total Specifications**: 64

## Baseline Groups

### G1: Higher corporate tax rates reduce foreign sales ratios (sales shifting to low-ta...
- **Expected sign**: -
- **Baseline spec(s)**: baseline
- **Outcome**: foreign_sales_ratio
- **Treatment**: corp_tax_rate
- **Notes**: Coef=-0.0054, p<0.001. Panel FE (country+industry+year). 96.6% of specs show negative coefficient.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 1 |
| Core tests (non-baseline) | 43 |
| Non-core tests | 20 |
| **Total** | **64** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_fe | 5 |
| core_funcform | 5 |
| core_inference | 6 |
| core_sample | 18 |
| noncore_alt_outcome | 4 |
| noncore_alt_treatment | 4 |
| noncore_heterogeneity | 9 |
| noncore_placebo | 3 |

## Global Notes

Panel study of US multinational affiliates 1999-2013. Uses simulated data (original BEA data restricted). 64 specs. Very robust negative relationship between tax rates and foreign sales ratios. Alternative treatments (tax haven dummy, low tax thresholds) show expected positive signs.
