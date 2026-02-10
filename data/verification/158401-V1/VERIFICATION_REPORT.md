# Verification Report: 158401-V1

## Paper Information
- **Title**: Market Access and Quality Upgrading: Evidence from Four Field Experiments
- **Total Specifications**: 54

## Baseline Groups

### G1: Providing market access for quality maize increases farmer yield (tons/hectare)....
- **Baseline spec(s)**: baseline_yield
- **Expected sign**: +
- **Outcome**: yield_ha_ton
- **Treatment**: buy_treatment

## Classification Summary

| Category | Count |
|----------|-------|
| core_controls | 8 |
| core_fe | 6 |
| core_funcform | 4 |
| core_inference | 2 |
| core_sample | 19 |
| noncore_alt_outcome | 5 |
| noncore_heterogeneity | 3 |
| noncore_placebo | 2 |
| unclear | 5 |
| **Total** | **54** |

## Key Notes

1. Yield effect (0.29 ton/ha) robust across most specifications.
2. Village FE roughly doubles the effect (~0.65), suggesting between-village variation was masking treatment effect.
3. Price effect strong (60 UGX/kg, p<0.01), confirming quality channel.
4. Harvest value and share sold not significant -- income effects more muted than productivity effects.
5. Pre-treatment placebo: coef=0.09, p=0.65 (passes).
6. First differences specification shows much smaller effect, suggesting level differences matter.
7. Jackknife stable across village drops (4/5 significant).

