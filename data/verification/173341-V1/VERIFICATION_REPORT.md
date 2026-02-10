# Verification Report: 173341-V1

## Paper Information
- **Title**: Vulnerability and Clientelism
- **Total Specifications**: 92

## Baseline Groups

### G1: Cistern Treatment -> Private Goods Requests
- **Coefficient**: -0.030 (p=0.018)
- **Outcome**: ask_private_stacked
- **Treatment**: treatment (cisterns)

### G2: Rainfall -> Private Goods Requests
- **Coefficient**: -0.023 (p=0.018)
- **Outcome**: ask_private_stacked
- **Treatment**: rainfall_std_stacked

## Classification Summary

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_fe | 2 |
| core_funcform | 3 |
| core_inference | 2 |
| core_method | 1 |
| core_sample | 44 |
| noncore_alt_outcome | 17 |
| noncore_alt_treatment | 5 |
| noncore_heterogeneity | 5 |
| noncore_placebo | 3 |
| **Total** | **92** |

## Key Notes

1. Cistern treatment effect robust across all 38 municipality jackknife drops.
2. Effect stable across controls and clustering choices.
3. Frequent interactors show much larger effects (-0.118 vs -0.030 overall).
4. Public goods requests (placebo) show no significant treatment effect.
5. Household wellbeing outcomes (happiness, health, food security) show positive effects of treatment.
6. Rainfall effects complement the cistern finding -- both vulnerability channels matter.
