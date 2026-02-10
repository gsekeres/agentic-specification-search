# Verification Report: 138922-V1

## Paper Information
- **Title**: The Long-Run Effects of Sports Club Vouchers for Primary School Children
- **Authors**: Marcus, Siedler, Ziebarth
- **Journal**: AEJ: Policy (2022)
- **Total Specifications**: 84

## Baseline Groups

### G1: Sports club vouchers (Saxony treatment) increase sports club membership. Differe...
- **Expected sign**: +
- **Baseline spec(s)**: baseline
- **Outcome**: sportsclub
- **Treatment**: treat
- **Notes**: Baseline: coef=0.009, p=0.635, insignificant. DID with year+state+city FE. The paper finds a null result for the behavioral outcome.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 1 |
| Core tests (non-baseline) | 39 |
| Non-core tests | 44 |
| **Total** | **84** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 21 |
| core_fe | 5 |
| core_inference | 3 |
| core_method | 1 |
| core_sample | 10 |
| noncore_alt_outcome | 27 |
| noncore_alt_treatment | 2 |
| noncore_heterogeneity | 12 |
| noncore_placebo | 3 |

## Global Notes

Paper finds a null effect on sports club membership despite strong first-stage (program awareness). The 84 specs include first-stage outcomes (6 specs), alternative behavioral outcomes, heterogeneity, and placebo tests. The FE variations (pooled OLS, unit-only FE) show significance but are biased due to omitted confounders.
