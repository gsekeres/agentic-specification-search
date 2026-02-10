# Verification Report: 147561-V3

## Paper Information
- **Title**: Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo
- **Authors**: Balancing, Bergeron, et al.
- **Journal**: AER
- **Total Specifications**: 60

## Baseline Groups

### G1: Local tax collectors increase tax compliance compared to central collectors
- **Expected sign**: +
- **Baseline spec(s)**: baseline
- **Outcome**: taxes_paid
- **Treatment**: t_l
- **Notes**: Coef=0.017, p<0.001. With stratum FE. Effect = 30-50% increase relative to control mean of ~5.4%.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 1 |
| Core tests (non-baseline) | 37 |
| Non-core tests | 22 |
| **Total** | **60** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 5 |
| core_fe | 7 |
| core_inference | 4 |
| core_method | 1 |
| core_sample | 21 |
| noncore_alt_outcome | 7 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 9 |
| noncore_placebo | 3 |

## Global Notes

RCT of local vs central tax collection in Kananga, DRC. 60 specs. Highly robust effect. Local collectors increase compliance by 1.7-2.9pp depending on specification.
