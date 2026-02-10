# Verification Report: 174781-V2

## Paper Information
- **Title**: Work and Mental Health among Rohingya Refugees
- **Journal**: AER
- **Total Specifications**: 78

## Baseline Groups
### G1: Work Treatment on Mental Health
- **Claim**: Cash-for-work improves mental health.
- **Baseline spec**: `baseline` (coef=0.211, p<0.001)
- **Expected sign**: Positive
- **Outcome**: `mental_health_index` | **Treatment**: `b_treat_work`

## Classification Summary
| Category | Count |
|----------|-------|
| Baselines | 8 |
| Core tests | 41 |
| Non-core | 29 |
| **Total** | **78** |

### By Category
| Category | Count |
|----------|-------|
| core_controls | 30 |
| core_fe | 3 |
| core_inference | 3 |
| core_sample | 13 |
| noncore_alt_outcome | 10 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 8 |
| noncore_placebo | 8 |

## Robustness Assessment
**STRONG** support. Coefficient robust across specifications.
