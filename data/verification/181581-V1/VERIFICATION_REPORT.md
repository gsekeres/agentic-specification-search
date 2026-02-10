# Verification Report: 181581-V1

## Paper Information
- **Title**: When a Doctor Falls from the Sky
- **Journal**: AER: Insights
- **Total Specifications**: 93

## Baseline Groups
### G1: Doctor Treatment on 7-day Mortality
- **Baselines**: `baseline`, `baseline_basic_controls`, `baseline_extended_controls` | **Sign**: - | **Outcome**: `mort7` | **Treatment**: `doctor`

## Classification Summary
| Category | Count |
|----------|-------|
| Baselines | 3 |
| Core tests | 68 |
| Non-core | 22 |
| **Total** | **93** |

### By Category
| Category | Count |
|----------|-------|
| core_controls | 38 |
| core_fe | 5 |
| core_inference | 3 |
| core_sample | 25 |
| noncore_alt_outcome | 6 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 9 |
| noncore_placebo | 4 |

## Robustness Assessment
**STRONG** support. 98.8% of mort7 specs negative.
