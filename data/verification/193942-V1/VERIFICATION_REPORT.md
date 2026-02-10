# Verification Report: 193942-V1

## Paper Information
- **Title**: Effective Health Aid: Evidence from Gavi's Vaccine Program
- **Total Specifications**: 59

## Baseline Groups
### G1: Vaccine Coverage
- **Baseline**: `baseline` | **Sign**: + | **Outcome**: `coverage` | **Treatment**: `post`
### G2: Mortality Reduction
- **Baseline**: `did/outcome/mortality_baseline` | **Sign**: - | **Outcome**: `rate` | **Treatment**: `post`

## Classification Summary
| Category | Count |
|----------|-------|
| Baselines | 2 |
| Core tests | 38 |
| Non-core | 19 |
| **Total** | **59** |

### By Category
| Category | Count |
|----------|-------|
| core_controls | 2 |
| core_fe | 4 |
| core_funcform | 3 |
| core_inference | 2 |
| core_sample | 29 |
| noncore_alt_treatment | 1 |
| noncore_heterogeneity | 17 |
| noncore_placebo | 1 |

## Robustness Assessment
**MODERATE** support. 92% positive, 78% significant at 5%.
