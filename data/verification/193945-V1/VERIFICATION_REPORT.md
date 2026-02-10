# Verification Report: 193945-V1

## Paper Information
- **Title**: Location, Location, Location
- **Authors**: Card, Rothstein, Yi
- **Journal**: AEJ: Applied
- **Total Specifications**: 99

## Baseline Groups
### G1: CZ Earnings Variance Decomposition
- **Baseline**: `baseline` (coef=0.293) | **Sign**: + | **Outcome**: `mean_log_earnings_cz` | **Treatment**: `cz_place_effect`
### G2: Size Elasticity of Place Premiums
- **Baseline**: `ols/firm_effect/on_log_size` | **Sign**: + | **Outcome**: `mean_firm_effect` | **Treatment**: `log_size`

## Classification Summary
| Category | Count |
|----------|-------|
| Baselines | 2 |
| Core tests | 37 |
| Non-core | 60 |
| **Total** | **99** |

### By Category
| Category | Count |
|----------|-------|
| core_controls | 13 |
| core_funcform | 4 |
| core_inference | 4 |
| core_method | 5 |
| core_sample | 13 |
| noncore_alt_outcome | 30 |
| noncore_diagnostic | 22 |
| noncore_heterogeneity | 8 |

## Robustness Assessment
**STRONG** support. 93% positive, 91% significant.
