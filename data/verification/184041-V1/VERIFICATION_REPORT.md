# Verification Report: 184041-V1

## Paper Information
- **Title**: The Common-Probability Auction Puzzle
- **Journal**: AER: Insights
- **Total Specifications**: 133 (CSV rows), 133 (disambiguated)

## Baseline Groups
### G1: CV vs CP Overbidding
- **Baselines**: 3 rows (BF, BEBF, NEBF) | **Sign**: + | **Treatment**: `CV`
- **Note**: 3 CSV rows share spec_id 'baseline'; disambiguated as baseline_BF, baseline_BEBF, baseline_NEBF.

## Classification Summary
| Category | Count |
|----------|-------|
| Baselines | 3 |
| Core tests | 101 |
| Non-core | 29 |
| **Total** | **133** |

### By Category
| Category | Count |
|----------|-------|
| core_controls | 36 |
| core_funcform | 7 |
| core_inference | 8 |
| core_method | 11 |
| core_sample | 42 |
| noncore_alt_outcome | 6 |
| noncore_heterogeneity | 19 |
| noncore_placebo | 4 |

## Robustness Assessment
**STRONG** support. 97.7% positive, 96.2% significant at 5%.
