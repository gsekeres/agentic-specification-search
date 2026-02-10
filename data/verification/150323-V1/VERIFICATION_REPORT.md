# Verification Report: 150323-V1

## Paper Information
- **Title**: Political Turnover, Bureaucratic Turnover and the Quality of Public Services
- **Authors**: Akhtari, Moreira, Trucco
- **Journal**: American Economic Review
- **Total Specifications**: 93

## Baseline Groups

### G1: 4th Grade Test Scores
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Coefficient**: -0.042 (p=0.10)

### G2: Teacher Turnover (newtchr)
- **Baseline spec**: `rd/bandwidth/optimal_newtchr`
- **Expected sign**: Positive
- **Coefficient**: 0.128 (p<0.001)

### G3: Headmaster Replacement
- **Baseline spec**: `rd/bandwidth/optimal_expthissch`
- **Expected sign**: Positive
- **Coefficient**: 0.274 (p<0.001)

## Classification Summary

| Category | Count |
|----------|-------|
| core_controls | 20 |
| core_funcform | 7 |
| core_inference | 2 |
| core_sample | 53 |
| noncore_alt_outcome | 3 |
| noncore_heterogeneity | 3 |
| noncore_placebo | 5 |
| **Total** | **93** |

## Key Notes

1. Teacher/headmaster turnover effects are highly robust across all bandwidths and control specifications.
2. Test score effects are consistently negative but often marginally significant, sensitive to bandwidth choice.
3. Placebo tests on non-municipal schools correctly show null effects.
4. Donut hole tests show sensitivity for test scores but not for bureaucratic turnover.
