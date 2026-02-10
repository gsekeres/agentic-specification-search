# Verification Report: 149882-V1

## Paper Information
- **Title**: Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India
- **Journal**: American Economic Review
- **Total Specifications**: 53

## Baseline Groups

### G1: Gender Attitudes Index
- **Claim**: The Breakthrough program improves students' gender-egalitarian attitudes.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.014 SD (SE: 0.018, p=0.42)
- **Outcome**: `E_Sgender_index2_std`
- **Treatment**: `B_treat`
- **Note**: Effect positive but not statistically significant at conventional levels.

## Classification Summary

| Category | Count |
|----------|-------|
| **Baselines** | 1 |
| **Core tests** | 34 |
| **Non-core tests** | 19 |
| **Total** | 53 |

Detailed breakdown:
- core_controls: 11
- core_fe: 1
- core_funcform: 3
- core_inference: 3
- core_sample: 16
- noncore_alt_outcome: 2
- noncore_heterogeneity: 14
- noncore_placebo: 3

## Key Notes

1. **Insignificant baseline**: The pooled treatment effect is positive but not significant (p=0.42). Significance emerges only in gender subsamples (boys: p=0.02).
2. **Control stability**: Coefficient is remarkably stable across control specifications (0.014-0.015 SD).
3. **Winsorization anomaly**: Winsorized specs show very different coefficient magnitudes (~0.002), suggesting the standardization interacts oddly with winsorization.
4. **Placebo tests pass**: Treatment does not predict baseline outcomes or demographics.
5. **All specs use same treatment and primary outcome** except 3 alternative outcome specs and 3 placebo specs.
