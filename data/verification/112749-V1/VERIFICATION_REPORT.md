# Verification Report: 112749-V1

## Paper Information
- **Title**: When the Levee Breaks: Black Migration and Economic Development in the American South
- **Journal**: AER
- **Total Specifications**: 57

## Baseline Groups

### G1: Flood Intensity and Black Population Share
- **Claim**: Flooded counties experienced larger declines in Black population share post-1927.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative (flood reduces Black share)
- **Baseline coefficient**: 0.026 (SE: 0.054, p = 0.633) -- positive and insignificant
- **Outcome**: `lnfrac_black`
- **Treatment**: `flood_intensity`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **39** | |
| core_fe | 7 | 2 baselines + county-only, year-only, none, two-way, state-year FE |
| core_controls | 8 | Lagged DV variations (7, most with missing results) + 1 baseline_lagged_dv |
| core_sample | 20 | Drop census years (5), drop states (10), early/late period (2), flooded only, winsorize (3), weights (2) |
| core_inference | 2 | HC1 robust SE, state clustering |
| core_funcform | 4 | Levels, IHS, quadratic intensity, frac_black in levels |
| core_method | 1 | Long differences |
| **Non-core tests** | **18** | |
| noncore_alt_outcome | 2 | Log Black population, log total population |
| noncore_alt_treatment | 1 | Binary flood indicator |
| noncore_heterogeneity | 5 | Plantation, high intensity, Black share, Mississippi interactions |
| noncore_placebo | 1 | Random permutation |
| noncore_diagnostic | 4 | Cross-sectional regressions (1930, 1950, 1960, 1970) |
| **Total** | **57** | |

## Robustness Assessment

The main finding has **weak** support. The baseline coefficient is positive (0.026) and insignificant (p=0.633), contrary to the paper's hypothesis. The two-way FE specification yields a negative coefficient (-0.072) but is also insignificant (p=0.122). Seven specifications failed to produce results due to data issues with lagged dependent variables. Cross-sectional regressions show large positive coefficients, but these capture level differences rather than within-county changes.
