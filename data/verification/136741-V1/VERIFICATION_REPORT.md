# Verification Report: 136741-V1

## Paper Information
- **Title**: Historical Lynchings and Black Voter Registration
- **Author**: Williams
- **Journal**: AEJ-Applied
- **Method**: Cross-sectional OLS with state FE
- **Total Specifications**: 70

## Baseline Groups

### G1: Black Voter Registration Rate
- **Claim**: Higher historical black lynching rates are associated with lower contemporary black voter registration.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.469 (SE: 0.154, p = 0.003)
- **Outcome**: `Blackrate_regvoters`
- **Treatment**: `lynchcapitamob`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **46** | |
| core_controls | 19 | Baseline + LOO (7) + control progression (8) + additional controls (3) + Stevenson data |
| core_sample | 13 | State LOO (6) + trim + winsor + reg caps + weights |
| core_inference | 6 | State cluster, county cluster, HC1/HC2/HC3, classical |
| core_funcform | 7 | Log/IHS outcome, quadratic, log/IHS/binary/tercile treatment, standardized |
| core_fe | 1 | No FE estimation |
| core_method | 3 | Quantile regressions (25th, 50th, 75th) |
| **Non-core tests** | **24** | |
| noncore_alt_outcome | 2 | Register count, white voter registration (in placebo) |
| noncore_alt_treatment | 3 | Lynch rate with 1910/1920/1930 population denominators |
| noncore_placebo | 3 | White lynching on black reg, white on white, black lynching on white reg |
| noncore_heterogeneity | 9 | Interactions (education, earnings, church, incarceration, slavery) + subgroups |
| **Total** | **70** | |

## Robustness Assessment

**STRONG** support. 98.6% negative, 78.6% significant at 5%. Stable coefficient (-0.39 to -0.50) across LOO states. Placebo tests pass perfectly. Sensitive to population denominator choice (1920/1930 attenuates substantially). Binary treatment (any lynching) shows opposite sign (artifact of high-registration counties having some lynchings).
