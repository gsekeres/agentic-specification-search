# Verification Report: 131981-V1

## Paper Information
- **Title**: Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey
- **Journal**: AEJ: Applied Economics
- **Method**: Sharp Regression Discontinuity Design
- **Total Specifications**: 87

## Baseline Groups

### G1: Depression Index (z-scored)
- **Claim**: Age-specific curfews increased mental distress in those over 65.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.140 (SE: 0.104, p = 0.18) -- NOT significant
- **Outcome**: `z_depression`
- **Treatment**: `before1955`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **37** | |
| core_controls | 7 | Baseline + no controls + female only + LOO (ethnicity, education, female) + duplicate |
| core_sample | 19 | Bandwidths (7), donut holes (4), gender, marital status, education, chronic disease, psych, winsorize/trim (5), symmetric |
| core_inference | 3 | Robust, modate, province clustering |
| core_funcform | 6 | Linear at 3 BWs + quadratic at 3 BWs |
| **Non-core tests** | **50** | |
| noncore_alt_outcome | 30 | Mental health indices (3), mobility (3), channel outcomes (9), individual symptoms (12), z_depression duplicate |
| noncore_heterogeneity | 4 | Gender, married, chronic, psych support interactions |
| noncore_placebo | 4 | Placebo cutoffs at +/-12 and +/-24 months |
| noncore_diagnostic | 8 | Covariate balance tests |
| **Total** | **87** | |

## Robustness Assessment

**WEAK** support. Baseline is NOT significant (p=0.18). Only 12.2% of main outcome specs significant at 5%. Donut holes make results significant (concerning). Strong gender heterogeneity (males only). First-stage mobility effects are highly significant, confirming treatment works.
