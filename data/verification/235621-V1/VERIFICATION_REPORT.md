# Verification Report: 235621-V1

## Paper Information
- **Paper ID**: 235621-V1
- **Total Specifications**: 70
- **Baseline Specifications**: 1
- **Core Test Specifications**: 49
- **Non-Core Specifications**: 21

## Baseline Groups

### G1: Experience has a positive return on wages in a Mincerian framework: an additiona
- **Expected sign**: +
- **Baseline spec(s)**: `baseline`
- **Outcome**: `lw`
- **Treatment**: `e`
- **Notes**: Coefficient: 0.049 (p<0.001). N=165,097. PSID 1968-2007. Cohort FE, clustered SE. Includes exp-squared to capture concavity.

**Global Notes**: Paper: 'The Price of Experience' (AER). Mincerian wage equation with PSID panel data. All 70 specifications are positive and significant at 1%. The coefficient ranges from 0.017 (without exp-squared, capturing average slope) to 0.074 (max exp 20, capturing steeper part of curve). The functional form assumption (quadratic experience) matters greatly: dropping exp-squared reduces the coefficient from 0.049 to 0.017. Returns to education (~0.099) are treated as a robustness check.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **49** | |
| core_controls | 14 | |
| core_fe | 5 | |
| core_funcform | 7 | |
| core_inference | 4 | |
| core_method | 1 | |
| core_sample | 18 | |
| **Non-core tests** | **21** | |
| noncore_alt_treatment | 1 | |
| noncore_heterogeneity | 20 | |
| **Total** | **70** | |

## Verification Checks

- Total specs in CSV: 70
- Unique spec_ids: 70
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
