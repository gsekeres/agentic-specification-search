# Verification Report: 208341-V1

## Paper Information
- **Paper ID**: 208341-V1
- **Total Specifications**: 69
- **Baseline Specifications**: 5
- **Core Test Specifications**: 40
- **Non-Core Specifications**: 29

## Baseline Groups

### G1: Land rental subsidies increase agricultural value added (the primary outcome) vi
- **Expected sign**: +
- **Baseline spec(s)**: `baseline/ETwadj_ag_va1_r6_qaB_1`
- **Outcome**: `ETwadj_ag_va1_r6_qaB_1`
- **Treatment**: `rental_subsidy`
- **Notes**: Coefficient: 19.84 (SE~14, p=0.15). Not significant at conventional levels. Strata and round FE, farmer-clustered SE, baseline controls.

### G2: Land rental subsidies increase land cultivation (extensive margin).
- **Expected sign**: +
- **Baseline spec(s)**: `baseline/ETd2_1_plot_use_cltvtd_1`
- **Outcome**: `ETd2_1_plot_use_cltvtd_1`
- **Treatment**: `rental_subsidy`
- **Notes**: Coefficient: 0.031 (p=0.14). First-stage mechanism: does subsidy increase land use?

**Global Notes**: Paper: Land Rental Subsidies and Agricultural Productivity by Acampora, Casaburi, Willis (AER). RCT with rental subsidy and cash drop arms vs control. ITT analysis with strata + round FE. Primary outcome (value added) is not significant at baseline (p=0.15). The cash drop arm provides a comparison treatment. First stage and mechanism specs confirm subsidy increases land rental. 69 total specs.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **40** | |
| core_controls | 17 | |
| core_fe | 3 | |
| core_funcform | 7 | |
| core_inference | 2 | |
| core_method | 5 | |
| core_sample | 6 | |
| **Non-core tests** | **29** | |
| noncore_alt_outcome | 4 | |
| noncore_alt_treatment | 7 | |
| noncore_diagnostic | 3 | |
| noncore_heterogeneity | 12 | |
| noncore_placebo | 3 | |
| **Total** | **69** | |

## Verification Checks

- Total specs in CSV: 69
- Unique spec_ids: 69
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
