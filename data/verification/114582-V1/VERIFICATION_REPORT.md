# Verification Report: 114582-V1

## Paper Information
- **Paper ID**: 114582-V1
- **Title**: Valuing the Wind: Renewable Energy and Environmental Externalities
- **Journal**: AER
- **Method**: Time Series OLS with Newey-West HAC

## Baseline Specification
Hourly OLS: wind generation on CO2 emissions, daily + hourly FE, cubic load controls, NW 24 lags

**Baseline spec IDs**: baseline, baseline_nox, baseline_so2

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 3 | 3.6% |
| Core | 62 | 73.8% |
| Non-core | 19 | 22.6% |
| **Total** | **84** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 3 |
| core/controls | 12 |
| core/funcform | 1 |
| core/inference | 2 |
| core/other | 6 |
| core/sample | 41 |
| non_core/alt_outcome | 14 |
| non_core/diagnostics | 5 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 84
- Unique spec identifiers classified: 84
- Baseline specifications identified: 3
- Core specifications: 62
- Non-core specifications: 19

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
