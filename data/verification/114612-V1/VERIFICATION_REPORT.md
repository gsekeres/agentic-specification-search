# Verification Report: 114612-V1

## Paper Information
- **Paper ID**: 114612-V1
- **Title**: ACA Health Insurance: Adverse Selection via Website Glitches
- **Journal**: AER
- **Method**: Difference-in-Differences

## Baseline Specification
TWFE DiD: Glitch vs No-Glitch states, state + date FE, clustered at state

**Baseline spec IDs**: baseline

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 1 | 1.3% |
| Core | 52 | 69.3% |
| Non-core | 22 | 29.3% |
| **Total** | **75** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 1 |
| core/combined | 11 |
| core/controls | 15 |
| core/fe | 5 |
| core/funcform | 3 |
| core/inference | 2 |
| core/other | 5 |
| core/sample | 11 |
| non_core/alt_outcome | 7 |
| non_core/alt_treatment | 6 |
| non_core/diagnostics | 1 |
| non_core/heterogeneity | 4 |
| non_core/placebo | 4 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 75
- Unique spec identifiers classified: 75
- Baseline specifications identified: 1
- Core specifications: 52
- Non-core specifications: 22

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
