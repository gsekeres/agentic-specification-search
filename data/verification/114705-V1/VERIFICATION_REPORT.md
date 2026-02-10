# Verification Report: 114705-V1

## Paper Information
- **Paper ID**: 114705-V1
- **Title**: The Role of Information in Disability Insurance Application
- **Journal**: AEJ: Economic Policy
- **Method**: Difference-in-Differences

## Baseline Specification
DiD: SS Statement receipt on disability application, age + year FE, clustered by birth year, WLS

**Baseline spec IDs**: baseline

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 1 | 1.4% |
| Core | 48 | 69.6% |
| Non-core | 20 | 29.0% |
| **Total** | **69** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 1 |
| core/controls | 12 |
| core/fe | 6 |
| core/funcform | 3 |
| core/inference | 3 |
| core/method | 5 |
| core/sample | 18 |
| core/weights | 1 |
| non_core/alt_outcome | 3 |
| non_core/heterogeneity | 11 |
| non_core/placebo | 6 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 69
- Unique spec identifiers classified: 69
- Baseline specifications identified: 1
- Core specifications: 48
- Non-core specifications: 20

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
