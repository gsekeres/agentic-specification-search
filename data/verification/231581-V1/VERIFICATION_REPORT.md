# Verification Report: 231581-V1

## Paper Information
- **Paper ID**: 231581-V1
- **Title**: Teacher Value-Added, Salary, and Contract Status in Pakistan
- **Journal**: AER
- **Method**: Cross-sectional OLS with absorbed FE

## Baseline Specification
OLS: TVA on log salary, district FE, clustered at school group

**Baseline spec IDs**: baseline

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 1 | 1.1% |
| Core | 59 | 66.3% |
| Non-core | 29 | 32.6% |
| **Total** | **89** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 1 |
| core/controls | 18 |
| core/fe | 2 |
| core/funcform | 5 |
| core/inference | 5 |
| core/method | 1 |
| core/sample | 17 |
| core/treatment | 11 |
| non_core/alt_outcome | 18 |
| non_core/heterogeneity | 6 |
| non_core/placebo | 5 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 89
- Unique spec identifiers classified: 89
- Baseline specifications identified: 1
- Core specifications: 59
- Non-core specifications: 29

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
