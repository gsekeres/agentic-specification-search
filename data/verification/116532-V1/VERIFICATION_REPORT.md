# Verification Report: 116532-V1

## Paper Information
- **Paper ID**: 116532-V1
- **Title**: Are Tax Incentives for Charitable Giving Efficient?
- **Journal**: AEJ: Policy
- **Method**: Difference-in-Differences (OLS)

## Baseline Specification
DiD: Tax reform impact on log donations, groupe + year FE, clustered at groupe

**Baseline spec IDs**: baseline

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 1 | 1.1% |
| Core | 66 | 71.7% |
| Non-core | 25 | 27.2% |
| **Total** | **92** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 1 |
| core/controls | 15 |
| core/fe | 4 |
| core/funcform | 6 |
| core/inference | 5 |
| core/method | 15 |
| core/sample | 20 |
| core/weights | 1 |
| non_core/alt_outcome | 4 |
| non_core/alt_treatment | 4 |
| non_core/heterogeneity | 13 |
| non_core/placebo | 4 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 92
- Unique spec identifiers classified: 92
- Baseline specifications identified: 1
- Core specifications: 66
- Non-core specifications: 25

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
