# Verification Report: 114643-V1

## Paper Information
- **Paper ID**: 114643-V1
- **Title**: Immigration, Employment Opportunities, and Criminal Behavior
- **Journal**: AEJ: Economic Policy
- **Method**: Triple-Difference (DiD)

## Baseline Specification
Triple-diff: hisp x imm_index x post-IRCA, clustered at block group

**Baseline spec IDs**: baseline, baseline/exLAW, baseline/exSAW

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 3 | 2.5% |
| Core | 64 | 53.8% |
| Non-core | 52 | 43.7% |
| **Total** | **119** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 3 |
| core/controls | 4 |
| core/funcform | 16 |
| core/inference | 3 |
| core/other | 3 |
| core/sample | 18 |
| core/treatment | 18 |
| core/weights | 2 |
| non_core/alt_outcome | 21 |
| non_core/alt_treatment | 1 |
| non_core/diagnostics | 1 |
| non_core/heterogeneity | 28 |
| non_core/placebo | 1 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 119
- Unique spec identifiers classified: 119
- Baseline specifications identified: 3
- Core specifications: 64
- Non-core specifications: 52

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
