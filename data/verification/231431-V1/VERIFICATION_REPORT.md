# Verification Report: 231431-V1

## Paper Information
- **Paper ID**: 231431-V1
- **Title**: Reported Effects vs Revealed-Preference Estimates: Tax Rebate MPC
- **Journal**: AER: Insights
- **Method**: Instrumental Variables (2SLS)

## Baseline Specification
2SLS: ESP amount on nondurable consumption, yymm FE, clustered by household

**Baseline spec IDs**: baseline, baseline_total_consumption

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 2 | 2.2% |
| Core | 54 | 60.7% |
| Non-core | 33 | 37.1% |
| **Total** | **89** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 2 |
| core/controls | 11 |
| core/fe | 1 |
| core/funcform | 13 |
| core/inference | 3 |
| core/instruments | 4 |
| core/method | 7 |
| core/sample | 15 |
| non_core/alt_outcome | 16 |
| non_core/diagnostics | 5 |
| non_core/heterogeneity | 12 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 89
- Unique spec identifiers classified: 89
- Baseline specifications identified: 2
- Core specifications: 54
- Non-core specifications: 33

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
