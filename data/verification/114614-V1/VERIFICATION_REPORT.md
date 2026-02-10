# Verification Report: 114614-V1

## Paper Information
- **Paper ID**: 114614-V1
- **Title**: Training Microentrepreneurs (SEWA Business Training RCT)
- **Journal**: AEJ: Economic Policy
- **Method**: OLS with clustered SE (RCT)

## Baseline Specification
Pooled treatment (Treated_All), center*baseline + month FE, clustered at training group

**Baseline spec IDs**: baseline_pooled, baseline_peer

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 75 | 5.3% |
| Core | 533 | 37.6% |
| Non-core | 809 | 57.1% |
| **Total** | **1417** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 75 |
| core/controls | 156 |
| core/fe | 175 |
| core/funcform | 43 |
| core/inference | 100 |
| core/sample | 9 |
| core/treatment | 50 |
| non_core/alt_treatment | 25 |
| non_core/heterogeneity | 784 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 1417
- Unique spec identifiers classified: 1417
- Baseline specifications identified: 75
- Core specifications: 533
- Non-core specifications: 809

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
