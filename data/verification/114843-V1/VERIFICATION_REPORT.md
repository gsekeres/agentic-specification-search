# Verification Report: 114843-V1

## Paper Information
- **Paper ID**: 114843-V1
- **Title**: On the Empirics of EU Regional Policy: Objective 1 on Growth
- **Journal**: AER
- **Method**: Fuzzy RDD (IV-Probit)

## Baseline Specification
IV-Probit: Objective 1 eligibility on growth, polynomial control function, clustered by region

**Baseline spec IDs**: baseline/pooled_iv/growth_hc_poly3

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 9 | 6.9% |
| Core | 48 | 36.6% |
| Non-core | 74 | 56.5% |
| **Total** | **131** | **100%** |

## Detailed Category Breakdown

| Classification/Category | Count |
|------------------------|-------|
| baseline/baseline | 9 |
| core/controls | 6 |
| core/funcform | 11 |
| core/inference | 4 |
| core/method | 8 |
| core/other | 4 |
| core/sample | 15 |
| non_core/alt_outcome | 12 |
| non_core/diagnostics | 18 |
| non_core/heterogeneity | 42 |
| non_core/placebo | 2 |

## Classification Definitions

- **Baseline**: The original paper specification(s) as reported in the primary results tables.
- **Core**: Variations that test the same claim through different controls, sample restrictions, inference methods, fixed effects structures, or functional forms. These are the "robustness" specs.
- **Non-core**: Specifications that test different claims or serve diagnostic purposes, including placebo tests, alternative outcomes, alternative treatments, heterogeneity analyses, and balance/diagnostic checks.

## Verification Notes

- Total specifications in CSV: 131
- Unique spec identifiers classified: 131
- Baseline specifications identified: 9
- Core specifications: 48
- Non-core specifications: 74

## Files Generated
- `verification_baselines.json`: Baseline identification and paper metadata
- `verification_spec_map.csv`: Full classification of all specifications
- `VERIFICATION_REPORT.md`: This report
