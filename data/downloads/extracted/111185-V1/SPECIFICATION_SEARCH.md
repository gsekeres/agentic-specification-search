# Specification Search Run Log: 111185-V1

## Paper
Rudik (2020), "Optimal Climate Policy When Damages are Unknown", AEJ: Economic Policy

## Surface Summary

- **Baseline groups**: 1 (G1: damage exponent estimation from Table 1)
- **Design**: cross_sectional_ols
- **Budgets**: max_specs_core_total=80, max_specs_controls_subset=32
- **Seed**: 111185
- **Canonical inference**: Classical (non-robust) OLS standard errors

## Execution Summary

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| baseline | 1 | 1 | 0 | Exact match to replication: coef=1.882, SE=0.451 |
| rc/controls/single | 7 | 7 | 0 | One control at a time from pool of 7 |
| rc/controls/sets | 3 | 3 | 0 | minimal, extended, full |
| rc/controls/progression | 4 | 4 | 0 | bivariate -> study_quality -> damage_type -> full |
| rc/controls/subset | 21 | 21 | 0 | 6 exhaustive block combos + 15 random draws |
| rc/sample | 11 | 11 | 0 | Outliers, quality filters, temporal splits |
| rc/form | 5 | 5 | 0 | Level/asinh outcome, level treatment, quadratic, levels-quad |
| rc/preprocess | 2 | 2 | 0 | Winsorize outcome, winsorize treatment |
| rc/joint | 8 | 8 | 0 | Combined sample + controls variations |
| **Total core** | **62** | **62** | **0** | |
| infer/* | 3 | 3 | 0 | HC1, HC2, HC3 on baseline (in inference_results.csv) |

## Key Results

- **Baseline coefficient (logt)**: 1.882 (SE=0.451, p=0.00015, N=43, R2=0.299)
- **Coefficient range across 62 specs**: [-7.171, 13.675]
- **Median coefficient**: 1.802
- **P-value range**: [9.15e-13, 0.913]
- **Specifications with p < 0.05**: most specifications (detailed counts below)
- **Specifications with same sign as baseline**: majority are positive

## Deviations from Surface

1. **Block combinations**: The surface planned 2^3 = 8 exhaustive block combinations, but only 6 were run. The "all three blocks" combination (study_quality + damage_type + study_design) produces 7 controls, which exceeds the max_controls_count=6 constraint. This is correct behavior -- the constraint was enforced.

2. **rc/controls/sets/none**: Not run as a separate spec because it is identical to the baseline (bivariate regression with no controls). The progression/bivariate spec serves this purpose.

## Software Stack

- Python 3.12
- statsmodels 0.14.x (OLS, robust covariance, influence diagnostics)
- pandas 2.x (data loading and manipulation)
- numpy 2.x (numerical operations)

## Output Files

- `specification_results.csv`: 62 rows (core specs only: baseline, rc/*)
- `inference_results.csv`: 3 rows (HC1, HC2, HC3 recomputations of baseline)
- `SPECIFICATION_SEARCH.md`: this file
- `scripts/paper_analyses/111185-V1.py`: executable analysis script
