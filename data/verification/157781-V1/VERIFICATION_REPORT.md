# Verification Report: 157781-V1

## Paper Summary

This paper studies the effect of the Grand Canal closure on rebellion incidence in imperial China using a difference-in-differences design. The treatment is proximity to the Grand Canal interacted with the post-closure period (1826 onwards). The outcome is asinh-transformed rebellion onset per capita (1600 population). The design exploits the sharp, simultaneous canal closure affecting all canal-adjacent counties.

## Baseline Groups

- **G1**: DiD effect of canal closure on rebellion in canal-adjacent vs non-adjacent counties
  - Baseline spec_run_ids: 157781-V1_run_001 through 157781-V1_run_005
  - Baseline spec_ids: baseline__table3_col1, baseline__table3_col2, baseline__table3_col3, baseline__table3_col4, baseline__table3_col5
  - Preferred specification: baseline__table3_col4 (full FE, no controls; coef=0.0422, SE=0.0175, p=0.016)
  - Expected sign: positive (canal closure increases rebellion)
  - Note: baseline__table3_col5 failed (singular matrix when all 15 controls added with full FE)

## Counts

- **Total rows**: 47
- **Valid (run_success=1)**: 31
- **Invalid (run_success=0)**: 16
- **Core tests (valid and core-eligible)**: 31
- **Non-core**: 0
- **Baselines**: 5 (4 valid, 1 failed)

## Category Breakdown

| Category | Total | Valid | Failed |
|----------|-------|-------|--------|
| core_method | 6 | 5 | 1 |
| core_controls | 20 | 16 | 4 |
| core_fe | 4 | 4 | 0 |
| core_sample | 11 | 0 | 11 |
| core_funcform | 6 | 6 | 0 |

## Sign and Significance (31 valid specs)

- All 31 valid specifications have a **positive** coefficient (100%)
- **p < 0.05**: 25/31 (81%)
- **p < 0.10**: 30/31 (97%)
- **p < 0.15**: 31/31 (100%)
- Coefficient range (same-outcome specs, ashonset_cntypop1600): [0.0322, 0.0443]
- The single spec with p > 0.10 is rc/fe/add/prefid_year (prefecture x year FE, p=0.113), which absorbs much of the identifying variation

## Duplicate Estimates

Several RC specs mechanically reproduce baseline columns due to the paper's FE-progression structure:
- run_021 (rc/controls/sets/none) = run_004 (baseline col 4): both use full FE, no controls
- run_026 (rc/fe/drop/ashprerebels_year) = run_001 (baseline col 1): both use county + year FE only
- run_027 (rc/fe/drop/provid_year) = run_002 (baseline col 2): both use county + year + prerebels*year FE
- run_028 (rc/fe/drop/prefid_trend) = run_003 (baseline col 3): both use county + provid*year + prerebels*year FE
- run_047 (design/did/twfe) = run_004 (baseline col 4): explicit TWFE confirmation

These duplicates are not errors; they reflect the mechanical overlap between the FE drop/add robustness and the paper's progressive FE structure. They correctly confirm internal consistency.

## Failure Analysis

### Singular matrix errors (4 failures)
- run_005 (baseline__table3_col5): full FE + all 15 controls causes collinearity
- run_006 (rc/controls/loo/drop_larea_after): same issue, 14 of 15 controls
- run_007 (rc/controls/loo/drop_rug_after): same issue
- run_014 (rc/controls/loo/drop_lpopdencnty1600_after): same issue
- run_025 (rc/controls/sets/full_controls): same issue

These failures indicate that the full control set + full FE structure creates perfect collinearity in pyfixest. The paper's original Stata implementation may handle this differently (e.g., Stata's reghdfe drops collinear variables automatically). This is a runner implementation issue, not a specification validity issue.

### Duplicate column name errors (11 failures)
- All 10 sample restriction specs (run_030 through run_039) failed with "Expected unique column names" error
- run_046 (outlier trim) failed with the same error
- The error refers to duplicate `apr_YYYY` columns, suggesting the data merging step in the runner script creates duplicated time-period columns when subsetting the sample

This is a systematic runner bug affecting all sample restriction and outlier trim specifications. The data construction step likely performs a merge that duplicates the ashprerebels*year interaction columns.

## Inference Variants (inference_results.csv)

Three inference variants were run on the preferred baseline (run_004):
- **infer/se/cluster/prefid**: coef=0.0422, SE=0.0208, p=0.046 (significant at 5%)
- **infer/se/cluster/provid**: coef=0.0422, SE=0.0286, p=0.200 (not significant; only ~6 province clusters)
- **infer/se/hetero/hc1**: coef=0.0422, SE=0.0126, p=0.001 (highly significant without clustering)

The province-level clustering (p=0.200) is expected to fail given the very small number of clusters (~6 provinces). The result is robust to prefecture-level clustering and heteroskedasticity-robust SEs.

## Surface Alignment

- The surface planned 1 baseline group (G1) with 5 baseline specs and 41 RC specs plus 1 design spec. All 47 planned specs were executed.
- No extra specs were executed beyond the surface plan.
- No baseline group reassignment was needed.
- The surface correctly identified baseline__table3_col4 as the preferred/primary specification.

## Assessment

**MODERATE-TO-STRONG robustness**: Among the 31 valid specifications, the positive effect of canal closure on rebellion is remarkably consistent. All coefficients are positive, 97% are significant at the 10% level, and 81% at the 5% level. The result survives all controls variations, FE variations, and outcome transformations. However, 16/47 specifications (34%) failed due to runner implementation issues, notably all sample restriction specs. The high failure rate limits the breadth of the robustness check, particularly along the sample restriction dimension.

## Recommendations

1. **Fix runner data construction**: The duplicate column name bug prevents all sample restriction and outlier trim specs from running. The merge step that creates the panel-level `apr_YYYY` interaction variables needs to be debugged (likely a merge creating duplicate columns from the ashprerebels*year interactions).
2. **Fix singular matrix issue**: The full-controls specs fail with collinearity. Consider using pyfixest's collinearity handling or dropping collinear variables before estimation, matching Stata's reghdfe behavior.
3. **Add Conley spatial HAC SEs**: The surface planned spatial HAC SEs (500km cutoff, 262-year lag) as an inference variant, but these were not executed. This would be valuable given the spatial structure of the data.
4. **Consider re-running with Stata**: Given the systematic pyfixest failures, a Stata-based runner might recover the 16 failed specifications.
