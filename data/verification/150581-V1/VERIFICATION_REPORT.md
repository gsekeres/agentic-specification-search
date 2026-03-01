# Verification Report: 150581-V1

## Baseline Groups
- **G1**: Wage cyclicality with job transitions and skill mismatch
  - Baseline spec_run_ids: ['150581-V1_run_001', '150581-V1_run_002', '150581-V1_run_003', '150581-V1_run_004', '150581-V1_run_005']
  - Baseline spec_ids: ['baseline', 'baseline__table2_col1', 'baseline__table2_col2', 'baseline__table2_col3', 'baseline__table2_col5']
  - Expected sign: negative (unemployment reduces wages)

## Counts
- **Total rows**: 40
- **Core**: 40 (conceptually; all invalid due to run failure)
- **Non-core**: 0
- **Invalid**: 40
- **Baselines**: 5

## Category Breakdown
| Category | Count |
|----------|-------|
| core_controls | 24 |
| core_method | 5 |
| core_sample | 4 |
| core_data | 4 |
| core_fe | 3 |

## Sign and Significance
- No results available: all 40 specifications have run_success=0.

## Assessment
- **FAILED RUN**: No specifications produced valid estimates.
- The failure is uniform and systematic: the raw NLSY79 data requires Stata-specific data construction (converting weekly labor status files to monthly panel, constructing job transition indicators, computing skill mismatch measures). The Python runner cannot perform these steps.
- The surface plan is coherent: 5 baseline specs (Table 2 Cols 1-5), 5 LOO specs, 5 control set variants, 4 progressions, 10 random subsets, 2 sample restrictions, 2 outlier trims, 3 FE variants, 4 data construction variants. This covers the standard robustness axes well.
- The spec_id assignments match the surface core_universe exactly.

## Issues
1. **Complete data construction failure**: All rows fail with the same error about requiring Stata to process raw NLSY .dct files. This is not a runner bug but a data availability issue.
2. **No numerical verification possible**: Cannot verify coefficients, signs, or significance.
3. **Inference results also minimal**: inference_results.csv has 2 rows (HC1 and two-way cluster), but these likely also failed.

## Recommendations
- This paper requires either (a) a Stata-based runner or (b) pre-processed analysis-ready data files to be provided.
- If Stata is available, re-run the full specification search from the existing surface plan.
- Consider flagging this paper as "data-blocked" in tracking metadata.
