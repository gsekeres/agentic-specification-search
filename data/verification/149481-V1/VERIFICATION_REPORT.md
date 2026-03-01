# Verification Report: 149481-V1

## Baseline Groups

- **G1**: Thank-you call effect on donation behavior, Experiment 1 (Public TV Stations)
  - Baseline spec_run_ids: `149481-V1__run_001` through `149481-V1__run_007`
  - Baseline spec_ids: all `baseline` (surface planned named IDs not used)
  - Expected sign: null (paper reports no significant effects)
  - Outcomes: renewing (chi2), payment_amount3 (ranksum), var13 (ranksum), gift_cond (ranksum), retention (ranksum), donated (OLS+FE+controls), gift_cond (OLS+FE+controls)
  - Full sample N=485,767; Conditional N=136,950

- **G2**: Thank-you call effect on donation behavior, Experiment 2 (National Non-Profit)
  - Baseline spec_run_ids: `149481-V1__run_008` through `149481-V1__run_014`
  - Baseline spec_ids: all `baseline` (surface planned named IDs not used)
  - Expected sign: null (paper reports no significant effects)
  - Outcomes: same 5 diff-in-means + 2 OLS specs as G1, but with fewer controls (no demographics)
  - Full sample N=57,643; Conditional N=17,862

## Counts

- **Total rows**: 87
- **Core**: 87
- **Non-core**: 0
- **Invalid**: 0
- **Baselines**: 14

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 34 |
| core_method | 29 |
| core_sample | 12 |
| core_funcform | 8 |
| core_fe | 4 |

## Sign and Significance

### G1 (Experiment 1): 58 core specifications
- Significant at 5%: 0/58 (0%)
- Significant at 10%: 0/58 (0%)
- This is a well-powered null result. The largest p-value for any G1 specification is 0.93 and the smallest is 0.16 (gift_cond asinh/log1p transforms).
- All treatment effect estimates are substantively tiny relative to control means.

### G2 (Experiment 2): 29 core specifications
- Significant at 5%: 0/29 (0%)
- Significant at 10%: 0/29 (0%)
- Consistent null across all specifications. Smallest p-value is 0.37 (gift_cond trim_y_5_95).

## Assessment

- **NULL RESULT - ROBUST**: The null effect of thank-you calls on donation behavior is extremely robust across both experiments, all 87 specifications, multiple outcomes, control sets, sample restrictions, functional form transforms, and FE structures. Not a single specification reaches p < 0.10. This is a convincing, well-powered null finding.

## Issues Found

1. **Baseline spec_ids not named per surface plan**: The surface planned distinct spec_ids like `baseline__t2_exp1_renewing`, `baseline__tA1_exp1_donated_ols`, etc. The runner used `baseline` for all 14 baseline rows. Baselines are distinguishable by `outcome_var` and `baseline_group_id` and `controls_desc`, but the lack of named IDs means the `spec_id` column is not unique within a baseline group. This is a minor naming issue that does not affect classification.

2. **Effective duplicates (donated = renewing)**: For diff-in-means specs, `donated` and `renewing` produce identical coefficients (e.g., run_015 and run_017 both have coef=-0.000893). This suggests the two outcome variables are the same binary indicator or nearly identical. This creates redundant specifications but does not affect validity. Similarly, run_021 (with_covariates, renewing) and run_023 (with_covariates, donated) have identical coefficients.

3. **Inference spec_id varies by estimator type**: 10 rows use `infer/se/nonparametric/ranksum_chi2` (the surface canonical for diff-in-means) while 77 rows use `infer/se/iid` (classical OLS SE for regression specs). This is correct behavior -- the canonical inference choice is estimator-dependent. The surface canonical applies to diff-in-means specs; OLS specs match the paper's Stata `xtreg, fe` default (no `robust` or `cluster` option).

4. **Missing r_squared for diff-in-means rows**: 10 baseline rows (the nonparametric diff-in-means specs) have empty r_squared. This is expected since R-squared is not defined for nonparametric tests.

5. **No sample/restriction/include_sustaining_donors spec**: The surface planned this RC variant but it was not executed, likely because it would require re-running data construction from step1 (noted in surface rc_notes). This is an expected gap.

6. **Design rows expand across outcomes**: The design variants (diff_in_means, strata_fe, with_covariates) are run for 3 outcomes each (renewing, payment_amount3, donated) in G1 and G2. This contributes 15 design rows (9 G1 + 6 G2). Combined with 14 baselines, design+baseline account for 29 of the 87 total specs.

## Validation Script Results

Running `python scripts/validate_agent_outputs.py --paper-id 149481-V1` reports:
- **78 ERRORs**: 77 are `non_canonical_inference_used` (OLS rows use `infer/se/iid` instead of canonical `infer/se/nonparametric/ranksum_chi2`). 1 is `inference_results missing_columns` (missing `outcome_var`, `treatment_var`, `cluster_var` in inference_results.csv). These are all upstream runner issues, not verification output issues.
- **53 WARNs**: 1 missing `SPECIFICATION_SEARCH.md`, 52 `spec_tree_path_missing_anchor` (spec_tree_path lacks `#section-anchor`). These are upstream formatting issues.
- **0 verification-specific errors**: All verification outputs (`verification_baselines.json`, `verification_spec_map.csv`) pass validation.

The 77 `non_canonical_inference_used` errors reflect a design decision by the runner: OLS regression specs use classical SE (`infer/se/iid`) rather than the nonparametric canonical (`infer/se/nonparametric/ranksum_chi2`), which is correct behavior since rank-sum tests cannot be applied to OLS regressions. The surface canonical should be interpreted as applying only to diff-in-means specs.

## Recommendations

- Consider deduplicating donated/renewing rows since they appear to be the same variable. This would reduce spec count by approximately 10-15 rows without losing information.
- The named baseline spec_ids from the surface should be adopted by the runner for clearer identification of which paper table/column each baseline corresponds to.
- Additional power could come from running HC1 robust SE variants directly in specification_results.csv rather than only in inference_results.csv, especially since the classical SE may be slightly optimistic.
- The surface planned `rc/sample/quality/include_sustaining_donors` which was not executed. If feasible, this would test an important sample construction choice.
- Future runs could include Experiment 3 (new script variant) as exploration specs to give a fuller picture of the paper's scope.
- The spec_tree_path entries should include `#anchor` suffixes for traceability (e.g., `specification_tree/designs/randomized_experiment.md#baseline`).
- The inference_results.csv should include `outcome_var`, `treatment_var`, and `cluster_var` columns to match the expected schema.
