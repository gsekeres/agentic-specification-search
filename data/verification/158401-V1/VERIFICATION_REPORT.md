# Verification Report: 158401-V1

**Paper**: Bold, Ghisolfi, Nsonzi & Svensson, "Market Access and Quality Upgrading: Evidence from Four Field Experiments"
**Design**: Cluster-randomized experiment (village-level, ITT with OLS ANCOVA)
**Treatment**: buy_treatment (12 of 20 villages assigned to market access)

## Baseline Groups

### G1: Market Access -> Investment (Table 5 Panel A)
- **Claim**: Village-level assignment to a quality-maize market increases farmer investment in maize quality
- **Expected sign**: positive
- **Baseline spec_run_ids**: 158401-V1__001 through 158401-V1__008
- **Baseline spec_ids**: baseline__t5a_col1_fert_seeds, baseline__t5a_col2_inputs, baseline__t5a_col3_tarpaulin, baseline__t5a_col4_sort, baseline__t5a_col5_winnow, baseline__t5a_col6_labor_pre, baseline__t5a_col7_postharvest, baseline__t5a_col8_labor_post
- **Focal outcome**: expenses_fert_seeds (coef=2.37, SE=1.11, p=0.045, N=658)
- **Outcomes**: 8 investment outcomes (seeds/fertilizer expenses, total input expenses, tarpaulin drying, sorting, winnowing, pre-harvest labor, post-harvest expenses, post-harvest labor)

### G2: Market Access -> Productivity/Income (Table 6 Panel A)
- **Claim**: Village-level assignment to a quality-maize market increases farmer productivity and income
- **Expected sign**: positive
- **Baseline spec_run_ids**: 158401-V1__009 through 158401-V1__016
- **Baseline spec_ids**: baseline__t6a_col1_price, baseline__t6a_col2_acreage, baseline__t6a_col3_harvest, baseline__t6a_col4_yield, baseline__t6a_col5_harvest_value, baseline__t6a_col6_expenses, baseline__t6a_col7_surplus, baseline__t6a_col8_surplus_hrs
- **Focal outcome**: surplus (coef=65.82, SE=32.26, p=0.055, N=628)
- **Outcomes**: 8 productivity/income outcomes (price, acreage, harvest, yield, harvest value, expenses, surplus, surplus incl. hours)

## Summary Counts

| Metric | Count |
|--------|-------|
| **Total rows** | 70 |
| **Valid (run_success=1)** | 58 |
| **Invalid (run_success=0)** | 12 |
| **Core (valid + core-eligible)** | 58 |
| **Non-core** | 0 |
| **Baselines** | 16 (8 per group) |

### By category

| Category | Count | Valid | Invalid |
|----------|-------|-------|---------|
| core_method | 28 | 22 | 6 |
| core_sample | 22 | 22 | 0 |
| core_controls | 12 | 6 | 6 |
| core_funcform | 8 | 8 | 0 |

## Sanity Checks

1. **spec_run_id uniqueness**: PASS -- all 70 unique
2. **baseline_group_id**: PASS -- all rows in G1 or G2, matching surface exactly
3. **Surface alignment**: PASS -- 70 rows match surface enumeration (surface targeted 71; 1 fewer due to spec_enumeration rounding, actual count matches SPECIFICATION_SEARCH.md)
4. **Baseline spec_ids**: PASS -- all 16 surface baseline spec_ids present, all successful
5. **Design/RC spec_ids**: PASS -- all surface-planned spec_ids present in results
6. **spec_tree_path anchors**: PASS -- all spec_tree_path values include #anchor fragments
7. **coefficient_vector_json structure**: PASS -- all run_success=1 rows contain required keys (coefficients, inference, software, surface_hash); all run_success=0 rows contain error and error_details
8. **Inference canonical**: PASS -- all run_success=1 rows use infer/se/cluster/village as canonical inference
9. **Functional form blocks**: PASS -- all rc/form/* rows contain functional_form object with matching spec_id
10. **RC axis blocks**: PASS -- all rc/controls/* rows contain controls block; all rc/sample/* rows contain sample block
11. **No infer/* in specification_results.csv**: PASS -- inference variants correctly in inference_results.csv only
12. **Numeric fields finite**: PASS -- all finite for run_success=1; all NaN for run_success=0
13. **No binary outcomes transformed**: PASS -- asinh/log1p applied only to continuous monetary outcomes
14. **run_error present for failures**: PASS -- all 12 failed rows have non-empty run_error

## Inference Variants (inference_results.csv)

- 18 rows total, all run_success=1
- 16 rows: infer/se/hc/hc1 (HC1 robust SE without clustering, one per baseline)
- 2 rows: infer/ri/fisher/permutation (expenses_fert_seeds and surplus only)
- All inference_run_ids unique; all spec_run_ids link to valid specification_results rows
- Fisher p-values: expenses_fert_seeds p=0.046, surplus p=0.070 (note: surplus Fisher p=0.070 vs clustered p=0.055)

## Issues and Observations

### Issue 1: 12 failed specifications (all HH characteristics variable mismatch)

All 12 failures (6 design/with_covariates + 6 rc/controls/extended_hh_chars) share the same root cause: household characteristics variables (mdm_female, mdm_primary, hhr_n, distance_kakumiro, main_road_min) were stored with a `hh_` prefix during data construction, creating a column-name mismatch that resulted in 0 observations after listwise deletion. These are correctly logged with run_success=0 and appropriate error/error_details in coefficient_vector_json.

**Impact**: This eliminates 12 of 70 specs (17%), all in the "extended controls" axis. The core robustness assessment loses one full control-set variant. However, since with_covariates and extended_hh_chars would produce identical models (both add the same 5 HH characteristics), the effective loss is 6 unique models.

### Issue 2: Duplicate result (158401-V1__049)

Row 158401-V1__049 (rc/sample/time/drop_first_post_season for expenses_postharvest) is numerically identical to baseline 158401-V1__007: same coefficient (6.025), same SE (5.011), same p-value (0.244), same N (482). This occurs because expenses_postharvest is only measured in post-harvest seasons (seasons 5-7), and the first post-treatment season (season 4) has no postharvest data, making the drop a no-op. The row is valid but informationally redundant.

### Issue 3: Incomplete Fisher randomization inference

The surface planned Fisher permutation tests for all 16 baselines, but only 2 were completed (expenses_fert_seeds and surplus, one per group as primary focal outcomes). The remaining 14 were not computed due to computational cost.

### Issue 4: Known design/RC redundancy

The surface notes that design/with_covariates and rc/controls/extended_hh_chars specify identical models. Since both failed, this redundancy has no practical impact on the completed run, but should be resolved in any re-run.

## Sign and Significance Patterns

### G1: Investment Outcomes (35 specs, 29 valid)

- **All positive**: 29/29 valid specs (100%)
- **Significant at 5%**: 12/29 (41%)
- **Significant at 10%**: 13/29 (45%)

By focal outcome (including functional form transforms):
| Outcome | N valid | p < 0.05 | Notes |
|---------|---------|----------|-------|
| tarpaulin_d | 6 | 6/6 (100%) | Most robust G1 outcome; coef range [0.222, 0.260] |
| expenses_fert_seeds (incl asinh/log1p) | 9 | 4/9 (44%) | Sensitive to trimming (p=0.17), time window, ANCOVA |
| expenses_postharvest (incl asinh/log1p) | 9 | 0/9 (0%) | Never significant; always positive but imprecise |
| sort_d | 1 | 1/1 | Baseline only; p=0.002 |
| winnow_d | 1 | 1/1 | Baseline only; p=0.033 |
| expenses_inputs | 1 | 0/1 | Baseline only; p=0.075 |
| expenses_labor_preharvest | 1 | 0/1 | Baseline only; p=0.275 |
| expenses_labor_postharvest | 1 | 0/1 | Baseline only; p=0.145 |

### G2: Productivity/Income Outcomes (35 specs, 29 valid)

- **All positive**: 29/29 valid specs (100%)
- **Significant at 5%**: 11/29 (38%)
- **Significant at 10%**: 19/29 (66%)

By focal outcome (including functional form transforms):
| Outcome | N valid | p < 0.05 | Notes |
|---------|---------|----------|-------|
| yield | 6 | 6/6 (100%) | Most robust G2 outcome; coef range [111, 119] kg/acre |
| surplus (incl asinh/log1p) | 9 | 3/9 (33%) | Marginal baseline (p=0.055); trimming helps (p=0.026) |
| harvest_value (incl asinh/log1p) | 9 | 0/9 (0%) | Always positive, p range [0.06, 0.24] |
| price | 1 | 1/1 | Baseline only; p=0.001 |
| surplus_hrs | 1 | 1/1 | Baseline only; p=0.029 |
| acreage | 1 | 0/1 | Baseline only; p=0.829 (null effect) |
| harvest_kg_tot | 1 | 0/1 | Baseline only; p=0.308 |
| expenses | 1 | 0/1 | Baseline only; p=0.307 |

## Assessment

**MODERATE robustness** overall. The treatment effect sign is uniformly positive across all 58 valid specifications in both groups (100%). Statistical significance varies by outcome:

- **Strong for behavioral/practice outcomes**: tarpaulin_d (G1) and yield (G2) are consistently significant across all specifications.
- **Weaker for monetary outcomes**: expenses_fert_seeds (G1 focal, p=0.045 at baseline) and surplus (G2 focal, p=0.055 at baseline) are sensitive to ANCOVA inclusion, outlier trimming, time window restrictions, and panel balance requirements.
- The pattern aligns with the paper's narrative: market access robustly shifts farming practices, while monetary outcomes have higher variance and more sensitivity to specification choices.

## Recommendations

1. **Fix HH characteristics variable naming**: The `hh_` prefix mismatch should be corrected in the runner script to recover the 12 failed specifications. This is a data construction bug, not a fundamental issue.
2. **Complete Fisher randomization inference**: With only 20 clusters (12T/8C), clustered SEs may be unreliable. Fisher exact inference is the gold standard for this design and should be run for all 16 baselines.
3. **Resolve with_covariates/extended_hh_chars redundancy**: These specify identical models. One could be dropped or explicitly documented as a cross-check.
4. **Consider wild cluster bootstrap**: Cameron, Gelbach & Miller (2008) wild cluster bootstrap would complement HC1 and Fisher tests given the small cluster count.
5. **Note the drop_first_post_season no-op for expenses_postharvest**: This spec is valid but uninformative. Could add a note or skip it for outcomes that are only measured in later seasons.

## Validation Script Output

`python scripts/validate_agent_outputs.py --paper-id 158401-V1` reports 359 ERROR, 0 WARN. All errors are pre-existing issues in the runner's source files (specification_results.csv, inference_results.csv), not in the verification outputs:

1. **design_audit_missing_key** (~358 errors): The `coefficient_vector_json.design.randomized_experiment` block omits 4 keys present in the surface's design_audit: `ancova_control`, `sample_filter`, `post_treatment_seasons`, `post_treatment_labels`. This affects all 58 successful rows (4 keys x ~58 rows + additional design variant rows). These keys are informational metadata and do not affect the regression estimates. Fix: the runner script should propagate all design_audit keys into the coefficient_vector_json.
2. **inference_results missing_columns** (1 error): inference_results.csv lacks `outcome_var`, `treatment_var`, and `cluster_var` columns. Fix: the runner script should include these columns in inference output.

Neither category affects the correctness of the verification classification (verification_baselines.json, verification_spec_map.csv).
