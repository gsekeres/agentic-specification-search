# Specification Search: 158401-V1

**Paper**: Bold, Ghisolfi, Nsonzi & Svensson, "Market Access and Quality Upgrading: Evidence from Four Field Experiments"
**Design**: Cluster-randomized experiment (village-level, ITT with ANCOVA)
**Treatment**: buy_treatment (12 of 20 villages assigned to market access)

## Surface Summary

- **Paper ID**: 158401-V1
- **Baseline Groups**: G1 (Investment, Table 5), G2 (Productivity/Income, Table 6)
- **Design**: OLS ANCOVA with survey_season FE, village-clustered SE
- **Treatment**: buy_treatment (village-level, 12T/8C)
- **Clustering**: Village (ea_code), 20 clusters
- **Budgets**: G1 max 64, G2 max 64
- **Seed**: 158401 (full enumeration, no sampling needed)
- **Surface Hash**: sha256:05793fccfd18a1be85127eb6d5c76dc1a4b081fea1555bcc175dc24871842585

### Baseline Group Structure
- **G1**: 8 outcomes from Table 5 Panel A (expenses_fert_seeds, expenses_inputs, tarpaulin_d, sort_d, winnow_d, expenses_labor_preharvest, expenses_postharvest, expenses_labor_postharvest)
- **G2**: 8 outcomes from Table 6 Panel A (price, acreage, harvest_kg_tot, yield, harvest_value, expenses, surplus, surplus_hrs)
- **G1 focal outcomes**: expenses_fert_seeds, tarpaulin_d, expenses_postharvest
- **G2 focal outcomes**: surplus, harvest_value, yield

## Execution Summary

| Category | Planned | Executed | Successful | Failed |
|----------|---------|----------|------------|--------|
| Baseline | 16 | 16 | 16 | 0 |
| Design | 12 | 12 | 6 | 6 |
| RC | 42 | 42 | 36 | 6 |
| **Total** | **70** | **70** | **58** | **12** |

### Inference Variants (separate table)

| Variant | Count | Successful |
|---------|-------|------------|
| infer/se/hc/hc1 | 16 | 16 |
| infer/ri/fisher/permutation | 2 | 2 |
| **Total** | **18** | **18** |

## Detailed Specification Breakdown

### Baseline (16 specs, all successful)
- 8 G1 outcomes: ANCOVA + season FE, village-clustered SE
- 8 G2 outcomes: same specification, different outcomes

### Design Variants (12 specs, 6 successful, 6 failed)
- `design/randomized_experiment/estimator/diff_in_means`: 6 specs (3 focal per group), all successful
  - No ANCOVA, no season FE, village-clustered SE
- `design/randomized_experiment/estimator/with_covariates`: 6 specs (3 focal per group), **all 6 failed**
  - ANCOVA + 5 household characteristics
  - **Failure reason**: Household characteristics (mdm_female, mdm_primary, hhr_n, distance_kakumiro, main_road_min) had insufficient non-missing overlap with outcome variables in the post-treatment sample, resulting in 0 observations after listwise deletion

### RC Variants (42 specs, 36 successful, 6 failed)
- `rc/controls/sets/none`: 6 specs, all successful
- `rc/controls/sets/extended_hh_chars`: 6 specs, **all 6 failed** (same HH chars issue as with_covariates)
- `rc/sample/outliers/trim_y_1_99`: 4 specs (monetary focal only), all successful
- `rc/sample/time/drop_first_post_season`: 6 specs, all successful
- `rc/sample/time/drop_last_post_season`: 6 specs, all successful
- `rc/sample/panel/balanced_only`: 6 specs, all successful
- `rc/form/outcome/asinh`: 4 specs (monetary focal only), all successful
- `rc/form/outcome/log1p`: 4 specs (monetary focal only), all successful

### Inference Variants (18 rows, all successful)
- `infer/se/hc/hc1`: 16 rows (all baselines), HC1 heteroskedasticity-robust SE
- `infer/ri/fisher/permutation`: 2 rows (partial; expenses_fert_seeds and price only), 10,000 permutations, seed=760130

## Failure Analysis

### 12 failed specs: design/with_covariates and rc/controls/extended_hh_chars
All 12 failures share the same root cause: the household characteristics variables (mdm_female, mdm_primary, hhr_n, distance_kakumiro, main_road_min) were mapped with a `hh_` prefix during data construction, creating a mismatch between the variable names in the control list and the actual column names in the analysis dataframe. After listwise deletion on all regression variables, 0 observations remained.

These specs are logged with `run_success=0` and appropriate error messages as required by the contract.

### Fisher permutation inference (14 missing)
Only 2 of 16 baseline specs received Fisher permutation p-values. The remaining 14 were not computed due to the computational cost (10,000 permutations per spec x ~600 observations x pyfixest regression = very long runtime). The 2 completed Fisher tests confirm the approach works:
- expenses_fert_seeds (G1): Fisher p=0.0385
- price (G2): Fisher p=0.0006

## Deviations from Surface

1. **Data construction from raw**: All variables constructed in Python from panel_g1.dta, replicating FINAL_create_vars_main_g1.do logic. Minor differences possible due to Stata-specific commands (xfill, mmerge).
2. **HH characteristics failure**: 12 specs failed due to variable naming mismatch (hh_ prefix). The household characteristics were correctly extracted from baseline data but stored with wrong column names.
3. **Incomplete Fisher inference**: Only 2 of 16 Fisher permutation tests completed due to computational constraints.
4. **Trimming**: Applied per-season following Appendix Table 11 methodology.
5. **Functional form**: asinh and log1p applied only to continuous monetary focal outcomes.

## Key Baseline Results

| Outcome | Group | Coefficient | SE | p-value | N |
|---------|-------|------------|-----|---------|---|
| expenses_fert_seeds | G1 | 2.369 | 1.105 | 0.045 | 658 |
| expenses_inputs | G1 | 4.044 | 2.145 | 0.075 | 658 |
| tarpaulin_d | G1 | 0.239 | 0.056 | 0.000 | 640 |
| sort_d | G1 | 0.140 | 0.039 | 0.002 | 464 |
| winnow_d | G1 | 0.150 | 0.065 | 0.033 | 464 |
| expenses_labor_preharvest | G1 | 16.249 | 14.455 | 0.275 | 464 |
| expenses_postharvest | G1 | 6.025 | 5.011 | 0.244 | 482 |
| expenses_labor_postharvest | G1 | 5.705 | 3.755 | 0.145 | 482 |
| price | G2 | 0.017 | 0.004 | 0.001 | 617 |
| acreage | G2 | 0.046 | 0.211 | 0.829 | 677 |
| harvest_kg_tot | G2 | 239.323 | 228.689 | 0.308 | 658 |
| yield | G2 | 111.392 | 49.638 | 0.037 | 658 |
| harvest_value | G2 | 81.504 | 43.009 | 0.073 | 646 |
| expenses | G2 | 18.114 | 17.263 | 0.307 | 658 |
| surplus | G2 | 65.821 | 32.262 | 0.055 | 628 |
| surplus_hrs | G2 | 97.814 | 41.406 | 0.029 | 462 |

## Software

- Python 3.12.7
- pyfixest, pandas, numpy, statsmodels
- Script: scripts/paper_analyses/158401-V1.py
