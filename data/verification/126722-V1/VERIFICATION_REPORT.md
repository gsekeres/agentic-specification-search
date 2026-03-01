# Verification Report: 126722-V1

## Paper
Lopez, Sautmann, Schaner (AEJ Applied) -- "Does Patient Demand Contribute to the Overuse of Prescription Drugs?"

## Baseline Groups

### G1: Effect of patient voucher on antimalarial treatment outcomes

| spec_run_id | spec_id | outcome_var | coef | SE | p | N |
|-------------|---------|-------------|------|----|---|---|
| 126722-V1_run_001 | baseline | RXtreat_sev_simple_mal | 0.060 | 0.026 | 0.023 | 2053 |
| 126722-V1_run_002 | baseline__treat_sev_simple_mal | treat_sev_simple_mal | 0.142 | 0.029 | 0.000 | 2053 |
| 126722-V1_run_003 | baseline__used_vouchers_admin | used_vouchers_admin | 0.346 | 0.030 | 0.000 | 2055 |
| 126722-V1_run_004 | baseline__RXtreat_severe_mal | RXtreat_severe_mal | -0.038 | 0.019 | 0.049 | 2053 |
| 126722-V1_run_005 | baseline__treat_severe_mal | treat_severe_mal | -0.017 | 0.021 | 0.431 | 2053 |

**Primary baseline**: run_001 (RXtreat_sev_simple_mal, Table 3 Col 2). Coefficient is positive and significant at 5%, matching the paper's direction and significance. The paper reports 0.052 (SE 0.025) using pdslasso; the replication obtains 0.060 (SE 0.026) using manual covariates from Table B10. This discrepancy is expected and documented in the surface.

**Sign pattern**: Positive for prescribing/purchasing any antimalarial (runs 001-003), negative for severe-only treatments (runs 004-005). This is consistent with the paper's finding that vouchers shift patients toward simple malaria treatment and away from severe-only treatment.

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 57 |
| Valid (run_success=1 and no extraction issues) | 57 |
| Invalid | 0 |
| Baseline | 5 |
| Core test | 57 |
| Non-core | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 15 |
| core_controls (LOO, sets, progression, subset) | 36 |
| core_sample (home survey restriction) | 2 |
| core_fe (add/drop FE) | 4 |

### Inference Results (inference_results.csv)

5 rows in inference_results.csv, all run_success=1. These are HC1 (no clustering) variants of the 5 baseline specs. Correctly segregated from specification_results.csv.

## Structural Validation

### Step 0: Sanity Checks

- **spec_run_id uniqueness**: All 57 spec_run_ids are unique. PASS.
- **baseline_group_id**: All rows have baseline_group_id = G1. PASS.
- **spec_tree_path anchors**: All spec_tree_paths contain a `#` anchor. PASS.
- **run_success**: All 57 rows have run_success=1. No failures. PASS.
- **Numeric fields**: All coefficient, std_error, p_value, n_obs, r_squared fields are finite for all rows. PASS.
- **coefficient_vector_json structure**: All rows have the required audit keys (coefficients, inference, software, surface_hash). No unexpected top-level keys. PASS.
- **Canonical inference**: All 57 rows use `infer/se/cluster/clinic` as the inference spec_id. PASS.
- **No infer/* rows in specification_results.csv**: PASS.

### Step 1: Surface Consistency

The surface defined one baseline group (G1) with 5 outcomes, one design code (randomized_experiment), and planned ~65 specs. The execution produced 57 estimate rows + 5 inference rows = 62 total. This is within budget (max 80).

The surface's `core_universe.baseline_spec_ids` lists 4 additional baselines (baseline__treat_sev_simple_mal, baseline__used_vouchers_admin, baseline__RXtreat_severe_mal, baseline__treat_severe_mal). All 4 are present plus the primary baseline. PASS.

The surface's `core_universe.design_spec_ids` includes diff_in_means and strata_fe variants. Both are present for all 5 outcomes (10 design rows total). PASS.

The surface's `core_universe.rc_spec_ids` includes LOO, sets, progression, subset, sample restriction, and FE variants. All are present. PASS.

No spurious baseline groups or missing groups. PASS.

### Step 2: Baseline Identification

5 baseline rows identified (runs 001-005). All match the surface plan. The primary baseline (run_001) matches the Table 3 Col 2 specification from the paper.

### Step 3: Row Classification

All 57 rows classified as core. The treatment variable (patient_voucher) is consistent across all rows. All outcomes are within the claim object's scope. No outcome drift, treatment drift, or population changes that would warrant non-core classification.

## Issues Found

### Issue 1: Mechanically Duplicate Specifications (MODERATE)

Seven specification pairs produce identical coefficient/SE/p-value/N values:

| Duplicate | Matches | Reason |
|-----------|---------|--------|
| run_029 (rc/controls/sets/none) | run_006 (design/diff_in_means) | Both are no-controls specs for primary outcome |
| run_031 (rc/controls/sets/extended) | run_001 (baseline) | Extended control set equals the full baseline control set |
| run_032 (rc/controls/progression/bivariate) | run_006 (design/diff_in_means) | Bivariate = no controls = diff_in_means |
| run_033 (rc/controls/progression/symptoms_only) | run_030 (rc/controls/sets/minimal) | Symptoms-only progression step = minimal control set |
| run_036 (rc/controls/progression/full) | run_001 (baseline) | Full progression step = baseline |
| run_056 (rc/fe/add/clinic) | run_007 (design/strata_fe) | Adding clinic FE = strata FE design variant |
| run_057 (rc/fe/add/clinic__treat_sev_simple_mal) | run_009 (design/strata_fe__treat_sev_simple_mal) | Same for secondary outcome |

These duplicates are valid specifications viewed from different conceptual axes, but they inflate the specification count without adding independent information. After deduplication, the effective unique specification count is **50** (57 - 7 duplicates).

**Recommendation**: For the specification curve, either flag duplicates or deduplicate prior to analysis. The `why` column in verification_spec_map.csv notes each duplicate.

### Issue 2: Concentration on Primary Outcome (MINOR)

Of 57 rows, 42 (74%) use the primary outcome RXtreat_sev_simple_mal. The 4 secondary outcomes have only 2-4 non-baseline rows each (mostly design variants plus a few RC specs). This is adequate for the primary claim but provides limited robustness evidence for secondary outcomes.

### Issue 3: pdslasso Approximation (ACKNOWLEDGED)

The baseline uses manual covariates (Table B10) instead of the paper's pdslasso-selected controls (Table 3). This is documented in the surface and is a reasonable approximation. Coefficients are directionally consistent with published values.

### Issue 4: No Functional Form or Weights Variants (NOT APPLICABLE)

All outcomes are binary (0/1), so functional form transforms are inappropriate. The paper does not use weights. These omissions are correct.

## Robustness Assessment

The primary baseline claim (patient_voucher -> RXtreat_sev_simple_mal positive) shows:
- **42 of 42 primary-outcome specs have positive coefficients** (100% sign agreement)
- **36 of 42 are significant at p<0.05** (86%)
- **40 of 42 are significant at p<0.10** (95%)
- Coefficient range: 0.035 to 0.075 (baseline: 0.060)
- The only non-significant primary-outcome spec is the clinic-FE variant (coef=0.035, p=0.150), which absorbs between-clinic variation

**Overall assessment**: STRONG support for the baseline claim.

## Recommendations

1. **Deduplicate before analysis**: Remove or flag the 7 mechanically duplicate rows to avoid inflating concordance rates.
2. **Expand secondary outcomes**: Future runs could add more RC variants for secondary outcomes (treat_sev_simple_mal, used_vouchers_admin) to strengthen multi-outcome robustness evidence.
3. **Consider wild cluster bootstrap**: With only ~60 clusters, CRV1 may slightly under-reject. Wild cluster bootstrap (if available) would be a valuable inference variant.
4. **Consider trimming/winsorization**: The surface planned sample outlier trimming (1%/99% and 5%/95%) but these were not executed, possibly because outcomes are binary. This omission is appropriate.
