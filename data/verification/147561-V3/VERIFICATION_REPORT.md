# Verification Report: 147561-V3

**Paper:** Local vs central property tax collection in Kananga, DRC (RCT)
**Verification date:** 2026-02-25
**Verifier:** Post-run audit (prompt 06)

---

## Baseline Groups

### G1: Local collection -> tax compliance and revenues (RCT)
- **Claim:** ITT effect of assigning neighborhood to local (chief-led) tax collection on individual property tax compliance probability and revenue amount. RCT randomized at polygon level in Kananga, DRC. FE: stratum + house type + bimonthly time. Clustered at polygon (a7).
- **Baseline specs:**
  - `147561-V3_run_001` (`baseline`): taxes_paid ~ t_l, compliance binary. coef=0.034, p<0.001, N=27643.
  - `147561-V3_run_002` (`baseline__revenues`): taxes_paid_amt, revenue amount. coef=72.11, p<0.001, N=27643.
- **Expected sign:** Positive (local collection increases compliance and revenues).

---

## Counts

| Metric | Count |
|--------|-------|
| Total rows | 51 |
| Valid | 51 |
| Invalid | 0 |
| Baseline | 2 |
| Core | 51 |
| Non-core | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_sample | 14 |
| core_data | 14 |
| core_fe | 11 |
| core_funcform | 8 |
| core_method | 4 |

---

## Sanity Checks

- **spec_run_id uniqueness:** PASS. All 51 spec_run_ids are unique.
- **baseline_group_id present:** PASS. All rows have G1.
- **run_success:** PASS. All 51 rows have run_success=1.
- **Numeric fields finite:** PASS. All numeric fields are finite.
- **No infer/* rows in specification_results.csv:** PASS.
- **coefficient_vector_json structure:** PASS. All rows have `coefficients`, `inference`, `software`, `surface_hash`. rc/fe rows have `fixed_effects`. rc/form rows have `functional_form`. rc/data rows have `data_construction` or treatment redefinition info.
- **Inference canonical match:** PASS. All spec_results rows use `infer/se/cluster/a7` matching surface canonical.
- **Inference variants in inference_results.csv:** PASS. 1 inference variant row (HC1 robust) in inference_results.csv only.

---

## Issues

### Minor Issues

1. **Duplicate: run_009 = baseline run_001:** `rc/fe/sets/stratum_month_house` is the exact same FE structure as the baseline (stratum + month + house). Coefficients are identical (0.03437, N=27643). Similarly, run_010 duplicates run_002 for revenues. Flagged with lower confidence (0.70).

2. **Compound spec_ids encode multiple dimensions:** Many spec_ids combine a robustness dimension with an outcome or FE variant (e.g., `rc/data/treatment/include_cli_arm__stratum_only`, `rc/form/outcome/log1p_amt__excl_exempt`). This is well-structured but means the total unique single-dimension variations is smaller than the 51 row count suggests.

3. **No control variations:** The surface explicitly sets controls_count envelope to [0, 0] since this is an RCT with no covariates in the baseline. This is correct per the experimental design.

4. **Pooled local_vs_central variants change treatment variable:** 4 specs use `t_local_type` instead of `t_l`, pooling local subtypes. This is classified as core_data since the surface explicitly includes it and the estimand concept (local vs central) is preserved.

5. **Two marginal p-values:** run_004 (diff_in_means__revenues, p=0.053) and run_050 (stratum_only__revenues__diff_means, p=0.053) cross the conventional threshold. run_043 (pooled_local_vs_central__revenues__stratum_only, p=0.26) is clearly insignificant, suggesting the pooled treatment definition weakens the revenue result with minimal FE.

6. **Polygon means specification has very small N:** run_015 and run_016 (polygon_means) have N=206, collapsing to polygon-level averages. This is a valid aggregation but dramatically reduces power.

---

## Recommendations

1. **Remove or flag exact duplicates:** run_009 and run_010 should be documented as redundant with the baselines. Consider removing them to avoid inflating the specification count.

2. **Add heterogeneity specifications:** No heterogeneity analysis appears in the core specs. The paper likely examines heterogeneity by property type, chief characteristics, or neighborhood income. These could be added as explore/* specs.

3. **Consider additional inference variants:** The paper uses cluster-robust SEs at the polygon level. Wild cluster bootstrap (Cameron-Gelbach-Miller) would be appropriate given the moderate number of clusters (~356).

4. **Add revenue outlier analysis for compliance:** The outlier trimming specs only apply to the revenue outcome. Adding compliance-outcome outlier analysis (e.g., dropping extreme polygons) would be useful.
