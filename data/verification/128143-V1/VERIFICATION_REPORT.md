# Verification Report: 128143-V1

**Paper:** Carbon tax acceptance and self-interest/environmental effectiveness beliefs (French survey RCT)
**Verification date:** 2026-02-25
**Verifier:** Post-run audit (prompt 06)

---

## Baseline Groups

### G1: Self-interest belief -> targeted tax acceptance (IV)
- **Claim:** LATE/IV effect of believing one does not lose from carbon tax+dividend on targeted tax acceptance, instrumented by random dividend eligibility assignment.
- **Baseline specs:**
  - `128143-V1_run_001` (`baseline`): Table 5.2 Col 1, subsample p10-p60. coef=0.557, p<0.001, N=1969.
  - `128143-V1_run_002` (`baseline__table52_col2_iv_fullsample`): Table 5.2 Col 2, full sample. coef=0.487, p<0.001, N=3002.
- **Expected sign:** Positive.

### G2: Environmental effectiveness belief -> initial tax approval (IV)
- **Claim:** LATE/IV effect of environmental effectiveness belief on initial carbon tax approval, instrumented by random information treatment.
- **Baseline specs:**
  - `128143-V1_run_014` (`baseline`): Table 5.4 Col 1, Yes/Yes coding. coef=0.436, p=0.009, N=3002.
  - `128143-V1_run_015` (`baseline__table54_col3_iv_notnoyes`): Table 5.4 Col 3, NotNo/Yes coding. coef=0.513, p=0.033, N=3002.
- **Expected sign:** Positive.

---

## Counts

| Metric | Count |
|--------|-------|
| Total rows | 57 |
| Valid | 57 |
| Invalid | 0 |
| Baseline | 4 |
| Core | 57 |
| Non-core | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 46 |
| core_method | 4 |
| core_weights | 4 |
| core_funcform | 3 |

### By Baseline Group

| Group | Total | Core | Non-core |
|-------|-------|------|----------|
| G1 | 35 | 35 | 0 |
| G2 | 22 | 22 | 0 |

---

## Sanity Checks

- **spec_run_id uniqueness:** PASS. All 57 spec_run_ids are unique.
- **baseline_group_id present:** PASS. All rows have G1 or G2.
- **run_success:** PASS. All 57 rows have run_success=1.
- **Numeric fields finite:** PASS. All coefficient, std_error, p_value, ci_lower, ci_upper, n_obs, r_squared are finite.
- **No infer/* rows in specification_results.csv:** PASS.
- **coefficient_vector_json structure:** PASS. All rows have `coefficients`, `inference`, `software`, `surface_hash`. rc/form rows have `functional_form`. rc/controls rows have `controls`. rc/weights rows have `weights`.
- **Inference canonical match:** PASS. All spec_results rows use `infer/se/ols_default` matching the surface canonical.
- **Inference variants in inference_results.csv:** PASS. 6 inference variant rows (HC1, HC2) appear only in inference_results.csv.

---

## Issues

### Minor Issues

1. **G2 run_016 duplicates baseline run_014:** `rc/form/outcome_approval_yes` for G2 has outcome=`outcome_initial_yes` and coef=0.4361, which is identical to baseline run_014 (same outcome coding). This appears to be a no-op form variant since the G2 primary baseline already uses the Yes coding. Marked as core_funcform with lower confidence (0.80).

2. **Sparse G2 LOO coverage:** G2 has only 22 specifications with 15 LOO controls. The surface budgeted up to 50 specs for G2, so coverage is moderate but not exhaustive.

3. **Some spec_ids include double suffixes:** Several spec_ids use compound suffixes like `rc/controls/loo/income_quadratic__approval_yes` to denote simultaneous changes (LOO + outcome recoding). These are still coherent and correctly categorized.

4. **Gaps in spec_run_id numbering:** Run IDs skip numbers (e.g., run_012 to run_014, run_021 to run_024). This likely reflects inference variants or failed runs that were removed, which is expected behavior.

---

## Recommendations

1. **Expand G2 LOO coverage:** The G2 group has fewer individual-variable LOO specs compared to G1. Consider adding LOO for individual demographic dummies (already done for G1 in runs 040-055).

2. **Add sample-restriction variants:** No `rc/sample/*` specs appear. The surface notes a subsample restriction (p10-p60) for G1. Adding sample boundary variations (e.g., p5-p65, p15-p55) would strengthen robustness.

3. **Address generated-regressor SE issue:** The surface notes that manual 2SLS SEs do not account for generated-regressor uncertainty. The inference variants (HC1, HC2) help but a bootstrap or proper 2SLS SE correction would be valuable.
