# Verification Report: 136741-V1

**Paper:** Historical lynching and contemporary black voter registration (cross-sectional OLS)
**Verification date:** 2026-02-25
**Verifier:** Post-run audit (prompt 06)

---

## Baseline Groups

### G1: Lynching -> black voter registration (OLS)
- **Claim:** Cross-sectional OLS: negative association between historical black lynching rate per capita (1882-1930) and contemporary black voter registration rate in Southern US counties, with state FE and historical controls.
- **Baseline specs:**
  - `136741-V1_run_001` (`baseline`): Table 2 Col 1. coef=-0.469, p=0.001, N=267, R2=0.548. Default OLS SEs.
- **Expected sign:** Negative (more lynching suppresses voter registration).

---

## Counts

| Metric | Count |
|--------|-------|
| Total rows | 52 |
| Valid | 52 |
| Invalid | 0 |
| Baseline | 1 |
| Core | 52 |
| Non-core | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 40 |
| core_sample | 4 |
| core_data | 4 |
| core_funcform | 2 |
| core_method | 1 |
| core_fe | 1 |

---

## Sanity Checks

- **spec_run_id uniqueness:** PASS. All 52 spec_run_ids are unique.
- **baseline_group_id present:** PASS. All rows have G1.
- **run_success:** PASS. All 52 rows have run_success=1.
- **Numeric fields finite:** PASS. All numeric fields are finite for all rows.
- **No infer/* rows in specification_results.csv:** PASS.
- **coefficient_vector_json structure:** PASS. All rows have `coefficients`, `inference`, `software`, `surface_hash`. rc/controls rows have `controls` block. rc/data rows have `data_construction`. rc/form rows have `functional_form`. rc/fe row has `fixed_effects`.
- **Inference canonical match:** PASS. All spec_results rows use `infer/se/iid/default` matching surface canonical.
- **Inference variants in inference_results.csv:** PASS. 2 inference variant rows (HC1, state cluster) in inference_results.csv only.

---

## Issues

### Minor Issues

1. **Duplicate results between progression/bivariate and sets/none:** Run_013 (bivariate) and run_009 (sets/none) produce identical results (coef=-0.629, p<0.001). Both are valid specifications but represent the same model (treatment only, no controls). Lower confidence assigned to run_013 (0.85).

2. **Progression/add_all_historical matches baseline:** Run_016 (add_all_historical) produces identical results to run_001 (baseline) because the baseline already includes all historical controls. Lower confidence assigned (0.85).

3. **Heavy reliance on control variations:** 40 of 52 specs (77%) are control variations. While thorough, this leaves limited coverage of sample, data, and functional form dimensions.

4. **All coefficients negative and 51 of 52 significant at p<0.05:** The one exception is run_043 (trim_y_5_95, p=0.074), which trims 10% of the outcome distribution. Strong directional consistency across specifications.

5. **Treatment variable alternatives change coefficient scale:** lynchcapitamob1920 (coef=-0.086) and lynchcapitamob1930 (coef=-0.040) have much smaller coefficients than the baseline (coef=-0.469) because the treatment variable has a different scale/denominator. These are still validly signed but caution is needed in pooled coefficient comparisons.

---

## Recommendations

1. **Add weight variants:** No rc/weights/* specs appear. Adding population-weighted or unweighted variants would diversify the specification surface.

2. **Consider spatial clustering or Conley SEs:** The surface notes only 6 states in the sample. State-clustered SEs (in inference variants) have very few clusters. Conley spatial HAC SEs would be more appropriate for cross-sectional spatial data.

3. **Add more sample variants:** Beyond outlier trimming and registration cap, consider state-level leave-one-out to check if results are driven by a single state.
