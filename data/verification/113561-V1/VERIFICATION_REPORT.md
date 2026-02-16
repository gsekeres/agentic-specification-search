# Verification Report: 113561-V1

**Paper**: Fong & Luttmer (2009), "What Determines Giving to Hurricane Katrina Victims?"

**Date**: 2026-02-15

## 1. Baseline Groups

Four baseline groups found, matching the surface definition:

| Group | Outcome | Baseline spec_run_id | Coefficient | p-value |
|-------|---------|---------------------|-------------|---------|
| G1 | giving | 113561-V1__run_0001 | -4.198 | 0.370 |
| G2 | hypgiv_tc500 | 113561-V1__run_0024 | -2.181 | 0.591 |
| G3 | subjsupchar | 113561-V1__run_0047 | -0.221 | 0.168 |
| G4 | subjsupgov | 113561-V1__run_0069 | -0.435 | 0.026 |

All baseline groups are correctly defined and match the surface.

## 2. Row Counts

| Metric | Count |
|--------|-------|
| Total rows | 90 |
| Valid (is_valid=1) | 90 |
| Core (is_core_test=1) | 90 |
| Non-core | 0 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 56 |
| core_sample | 16 |
| core_method | 8 |
| core_weights | 8 |
| core_funcform | 2 |

## 3. Sanity Checks

### 3.1 spec_run_id uniqueness
All 90 spec_run_ids are unique. Verified.

### 3.2 baseline_group_id consistency
All rows have valid baseline_group_id values (G1, G2, G3, G4). Each group has the expected number of specs.

### 3.3 spec_id typing
All spec_ids follow the typed namespace convention. No `infer/*` rows appear in specification_results.csv (the single HC3 inference variant is correctly in inference_results.csv).

### 3.4 run_success
All 90 rows have run_success=1 with empty run_error. No failures.

### 3.5 Numeric fields
All coefficient, std_error, p_value, ci_lower, ci_upper, n_obs, r_squared values are finite for all rows.

### 3.6 coefficient_vector_json
All rows have valid JSON in coefficient_vector_json with non-empty content.

## 4. Estimand Preservation Check

### 4.1 Outcome/treatment drift
No outcome or treatment concept changes detected:
- All G1 rows use outcome=giving (or giving_tc50 for the topcode variant, which preserves concept)
- All G2 rows use outcome=hypgiv_tc500 (or hypgiv_tc200 for tighter topcode)
- All G3 rows use outcome=subjsupchar
- All G4 rows use outcome=subjsupgov
- All rows use treatment=picshowblack

### 4.2 Population changes
Sample restrictions (main_survey_only, slidell_only, biloxi_only, race_shown_only) are within-population robustness checks, not estimand-changing population switches. The target population remains "white respondents" in all cases; these are subsets used as stability checks matching the paper's own Table 5.

## 5. Top Issues

None. The specification search is clean:
- All rows are valid and core-eligible
- No outcome/treatment drift
- No duplicated spec_run_ids
- No missing baseline groups
- No `infer/*` rows in specification_results.csv

## 6. Recommendations

1. **Consider adding more functional form variants**: Only 2 of 90 specs explore functional form. For the dollar-denominated outcomes (G1, G2), log(1+giving) or asinh(giving) could be informative.

2. **Consider interaction-based robustness**: The paper reveals interaction effects in Table 6 with racial attitude moderators. While these are heterogeneity/exploration (not core), they could be included as `explore/*` in a future round.

3. **Multiple testing correction**: With 4 outcomes tested, a Bonferroni or BH correction could be applied as a post-processing step. The sole significant result (G4, p=0.026) would not survive Bonferroni correction (threshold = 0.0125).

## 7. Final Assessment

The specification search is **approved**. All 90 specifications are valid, correctly typed, and preserve their respective estimands. The search faithfully implements the approved surface and the results are auditable.
