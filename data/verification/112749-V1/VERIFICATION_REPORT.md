# Verification Report: 112749-V1

**Paper**: Hornbeck & Naidu (2014), "When the Levee Breaks: Black Migration and Economic Development in the American South", AER 104(3): 963-990.

**Verified**: 2026-02-13

---

## 1. Baseline Groups

Two baseline groups were identified, matching the surface plan exactly.

### G1: Flood impact on Black share of population
- **Outcome**: `lnfrac_black` (log fraction Black population)
- **Treatment**: `f_int_{year}` (flood intensity x post-flood year dummies)
- **Focal parameter**: f_int_1950 (peak effect period)
- **Expected sign**: Negative (flood caused Black out-migration)
- **Baseline spec_run_ids**:
  - `112749-V1_G1_0001` (Table 2 Col 1, geography+lags): coef = -0.202, p = 0.0008, N = 978
  - `112749-V1_G1_0002` (Table 2 Col 2, geography+lags+NewDeal): coef = -0.240, p = 6.3e-05, N = 978
- **Both baselines succeeded**.

### G2: Flood impact on agricultural capital intensity
- **Outcome**: `lnvalue_equipment` (log value of farm equipment)
- **Treatment**: `f_int_{year}` (flood intensity x post-flood year dummies)
- **Focal parameter**: f_int_1940 (deviated from surface plan of f_int_1950 due to data availability)
- **Expected sign**: Positive (labor outflow induced mechanization)
- **Baseline spec_run_ids**:
  - `112749-V1_G2_0036` (Table 4 Col 3, geography+lags): coef = 0.378, p = 0.000236, N = 815
  - `112749-V1_G2_0037` (Table 4 Col 4, geography+lags+NewDeal): **FAILED** (collinearity)
- **One of two baselines failed**. The failure is a legitimate collinearity issue: equipment data is available for only 5 of 13 panel years, severely constraining degrees of freedom when adding year-interacted New Deal controls.

---

## 2. Row Counts

| Metric | Count |
|--------|-------|
| Total rows | 72 |
| Core rows | 72 |
| Non-core rows | 0 |
| Invalid rows | 0 |
| Baseline rows | 4 |
| Succeeded (non-empty coefficient) | 57 |
| Failed (empty coefficient, collinearity) | 15 |
| Partial (coef present, SE missing: Conley) | 6 |

### By Group

| Group | Total | Succeeded | Failed | Baselines OK |
|-------|-------|-----------|--------|--------------|
| G1 | 35 | 32 | 3 | 2/2 |
| G2 | 37 | 25 | 12 | 1/2 |

---

## 3. Category Counts

| Category | Count |
|----------|-------|
| core_method | 4 |
| core_controls | 30 |
| core_sample | 14 |
| core_weights | 4 |
| core_funcform | 8 |
| core_inference | 12 |

All 72 rows are classified as core. No rows drifted from the baseline claim object in either group: all G1 rows test the same outcome (lnfrac_black) and treatment concept (flood intensity), and all G2 rows test lnvalue_equipment with the same treatment concept. The alternative treatment measures (RedCross acres, RedCross people) are different operationalizations of the same flood intensity concept, consistent with the paper's own robustness checks.

---

## 4. Top Issues

### 4.1 High G2 failure rate (12/37 = 32%)
The most significant issue is the high failure rate for G2 specifications. Equipment data is available for only 5 of 13 panel years (1920, 1925, 1930, 1940, 1970), which severely constrains degrees of freedom. Specifications with many year-interacted controls (New Deal block, tenancy/mfg block) frequently hit collinearity. This is documented in SPECIFICATION_SEARCH.md and is a legitimate data constraint, not a coding error.

### 4.2 G2 baseline 2 failure
The G2 baseline 2 (geo+lags+ND) failed entirely. This means G2 has only one functioning baseline (geo+lags). The runner noted this deviation and adapted G2 inference variants to use the geo+lags baseline instead. This is appropriate but reduces the effective comparison set for G2.

### 4.3 Focal parameter deviation for G2
The surface plan specified f_int_1950 as the focal parameter for G2, but the runner changed it to f_int_1940 because equipment data is unavailable at 1950. This is well-documented and justified. However, it means the G2 focal coefficient captures a shorter post-flood effect (13 years post-flood rather than 23 years).

### 4.4 Conley SE not computable (6 rows)
Six Conley spatial HAC SE specifications (3 per group) report point estimates but no standard errors, because pyfixest does not support Conley SE computation. These are partial results. The point estimates are identical to the baseline, providing no additional information beyond confirming the coefficient.

### 4.5 Redundant LOO specifications
Two LOO specifications produce results identical to existing baselines:
- G1_0012 (`drop_new_deal`): identical to G1_0001 (baseline 1 has no New Deal controls)
- G1_0017 (`drop_tenancy_mfg`): identical to G1_0002 (baseline 2 has no tenancy/mfg controls)
- G2_0047 (`drop_new_deal`): identical to G2_0036 (baseline 1 has no New Deal controls)

These are not errors but are redundant with existing baselines.

### 4.6 Duplicate spec_ids within groups
Many RC variations are run twice within each group: once with geo+lags controls and once with geo+lags+ND controls. This is a deliberate choice that doubles the spec count but produces genuinely distinct specifications. The `spec_id` field is not unique within groups, but `spec_run_id` is globally unique. This is acceptable but could cause confusion if downstream analysis assumes spec_id uniqueness within groups.

### 4.7 G2 inference specs use baseline 1 only
The G2 inference variants (HC1, state-clustered, county-clustered, Conley) all use the geo+lags controls (baseline 1) because the geo+lags+ND controls (baseline 2) failed. This is noted in SPECIFICATION_SEARCH.md deviation 4. While reasonable, it means inference robustness is only checked against one control configuration for G2.

---

## 5. Surface Consistency

The SPECIFICATION_SURFACE.json planned the following core universe for each group:
- 1 design spec (TWFE)
- ~23 RC specs (controls progressions, LOO, weights, sample, treatment form)
- 6 inference specs

**Comparison**:
- All surface-planned spec categories appear in the results.
- G1 omits `rc/sample/time/pre1960_only` as documented (equivalent to `drop_1970` for G1's decadal data).
- G2 includes `rc/sample/time/pre1960_only` as planned.
- No spurious baseline groups or unexpected spec categories.
- The surface-planned `rc/controls/progression/geography_and_lags` and `rc/controls/progression/geography_lags_newdeal` are not explicitly present as separate spec_ids in the results because they are the baseline specifications themselves (Table2-Col1/Col2 for G1, Table4-Col3/Col4 for G2). This is correct: the baselines serve as those progression steps.

---

## 6. Recommendations

1. **Distinguish paired specs more clearly**: Many spec_ids appear twice within a group (once with geo+lags, once with geo+lags+ND). Consider appending a suffix to spec_id (e.g., `rc/sample/time/drop_1970__ctrl_geolags` vs `rc/sample/time/drop_1970__ctrl_geolagsnd`) to make spec_ids unique within groups.

2. **Remove redundant LOO specs**: The `drop_new_deal` and `drop_tenancy_mfg` LOO specs are redundant when the base specification (baseline 2) does or does not already include those blocks. Pre-check whether the dropped block is present before running.

3. **G2 data recovery**: The high G2 failure rate stems from having only 5 time periods for equipment data. If possible, investigate whether the original Stata data files contain intercensus equipment observations that the Python data pipeline failed to recover.

4. **Conley SE**: Consider using a dedicated spatial econometrics package (e.g., `spreg` or custom implementation) for Conley SE computation, or mark these specs as "planned but not executable" in the surface rather than running them with partial output.

5. **G2 baseline 2 fallback**: Since G2 baseline 2 failed, consider whether the geo+lags specification (baseline 1) should be the sole canonical baseline for G2, with appropriate documentation. Currently the LOO specs reference baseline 2 as their base, which is confusing since baseline 2 itself failed.

---

## 7. Overall Assessment

The specification search is well-executed and well-documented. The 72 specifications across 2 baseline groups cover a thorough range of robustness checks: control progressions, leave-one-out blocks, weight variations, sample restrictions, alternative treatment measures, and inference alternatives. All rows are correctly classified as core; no outcome or treatment drift was detected.

The main limitation is the high G2 failure rate (32%), which is driven by a legitimate data constraint (limited equipment data time periods) rather than coding issues. The runner's documentation of deviations and failures in SPECIFICATION_SEARCH.md is thorough and accurate.

**Effective core specifications**: 57 succeeded + 6 Conley partial = 51 fully informative core specs (ignoring 6 Conley rows with no SE and 15 failed rows). Of these:
- G1: 32 succeeded (all fully informative except 3 Conley partials = 29 fully informative)
- G2: 25 succeeded (all fully informative except 3 Conley partials = 22 fully informative)
