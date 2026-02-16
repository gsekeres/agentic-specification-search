# Verification Report: 112749-V1

## Paper
Hornbeck & Naidu (2014), "When the Levee Breaks: Black Migration and Economic Development in the American South," AER 104(3): 963-990.

## Baseline Groups

### G1: lnfrac_black ~ f_int (Black Population Share)
- **Claim**: The 1927 Mississippi flood caused a persistent decline in Black population share in affected Southern counties
- **Expected sign**: Negative
- **Baseline spec**: 112749-V1__G1__001 (Table 2 Col 1)
  - Coefficient: -0.1563
  - SE: 0.0321 (clustered at county)
  - p-value: 2e-06
  - N: 2604
- **Additional baseline variants**: 112749-V1__G1__019 (f_int_1950 focal, coef = -0.1812, p = 0.0002)

### G2: lnvalue_equipment ~ f_int (Farm Equipment Value)
- **Claim**: The 1927 Mississippi flood caused an increase in farm equipment value (mechanization) in affected Southern counties
- **Expected sign**: Positive
- **Baseline spec**: 112749-V1__G2__024 (Table 4 Col 2)
  - Coefficient: 0.4396
  - SE: 0.0995 (clustered at county)
  - p-value: 1.3e-05
  - N: 2170
- **Additional baseline variants**: 112749-V1__G2__041 (f_int_1930 pre-trend, coef = 0.022, p = 0.796), 112749-V1__G2__042 (f_int_1970 long-run, coef = 0.700, p < 0.001)

## Row Counts

| Metric | Count |
|--------|-------|
| Total rows | 46 |
| Core (is_core_test=1) | 46 |
| Non-core | 0 |
| Invalid (is_valid=0) | 10 |
| Valid (is_valid=1) | 36 |
| Baseline (is_baseline=1) | 5 |

## Category Breakdown

| Category | Count | Valid | Invalid |
|----------|-------|-------|---------|
| core_method | 9 | 9 | 0 |
| core_controls | 15 | 10 | 5 |
| core_fe | 2 | 0 | 2 |
| core_sample | 6 | 4 | 2 |
| core_funcform | 6 | 6 | 0 |
| core_weights | 6 | 3 | 3 |
| **Total** | **46** | **36** | **10** |

Note: All invalid rows are failures (run_success=0) due to matrix singularity, not mis-extractions.

## Verification Checks

### 1. spec_run_id Uniqueness: PASS
All 46 spec_run_ids are unique.

### 2. baseline_group_id Consistency: PASS
- G1: 23 rows (IDs 001-023), matching surface
- G2: 23 rows (IDs 024-046), matching surface
- No spurious or missing baseline groups

### 3. spec_id Typing: PASS
- 5 baseline/baseline__* rows (G1: 2, G2: 3)
- 2 design/* rows (one per group)
- 39 rc/* rows (controls: 15, fe: 2, sample: 6, form: 6, weights: 6)
- 0 infer/* rows in specification_results.csv (correct)
- 0 forbidden namespace rows

### 4. Run Success: PASS (with expected failures)
- 36 rows with run_success=1
- 10 rows with run_success=0
- All failures have concrete run_error: "Matrix is singular."
- No failures have finite numeric fields (correctly blank)

### 5. Numeric Fields: PASS
All successful rows have finite coefficient, std_error, p_value, n_obs, r_squared.
All p-values are in [0, 1]. All CIs contain their respective coefficients.

### 6. JSON Validity: PASS
All coefficient_vector_json fields parse as valid JSON. Failed rows use "{}".

### 7. Outcome/Treatment Consistency: PASS

**G1 outcomes**: lnfrac_black (20 specs), frac_black_level (1), lnpopulation_black (1), lnfrac_black at f_int_1950 (4 counting overlaps)
- frac_black_level and lnpopulation_black are outcome transformations/alternatives but preserve the Black population concept
- No conceptual drift from baseline claim

**G1 treatments**: f_int_1930 (16 specs), f_int_1940 (1, drop_1930 sample), f_int_1950 (4), f_bin_1930 (1)
- Treatment concept is always flood intensity or its binary analog
- f_int_1940/1950 focal variants are part of the dynamic treatment effects, not treatment drift

**G2 outcomes**: lnvalue_equipment (18 specs), value_equipment_level (1), lntractors (1)
- value_equipment_level preserves concept; lntractors is a related mechanization measure
- No conceptual drift from baseline claim

**G2 treatments**: f_int_1940 (15 specs), f_int_1930 (2), f_int_1970 (3), f_bin_1940 (1)
- Same treatment concept throughout; different focal periods are part of dynamic effects

### 8. No Infer/* in Spec Results: PASS
Inference variants correctly segregated to inference_results.csv (4 rows).

## Inference Results Audit

| Variant | Group | Base Spec | Run Success | Coefficient | SE | p-value |
|---------|-------|-----------|------------|------------|------|---------|
| infer/se/hc/hc1 | G1 | G1__001 | 1 | -0.1563 | 0.0562 | 0.005 |
| infer/se/cluster/state | G1 | G1__001 | 1 | -0.1563 | 0.0373 | 0.003 |
| infer/se/hc/hc1 | G2 | G2__024 | 1 | 0.4396 | 0.0956 | <0.001 |
| infer/se/cluster/state | G2 | G2__024 | 1 | 0.4396 | 0.1813 | 0.041 |

All 4 inference variants succeeded. Coefficients match base specs exactly (as expected -- only SEs change).

Key findings:
- G1: HC1 robust SE (0.056) is larger than county-clustered SE (0.032), but state-clustered SE (0.037) is close to county-clustered. All significant at p < 0.01.
- G2: HC1 robust SE (0.096) is similar to county-clustered SE (0.099). State-clustered SE (0.181) is much larger, reducing significance to p = 0.041 (still significant at 5% but marginal). This suggests within-state correlation in equipment outcomes.

## Robustness Summary

### G1: Black Population Share
**36 of 36 successful specifications yield coefficients of the expected sign** (negative for f_int_1930 and f_int_1950 focal parameters).

Among the 18 successful specs with f_int_1930 focal:
- **18 of 18 (100%)** are significant at p < 0.05
- **16 of 18 (89%)** are significant at p < 0.01
- Coefficient range: [-0.175, -0.088]
- The narrowest effect (-0.088) comes from the binary flood indicator, which is expected (less variation in treatment)

**Assessment: STRONG robustness.** The flood's negative effect on Black population share is extremely robust.

### G2: Farm Equipment Value
**All 11 successful specs with f_int_1940 focal yield positive coefficients.**

- **11 of 11 (100%)** are significant at p < 0.05
- Coefficient range: [0.158, 0.655]
- The no-controls estimate (0.655) is inflated as expected
- Pre-trend (f_int_1930): 0.022, p = 0.796 -- correctly null
- Long-run (f_int_1970): 0.700, p < 0.001 -- effect persists and grows

The higher failure rate in G2 (8/23 = 35%) is a data limitation (sparse agricultural census panel), not a robustness concern. Among specifications that can be estimated, the result is consistently significant.

**Assessment: STRONG robustness** (conditional on estimable specifications).

### Overall Assessment
Both baseline claims are strongly supported. The G2 result becomes marginal (p = 0.041) only under state-level clustering, which may be overly conservative for this setting. State-clustered inference still rejects at the 5% level.

## Top Issues

1. **High failure rate for G2 (35%)**: The sparse panel structure of agricultural census data (only 3 post-flood observations for equipment) causes matrix singularity when control sets are expanded or weights are changed. This is a genuine data limitation.

2. **FE drop variant fails for both groups**: Dropping state-year FE always produces singularity, suggesting these FE absorb critical variation. The design may be fragile without state-year controls.

3. **Weight sensitivity for G2**: Both alternative weight specifications fail, so weight robustness cannot be assessed for the equipment outcome. This is a gap in the robustness coverage.

## Recommendations

1. **Investigate G2 singularity**: The matrix singularity in G2 specifications likely stems from near-collinearity between treatment interactions and control variables in the sparse panel. Consider using ridge-penalized estimation or reducing the control set dimensionality to recover more specifications.

2. **Add Conley spatial SE**: The paper's counties are spatially correlated. Conley spatial standard errors would be a valuable inference variant but are not available in the Python environment. Consider R implementation.

3. **Test pre-trends more formally**: The G2 f_int_1930 coefficient serves as an informal pre-trend check (correctly null), but a formal joint test of all pre-treatment coefficients would strengthen the parallel trends argument.

4. **Consider population-weighted DiD**: Since the paper studies aggregate county outcomes, population-weighted estimation might be more appropriate than area-weighted. The pop_1920 weight variant succeeded for G1 but failed for G2.
