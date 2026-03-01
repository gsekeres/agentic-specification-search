# Verification Report: 112749-V1

## Paper
Hornbeck & Naidu (2014), "When the Levee Breaks: Black Migration and Economic Development in the American South," AER 104(3): 963-990.

## Baseline Groups

### G1: Black Population Share / Black Population ~ Flood Intensity
- **Claim**: The 1927 Mississippi flood caused a persistent decline in Black population share in affected Southern US counties.
- **Expected sign**: Negative (for panel DiD on lnfrac_black and lnpopulation_black)
- **Primary baseline spec**: 112749-V1_run_0001 (Table 2 Column 1, lnfrac_black ~ f_int_YEAR)
  - Coefficient (f_int_1930): -0.1563
  - SE: 0.0464 (clustered at fips)
  - p-value: 0.000819
  - N: 2604
- **Additional baselines**:
  - run_0002: with New Deal controls (coef=-0.166, p=0.000002)
  - run_0007: lnpopulation_black outcome (coef=-0.138, p=0.018)
  - run_0008: lnpopulation_black + New Deal (coef=-0.142, p=0.014)
  - run_0025 to run_0030: Table 1 cross-sectional long-difference baselines (flood_intensity treatment)
- **Failed baselines**: run_0013 (lnpopulation, singular matrix), run_0019/run_0020 (lnfracfarms_nonwhite, singular matrix)

### G2: Agricultural Capital Intensity ~ Flood Intensity
- **Claim**: The 1927 Mississippi flood caused increases in agricultural capital intensity (tractors, equipment) in affected counties.
- **Expected sign**: Positive
- **Successful baselines**:
  - run_0037: lnvalue_equipment_nd (coef=-0.0006, p=0.994) -- near-zero, not supporting claim
  - run_0041: lntractors (coef=0.511, p=0.001) -- strong positive
  - run_0042: lntractors_nd (coef=0.517, p=0.001) -- strong positive
  - run_0046: lnmules_horses (coef=0.079, p=0.116) -- weak positive, not significant
- **Failed baselines**: run_0031/0032 (lnavfarmsize), run_0036 (lnvalue_equipment), run_0047 (lnmules_horses_nd), run_0051-0056 (Table 5 land/building outcomes) -- all singular matrix

## Row Counts

| Metric | Count |
|--------|-------|
| Total rows | 56 |
| Core (is_core_test=1) | 48 |
| Non-core (is_core_test=0) | 8 |
| Valid (is_valid=1) | 31 |
| Invalid (is_valid=0, all due to singular matrix) | 25 |
| Baseline (is_baseline=1) | 28 |

## Category Breakdown

| Category | Count | Valid | Invalid |
|----------|-------|-------|---------|
| core_method | 24 | 14 | 10 |
| core_controls | 12 | 6 | 6 |
| core_funcform | 6 | 1 | 5 |
| core_weights | 6 | 3 | 3 |
| noncore_alt_outcome | 8 | 4 | 4 |
| **Total** | **56** | **28** | **28** |

Note: The core_method category includes 4 rows with spec_id=`rc/inference/hc1_*` which are effectively inference variants (same coefficient, different SE from HC1 robust standard errors). These should ideally be in inference_results.csv.

## Verification Checks

### 1. spec_run_id Uniqueness: PASS
All 56 spec_run_ids are unique (112749-V1_run_0001 through 112749-V1_run_0056).

### 2. baseline_group_id Consistency: PASS
- G1: 30 rows (runs 0001-0030), matching surface plan
- G2: 26 rows (runs 0031-0056), matching surface plan
- Both groups match the surface's baseline_groups definition

### 3. spec_id Typing: PASS
- 28 baseline/baseline__* rows
- 0 design/* rows
- 24 rc/* rows (controls: 12, weights: 6, inference: 4, funcform implicitly via lnfracfarms_nonwhite)
- 0 infer/* rows in specification_results.csv (correct)
- 4 rc/inference/* rows are present -- technically valid namespace but conceptually inference variants

### 4. Run Success: PASS (with expected failures)
- 31 rows with run_success=1 (55.4%)
- 25 rows with run_success=0 (44.6%)
- All 25 failures have error: "Matrix is singular." with proper error_details JSON
- No failed rows have any finite numeric scalar fields

### 5. Numeric Fields: PASS
All 31 successful rows have finite coefficient, std_error, p_value, n_obs, r_squared.
All p-values are in [0, 1]. All confidence intervals contain their respective point estimates.

### 6. JSON Validity: PASS
All coefficient_vector_json fields parse as valid JSON.
- Successful rows include required keys: coefficients, inference, software, surface_hash
- Failed rows include error and error_details objects

### 7. No infer/* in Spec Results: PASS
No spec_ids begin with "infer/". The 4 rc/inference/* rows are namespace-valid but noted as a concern.

### 8. Outcome/Treatment Consistency

**G1 outcomes**: lnfrac_black (12 specs), lnpopulation_black (6 specs), lnpopulation (6 specs, classified noncore), lnfracfarms_nonwhite (6 specs)
- lnpopulation (total population) is a different claim (placebo/comparison) -- classified as noncore_alt_outcome
- lnfracfarms_nonwhite is a related racial composition measure -- classified as core_funcform

**G1 treatments**: f_int_1930 (24 panel specs), flood_intensity (6 cross-sectional specs)
- Panel specs use year-interacted flood intensity; cross-sectional Table 1 specs use simple flood_intensity
- Both measure the same underlying treatment concept

**G2 outcomes**: lnavfarmsize (4 specs), lnvalue_equipment (4 specs), lntractors (4 specs), lnmules_horses (4 specs), lnfarmland_a (2 specs), lnlandbuildingvaluef (2 specs), lnlandbuildingvalue (2 specs)
- All are agricultural capital/land measures consistent with the mechanization claim

**G2 treatment**: f_int_1930 throughout (26 specs)

## Inference Results Audit

62 inference variant rows in inference_results.csv, covering both HC1 robust and state-level clustering variants for all successful specification_results.csv rows. All 62 inference runs succeeded.

Key findings by baseline:
- G1 run_0001 (lnfrac_black): HC1 SE=0.067 (p=0.021), state-cluster SE=0.042 (p=0.006) vs. baseline fips-cluster SE=0.046 (p=0.001)
- G1 run_0007 (lnpopulation_black): HC1 SE=0.095 (p=0.149), state-cluster SE=0.052 (p=0.029) vs. baseline SE=0.058 (p=0.018)
- G2 run_0041 (lntractors): HC1 SE=0.177 (p=0.004), state-cluster SE=0.133 (p=0.005) vs. baseline SE=0.158 (p=0.001)

## Robustness Summary

### G1: Black Population Share (lnfrac_black panel specs)
Among the 7 successful core lnfrac_black panel specs (runs 0001-0006, with treatment f_int_1930):
- **6 of 6 valid specifications yield negative coefficients** (all expected sign)
- **6 of 6 (100%) significant at p < 0.05** under baseline clustering
- Coefficient range: [-0.166, -0.147]
- The results are highly robust across control variations (LOO geography, New Deal controls) and weighting

The Table 1 cross-sectional baselines (runs 0025-0030) show positive coefficients for lnfrac_black, but this is a different design (pre-flood cross-section) measuring the initial correlation between flood exposure and Black population, not the causal effect.

**Assessment: STRONG robustness** for the panel DiD results on lnfrac_black.

### G1: Black Population (lnpopulation_black panel specs)
Among 4 successful specs (runs 0007-0009, 0011):
- All negative coefficients, consistent with expected sign
- 3 of 4 significant at p < 0.05 (run_0012 marginally insignificant under HC1)
- Coefficient range: [-0.142, -0.124]

**Assessment: MODERATE robustness** (smaller sample of successful specs, one marginally insignificant under HC1).

### G2: Capital Outcomes
Results are mixed across different capital measures:
- **lntractors**: 3 of 3 successful specs positive and significant (coef range: [0.511, 0.650])
- **lnvalue_equipment**: Only 2 of 4 specs succeed; the _nd baseline shows near-zero effect (coef=-0.001, p=0.994)
- **lnmules_horses**: 3 of 4 succeed; coefficients positive but only 1 of 3 significant (coef range: [0.058, 0.114])
- **lnavfarmsize**: 2 of 4 succeed; both insignificant (coef range: [-0.026, 0.022])
- **Table 5 outcomes** (land, building values): All 6 specs fail (singular matrix)

**Assessment: WEAK to MODERATE robustness.** Strong for tractors, weak/null for equipment value, farm size, and mules/horses. The high failure rate (14/26 = 54%) prevents thorough assessment.

## Top Issues

1. **Very high failure rate (25/56 = 44.6%)**: Matrix singularity is pervasive, especially in G2 (14/26 = 54%) and for lagged DV leave-one-out and New Deal control additions. This is a genuine data limitation from the time-interacted control structure creating near-collinearity.

2. **rc/inference/hc1_* specs in specification_results.csv**: Runs 0006, 0012, 0018, 0024 are inference variants (identical coefficients to their parent baseline with HC1 SEs). These should be in inference_results.csv. They duplicate information already present in the inference_results.csv file.

3. **G2 equipment value baseline is null**: The only successful equipment value baseline (run_0037, _nd variant) has coefficient -0.0006 with p=0.994. This contradicts the paper's Table 4 finding. The non-_nd baseline (run_0036) failed. This suggests the Python replication may not exactly match the Stata results.

4. **lnpopulation outcomes classified as noncore**: Total population (lnpopulation) is used in the paper as a comparison outcome but is a different claim from Black population. 8 rows (6 panel + 2 in cross-section overlap) are classified as noncore_alt_outcome.

5. **Table 1 cross-sectional specs use different design**: Runs 0025-0030 use flood_intensity in a cross-sectional long-difference design rather than the main panel FE design. They are correctly classified as baselines but represent a fundamentally different estimand.

## Recommendations

1. **Investigate G2 equipment value replication**: The near-zero coefficient on lnvalue_equipment_nd (run_0037) may indicate a data loading or variable construction issue compared to the paper's Stata code.

2. **Move rc/inference/hc1_* to inference_results.csv**: These 4 rows are conceptually inference variants and create redundancy with the existing inference_results.csv entries.

3. **Consider reducing control dimensionality**: The 80-135 time-interacted controls create persistent singularity issues. A more parsimonious control specification (e.g., using fewer interaction terms) could recover more specifications.

4. **Add design variants**: The surface planned design/panel_fixed_effects/estimator/long_difference but no such specs appear in specification_results.csv. A long-difference estimator could complement the panel FE approach.
