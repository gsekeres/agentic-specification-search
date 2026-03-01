# Verification Report: 181581-V1

## Paper
Okeke, Abubakar, & Guo, "The Effect of Deploying Doctors to Primary Health Centers on Infant Mortality" (AEJ: Applied Economics)

## Baseline Groups Found

### G1: Doctor deployment effect on 7-day neonatal mortality (Table 4)
- **Baseline spec_run_ids**: 181581-V1_run_001 (no controls), 181581-V1_run_002 (basic controls)
- **Baseline spec_ids**: baseline, baseline__basic_controls
- **Claim**: ITT effect of deploying a doctor to a primary health center on 7-day neonatal mortality
- **Baseline coefficient (no controls)**: -0.00530 (p = 0.143, N = 9,126)
- **Baseline coefficient (basic controls)**: -0.00687 (p = 0.056, N = 9,124)
- **Expected sign**: Negative (doctor deployment reduces infant mortality)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 45 |
| Valid (run_success=1) | 45 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 45 |
| Non-core | 0 |
| Baseline rows | 2 |
| Inference variants (inference_results.csv) | 2 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 3 |
| core_controls | 29 |
| core_fe | 2 |
| core_sample | 3 |
| core_data | 8 |

## Robustness Assessment

### Sign consistency
- **45/45 specifications (100%)** produce a negative coefficient, consistent with the expected sign that doctor deployment reduces mortality. The sign is fully robust.

### Statistical significance
- **22/45 (48.9%)** significant at 5%.
- **32/45 (71.1%)** significant at 10%.
- **1/45 (2.2%)** significant at 1%.

The baseline result is statistically fragile. The no-controls specification (run_001, p=0.143) is not significant at any conventional level. The basic-controls specification (run_002, p=0.056) is borderline at 10%.

### Controls sensitivity (main mort7/doctor claim, 37 specs)
- All 37 mort7/doctor specs produce negative coefficients.
- 22/37 (59.5%) significant at 5%; 30/37 (81.1%) significant at 10%.
- Extended controls and LOO variations generally strengthen significance relative to the no-controls baseline, suggesting that covariate adjustment improves precision.
- Random control subsets show moderate variation in p-values (range: 0.010 to 0.090), reflecting sensitivity to specific control set composition.

### Outcome variant: 30-day mortality (mort30)
- 4 specifications use mort30 instead of mort7.
- All produce negative coefficients but none are significant at 10% (p-values: 0.153 to 0.498).
- The treatment effect fades when extending the mortality window from 7 to 30 days.

### Treatment variant: Doctor only (dropping mlp arm)
- 4 specifications drop the mlp treatment arm.
- Coefficients are negative but generally not significant (p-values: 0.059 to 0.230).
- Dropping the mlp arm reduces precision, likely because fewer facilities enter the comparison.

### Fixed effects sensitivity
- Adding quarter FE (run_033): coef = -0.00550, p = 0.131 (not significant).
- Dropping quarter FE from basic controls (run_034): coef = -0.00685, p = 0.056 (borderline).

### Sample sensitivity
- Excluding multiple births (run_035, 040, 045): coefficients range from -0.00517 to -0.00744, p-values range from 0.041 to 0.160.
- Restricting to singletons does not materially change the result.

### Inference sensitivity (from inference_results.csv)
- **Cluster at facility (canonical)**: coef = -0.00530, p = 0.143
- **HC1 (robust)**: coef = -0.00530, p = 0.261 (less significant)
- **Cluster at strata**: coef = -0.00530, p = 0.140 (similar to facility clustering)

Robust SEs without clustering make the result less significant, as expected given the cluster-randomized design.

## Top Issues

1. **Baseline not significant at conventional levels**: The primary baseline (no controls, strata FE only) has p = 0.143. Significance is achieved only with covariate adjustment (p = 0.027 with extended controls). This is a meaningful fragility: the result depends on the control set.

2. **30-day mortality shows no significant effect**: All mort30 specifications have p > 0.15, suggesting the treatment effect on mortality is concentrated in the first 7 days and does not extend to 30 days.

3. **Treatment definition sensitivity**: Dropping the mlp arm (doctor_only specs) reduces the sample/power and makes the result insignificant, though the sign remains negative.

4. **Non-unique spec_id not an issue**: All spec_ids are unique within this paper. The compound spec_ids (e.g., mort30__basic, doctor_only__basic_singleton) correctly encode the joint variations.

## Structural Checks

- **spec_run_id uniqueness**: Confirmed unique across all 45 rows.
- **baseline_group_id present**: Yes, all rows are G1.
- **No infer/* rows in specification_results.csv**: Confirmed.
- **coefficient_vector_json structure**: All rows contain required keys (coefficients, inference, software, surface_hash, design).
- **Inference canonical match**: All rows use infer/se/cluster/fid, matching the surface's canonical inference plan.
- **Numeric fields finite**: All finite for all run_success=1 rows.

## Recommendations

1. The finding should be characterized as suggestive rather than definitive: the sign is robust but statistical significance depends on the control set.
2. The 30-day mortality null result is informative and should be reported as context.
3. Future runs could add wild bootstrap inference given the moderate number of clusters (~124 facilities).

## Conclusion

The specification search provides **WEAK-TO-MODERATE support** for the baseline claim. The sign of the effect (negative, i.e., doctor deployment reduces mortality) is 100% consistent across all 45 specifications. However, statistical significance at the 5% level is achieved in only about half the specifications. The result is strongest with extended controls and weakest with no controls, alternative outcome definitions (mort30), or alternative treatment definitions (doctor_only). The effect is concentrated in 7-day mortality and depends on covariate adjustment for statistical significance.
