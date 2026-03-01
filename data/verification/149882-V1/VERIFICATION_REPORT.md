# Verification Report: 149882-V1

## Paper
Dhar, Jain, & Jayachandran, "Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India" (AER)

## Baseline Groups Found

### G1: Gender attitude index (Table 1.2)
- **Baseline spec_run_id**: 149882-V1_G1_baseline
- **Baseline spec_id**: baseline
- **Claim**: ITT effect of Breakthrough gender-equality curriculum on gender attitude index (E_Sgender_index2, standardized inverse-covariance-weighted index)
- **Baseline coefficient**: 0.2179 (p < 0.001, N=13,799)
- **Expected sign**: Positive (curriculum improves gender attitudes)

### G2: Behavior index (Table 1.2)
- **Baseline spec_run_id**: 149882-V1_G2_baseline
- **Baseline spec_id**: baseline
- **Claim**: ITT effect of Breakthrough gender-equality curriculum on self-reported behavior index (E_Sbehavior_index2, standardized inverse-covariance-weighted index)
- **Baseline coefficient**: 0.1844 (p = 2.0e-05, N=13,784)
- **Expected sign**: Positive (curriculum improves gender-related behaviors)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 32 |
| Valid (run_success=1) | 32 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 32 |
| Non-core | 0 |
| Baseline rows | 2 |
| Inference variants (inference_results.csv) | 6 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 6 |
| core_controls | 22 |
| core_sample | 2 |
| core_data | 2 |

## Robustness Assessment

### G1: Gender attitude index

- **Sign consistency**: 16/16 specifications (100%) produce a positive coefficient.
- **Statistical significance**: 16/16 (100%) significant at 1%.
- **Coefficient range**: [0.2051, 0.2656] (baseline = 0.2179).
- **Inference variants**: HC1 (p < 0.001), HC3 (p < 0.001), cluster at district (p = 0.001). All remain highly significant, even with coarse 4-district clustering.

### G2: Behavior index

- **Sign consistency**: 16/16 specifications (100%) produce a positive coefficient.
- **Statistical significance**: 16/16 (100%) significant at 1%.
- **Coefficient range**: [0.1824, 0.2033] (baseline = 0.1844).
- **Inference variants**: HC1 (p < 0.001), HC3 (p < 0.001), cluster at district (p = 0.010). G2 is slightly less significant under coarse district clustering than G1, but still significant at 1%.

### Controls sensitivity
- The 11 control specifications per group (progression, LOO, sets) produce coefficients tightly clustered around the baseline.
- The LASSO extended controls (run_008/run_024 for each group) produce slightly smaller coefficients due to smaller sample (N=10,697 vs 13,799) from requiring non-missing LASSO-selected covariates.

### Sample sensitivity
- Trimming outcome at 1/99th percentile preserves the result in both groups, with near-identical coefficients to the baseline.

### Data construction sensitivity
- Equal-weight index construction (rc/data/index_construction/equal_weight_index) preserves the result in both groups, with similar coefficients.

## Top Issues

1. **No major issues identified.** All 32 specifications produce positive, highly significant coefficients. The result is extremely robust across control sets, estimator variants, sample restrictions, and inference methods.

2. **Moderate spec count (32)**: With 16 specs per baseline group, the specification search is reasonably thorough but below the 50-spec target. The surface planned explore/* specs (subgroup analyses by gender/grade/school type) that were not included in specification_results.csv, which is correct per the protocol.

3. **District-level clustering for G2**: The only partial weakness is that G2's baseline becomes p=0.010 under district-level clustering (only 4 clusters), which is still significant but relies on very few clusters for asymptotic validity.

4. **No functional form variations**: The paper uses an RCT with standardized indices as outcomes, so there is limited scope for functional form variation. This is not a deficiency but a feature of the design.

## Structural Checks

- **spec_run_id uniqueness**: Confirmed unique across all 32 rows.
- **baseline_group_id present**: Yes, all rows have G1 or G2.
- **No infer/* rows in specification_results.csv**: Confirmed.
- **coefficient_vector_json structure**: All rows contain required keys (coefficients, inference, software, surface_hash, design).
- **Inference canonical match**: All rows use infer/se/cluster/school, matching the surface's canonical inference plan.
- **Numeric fields finite**: All finite for all run_success=1 rows.

## Recommendations

1. The surface and runner performed well. No corrections needed.
2. Future runs could add explore/* specs (gender/grade subgroups) to a separate exploration_results.csv for additional context.
3. Wild bootstrap inference could be added as an inference variant given the 4-district stratification.

## Conclusion

The specification search provides **STRONG support** for both baseline claims. Both the gender attitude effect (G1) and the behavior effect (G2) are robust to all 16 specification variants tested, with 100% sign consistency and 100% significance at the 1% level. The results are also robust to alternative inference methods including conservative district-level clustering.
