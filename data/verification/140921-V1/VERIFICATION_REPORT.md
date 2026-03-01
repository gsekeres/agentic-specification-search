# Verification Report: 140921-V1

## Baseline Groups

- **G1**: Effect of London Season interruption on out-marriage (peers' daughters marrying commoners)
  - Baseline spec_run_ids: `140921-V1_run_001`, `140921-V1_run_002`
  - Baseline spec_ids: `baseline`, `baseline__table2_panelb_col1`
  - Expected sign: positive (interruption increases out-marriage probability)
  - Estimator: probit marginal effects, cluster(byear)
  - Panel A: N=644; Panel B: N=484 (adds distlondon control)

- **G2**: Effect of London Season interruption on wealth mismatch between spouses
  - Baseline spec_run_ids: `140921-V1_run_023`, `140921-V1_run_024`
  - Baseline spec_ids: `baseline`, `baseline__table2_panelb_col3`
  - Expected sign: positive (interruption increases mismatch in landholding ranks)
  - Estimator: OLS, cluster(byear)
  - Panel A: N=324; Panel B: N=260 (adds distlondon control)

## Counts

- **Total rows**: 50
- **Core**: 49
- **Non-core**: 1
- **Invalid**: 0
- **Baselines**: 4

## Category Breakdown

| Category | Count |
|----------|-------|
| core_sample | 17 |
| core_controls | 14 |
| core_funcform | 10 |
| core_method | 6 |
| core_fe | 2 |
| noncore_alt_outcome | 1 |

## Sign and Significance

### G1 (cOut ~ syntheticT): 24 core specifications
- Positive coefficient: 24/24 (100%)
- Significant at 5%: 14/24 (58%)
- Coefficient range: [0.0015, 0.0546]
- Baseline coefficients: 0.0045 (Panel A, p=0.029), 0.0062 (Panel B, p=0.013)
- Non-significant specs are primarily subsample restrictions (age_18_30, age_18_33, age_16_32) which reduce power, the no-controls specification, the birth-year FE specification which absorbs identifying variation, and the quadratic treatment form.

### G2 (fmissmatch ~ syntheticT): 25 core specifications
- Positive coefficient: 23/25 (92%)
- Significant at 5%: 19/25 (76%)
- Coefficient range: [-0.9741, 42.2094]
- Baseline coefficients: 0.524 (Panel A, p=0.014), 0.512 (Panel B, p=0.025)
- Two negative coefficients: run_034 (fmissmatch2_signed, coef=-0.537) uses the signed mismatch where negative indicates the same directional effect; run_047 (quadratic_syntheticT, coef=-0.974) reflects the linear term of a quadratic specification.
- The birth-year FE specification (run_036) is non-significant (p=0.30), expected since FE absorbs variation with only 22 clusters.
- The aggressive trim (5-95%) and narrow age subsample also lose significance due to reduced N.

## Assessment

- **G1: MODERATE robustness**. The positive effect of Season interruption on out-marriage is consistently positive across all 24 core specs (100%), but only 58% reach significance at 5%. The effect is fragile to power reductions from sample restrictions and absorbing birth-year FE. This is expected given the small sample (N=644 max) and limited cluster count (22 birth years).

- **G2: STRONG robustness**. The positive effect on wealth mismatch is robust across 92% of core specs (positive coefficient) with 76% significant at 5%. The result is stable across control variations, sample trimming, and outcome transforms. Only the birth-year FE, aggressive trimming, and quadratic treatment form lose significance.

## Issues Found

1. **Non-core row**: run_022 changes the G1 outcome from `cOut` (married commoner) to `mheir` (married heir), which is a different claim object. Correctly classified as `noncore_alt_outcome`.

2. **No infer/* rows in specification_results.csv**: Correct. Inference variants are in inference_results.csv (5 rows with HC1 variants for both baseline groups).

3. **Inference coefficient discrepancy (WARN)**: Inference results rows for G1 baselines (infer_001, infer_002) show slightly different coefficients from the specification_results baselines. This is because the baselines use probit marginal effects while the HC1 inference variants use OLS (the coefficient changes from marginal effect to OLS coefficient). This is expected behavior, not an error.

4. **Duplicate effective specifications**: Some control variations produce identical results to baselines (e.g., dropping distlondon from Panel B reverts to Panel A; minimal_no_distlondon equals Panel A baseline).

5. **Quadratic treatment specifications**: run_021 (G1) and run_047 (G2) use quadratic syntheticT. The linear coefficient in a quadratic model is not directly comparable to the baseline linear-only coefficient. These specs are validly core but their coefficients should be interpreted cautiously.

## Recommendations

- Consider adding the logit estimator variant for G1 (planned in surface but not executed).
- The duplicate-effective specs (LOO drop_distlondon = Panel A baseline) could be flagged to avoid double-counting in spec curve analyses.
- Birth-year FE specifications absorb the identifying variation and should be interpreted as stress tests rather than preferred alternatives.
- The quadratic treatment form coefficients are difficult to interpret; consider reporting the marginal effect at the mean instead.
