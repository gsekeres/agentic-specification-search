# Verification Report: 111185-V1

## Paper
Rudik (2020), "Optimal Climate Policy When Damages are Unknown", AEJ: Economic Policy

## Baseline Groups Found

### G1: Damage exponent estimation (Table 1)
- **Baseline spec_run_id**: 111185-V1_spec_001
- **Baseline spec_id**: baseline
- **Claim**: Elasticity of climate damages with respect to temperature (d2 in power-law damage function D = exp(d1) * T^d2)
- **Baseline coefficient (logt)**: 1.882 (SE=0.451, p=0.00015, N=43, R2=0.299)
- **Expected sign**: Positive (higher temperature -> higher damages)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 62 |
| Valid (run_success=1) | 62 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 62 |
| Non-core | 0 |
| Baseline rows | 1 |
| Inference variants (inference_results.csv) | 3 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline) | 1 |
| core_controls | 43 |
| core_sample | 11 |
| core_funcform | 5 |
| core_preprocess | 2 |

## Robustness Assessment

### Sign consistency
- **59 of 62** specifications (95.2%) produce a positive coefficient, consistent with the baseline sign.
- 3 specifications produce negative coefficients, all in the functional form category:
  - `rc/form/model/levels_quadratic` (correct_d ~ t + t^2): The linear temperature term is negative because the quadratic term captures the convex damage relationship.
  - Two joint specs with levels treatment + restricted samples also show near-zero or negative coefficients.

### Statistical significance
- **53 of 62** specifications (85.5%) are significant at the 5% level.
- **51 of 62** specifications (82.3%) are significant at the 1% level.
- 9 specifications have p >= 0.05, primarily from:
  - Temporal splits (small subsamples, N ~ 20)
  - Joint specs with small subsamples and controls
  - Functional form changes (levels outcome/treatment)

### Controls sensitivity
- All 35 control specifications are positive and significant at 5%.
- Coefficient range across control specs: [1.384, 2.182]
- The result is highly robust to any combination of meta-analytic study-characteristic controls.

### Sample sensitivity
- 11 sample restrictions: all positive, 6 of 11 significant at 5%.
- Non-significant specs are due to small subsamples (temporal splits reduce N to ~20).
- Dropping catastrophic estimates or grey literature preserves significance.

### Inference sensitivity (from inference_results.csv)
- **Classical SE**: 0.451 (p = 0.00015) -- baseline
- **HC1 (robust)**: 0.924 (p = 0.048) -- still significant at 5%
- **HC2**: 1.012 (p = 0.070) -- borderline, not significant at 5%
- **HC3**: 1.136 (p = 0.105) -- not significant at 10%

This is a notable finding: the heteroskedasticity-robust standard errors are approximately double the classical SEs. With N=43 and potential heteroskedasticity in the meta-analysis data, the classical SEs may substantially overstate precision. The result remains directionally strong but statistically fragile under conservative inference.

## Top Issues

1. **Inference fragility**: The baseline uses classical (homoskedastic) SEs. Heteroskedasticity-robust SEs (HC1-HC3) roughly double the standard error. HC2 and HC3 make the baseline result insignificant at conventional levels. This is the primary vulnerability.

2. **Functional form sensitivity**: Level-outcome and polynomial specifications produce very different coefficient magnitudes and interpretations. The power-law (log-log) specification is a strong modeling choice that drives the result.

3. **Small sample**: N=43 limits the power of many robustness checks and makes the results sensitive to individual observations (the Cook's D filter drops observations and preserves significance).

4. **No outcome/treatment drift in core controls/sample specs**: All control and sample RC specs maintain log_correct ~ logt, so there is no concept drift in the large majority of specifications.

## Recommendations

1. The paper should report heteroskedasticity-robust standard errors as a robustness check, given the substantial difference from classical SEs.
2. Sensitivity to the power-law functional form assumption should be acknowledged.
3. Influence diagnostics (Cook's D) suggest checking which studies drive the result most.

## Conclusion

The specification search confirms that the baseline result (d2 = 1.88) is **robust to control specification and most sample restrictions** but **sensitive to the inference method** (classical vs robust SEs). Under heteroskedasticity-robust inference, the result is borderline significant. The functional form (log-log) is a strong assumption that drives the power-law interpretation.

Overall assessment: **MODERATE support** for the baseline claim. The point estimate is stable across specifications, but statistical significance depends on the inference assumption.
