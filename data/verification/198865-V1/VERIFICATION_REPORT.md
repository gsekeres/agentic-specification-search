# Verification Report: 198865-V1

## Paper
- **Title**: Estimating Models of Supply and Demand: Instruments and Covariance Restrictions
- **Authors**: Alexander MacKay and Nathan H. Miller
- **Journal**: American Economic Journal: Microeconomics

## Baseline Groups

### G1: Cement Demand Elasticity
- **Baseline spec_id**: baseline
- **Claim**: Price elasticity of cement demand is negative (OLS, log-log, market FE)
- **Coefficient**: -0.466 (SE = 0.176, p = 0.008)
- **Expected sign**: Negative

### G2: Airlines Demand Elasticity
- **Baseline spec_id**: baseline_airlines
- **Claim**: Price elasticity of airline demand is negative (nested logit, route FE, product controls)
- **Coefficient**: -0.107 (SE = 0.005, p < 0.001)
- **Expected sign**: Negative

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **57** |
| Baselines | 2 |
| Core tests (non-baseline) | 46 |
| Non-core tests | 6 |
| Invalid | 3 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Group |
|----------|-------|-------|
| core_controls | 16 | G1: 10, G2: 6 |
| core_sample | 16 | G1: 9, G2: 7 |
| core_inference | 4 | G1: 2, G2: 2 |
| core_fe | 3 | G1: 3, G2: 0 |
| core_funcform | 4 | G1: 2, G2: 2 |
| core_method | 5 | G1: 5, G2: 0 |
| noncore_heterogeneity | 4 | G1: 0, G2: 4 |
| noncore_alt_treatment | 2 | G1: 0, G2: 2 |
| noncore_placebo | 0 | -- |
| noncore_alt_outcome | 0 | -- |
| noncore_diagnostic | 0 | -- |
| invalid | 3 | G1: 3, G2: 0 |

## Top 5 Most Suspicious Rows

1. **iv/method/ols** (Invalid): Exact duplicate of baseline. Coefficient (-0.4663), SE (0.1763), and all other fields are identical. Should be removed or flagged as redundant.

2. **iv/instruments/single_cpi_wage_hr** (Invalid): First-stage F-statistic is 2.43, far below the weak instrument threshold (F < 10). The coefficient is +6.51, a sign reversal driven by weak instruments. This is not a meaningful test of the demand elasticity.

3. **robust/form/quadratic** (Invalid): Reports the linear term (19.37) of a quadratic-in-log-price specification. This is not an elasticity and cannot be compared to the baseline without evaluating at specific price levels. The coefficient is mechanically different in interpretation.

4. **robust/sample/direct_only** (Non-core): This subsample switches from nested logit to simple logit (no nesting parameter), changing the model structure. The coefficient (-1.12) is an order of magnitude larger than the baseline (-0.107), confirming the model is fundamentally different.

5. **robust/sample/connecting_only** (Non-core): Same issue as direct_only -- switches to simple logit. The model_type field confirms "OLS Logit" vs baseline "OLS Nested Logit".

## Additional Notes

### Interpretive Gaps in Cement Specifications

- The leave-one-out (LOO) controls for cement start from the **full-controls model** (iv/controls/full), not from the baseline (no controls). This means the LOO specs are dropping controls from a model with coefficient -0.017 (essentially zero), not from the baseline coefficient of -0.466. Users should be aware that the LOO specs are robustness checks around the full-controls model, not the baseline.

- The control progression specs (robust/control/add_*) build up from no controls to full controls, with the final step (add_log_buildings) producing the same result as iv/controls/full. This is appropriate.

### IV Instrument Variations

- The IV instrument variations test the same demand claim but with different identifying assumptions. The natural gas price IV (cpi_ng_price) produces a **positive** coefficient (+2.08), which contradicts the demand law. This likely reflects instrument invalidity (the instrument may be correlated with demand shocks). The first-stage F is 57.6, so this is not a weak instrument problem but rather an exclusion restriction violation.

- The wage IV (cpi_wage_hr) has both weak instrument and likely exclusion restriction problems.

### Heterogeneity Specifications

- The four heterogeneity specs (legacy_carriers, lcc_carriers, long_haul, short_haul) restrict the airlines sample to specific subgroups. While these use the same outcome and treatment variables, they are primarily testing whether the elasticity differs across carrier types or route distances -- i.e., heterogeneity analysis rather than robustness of the main claim. Classified as non-core.

### Recommendations for Spec Search Script

1. **Remove iv/method/ols**: It is an exact duplicate of baseline and adds no information.
2. **Flag weak instruments**: Specs with first-stage F < 10 should be automatically flagged as potentially invalid.
3. **Handle quadratic terms properly**: When adding quadratic functional form variations, either report the marginal effect at the mean or clearly note that the linear term alone is not comparable.
4. **Track model structure changes**: When a sample restriction forces a change in model type (nested logit to simple logit), this should be flagged as a structural change, not a simple sample restriction.
5. **Align LOO specs with baseline**: LOO controls should ideally be centered on the baseline specification, or the documentation should clearly state which model they are perturbing.
