# Specification Search Report: 111185-V1

## Paper
**Title**: Optimal Climate Policy When Damages are Unknown
**Author**: Ivan Rudik
**Journal**: American Economic Journal: Economic Policy, 2020
**Paper Classification**: Structural calibration

## Surface Summary

- **Baseline groups**: 1 (G1: Table 1 damage parameter estimation)
- **Design code**: cross_sectional_ols
- **Baseline spec**: `reg log_correct logt` (OLS, classical SE, N=43)
- **Budget**: 65 core specs max, 5 exploration specs max
- **Seed**: 111185
- **Controls subset sampler**: exhaustive_blocks (no random sampling needed)

### Regression Scope Note
This paper is primarily a structural/calibration paper. The ONLY reduced-form regression is Table 1: an OLS of log damages on log temperature (N=43) using Howard & Sterner (2017) meta-analysis data. This regression estimates damage function parameters that feed into the structural dynamic programming model. All specification search operates on this single calibration regression.

## Baseline Reproduction

| Parameter | Original | Replicated | Match |
|-----------|----------|------------|-------|
| Coefficient (d2 = logt) | 1.88 | 1.8820 | EXACT |
| SE (logt) | 0.45 | 0.4505 | EXACT |
| N | 43 | 43 | EXACT |
| R-squared | 0.299 | 0.2986 | EXACT |
| p-value | 0.00015 | 1.503e-04 | EXACT |

The baseline is exactly reproduced at reported precision.

## Execution Summary

### Counts

| Category | Planned | Executed | Failed | Skipped |
|----------|---------|----------|--------|---------|
| baseline | 1 | 1 | 0 | 0 |
| design/* | 1 | 1 | 0 | 0 |
| rc/controls/single | 10 | 10 | 0 | 0 |
| rc/controls/sets | 2 | 2 | 0 | 0 |
| rc/controls/progression | 3 | 3 | 0 | 0 |
| rc/sample | 8 | 8 | 0 | 0 |
| rc/form | 1 | 1 | 0 | 0 |
| rc/preprocess | 4 | 4 | 0 | 0 |
| rc/weights | 2 | 2 | 0 | 0 |
| rc/joint (interactions) | 6 | 6 | 0 | 0 |
| infer/* | 4 | 4 | 0 | 0 |
| **Total core** | **42** | **42** | **0** | **0** |
| explore/* | 3 | 0 | 0 | 3 |
| diag/* | 4 | 4 | 0 | 0 |

### Exploration specs (not executed per contract)
- `explore/form/outcome/level`: Levels regression changes the estimand concept
- `explore/form/model/levels_quadratic`: Quadratic-in-levels changes the estimand
- `explore/preprocess/treatment/include_zero_damage_obs`: Changes sample + outcome transformation (asinh)

These are not included in `specification_results.csv` per the surface contract.

## Key Findings

### Coefficient Distribution
- **Baseline**: 1.882 (SE=0.451, p=1.50e-04)
- **Median across all specs**: ~1.56
- **Range**: [-1.663, 2.418]
- **Interquartile range**: ~[0.94, 1.90]

### Significance Patterns
- **Significant at p<0.05**: 32 of 42 specs (76%)
- **Significant at p<0.01**: 23 of 42 specs (55%)
- **Not significant at p<0.10**: 3 of 42 specs (7%)

### Notable Sensitivity

1. **Outlier sensitivity is extreme**: The Weitzman 12C observation (Cook's D=2.41) drives much of the baseline result. Dropping it alone reduces the coefficient from 1.88 to 0.94 (p=0.026). Trimming to [5%, 95%] percentiles gives coefficient 0.37 (p=0.287, insignificant).

2. **Robust SEs roughly double the classical SE**: HC1 SE=0.92 (vs classical 0.45); HC3 SE=1.14; cluster SE=1.14. The baseline p-value moves from 0.0002 to ~0.10 with HC3 or clustered SEs, just at the boundary of conventional significance.

3. **Temperature adjustments attenuate the coefficient**: FUND/NASA/AVG temperature adjustments reduce the coefficient to 1.11-1.28, though still significant.

4. **WLS with inverse-variance proxy flips the sign**: WLS with 1/t^2 weights gives coefficient -0.22 (p=0.42). This extreme sensitivity reflects that the WLS dramatically down-weights the high-temperature observations that drive the positive relationship. This weighting scheme may not be appropriate given the log-log functional form.

5. **Quadratic treatment term is highly significant**: Adding logt^2 reveals strong nonlinearity (R^2 jumps from 0.30 to 0.67), but the linear term on logt flips to -1.66. This suggests the power-law exponent is not constant across the temperature range.

6. **Controls for study methodology matter**: Adding Method_1 (CGE/enumerative) reduces the coefficient to 1.18. The full control progression (all 10 controls) gives 1.11 (p=0.031).

7. **Dropping grey literature strengthens the result**: Coefficient increases to 2.35 (p<0.001) when restricting to peer-reviewed published studies only.

### Robustness Assessment
The baseline result (d2 = 1.88) is **moderately robust** with important caveats:
- The coefficient is consistently positive across most specifications (38/42)
- Statistical significance at p<0.05 holds in the majority of specs (32/42)
- However, the result is highly sensitive to a single extreme observation (Weitzman 12C)
- Robust/clustered standard errors erode significance substantially (p goes to ~0.10)
- The functional form assumption (constant power-law exponent) is rejected by the quadratic test

## Diagnostics Summary

| Diagnostic | Result | Interpretation |
|------------|--------|----------------|
| Cook's D | max=2.41, 4 obs > 4/N | Severe influence problems; Weitzman 12C observation is an extreme outlier |
| Jarque-Bera | stat=18.9, p<0.001 | Residuals are strongly non-normal (high skew and kurtosis) |
| Breusch-Pagan | LM=2.26, p=0.133 | No strong evidence of heteroskedasticity with classical test |
| Ramsey RESET | F=22.3, p<0.001 | Strong evidence of functional form misspecification |

The diagnostics paint a concerning picture: non-normal residuals, extreme influential observations, and clear functional form misspecification. These issues are consistent with the sensitivity findings above.

## Deviations and Notes

1. **rc/controls/sets/study_characteristics_basic** is identical to **rc/controls/progression/study_type** (both use Market + Grey + Preindustrial). Both are retained as planned in the surface since they occupy different spec_id slots.

2. **rc/weights/main/wls_inverse_variance_proxy**: No direct study-level standard errors or precision measures are available in the Howard & Sterner dataset. We used 1/t^2 as a rough precision proxy (higher-temperature damage estimates are plausibly less precise). The resulting sign flip to -0.22 reflects the extreme down-weighting of high-temperature observations and should be interpreted with caution. An alternative proxy (e.g., Year, study replication count) might yield different results.

3. **Cluster SE by study**: With 23 unique Primary_Author clusters from 43 observations, many clusters are singletons. The cluster SE (1.14) is similar to HC3 (1.14), as expected when many clusters have only 1 observation.

4. **Full control progression (10 controls, N=43)**: With only 32 degrees of freedom remaining, this is at the margin of feasibility. No perfect collinearity was detected, but the model may be overfitting. Method_4 was excluded from the pool because it is all zeros in the regression sample.

## Software Stack

- **Python** 3.x
- **pandas** (data loading and manipulation)
- **numpy** (numerical operations)
- **statsmodels** 0.14.x (OLS, WLS, robust SEs, diagnostics)
- **scipy** (Jarque-Bera test, skewness/kurtosis)

## Output Files

- `specification_results.csv`: 42 specification rows (baseline + design + rc + infer)
- `diagnostics_results.csv`: 4 diagnostic runs
- `spec_diagnostics_map.csv`: 4 linkages (all baseline_group scope)
- `scripts/paper_analyses/111185-V1.py`: Executable analysis script
