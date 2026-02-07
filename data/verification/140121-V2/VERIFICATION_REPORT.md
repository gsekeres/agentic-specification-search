# Verification Report: 140121-V2

## Paper
**Title**: The Labor Market Impacts of Universal and Permanent Cash Transfers: Evidence from the Alaska Permanent Fund  
**Authors**: Damon Jones, Ioana Marinescu  
**Journal**: AEJ: Policy (2023)  
**Method**: Synthetic Control Method (SCM) + Difference-in-Differences (DiD)

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | PFD does not reduce employment rates | 0 (null) | baseline/employed/ak_vs_wa, baseline/employed/ak_vs_pacificnw, baseline/employed/all_states |
| G2 | PFD increases part-time employment | + | baseline/parttime/ak_vs_wa, baseline/parttime/ak_vs_pacificnw, baseline/parttime/all_states |
| G3 | PFD slightly reduces labor force participation | - | baseline/activelf/ak_vs_wa, baseline/activelf/ak_vs_pacificnw, baseline/activelf/all_states |
| G4 | PFD reduces hours worked per week | - | baseline/hourslw/ak_vs_wa, baseline/hourslw/ak_vs_pacificnw |

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **134** |
| Baselines | 11 |
| Core tests (non-baseline) | 96 |
| Non-core tests | 27 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 70 |
| core_sample | 18 |
| core_fe | 9 |
| core_inference | 6 |
| core_funcform | 4 |
| noncore_heterogeneity | 20 |
| noncore_placebo | 5 |
| noncore_alt_outcome | 2 |

## Top 5 Most Suspicious Rows

### 1. robust/sample/post_1985 (employed)
**Issue**: Coefficient = 0.20, implausibly large for an employment rate variable (range [0,1]). All other employed coefficients in 2-state models are on the order of 1e-08. This spec restricts to post-1985 observations only (n=60) and likely suffers from a different model specification issue. The p-value is also missing (NaN SE).  
**Recommendation**: Flag as potentially invalid. Investigate whether the post-1985 restriction changes the model structure in a way that produces unreliable estimates.

### 2. Near-perfect R-squared in AK vs WA 2-state models
**Issue**: Many employed specifications using Alaska vs Washington only (2 states) have R-squared > 0.9999999999 and coefficients on the order of 1e-08. These are numerical artifacts of near-perfect collinearity when running OLS with many controls on only 76 observations across 2 states. The p=0.0 values reported for some of these are also numerical artifacts.  
**Recommendation**: The spec search script should recognize that 2-cluster models produce unreliable inference and flag these accordingly. Consider using the all_states or ak_vs_pacificnw baselines as the primary comparison.

### 3. robust/heterogeneity/tradeable/employed
**Issue**: Reports p = 1.47e-181, an astronomically small p-value that is clearly a numerical artifact. The coefficient (2.6e-08) is near-zero, making such a p-value implausible.  
**Recommendation**: This is a heterogeneity spec (non-core), but the extreme p-value should be flagged as invalid inference.

### 4. robust/oil/vs_oil_states (employed and parttime)
**Issue**: These are exact duplicates of robust/sample/comparison_oil_states/employed and robust/sample/comparison_oil_states/parttime respectively (identical coefficients and p-values). The same specification appears under two different spec_ids.  
**Recommendation**: Remove one of the duplicate pairs. The oil/vs_oil_states spec_ids are redundant.

### 5. 57 specifications with empty p-values
**Issue**: 57 of 134 specifications (43%) have empty/missing p-values. These are overwhelmingly from the AK vs WA comparison (2 states), where clustering standard errors at the state level with only 2 clusters produces NaN standard errors and thus no valid p-values.  
**Recommendation**: The spec search script should either (a) use the Conley-Taber or other small-cluster correction, (b) report wild cluster bootstrap p-values, or (c) clearly flag that 2-cluster inference is unreliable and avoid reporting specs without valid inference.

## Recommendations for the Spec-Search Script

1. **Primary baseline choice**: The ak_vs_pacificnw or all_states comparisons should be preferred over ak_vs_wa as the primary baseline, since the 2-state comparison produces degenerate inference. Many of the 57 missing p-values come from this issue.

2. **Remove duplicate specs**: robust/oil/vs_oil_states duplicates robust/sample/comparison_oil_states. Only one should be kept.

3. **Investigate post_1985 coefficient**: The 0.20 employed coefficient for the post-1985 restriction is orders of magnitude larger than all other employed coefficients and likely reflects a specification error.

4. **Collinearity warnings**: When R-squared exceeds 0.9999, the spec search should flag potential collinearity and treat the resulting coefficients and p-values with caution.

5. **Hours outcome coverage**: The hours worked outcome (G4) has only 2 baseline specs and few robustness variations. Adding more specifications for this outcome (e.g., control variations, sample restrictions) would improve coverage.
