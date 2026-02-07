# Verification Report: 114398-V1

## Paper: Competition and the Strategic Choices of Churches
**Authors**: Adam Rennhoff and Mark Owens
**Journal**: AEA

---

## Baseline Groups

### G1: Same-denomination competition effect on childcare provision
- **Baseline spec_id**: baseline
- **Outcome**: ccare (binary: church provides childcare)
- **Treatment**: same_denom_4mi (count of same-denomination churches within 4 miles)
- **Baseline coefficient**: 0.067 (p=0.170)
- **Model**: Logit with full controls and county FE, N=424

### G2: Different-denomination competition effect on childcare provision
- **Baseline spec_id**: baseline_diff_denom
- **Outcome**: ccare
- **Treatment**: diff_denom_4mi (count of different-denomination churches within 4 miles)
- **Baseline coefficient**: 0.025 (p=0.375)
- **Model**: Same regression as G1, reporting the other treatment variable
- **Note**: No robustness specs directly test this coefficient

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **60** |
| Baselines (is_baseline=1) | 2 |
| Core tests (is_core_test=1, incl. baselines) | 47 |
| Non-core tests (is_core_test=0) | 13 |
| Invalid | 1 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 16 |
| core_sample | 16 |
| core_method | 5 |
| core_funcform | 6 |
| core_fe | 2 |
| core_inference | 2 |
| noncore_heterogeneity | 8 |
| noncore_placebo | 2 |
| noncore_alt_outcome | 1 |
| noncore_alt_treatment | 1 |
| invalid | 1 |

---

## Top 5 Most Suspicious Rows

### 1. robust/sample/city_wm_rural (INVALID)
- **Issue**: Perfect/near-perfect separation. Coefficient = -29.42, SE = 749.45. This estimate is completely unreliable.
- **Sample**: N=60 churches in Williamson County rural areas.
- **Action**: Marked as invalid. Should not be included in any aggregation.

### 2. robust/cluster/county (Suspicious p-value)
- **Issue**: This LPM with county-clustered SEs reports p < 1e-14 for the treatment effect (coef=0.0027). With only 2 clusters (2 counties), clustered SEs are unreliable and may be severely downward-biased. The same coefficient in the unclustered LPM has p=0.326.
- **Action**: Kept as core_inference but with lower confidence (0.85). The extreme significance is likely an artifact of 2-cluster clustering.

### 3. robust/weights/population (Suspicious p-value)
- **Issue**: Population-weighted GLM reports coef=0.062, p < 1e-21. The standard error (0.006) is an order of magnitude smaller than in the unweighted baseline (0.049). Population weighting may dramatically change the effective sample by up-weighting large population areas.
- **Action**: Kept as core_method but flagged. The coefficient magnitude is similar to baseline but the artificially tight SEs are concerning.

### 4. robust/treatment/total_comp_4mi (Non-core alt treatment)
- **Issue**: total_comp_4mi combines same- and different-denomination competition into a single count. This changes the causal object from "same-denomination competition" to "all competition," which is a different claim.
- **Action**: Classified as noncore_alt_treatment.

### 5. robust/sample/denom_meth (Small sample separation risk)
- **Issue**: Methodist denomination subsample has only N=46. Some control coefficients show NaN SEs (large and New variables), indicating separation issues similar to city_wm_rural but less severe.
- **Action**: Kept as core_sample with low confidence (0.70).

---

## Notes on Heterogeneity Specifications

All 8 heterogeneity specs (same_x_large, same_x_williamson, same_x_highpop, same_x_highincome, same_x_new, same_x_highfp, same_x_married, same_x_under5) report the coefficient on the **interaction term**, not the main effect of competition. The treatment_var column contains the interaction variable name (e.g., same_x_highinc), not same_denom_4mi. Since the interaction coefficient is a different estimand than the main competition effect, these are classified as noncore_heterogeneity. The main effect of same_denom_4mi is still included in these models as a control, but its coefficient is not the one reported.

---

## Recommendations for Spec-Search Script

1. **Cluster specification with 2 clusters**: The county-clustered LPM (robust/cluster/county) should flag that 2-cluster clustering is unreliable. Consider either dropping this specification or noting that the inference is invalid with so few clusters.

2. **Separation detection**: The script should detect and flag specifications where coefficients or standard errors are extreme (e.g., |coef| > 20 or SE > 100), indicating quasi-complete separation. The city_wm_rural spec is unusable.

3. **Heterogeneity reporting**: When the treatment_var is an interaction term, the spec_tree_path correctly identifies it as heterogeneity. However, it would be clearer if the main effect coefficient were also extractable from these regressions (it is available in the coefficient_vector_json).

4. **Weighted estimation**: The population-weighted spec produces dramatically different standard errors. Consider whether population weighting is appropriate given the unit of analysis (churches, not populations).

5. **baseline_diff_denom as G2**: This second baseline has no dedicated robustness variations. If the diff_denom_4mi effect is a separate claim of interest, the spec search should generate robustness checks for it too.
