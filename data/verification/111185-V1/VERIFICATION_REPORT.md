# Verification Report: 111185-V1

**Paper**: "Optimal Climate Policy When Damages are Unknown" (Rudik, 2020, AEJ: Economic Policy)
**Verification date**: 2026-02-13
**Verifier**: Post-run verification agent

---

## 1. Baseline Groups

### G1: Table 1 Damage Parameter Estimation

- **Claim object**: Elasticity of climate damages with respect to temperature (power-law exponent d2) estimated via OLS of log damages on log temperature using Howard & Sterner (2017) meta-analysis data (N=43).
- **Expected sign**: Positive (higher temperatures increase climate damages).
- **Baseline spec_run_id**: `111185-V1_G1_baseline`
- **Baseline spec_id**: `baseline`
- **Baseline coefficient**: 1.882, SE=0.451, p=0.00015, N=43, R2=0.299
- **Reproduction quality**: EXACT at reported precision for all five parameters.

This is the only reduced-form regression in the paper. All other results are structural/calibration outputs that depend on the d2 parameter estimated here.

---

## 2. Counts

| Metric | Count |
|--------|-------|
| Total rows | 42 |
| Baseline | 1 |
| Core tests | 41 |
| Non-core | 0 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| baseline | 1 |
| core_controls | 16 |
| core_sample | 10 |
| core_preprocess | 5 |
| core_inference | 4 |
| core_weights | 3 |
| core_funcform | 2 |
| core_method | 1 |

---

## 3. Surface Alignment

### Surface completeness

All spec_ids planned in `SPECIFICATION_SURFACE.json` appear in `specification_results.csv`:
- 1 baseline spec: present
- 1 design spec: present
- 30 rc spec_ids: all present
- 4 infer spec_ids: all present
- 6 joint/interaction spec_ids: all present
- 3 explore spec_ids: correctly excluded from results

No spurious spec_ids appear in results that are not in the surface.

### Baseline group alignment

The surface defined one baseline group (G1). All 42 rows are assigned to G1 in the results. No missing or spurious baseline groups.

### Duplicate-value specs (noted, not flagged as invalid)

1. **design/cross_sectional_ols/estimator/ols** and **rc/weights/main/unweighted** are both mechanically identical to the baseline (same coefficient, SE, p-value, N, R2). This is expected per the surface design: the design spec confirms the estimator choice, and the unweighted spec is the explicit reference point for the weights axis.

2. **rc/controls/sets/study_characteristics_basic** and **rc/controls/progression/study_type** produce identical results (coef=1.699, same controls: Market, Grey, Preindustrial). This was noted in `SPECIFICATION_SEARCH.md` and was retained by design since they occupy different spec_id slots.

---

## 4. Claim Object Preservation

All 42 rows preserve the baseline claim object (log-log elasticity of climate damages w.r.t. temperature). Specific notes on borderline cases:

### Treatment variable variants (preserved)
- **Temperature adjustment specs** (FUND, NASA, AVG): Treatment changes from `logt` to `log(t - Temp_adj_X)`. The conceptual estimand (log-log elasticity) is preserved; the temperature baseline shifts. Classified as core_preprocess with confidence 0.95.

### Functional form variants (preserved with caveats)
- **Quadratic treatment** (`rc/form/model/quadratic_treatment`): Treatment listed as `logt + logt^2`. The focal coefficient is the linear term on logt, which in the presence of a quadratic is no longer a constant elasticity but a conditional partial derivative. The surface explicitly planned this as core, and the coefficient_vector_json includes the logt^2 term and joint F-test. Classified as core_funcform with confidence 0.85.
- **Joint drop Weitzman + quadratic** (`rc/joint/outlier_form/drop_weitzman_quadratic`): Same interpretation issue. Classified as core_funcform with confidence 0.80.

### Outcome variable variants (preserved)
- **Winsorized outcome** (`rc/preprocess/outcome/winsor_1_99`): Outcome listed as `log_correct (winsorized 1%/99%)`. Same variable concept with tails capped. Classified as core_preprocess with confidence 1.0.

### No outcome/treatment drift to different concepts
No rows change the outcome concept (e.g., to levels) or the treatment concept (e.g., to a different regressor). The three explore specs that would have changed the estimand concept were correctly excluded.

---

## 5. Issues and Observations

### Issue 1: Extreme outlier sensitivity
The Weitzman 12C observation (Cook's D=2.41, compared to threshold 4/N=0.093) dominates the baseline result. Dropping it alone halves the coefficient from 1.88 to 0.94. This is the most important sensitivity finding and is well-documented.

### Issue 2: Classical SE may be inappropriate
The baseline uses classical (homoskedastic) standard errors. The Breusch-Pagan test does not reject (p=0.133), but the RESET test strongly rejects functional form (F=22.3, p<0.001), and HC1/HC2/HC3 SEs are 2-2.5x larger than classical SEs. The baseline p-value of 0.00015 degrades to 0.042 (HC1), 0.063 (HC2), 0.098 (HC3), and 0.100 (clustered). This is a substantively important finding.

### Issue 3: WLS sign flip
The WLS with 1/t^2 weights flips the coefficient sign to -0.22 (p=0.42). The SPECIFICATION_SEARCH.md correctly notes this is because the weighting heavily down-weights high-temperature observations, and the proxy may not be appropriate for the log-log form. The result is valid as executed but should be interpreted cautiously as noted.

### Issue 4: Functional form misspecification
The Ramsey RESET test (F=22.3, p<0.001) strongly rejects the linear log-log specification. Consistent with this, the quadratic treatment spec shows that adding logt^2 dramatically improves R2 (0.30 to 0.67) and flips the linear coefficient to -1.66. This suggests the constant-elasticity assumption (power law with fixed exponent) is rejected by the data.

### Issue 5: Numerically identical spec pairs
Two pairs of specs produce identical results:
- `design/...ols` = `baseline` (by construction)
- `rc/weights/main/unweighted` = `baseline` (by construction)
- `rc/controls/sets/study_characteristics_basic` = `rc/controls/progression/study_type` (same controls)

These are documented and intentional per the surface design.

---

## 6. Diagnostics Coverage

All four planned diagnostics were executed and linked to the baseline via `spec_diagnostics_map.csv`:

| Diagnostic | Result | Concern Level |
|------------|--------|---------------|
| Cook's D | max=2.41, 4 obs > 4/N | HIGH |
| Jarque-Bera | stat=18.9, p<0.001 | MODERATE |
| Breusch-Pagan | LM=2.26, p=0.133 | LOW |
| Ramsey RESET | F=22.3, p<0.001 | HIGH |

The diagnostics are well-integrated with the specification battery: the outlier sensitivity (Cook's D) is directly tested via sample restriction specs, and the functional form concern (RESET) is addressed by the quadratic treatment spec.

---

## 7. Validity Summary

| Criterion | Status |
|-----------|--------|
| spec_run_id uniqueness | PASS (42 unique) |
| baseline_group_id present | PASS (all G1) |
| spec_id consistent with spec_tree_path | PASS (identical for all rows) |
| Numeric fields finite | PASS (all non-null, finite) |
| Surface-to-results alignment | PASS (all planned specs present, no spurious) |
| Explore specs correctly excluded | PASS |
| Baseline reproduction | PASS (exact match) |
| No invalid rows | PASS |
| No claim-object drift | PASS (no rows reclassified to non-core) |

---

## 8. Recommendations

1. **Consider dropping one of the duplicate spec pairs**: `rc/controls/sets/study_characteristics_basic` and `rc/controls/progression/study_type` produce identical results. A future surface iteration could consolidate these into a single spec_id.

2. **Consider dropping `rc/weights/main/unweighted`**: This is mechanically identical to the baseline and adds no information. It was included as the explicit weights-axis reference point, but the baseline already serves that role.

3. **Quadratic spec coefficient reporting**: For `rc/form/model/quadratic_treatment`, the reported coefficient (-1.66) is the partial on logt conditional on logt^2. The `coefficient_vector_json` correctly includes the logt^2 coefficient and joint F-stat. Consider adding a note in the surface that the focal parameter interpretation changes in this spec.

4. **WLS proxy quality**: The 1/t^2 inverse-variance proxy is acknowledged as rough. Future iterations could explore alternative precision proxies (e.g., inverse of the range of reported damage estimates, or study-level sample size if available).

5. **No block-combination subset sampling was needed**: The surface correctly identified that with 4 control blocks and a 4-control cap, exhaustive block enumeration suffices. The `sampling.controls_subset_sampler = "exhaustive_blocks"` setting was appropriate.
