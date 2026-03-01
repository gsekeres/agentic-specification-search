# Verification Report: 140161-V1

**Paper**: Henry et al. — Fake news / fact-check sharing experiment (2019 French EU election)
**Paper ID**: 140161-V1
**Design**: randomized_experiment
**Verified**: 2026-02-24
**Verifier**: post-run-verifier agent (automated audit)

---

## 1. Baseline Groups Found

### G1 — Alt-facts sharing (main 3-arm experiment)

| Spec Run ID | Spec ID | Focal Arm | Outcome | Coef | SE | p-value | N |
|---|---|---|---|---|---|---|---|
| 140161-V1__spec__0001 | baseline__intent_share_fb | survey2 (Imposed) | want_share_fb | -0.0451 | 0.0160 | 0.0049 | 2537 |
| 140161-V1__spec__0002 | baseline__intent_share_fb | survey3 (Voluntary) | want_share_fb | -0.0380 | 0.0162 | 0.0187 | 2537 |
| 140161-V1__spec__0003 | baseline__share_action_g1 | survey2 (Imposed) | share_click2 | -0.0202 | 0.0060 | 0.0008 | 2537 |
| 140161-V1__spec__0004 | baseline__share_action_g1 | survey3 (Voluntary) | share_click2 | -0.0214 | 0.0055 | 0.0001 | 2537 |

**Claim object**: ATE of imposed and voluntary fact-checking on intent/action to share alt-facts on Facebook among French survey participants in the main experiment (survey < 4). Strata controls: male, low_educ, mid_educ. HC1 standard errors. All four baseline coefficients are negative (consistent with expected reduction in fake news sharing).

**Surface match**: Exact. Surface specifies Table2-PanelA-Col2 (want_share_fb, strata controls) as primary baseline and Table2-PanelA-Col6 (share_click2, strata controls) as co-baseline. Both replicated.

### G2 — Fact-check sharing (2-arm comparison)

| Spec Run ID | Spec ID | Focal Arm | Outcome | Coef | SE | p-value | N |
|---|---|---|---|---|---|---|---|
| 140161-V1__spec__0005 | baseline__factcheck_action | survey3 (Voluntary vs Imposed) | want_share_facts | -0.0284 | 0.0160 | 0.0764 | 1692 |

**Claim object**: ATE of voluntary vs imposed fact-checking on intent to share the fact-check on Facebook, restricted to the two fact-check arms (survey 2 and 3). Strata controls: male, low_educ, mid_educ. HC1 standard errors. Coefficient is marginally significant (p=0.076); negative sign means voluntary arm shares the fact-check less than imposed.

**Surface match**: Exact. Surface specifies Table2-PanelB-Col2 as the baseline. Replicated exactly.

---

## 2. Row Counts

| Metric | Count |
|---|---|
| Total rows | 222 |
| run_success = 1 | 222 |
| run_success = 0 | 0 |
| is_valid = 1 | 222 |
| is_valid = 0 | 0 |
| is_baseline = 1 | 5 |
| is_core_test = 1 | 222 |
| is_core_test = 0 | 0 |
| G1 rows | 196 |
| G2 rows | 26 |

---

## 3. Category Breakdown

| Category | Count | Description |
|---|---|---|
| core_method | 27 | Baseline specs (5) + design variants (14) + treatment variants (8) |
| core_controls | 189 | Control set (20), control progression (24), LOO (84), random subsets (56), G2 control RCs (5) |
| core_funcform | 6 | Alternative outcome measures: share_click2 (2), share_click3 (2), share_facts_click2 (1), share_fact_click3 (1) |
| noncore | 0 | None |
| invalid | 0 | None |

**Detailed core_method breakdown**:
- baseline__intent_share_fb: 2 rows (survey2 arm, survey3 arm)
- baseline__share_action_g1: 2 rows (survey2 arm, survey3 arm)
- baseline__factcheck_action: 1 row (survey3 arm)
- design/randomized_experiment/estimator/diff_in_means: 5 rows (G1: 4 rows = 2 outcomes x 2 arms; G2: 1 row)
- design/randomized_experiment/estimator/with_covariates: 5 rows (G1: 4; G2: 1)
- design/randomized_experiment/estimator/strata_fe: 4 rows (G1 only, 2 outcomes x 2 arms; G2 surface excludes strata_fe)
- rc/treatment/pairwise/imposed_vs_control: 2 rows (want_share_fb, share_click2)
- rc/treatment/pairwise/voluntary_vs_control: 2 rows (want_share_fb, share_click2)
- rc/treatment/pairwise/imposed_vs_voluntary: 2 rows (want_share_fb, share_click2)
- rc/treatment/binary/any_factcheck_vs_control: 2 rows (want_share_fb, share_click2)

---

## 4. Sanity Checks

| Check | Result |
|---|---|
| spec_run_id unique | PASS (222 unique / 222 rows) |
| baseline_group_id present | PASS |
| No infer/* in specification_results.csv | PASS |
| inference_results.csv contains only infer/* | PASS (5 rows, all infer/se/hc/hc3) |
| All run_success=1 numeric fields non-null | PASS |
| All p_values in [0,1] | PASS |
| All CVJ have required audit keys (coefficients, inference, software, surface_hash) | PASS |
| All inference.spec_id = infer/se/hc/hc1 | PASS (matches both G1 and G2 canonical) |
| rc/form/* have functional_form block | PASS (6/6 rows) |
| rc/controls/* have controls block with matching spec_id | PASS (all 181 rc/controls rows) |
| rc/treatment/pairwise/* have sample block with matching spec_id | PASS |
| No unapproved top-level CVJ keys | PASS |
| Single surface_hash across all rows | PASS (sha256:d01ac5fab4502dc8f055a081ecffe9647708cb603128ed01a5cd451ff493e207) |
| G1 surface spec_ids fully covered | PASS (all 46 surface rc_spec_ids present) |
| G2 surface spec_ids fully covered | PASS (all 25 surface spec_ids present) |
| No extra spec_ids beyond surface | PASS |

---

## 5. Top Issues / Notes

### Issue 1: rc/form/outcome/share_click2 is numerically identical to baseline__share_action_g1

**Rows**: 140161-V1__spec__0217, 140161-V1__spec__0218

The surface lists `share_click2` as both a co-baseline for G1 (`baseline__share_action_g1`) and an `rc/form` functional-form variant relative to `want_share_fb`. As a result, these two pairs of rows run the identical regression and produce identical coefficients, SEs, and p-values. This is **not an error**: the surface deliberately treats `share_click2` as a co-equal baseline (Table 2 Panel A col 6) *and* as a form robustness check against the primary intent-to-share outcome. The `functional_form` block in the rc/form rows correctly documents the semantic distinction. Recommendation: future surface revisions could omit the rc/form/share_click2 entry when it is already a co-baseline, to avoid numeric duplication.

**Validity**: Both rows are valid (is_valid=1). The rc/form rows are classified `core_funcform` and the baseline rows are classified `core_method` (is_baseline=1).

### Issue 2: G2 sample drops with extended controls

When `rc/controls/sets/strata_socio_vote_fb` or `rc/controls/sets/full` is used for G2, N drops from 1692 (strata-only) to 920 (strata+socio+vote+fb) and 854 (full). This is due to missing values in sociodemographic variables within the G2 subsample (survey 2 and 3 only). LOO specs for G2 use the full-control N=854 sample and drop one variable at a time. This is expected behavior for this survey dataset and does **not** indicate an error.

### Issue 3: rc/treatment/* uses `sample` block in CVJ (not a `treatment` block)

The `rc/treatment/pairwise/*` specs restrict the analysis sample to one or two arms to compute pairwise comparisons. The runner stores this restriction under the `sample` key in `coefficient_vector_json`, which is axis-appropriate (the variation is effectively a sample restriction dropping one arm). The `sample.spec_id` matches the row `spec_id` in all four pairwise rows. The `rc/treatment/binary/any_factcheck_vs_control` rows use `extra` block since the treatment variable itself is constructed (any_factcheck = survey2 | survey3). No invalidity.

### Issue 4: G2 `with_covariates` N=920 (much smaller than G2 baseline N=1692)

The `design/randomized_experiment/estimator/with_covariates` spec for G2 uses the full covariate set (strata + socio + vote + fb + i.educ), which reduces N from 1692 to 920 due to listwise deletion on sociodemographic controls. This is the same pattern as G1 `rc/controls/sets/strata_socio_vote_fb`. The surface's `with_covariates` design spec is defined as "OLS with the full published control set," which includes these socio controls. No invalidity.

---

## 6. Inference Results Check

`inference_results.csv` contains 5 rows (one HC3 variant per baseline spec_run_id):

| Inference Run ID | Linked spec_run_id | Spec ID | Coef | SE (HC3) | p-value (HC3) |
|---|---|---|---|---|---|
| 140161-V1__infer__0001 | 140161-V1__spec__0001 | infer/se/hc/hc3 | -0.0451 | 0.01603 | 0.0049 |
| 140161-V1__infer__0002 | 140161-V1__spec__0002 | infer/se/hc/hc3 | -0.0380 | 0.01618 | 0.0187 |
| 140161-V1__infer__0003 | 140161-V1__spec__0003 | infer/se/hc/hc3 | -0.0202 | 0.00600 | 0.0008 |
| 140161-V1__infer__0004 | 140161-V1__spec__0004 | infer/se/hc/hc3 | -0.0214 | 0.00549 | 0.0001 |
| 140161-V1__infer__0005 | 140161-V1__spec__0005 | infer/se/hc/hc3 | -0.0284 | 0.01603 | 0.0768 |

HC3 SEs are negligibly larger than HC1, consistent with no high-leverage observations. All baseline results remain signed in the expected direction under HC3. Correct: inference variants are in `inference_results.csv` only, not in `specification_results.csv`.

---

## 7. Substantive Summary

**G1 (alt-facts sharing)**: All four baseline specs show negative and statistically significant effects (p < 0.05) for both imposed and voluntary fact-checking, on both the intent-to-share and behavioral action outcomes. The treatment effects are robust across:
- 196 total specifications
- All run successfully
- Consistent negative sign throughout
- Controls variations cause minor N drops due to listwise deletion on socio variables
- Treatment pairwise comparisons confirm individual arm significance
- Behavioral outcomes (share_click2, share_click3) show smaller but consistently negative effects

**G2 (fact-check sharing)**: The single baseline shows a marginally significant negative effect (p=0.076) of voluntary vs imposed fact-checking on intent to share the fact-check. This result is less robust — several G2 control specifications push p above 0.10 — but sign is consistent throughout.

---

## 8. Recommendations

1. **Remove rc/form/outcome/share_click2 from future surfaces when share_click2 is already a co-baseline**: The duplicate numerics add no independent information. Instead, document the behavioral/intent distinction in the claim object.

2. **Add an explicit `treatment` CVJ block for rc/treatment/* specs**: Using `sample` for pairwise arm-restriction specs is acceptable but slightly ambiguous. A dedicated `treatment` block with spec_id and arm labels would be cleaner.

3. **Consider adding G2 strata_fe design variant**: The surface omits strata_fe for G2. Given that strata controls are standard for the stratified RCT design, including the strata_fe variant for G2 would improve symmetry with G1.

4. **G2 LOO specs use full-control sample (N=854)**: The LOO specs for G2 drop one control from the full set, but the full-control N is already substantially smaller (854 vs 1692 baseline). This is valid but should be noted in any meta-analysis as LOO results for G2 are based on a different (smaller) subsample than the G2 baseline.
