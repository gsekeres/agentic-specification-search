# Spec Surface Review: 140161-V1

**Paper**: "Checking and Sharing Alt-Facts" — Henry, Zhuravskaya, Guriev (AEJ: Policy)
**Reviewer**: Pre-run auditor
**Date**: 2026-02-24

---

## Summary

The surface defines two baseline groups:
- **G1**: Effect of fact-checking interventions on intent/action to share alt-facts (Table 2, Panel A)
- **G2**: Effect of voluntary vs. imposed fact-checking on intent/action to share the fact-check (Table 2, Panel B)

The two-group structure is justified: G1 uses all three arms (survey<4), G2 uses only the fact-check arms (survey>1 & survey<4), and the outcomes are conceptually distinct. The surface is broadly faithful to the paper's design. Several corrections were needed and are documented below.

---

## A) Baseline Groups

### Issues Found

**A1. Strata control set definition is ambiguous in the JSON.**

The `variable_definitions.control_sets.strata` field lists only `["male"]`, with `strata1` holding `["low_educ", "mid_educ"]`. However, the Stata code defines:
```stata
global strata "male"
global strata1 "low_educ mid_educ"
```
And the baseline spec uses both `$strata $strata1` together. The baseline_specs correctly list `controls: ["male", "low_educ", "mid_educ"]` with `n_controls: 3`, which is accurate. The split into strata/strata1 in variable_definitions is faithful to the code and fine as documentation.

**A2. Baseline spec notes (G1) are slightly misleading.**

The G1 baseline_spec notes say "col 3 adds socio+vote+fb." In the actual code (line 788), column 3 is: `$strata $strata1 $socio $vote $fb i.educ`. The `i.educ` (education factor variable) is separate from the `educ` scalar in `$socio`. This matters for the control count. The surface's `controls_count_max: 30` is approximate and acceptable as noted.

**A3. G2 baseline_spec sample filter inconsistency.**

The G2 baseline_spec `sample_filter` says `"survey > 1 & survey < 4"` but the actual Stata code (line 868) uses `survey<4&survey>1`. These are equivalent; no change needed.

---

## B) Design Selection

### Issues Found

**B1. G1 includes `rc/form/outcome/share_click2` and `rc/form/outcome/share_click3` as rc/form/outcome specs.**

`share_click2` is the action-level outcome (probability of navigating to sharing page, tracked via Google Analytics). It is equally prominent as `want_share_fb` — Columns 5-8 of Table 2 Panel A directly parallel Columns 1-4 using this outcome. The surface already lists it as an "additional baseline" in the SPECIFICATION_SURFACE.md but treats it as an rc/form/outcome in the JSON. This is a **classification issue**: `share_click2` is arguably a separate baseline outcome (different measurement level of the same underlying construct), not a mere functional form change. Similarly, `share_click3` (reconfirm sharing) is a third measurement level.

**Decision**: Keep `share_click2` as `rc/form/outcome` is defensible since it is a different measurement-level outcome of the same sharing act, not a different estimand. The surface correctly notes these are continuous [0,1] rather than binary. However, given that the paper reports them equally prominently (same Table 2, Panel A Columns 5-8), `share_click2` should be listed as a second baseline spec within G1, not merely an rc/form/outcome. This ensures it gets the same control-progression robustness as `want_share_fb`.

**CHANGE MADE in JSON**: The G1 `baseline_specs` array is updated to include a second entry for `share_click2` as `baseline__share_action_g1`. The `baseline_spec_ids` in `core_universe` is updated accordingly.

**B2. G1 design_audit is correct.**

Randomization is at the individual level, three treatment arms (1=Alt-Facts, 2=Imposed, 3=Voluntary), stratified on male + low_educ + mid_educ, no clustering (individual-level HC1 robust SEs). This matches the code exactly.

**B3. G1 `design_spec_ids` includes `strata_fe` variant.**

The surface proposes treating stratification variables as fixed effects rather than linear controls. This is a legitimate RC for stratified RCTs but should be noted: in this design, with only 3 strata variables (male, low_educ, mid_educ), the strata FE approach is identical to linear controls with dummies. Still acceptable as a specification variant.

---

## C) RC Axes

### Issues Found

**C1. Duration filter RC specs (`rc/sample/duration_filter/trim_300` and `trim_200`) are NOT implementable post-hoc.**

The baseline sample filter in the data construction code (`1.infile_data.do`, line 46) applies `keep if durationinseconds >= 250` at the raw CSV import stage. After this step, `durationinseconds` is converted to **minutes** (line 73: `replace durationinseconds = durationinseconds/60`). The final `surveys.dta` dataset no longer contains the raw seconds variable.

- `trim_300`: Would require restricting to respondents who took >= 300 seconds. Since the variable is converted to minutes and observations below 250s are already dropped, the original seconds value is lost.
- `trim_200`: This would be a *looser* threshold than the baseline 250s, but since observations with <250s are already excluded from the dataset, no additional obs can be added back.

Both RC specs are therefore **inoperable** with the available generated dataset. They would only be valid if one regenerated the dataset from raw CSVs.

**CHANGE MADE**: Remove `rc/sample/duration_filter/trim_300` and `rc/sample/duration_filter/trim_200` from G1's `rc_spec_ids`. Add a note to the surface that the sample filter (250s minimum + gc==1) is baked into the generated data and cannot be varied post-hoc.

**C2. G1 LOO control list is complete and correct.**

The 22 LOO specs cover: male, low_educ, mid_educ (strata), age (drops age+age_sqrd pair), income, married, single, village, town, children, catholic, muslim, no_religion, religious (socio), use_FB, often_share_fb, log_nb_friends_fb (fb), second_mlp, negative_image_UE (vote), altruism, reciprocity, image (behavioral). This matches the `$socio`, `$vote`, `$fb`, and `$behavioral` globals in the code. Note that `educ` (used as `i.educ` factor in column 3+) is not included in the LOO — this is acceptable since it is always part of the extended specs and is collinear with low_educ/mid_educ at the strata level.

**C3. G2 LOO list is a subset of G1's.**

G2 drops 8 variables: male, low_educ, mid_educ, religious, second_mlp, negative_image_UE, use_FB, log_nb_friends_fb. This is a reasonable reduced set for the smaller G2 subsample. However, it omits `often_share_fb` and several socio controls. This reduces the LOO coverage for G2 but is within acceptable range given the smaller budget.

**C4. The `rc/treatment/pairwise/imposed_vs_voluntary` spec in G1 changes the claim object.**

The `imposed_vs_voluntary` pairwise comparison (survey 2 vs 3) in G1 is very close to G2's main estimate. It is listed as an RC for G1 (which is framed as treatment vs. control). This pairwise comparison changes the estimand from "effect of fact-checking vs. no fact-check" to "difference between fact-check modes." It should not be classified as a standard RC. However, it is informative as an explore-level spec. Given the surface's existing structure and the fact that it is labeled as a treatment-coding variant, it is tolerable here if clearly understood as a between-treatment comparison, not a standard ATE robustness check.

**C5. `rc/treatment/binary/any_factcheck_vs_control` in G1 is a valid pooled estimate.**

This collapses surveys 2+3 vs. 1, which is a legitimate pooled ATE. This is appropriate as an RC.

---

## D) Controls Multiverse Policy

### Issues Found

**D1. Strata controls should be mandatory in all specs except pure diff-in-means.**

The surface notes in Section 8 (SPECIFICATION_SURFACE.md): "Mandatory strata controls: the stratification variables (male, low_educ, mid_educ) should be included in all specifications except the pure diff-in-means design variant, as is standard for stratified RCTs." This is correct and consistent with best practice. However, the JSON `constraints` block for both G1 and G2 does not explicitly encode this requirement — it only states `controls_count_min: 0`. This is slightly inconsistent since 0 controls is only valid for the diff-in-means variant.

**CHANGE MADE**: Add a `notes` field to G2 `constraints` (matching the G1 notes) to make the mandatory strata requirement explicit.

**D2. The `socio` control set in the JSON includes `educ` as a scalar.**

In the paper's code, column 3 uses `i.educ` (Stata factor syntax), which expands to 8 dummies (education categories 1-9, with one reference). The surface lists `educ` as a control but the actual implementation expands it. This means the control count in column 3 is underestimated. The `controls_count_max: 30` with the "approximate" caveat covers this. No change required, but the analysis script should use factor treatment for `educ`.

---

## E) Inference Plan

### Assessment

The canonical inference is HC1 heteroskedasticity-robust SEs, matching the paper's `r` (robust) option throughout Stata. There is no clustering (individual-level randomization). The HC3 variant is a reasonable stress test. The inference plan is correctly specified for both G1 and G2.

No issues found.

---

## F) Budgets and Sampling

### Assessment

- **G1**: 80 max specs core total, 10 random control subsets. Given ~52 named specs in the JSON rc_spec_ids (after removing the 2 inoperable duration filter specs), the budget is appropriate. With the baseline and design variants, the total is around 57-65 actual specs, comfortably within the 80 limit.
- **G2**: 40 max specs core total, 5 random control subsets. The named specs are ~27, which is feasible within budget.

Seeds (140161, 140162) are distinct and reproducible. Sampler type is `stratified_size`, which is appropriate for random control subset draws.

**F1. G1 LOO set references variables from the full control set (strata+socio+vote+fb+behavioral), but the baseline spec only includes strata.**

The LOO specs drop one variable from the extended control set. The implicit assumption is that LOO is computed relative to the maximum practical control set (strata+socio+vote+fb+behavioral = ~21 controls + educ factor). This is correct and consistent with how the paper's Table A3 structure works.

---

## G) Diagnostics Plan

### Assessment

Both G1 and G2 list the covariate balance check (`diag/randomized_experiment/balance/covariates`) with scope `baseline_group`. The paper's Table 1 provides balance for all three arms and Table A1 provides the omnibus mlogit test. The diagnostic is appropriately scoped. No issues.

---

## Changes Made to SPECIFICATION_SURFACE.JSON

1. **Removed** `rc/sample/duration_filter/trim_300` and `rc/sample/duration_filter/trim_200` from G1 `rc_spec_ids`. These cannot be implemented post-hoc since the duration filter is baked into the generated dataset at the raw data construction stage.

2. **Added** a second baseline spec entry in G1 (`Table2-PanelA-Col6`) for `share_click2` (the behavioral action outcome from Columns 5-8 of Table 2 Panel A), with corresponding entry in `baseline_spec_ids`. This promotes the action-level outcome to co-equal baseline status, which better reflects the paper's equal emphasis on intent vs. action outcomes.

3. **Added** `notes` field to G2 `constraints` block (making explicit the mandatory strata controls requirement, consistent with G1).

4. **Updated** `variable_definitions.sample_filters` to clarify that `duration_250` and `gc_filter` are applied at data construction and cannot be varied post-hoc in the generated dataset.

---

## What's Missing

- **No share_click3 baseline spec**: `share_click3` (reconfirm sharing, 3rd click) is included only as an rc/form/outcome. The paper reports these in the same Table 2 columns. This is acceptable — 3-click is the lowest-frequency outcome and is clearly a robustness variant of the 2-click measure.

- **No quantile/nonlinear estimator RCs**: The outcomes are binary (want_share_fb, want_share_facts) or continuous [0,1] (click2, click3). The paper uses linear probability models throughout. No probit/logit robustness is proposed. This could be added to explore/ but is not missing from the core universe per the i4r methodology.

- **No sub-arm analysis (Viewer vs. Nonviewer split in G2)**: The paper uses `survey_alt` to split voluntary arm into viewers and nonviewers. This is a secondary analysis that would require conditioning on a post-treatment variable and is correctly excluded from the core universe.

---

## Final Assessment

**APPROVED TO RUN** with the changes documented above.

The surface is conceptually coherent, statistically principled, and faithful to the revealed manuscript surface. The two key corrections (removing inoperable duration-filter RCs and adding the action-level outcome as a co-baseline) make the surface more accurate and executable. The inference plan, budgets, and sampling are appropriate.

Estimated total specs across G1+G2 after changes: approximately 80-90 specs (G1: ~55-60; G2: ~27-30), meeting the 50+ target.
