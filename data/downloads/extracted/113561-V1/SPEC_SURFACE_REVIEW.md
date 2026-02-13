# Specification Surface Review: 113561-V1

**Paper**: Fong and Luttmer (2009), "What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
**Reviewer**: Specification Surface Verifier Agent
**Date**: 2026-02-13
**Verdict**: APPROVED TO RUN (after edits described below)

---

## Summary of Baseline Groups

Four baseline groups, each targeting a different outcome for white respondents, all using the same RCT design and treatment (pictures showing black Katrina victims):

| Group | Outcome | Baseline Coef | Baseline p | N |
|-------|---------|--------------|-----------|---|
| G1 | Experimental giving (0-100 dictator game) | -4.198 | 0.370 | 915 |
| G2 | Hypothetical giving (topcoded at $500) | -2.181 | 0.591 | 913 |
| G3 | Subjective charity support (1-7 Likert) | -0.221 | 0.167 | 907 |
| G4 | Subjective government support (1-7 Likert) | -0.435 | 0.026 | 913 |

Only G4 is statistically significant at conventional levels. This is an important context for interpretation: the specification search is probing robustness of largely null or marginal results (with the exception of G4).

All four groups are correctly identified as distinct baseline claim objects (distinct outcome concepts, same treatment/estimand/population). This matches the paper's Table 4 structure where each outcome gets its own panel.

---

## Checklist Results

### A) Baseline Groups

**Issues found and fixed:**

1. **G1 had a misplaced secondary baseline spec (Table3-Col2-FullSample).** The Table 3 Col 2 result is a full-sample regression (N=1343) while G1's target population is white respondents (N=915). A full-sample regression changes the target population. It was already included as `rc/sample/subpopulation/full_sample` in the RC list. Listing it as a baseline spec for G1 is inconsistent with the claim object definition. **Removed from G1's baseline_specs.**

2. **No missing baseline groups.** The paper's four main outcomes are all represented. Black respondents are correctly excluded from core (small N, different population, mostly insignificant). Full-sample results are correctly treated as sample RCs.

3. **No exploration contamination.** Table 3 interaction specs (cols 3-6), Table 6 heterogeneity specs, and censored/ordered probit alternatives are all correctly classified as `excluded_from_core`. The reasoning is sound (interaction analyses change the estimand; censored regression targets a latent-variable parameter; ordered probit is infeasible in Python).

### B) Design Selection

**Issues found and fixed:**

1. **G1 listed `design/randomized_experiment/estimator/with_covariates` as a design variant, but this is identical to the baseline estimator.** The G1 baseline IS an OLS regression with covariates. There is no separate spec to run. **Removed from G1's design_spec_ids.** G2-G4 correctly list only `diff_in_means`.

2. Design code `randomized_experiment` is correct for all groups. The paper uses individual-level random assignment to picture treatment arms.

### C) RC Axes (Core Robustness)

**Issues found and fixed:**

1. **Baseline-identical RC entries removed.** The following specs were listed as RCs but produce regressions identical to the baseline:
   - `rc/controls/sets/baseline` -- This IS the baseline control set.
   - `rc/controls/manipulation_coding/nraudworthy` -- This IS the baseline manipulation coding.
   - `rc/weights/main/paper_weights` -- This IS the baseline weight.
   - `infer/se/hc/hc1` -- This IS the baseline inference.

   **All removed from rc_spec_ids and infer_spec_ids respectively.**

2. **Duplicate aliases de-duplicated.**
   - `rc/controls/sets/none` and `rc/controls/progression/bivariate` are identical (0 controls). Kept `sets/none`, removed `progression/bivariate`.
   - `rc/controls/sets/minimal` and `rc/controls/progression/manipulation_only` are identical (nraud controls only). Kept `progression/manipulation_only` in G1 (which has the fuller progression), removed `sets/minimal`. For G2-G4 which have a shorter progression, `sets/none` is kept.

3. **No missing high-leverage axes.** The surface covers:
   - Controls: progression + LOO + manipulation coding variant (adequate)
   - Sample: full sample, main variant, city split, race-shown (all revealed by Table 5)
   - Weights: unweighted (revealed by Table 3 and Table 6)
   - Outcome preprocessing: topcode/winsor for G1, topcode/no-topcode for G2 (appropriate for bounded/skewed outcomes; not applicable to Likert-scale G3/G4)
   - Inference: HC2, HC3, classical as alternatives to baseline HC1 (appropriate)

4. **No axes incorrectly included.** All RC axes preserve the claim object. The full-sample RC is borderline (changes target population), but the paper presents full-sample and white-only results in parallel as complementary analyses, so including it as an RC is defensible and matches the paper's own framing.

### D) Controls Multiverse Policy

**Issues found and fixed:**

1. **`controls_count_min` corrected from 11 to 9.** The original surface stated min=11 based on "nraud + race dummies = 11." However, on the white subsample (which is the target population for all baseline groups), `black` and `other` are collinear zeros and are automatically dropped. The effective minimum is 9 (just the nraud manipulation controls). This matters for any subset sampling that respects the envelope. The bivariate spec (0 controls) is explicitly enumerated and correctly sits outside the envelope.

2. **LOO count corrected from 18 to 17.** The surface originally stated 18 LOO drops, but `age` and `age2` must be dropped together (since `age2 = age^2`). With 18 individual variables but one paired drop, there are 17 distinct LOO specs.

3. **Mandatory controls correctly identified.** The nraud manipulation controls (9 vars) are correctly marked as mandatory. They correspond to the experimental design (randomized audio manipulations) and must always be included.

4. **No bundled estimator.** Correctly set `linked_adjustment: false`. This is a simple OLS/WLS regression, not a multi-component estimator.

5. **Education dummies note.** The three education dummies (edudo, edusc, educp) are individual LOO drops rather than a group drop. This is defensible since each dummy captures a distinct education level (dropout, some college, college plus). However, one could argue they should be dropped as a group. The current individual-drop approach is more conservative (generates more specs) and is fine.

### E) Budgets and Sampling

**Changes made:**

1. **Budgets revised downward after deduplication.**
   - G1: 80 -> 37 (removed ~7 baseline-identical and duplicate specs)
   - G2: 45 -> 34
   - G3: 40 -> 32
   - G4: 40 -> 32
   - Total: 153 -> 135

2. **Full enumeration remains feasible.** With ~135 total specs across 4 baseline groups, no random sampling is needed. The seed (113561) is correctly set for reproducibility if any sampling were added later.

3. **Spec count verification (G1 detailed breakdown):**
   - 1 baseline
   - 1 design variant (diff_in_means)
   - 7 unique control specs: sets/none (0 controls), sets/extended (32), progression/manipulation_only (9), progression/manipulation_plus_demographics (~23), progression/manipulation_plus_demographics_plus_charity (~27), progression/full (~30), manipulation_coding/separate_worthiness (~32 with $manip)
   - 17 LOO drops (age/age2 paired)
   - 5 sample variants (full_sample, main_variant_only, slidell_only, biloxi_only, race_shown_only)
   - 1 weight variant (unweighted)
   - 2 outcome preprocessing (topcode_99, winsor_1_99)
   - 3 inference variants (HC2, HC3, classical)
   - **Total G1: ~37**

### F) Diagnostics Plan

1. **Balance check is appropriate.** `diag/randomized_experiment/balance/covariates` at baseline_group scope is the standard RCT diagnostic.

2. **Attrition diagnostic correctly omitted.** Only 5 observations have missing giving data out of 1348, making attrition negligible.

3. **No noncompliance diagnostic needed.** This is a clean ITT analysis with no compliance/take-up issue (all respondents were shown pictures as assigned).

---

## Key Constraints and Linkage Rules

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Treatment arms always included | picshowblack, picraceb, picobscur | Experimental design |
| Focal coefficient | picshowblack | Captures "shown black vs shown white" contrast |
| Manipulation controls mandatory | nraud (9 vars) | Experimental manipulation controls |
| Controls count envelope | [9, 32] (for subset sampling) | Derived from Table 5 s5 (min) and s6 (max) |
| One-axis-at-a-time | Enforced | No cross-product combinations |
| Linked adjustment | false | Not a bundled estimator |
| Race-shown subsample collinearity | picobscur constant, picraceb = picshowblack | Must drop one; document in coefficient_vector_json |
| White subsample collinearity | black, other = 0 | Automatically dropped; effective n_controls = n_formula - 2 |
| age/age2 pairing | Drop together in LOO | age2 = age^2 |

---

## What Is Missing (Optional Enhancements)

The following are not blocking issues but could enrich the surface if desired:

1. **No `rc/controls/progression/manipulation_only` in G2-G4.** G1 has the full 5-step progression; G2-G4 only have 3 steps (none, manip+demo+charity, full). Adding the full progression to G2-G4 would cost 2 extra specs per group (6 total) and would make the control build-up parallel across outcomes.

2. **No `infer/se/hc/classical` in G2-G4.** G1 has 3 non-baseline inference variants (HC2, HC3, classical); G2-G4 have only 2 (HC2, HC3). Adding classical SE to G2-G4 would add 3 specs total.

3. **No randomization inference.** For a small experiment (N~915), randomization inference or wild bootstrap could provide better finite-sample properties than asymptotic HC SEs. This is not in the paper's revealed surface but is a standard i4r-style robustness check for RCTs. Could add `infer/resampling/permutation_test` as a future enhancement.

4. **No Lee bounds for potential selective attrition.** Although attrition is minimal (5/1348), Lee (2009) bounds could formalize the argument. This would be a `sens/*` object, not core.

5. **Education dummies could be grouped.** Dropping edudo/edusc/educp as a block (rather than individually) would test sensitivity to the entire education adjustment. This would be an additional LOO-block spec.

---

## Changes Made to SPECIFICATION_SURFACE.json

| Change | Section | Rationale |
|--------|---------|-----------|
| Removed Table3-Col2-FullSample from G1 baseline_specs | G1.baseline_specs | Different target population (full sample vs white-only) |
| Added n_controls_effective_white_subsample=27 to G1 baseline spec | G1.baseline_specs | Clarify that black/other are collinear zeros |
| Removed `with_covariates` from G1 design_spec_ids | G1.core_universe | Identical to baseline estimator |
| Removed `rc/controls/sets/baseline` from all groups | G1-G4.rc_spec_ids | Identical to baseline control set |
| Removed `rc/controls/manipulation_coding/nraudworthy` from G1 | G1.rc_spec_ids | Identical to baseline coding |
| Removed `rc/weights/main/paper_weights` from G1 | G1.rc_spec_ids | Identical to baseline weights |
| Removed `rc/controls/sets/minimal` from G1 | G1.rc_spec_ids | Alias for progression/manipulation_only |
| Removed `rc/controls/progression/bivariate` from all groups | G1-G4.rc_spec_ids | Alias for sets/none |
| Removed `infer/se/hc/hc1` from all groups | G1-G4.infer_spec_ids | Identical to baseline inference |
| Fixed controls_count_min from 11 to 9 for all groups | G1-G4.constraints | Effective min on white subsample after collinearity drop |
| Fixed max_specs_controls_loo from 18 to 17 | G1.budgets | age/age2 dropped as pair |
| Revised all budget totals downward | G1-G4.budgets, total_estimated_specs | After deduplication |
| Added collinearity note for race_shown_only | shared_definitions.sample_definitions | picobscur/picraceb collinearity |
| Added rc_dedup_notes to all groups | G1-G4.core_universe | Document which specs were removed and why |

---

## Final Verdict

**APPROVED TO RUN.**

The specification surface is conceptually coherent, statistically principled, and faithful to the manuscript's revealed search space after the corrections above. The four baseline groups correctly capture the paper's main claims. The RC axes are appropriate for a randomized experiment design. Budgets are feasible (full enumeration, ~135 total specs). No blocking issues remain.

The surface correctly handles the key challenge of this paper: distinguishing between the core ITT analysis (white respondents, four outcomes) and the extensive heterogeneity/interaction analyses (Tables 3 cols 3-6, Table 6) that change the estimand and belong in exploration. The exclusion of censored regression and ordered probit from core is well-justified by both feasibility and estimand concerns.
