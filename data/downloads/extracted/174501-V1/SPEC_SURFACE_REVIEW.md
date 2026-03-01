# Specification Surface Review: 174501-V1

**Paper**: "Interaction, Stereotypes and Performance: Evidence from South Africa" (Corno, La Ferrara, Burns)

**Review date**: 2026-02-24

---

## Summary of Baseline Groups

The surface defines three baseline groups corresponding to three families of outcomes in this randomized roommate-assignment natural experiment:

- **G1 (Stereotypes)**: Race IAT D-score, White and Black subsamples. ANCOVA with lagged outcome.
- **G2 (Academic performance)**: GPA, exams passed, continuation, PCA index. Black subsample primary.
- **G3 (Social outcomes)**: PCA indices for friendships, attitudes, communication, pro-social behavior. White subsample primary.

All three share the same treatment (mixracebas), randomization unit (room), clustering (roomnum_base), and strata (Res_base).

### Changes Made

1. **G1**: Removed `rc/form/academic_iat_outcome` (DscoreacaIAT). This changes the outcome concept from Race IAT to Academic IAT and belongs in `explore/outcome/academic_iat`.

2. **G1**: Removed `rc/sample/balanced_panel_only`. The analysis data file (`uctdata_balanced.dta`) is already the balanced panel, so this is the default state, not an RC variant.

3. **G1**: Fixed `controls_count_min` from 11 to 12. The lagged outcome (L.DscoreraceIAT) is mandatory for the ANCOVA structure; with 11 own controls plus the mandatory lagged outcome, the minimum is 12. Added explicit `mandatory_controls` field.

4. **G2**: Removed `baseline__table4_white_gpa` and `baseline__table4_full_gpa` from `baseline_spec_ids`. These change the target population from Black students to White/full sample. Per claim grouping rules, the paper's primary G2 claim is about Black student improvement; White and full-sample results are secondary and belong in `explore/population/*`.

5. **G2**: Removed `rc/controls/add_interaction_mixsamefaculty` (Table A11). This adds `mixsamefaculty` and `samefaculty` interaction terms, which decomposes the treatment effect by faculty overlap -- a mechanism exploration that changes the treatment concept. Moved to `explore/mechanism/same_faculty_interaction`.

6. **G2**: Removed `rc/fe/add_program_fe`. Redundant since the baseline already includes `i.regprogram`.

7. **G2**: Removed `rc/sample/restrict_white_only`. This changes the target population and belongs in `explore/population/white`.

8. **G3**: Removed `baseline__table5_black_pcafriend` and `baseline__table5_full_pcafriend` from `baseline_spec_ids`. These change the target population from White students to Black/full sample. The paper's primary G3 claim is about White students.

---

## Checklist Results

### A) Baseline Groups

- **G1**: Correctly defined. Two baseline specs (White and Black Race IAT) within one baseline group, sharing the same outcome concept. The Black spec is listed as an additional baseline_spec_id, appropriately.
- **G2**: Correctly defined after removing White/Full population variants. Four outcome variables (GPA, examspassed, continue, PCAperf) are components/proxies of one "academic performance" concept. The paper treats them as co-equal in Table 4, so listing them as additional baselines within the same group is appropriate.
- **G3**: Correctly defined after removing Black/Full population variants. Four PCA indices (friend, attitude, comm, social) are components of one "social outcomes" concept.
- **No missing baseline groups**: Table 6 (residential choices) is a different outcome family, but it is a downstream/mechanism outcome, not a primary claim. Correctly excluded.

### B) Design Selection

- `randomized_experiment` is correct for all three groups. The identification relies on as-good-as-random roommate assignment within residence halls.
- Design variants are appropriate: diff_in_means, ANCOVA (G1 only, where lagged IAT is available), with_covariates, strata_fe.
- `design_audit` blocks are present and informative for all groups. They include randomization_unit, strata_variable, cluster_var, and sample restrictions.

### C) RC Axes

- **Controls**: Appropriate. LOO over 6 own-control variable groups and 5 roommate-control groups (pairs with missingness indicators). The "drop all roommate controls" variant is revealed in Tables A7/A9.
- **Sample**: `full_sample_with_race_fe` is a legitimate RC (changes control set to add race dummies and race x residence FE but keeps the same claim object). `restrict_white_black_only` restricts the roommate composition and is revealed in Table A8 context.
- **FE**: `drop_program_fe` for G2 is appropriate (tests sensitivity to program FE inclusion).
- **Form**: `pca_performance_index` for G2 is a measurement variant (PCA composite vs individual outcomes), appropriate as RC. `second_year_outcomes` for G2 changes the measurement horizon (year 2 vs year 1); the paper presents this as a robustness check (Table A10), so keeping it as RC is defensible. `nomiss_pca_indices` for G3 is a coding variant (how missing values enter PCA construction), appropriate as RC.
- **No major missing axes**: The revealed control structure is well-captured.

### D) Controls Multiverse Policy

- **G1**: `controls_count_min=12` (11 own + 1 mandatory lagged), `controls_count_max=22` (all). Corrected from original min=11. Mandatory controls field added.
- **G2**: `controls_count_min=11`, `controls_count_max=21`. Correct (no lagged outcome for academic specs).
- **G3**: `controls_count_min=11`, `controls_count_max=21`. Correct.
- `linked_adjustment=false` for all groups. Correct -- no bundled estimator.

### E) Inference Plan

- **Canonical**: Cluster at room level (`roomnum_base`) for all groups. Matches the code: `robust cluster(roomnum_base)` throughout.
- **Variants**: HC1 (no clustering) and cluster at residence level (`Res_base`). Both are reasonable stress tests.
- Note: The code uses `robust cluster(roomnum_base)` which in Stata produces CRV1 standard errors. This is correctly captured.

### F) Budgets and Sampling

- **G1**: 60 core specs, 30 controls-subset. Feasible with full enumeration (LOO gives ~11 variants, plus design variants and sample variants).
- **G2**: 80 core specs, 40 controls-subset. Higher budget reflects 4 outcome variables. Feasible.
- **G3**: 60 core specs, 30 controls-subset. Feasible with 4 outcome variables and LOO.
- Seeds specified (174501). Sampling plan is full_enumeration, which is appropriate given the modest combinatorial space.

### G) Diagnostics Plan

- **G1**: Balance tests (Table 1/2) and attrition analysis (Table A1). Appropriate for RCT.
- **G2**: Balance tests (Table 1). Appropriate. Could add attrition diagnostic here too (G2 uses the same sample selection mechanism), but not blocking.
- **G3**: Empty. Could add balance tests, but the same randomization applies across all groups, so G1's diagnostics cover this.

---

## Key Constraints and Linkage Rules

1. **Missingness indicator pairing**: Variables like `privateschool_nomiss` and `privateschool_miss` must be dropped/added together in LOO. The surface implicitly handles this by naming LOO by the `_nomiss` variable, and the runner should drop both `_nomiss` and `_miss` together.
2. **Lagged outcome is mandatory for G1**: `L.DscoreraceIAT` must always be included in G1 ANCOVA specs. Now explicitly noted in `mandatory_controls`.
3. **Race subsample determines control set**: Full-sample specs use `controls` (with race dummies: white, coloured, Else) plus race x residence FE (blackRes, whiteRes). Subsample specs use `controls_subsample` (without race dummies). The runner must handle this conditional logic.
4. **Program FE mandatory for G2**: `i.regprogram` is included in all Table 4 academic specifications.

---

## What's Missing

- **Placebo regressions** (Table A6): Using baseline IAT as outcome (DscoreraceIATbas) as a falsification test. Could be added to G1's diagnostics_plan as `diag/randomized_experiment/placebo/baseline_outcome`. Not blocking.
- **Unbalanced panel variant**: The analysis uses `uctdata_balanced.dta` exclusively. An unbalanced panel variant using `uctdata_clean.dta` would test sensitivity to attrition-based sample selection. Not revealed in the paper's tables but would be a meaningful RC. Not blocking.
- **Non-White subsample**: The code defines `controls_nonwhite` for Coloured/Other students. Some tables include non-White subsamples but the surface does not. This is a population change, correctly excluded from core.

---

## Final Assessment

**Approved to run.** The surface correctly identifies three baseline groups with well-defined claim objects. The RC axes are faithful to the paper's revealed specification space. Control count envelopes are now correctly specified. Population-switching variants have been appropriately moved to exploration. Budgets are feasible for full enumeration. No blocking issues remain.
