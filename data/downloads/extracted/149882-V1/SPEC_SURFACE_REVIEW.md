# Specification Surface Review: 149882-V1

**Paper**: "Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India" by Dhar, Jain, and Jayachandran (AER 2022)

**Reviewed**: 2026-02-24

---

## Summary of Baseline Groups

### G1: ITT effect on gender attitudes (endline 1)
- **Design**: randomized_experiment (school-level RCT, OLS with covariates)
- **Claim object**: Well-defined. Combined (boys + girls) ITT effect of the Breakthrough curriculum on the gender attitude index (E_Sgender_index2), standardized inverse-covariance-weighted index.
- **Verified against code**: `04b_tables.do` line 535 confirms `reg E_Sgender_index2 B_treat B_Sgender_index2 district_gender_* gender_grade_* ${el_gender_flag} if !mi(E_Steam_id) & attrition_el==0, cluster(Sschool_id)`. This is the "Combined" column of Table 1.2.
- **Baseline spec verified**: Treatment variable is `B_treat`, outcome is `E_Sgender_index2`, clustering at `Sschool_id`.

### G2: ITT effect on self-reported behavior (endline 1)
- **Design**: randomized_experiment (same as G1)
- **Claim object**: Well-defined. Combined ITT effect on the behavior index (E_Sbehavior_index2), a separate family from gender attitudes.
- **Verified against code**: Same regression structure with `E_Sbehavior_index2` and `B_Sbehavior_index2` as baseline control, plus `${el_behavior_common_flag}`.

---

## Changes Made to SPECIFICATION_SURFACE.json

### 1. Moved subpopulation specs from `rc/*` to `explore/*`

The following were reclassified from `rc/sample/subvariant/` to `explore/population/`:
- `girls_only`, `boys_only`, `grade6_only`, `grade7_only`, `coed_schools_only`

**Rationale**: Per CLAIM_GROUPING.md Section 3B, restricting to a gender subgroup ("girls only", "boys only") changes the **target population** and hence the estimand. The paper's baseline claim is the combined (boys + girls) ITT effect. Table 1.4 reports gender-specific results as separate panels, but these are heterogeneity analyses -- the paper's headline finding is the combined effect (Abstract: "0.18 standard deviations"). Grade-level and school-type subsets similarly constitute population changes, not within-population robustness.

### 2. Moved sub-index outcomes from `rc/form/outcome/` to `explore/outcome/`

The following were reclassified:
- G1: `gender_subindex_education`, `gender_subindex_employment`, `gender_subindex_subjugation`
- G2: `behavior_subindex_oppsex`, `behavior_subindex_hhchores`, `behavior_subindex_relatives`

**Rationale**: Per CLAIM_GROUPING.md Section 2, sub-index components change the outcome concept. The baseline claim object is the overall gender attitude index (or overall behavior index), which is an inverse-covariance-weighted aggregate. Individual sub-indices (education attitudes, employment attitudes, subjugation attitudes) measure substantively different constructs. They are useful for understanding which domains drive the main result, but they are exploration, not estimand-preserving RC.

### 3. Removed duplicate control-set specs

Removed the following duplicative specs from `rc_spec_ids`:
- `rc/controls/sets/none` -- duplicates both `design/randomized_experiment/estimator/diff_in_means` and `rc/controls/progression/bivariate`
- `rc/controls/sets/strata_only` -- duplicates both `design/randomized_experiment/estimator/strata_fe` and `rc/controls/progression/strata_fe_only`
- `rc/controls/sets/strata_plus_baseline_index` -- duplicates `rc/controls/progression/strata_fe_plus_bl_index`

**Rationale**: These would produce identical regression runs with different spec_ids, which is wasteful and introduces spurious variation in spec counts. The progression series already covers the full control-building ladder; the redundant "sets" entries are not needed.

### 4. Updated n_controls to include endline missing flags

The baseline spec for G1 listed `n_controls: 13`, but the actual regression code includes `${el_gender_flag}` which adds ~17 endline component missing flag variables. Updated to `n_controls: 30` with an explanatory note. Similarly for G2, updated to `n_controls: 19` (13 + ~6 behavior flags).

**Rationale**: The surface must accurately reflect what the code actually includes. The endline missing flags are nuisance controls that the paper includes to avoid dropping observations with partially missing index components.

### 5. Added `n_clusters_approx` to design_audit

Added approximate cluster count (~314 schools) and district count (4) to both baseline groups.

### 6. Added warning to district-level clustering inference variant

The `infer/se/cluster/district` variant clusters at the district level with only 4 districts. Added note that this will produce unreliable inference and is included only as an extreme stress test.

---

## Key Constraints and Linkage Rules

- **Control-count envelope**: [0, 50] -- the range from no controls (difference-in-means) to the full LASSO-selected extended set (cntrls_all_8, ~50 variables). This is a wide envelope, but it is genuinely revealed by the paper: Table 1.2 uses strata FE + baseline index (~13-30 controls), while Appendix Table 1.10 uses LASSO-selected extended controls. The difference-in-means estimator (no controls) is a standard design alternative for RCTs.
- **No linked adjustment**: Correct. Single-equation OLS, no bundled components.
- **Strata FE vs. district dummies**: The combined regression uses `district_gender_*` (8 dummies for district-by-gender strata) and `gender_grade_*` (4 dummies for gender-by-grade), while the gender-specific regressions use `district_?` (4 district dummies) and `B_Sgrade6`. This is because the gender-specific samples don't need the gender interaction. The runner should use the appropriate strata FE for each subsample.
- **LASSO seed**: `set seed 5212021` -- control selection is deterministic given this seed.
- **Excluded observations**: child_id 3205037 (blind at baseline) and School_ID 2711 (missing at baseline) are excluded, and School_ID 2704 (wrong school) is dropped in baseline cleaning.

---

## Budget and Sampling Assessment

- **G1**: 50 core specs. After deduplication: 1 baseline + 2 design variants + 13 RC specs = 16 core specs. With exploration, total rises to 16 + 8 explore = 24. Budget of 50 is adequate with room for combinatorial expansion (e.g., LOO controls crossed with progression).
- **G2**: Same structure. 50 is adequate.
- Full enumeration is appropriate; no random subset sampling needed.

---

## What's Missing (minor)

1. **Endline 2 (medium-run) results**: Tables 1.8 and 1.10 report 2-year follow-up effects. These are correctly excluded as a separate time horizon, but could be added as a separate baseline group if persistence of treatment effects is a distinct claim.

2. **Aspirations index (girls only)**: This is a third primary outcome in Table 1.2 but measured only for girls. Correctly excluded per the surface notes. If included, it would need its own baseline group with a girls-only target population.

3. **Social desirability heterogeneity (Table 1.3)**: The paper interacts treatment with social desirability score. This is correctly identified as heterogeneity exploration, not RC.

4. **IAT (Implicit Association Test) outcomes (Appendix Table 1.9)**: These measure implicit gender attitudes using a different measurement technology. Correctly excluded as a different outcome concept.

5. **School-level outcomes (Appendix Table 1.26)**: Different unit of observation (school aggregate). Correctly excluded.

---

## Diagnostics Assessment

G1 includes two appropriate diagnostics:
- `diag/randomized_experiment/balance/covariates` (scope: baseline_group) -- standard covariate balance test
- `diag/randomized_experiment/attrition/attrition_diff` (scope: baseline_group) -- differential attrition test

G2 has an empty diagnostics_plan. The same diagnostics apply (same experiment, same sample). Consider adding them for completeness, though since this is the same experiment, the G1 diagnostics cover G2 by implication.

---

## Approved to Run

**Status**: APPROVED with revisions applied.

The specification surface is well-constructed for an RCT paper. The two baseline groups (attitudes vs. behavior) are conceptually distinct and correctly separated. The key revisions correct three conceptual issues: (a) subpopulation analyses should be exploration, not RC, since they change the target population; (b) sub-index outcomes should be exploration since they change the outcome concept; (c) duplicate control-set specs needed to be deduplicated. The control progression ladder is well-designed and reveals the paper's own reported variation. The inference plan is appropriate with school-level clustering as canonical.
