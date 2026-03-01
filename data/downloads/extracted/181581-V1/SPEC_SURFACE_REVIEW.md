# Specification Surface Review: 181581-V1

**Paper**: When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality (Okeke)

**Review date**: 2026-02-25

---

## Summary of Baseline Groups

**G1**: Effect of deploying a doctor to a primary health center on 7-day neonatal mortality.

- **Outcome**: `mort7` (binary: infant died within 7 days of birth)
- **Treatment**: `doctor` (indicator for Doctor Arm assignment), with `mlp` always included as a co-regressor
- **Estimand**: ITT
- **Population**: Live births (alive==1) to consenting women enrolled at participating primary health centers in Kaduna State, Nigeria
- **Design**: Randomized experiment (stratified by strata, clustered at facility fid)
- **Baseline specs**: Table 4 Col 1 (strata FE only, no controls) and Col 2 (strata + qtr FE, basic controls)

One baseline group is appropriate. The paper has one focal claim: deploying doctors reduces early neonatal mortality. Tables 2-3 (staffing effects, probability of doctor care) are intermediate/first-stage outcomes. Table 5 (dosage) is heterogeneity analysis. Tables 6-7 (clinical quality, qualitative impacts) use different datasets. Table 8 (IV) changes the estimand to LATE. All correctly excluded from the core universe.

---

## Changes Made to SPECIFICATION_SURFACE.json

### 1. Removed redundant design spec IDs

Removed `design/randomized_experiment/estimator/strata_fe` and `design/randomized_experiment/estimator/with_covariates`:

- **`strata_fe`**: The baseline (Table 4 Col 1) already includes strata FE via `abs(strata)`. This design variant is the baseline itself, creating a duplicate.
- **`with_covariates`**: This is equivalent to `rc/controls/sets/basic` (Table 4 Col 2) and `rc/controls/sets/extended` (Table 4 Col 3), which are already in `rc_spec_ids`. No need to double-list.

Retained `design/randomized_experiment/estimator/diff_in_means` (no FE at all), which is the only genuinely distinct design estimator.

**Rationale**: Avoid inflating spec counts with duplicates. The RC control-set and FE axes already capture the variation shown in Table 4 Cols 1-3.

### 2. Reclassified dosage and arm-restriction specs from RC to exploration

Moved three specs from `rc_spec_ids` to a new `exploration_universe`:
- `rc/sample/restriction/doctor_arm_only` -> `explore/population/doctor_arm_only`
- `rc/sample/restriction/high_dose` -> `explore/population/high_dose`
- `rc/sample/restriction/low_dose` -> `explore/population/low_dose`

**Rationale**: Per CLAIM_GROUPING.md, sample restrictions that change the target population are not estimand-preserving RCs. High-dose and low-dose pregnancies are definitionally different subpopulations (pregnancies with different exposure durations). These are heterogeneity analyses (Table 5), not robustness checks of the main ITT estimate. Similarly, restricting to Doctor-vs-Control-only facilities changes the comparison set.

### 3. Added `estimand`, `n_clusters`, and `mandatory_treatment_vars` to design_audit/constraints

- Added `"estimand": "ITT"` to `design_audit` per DESIGN_AUDIT_FIELDS.md.
- Added `"n_clusters"` (approximate) for interpretability.
- Added `mandatory_treatment_vars` constraint: both `mlp` and `doctor` must appear in all regressions (the paper always includes both arms). Only `rc/data/treatment/doctor_only` explicitly deviates from this.

### 4. Added `control_variable_notes` to constraints

Documented the exact Stata global macro definitions from tables.do:
- `cont_ind`, `cont_base`, `cont_hc`, `cont_all`

This ensures the runner can reconstruct the exact control sets without re-reading the Stata code.

### 5. Added note about factor variables in baseline_specs

The control list for Table 4 Col 2 now includes `i.magedum` and `i.mschool` (Stata factor notation) to clarify that these are categorical variables with multiple dummy indicators, not single continuous controls.

### 6. Added `reviewed_at` timestamp

---

## Key Constraints and Linkage Rules

- **Control-count envelope**: [0, 25]. Table 4 ranges from no controls (Col 1) to full cont_all (~25 vars, Col 4 double-lasso).
- **No linkage constraints**: Single-equation OLS/RCT design.
- **Both treatment dummies always present**: `mlp` and `doctor` are always included together as regressors (except in the `doctor_only` treatment RC). This is important: the `doctor` coefficient is always estimated conditional on controlling for `mlp`.
- **Child file restriction**: The child.dta analysis file is restricted to `alive==1` (live births only). This is a fundamental sample definition, not a robustness axis.
- **Quarter FE (qtr)**: Present in Table 4 Cols 2-4 but not Col 1. The FE axis `rc/fe/add/qtr` adds it to the baseline (Col 1 -> Col 2 structure), and `rc/fe/drop/qtr` removes it from the enriched specs.
- **Double-lasso (dsregress, Table 4 Col 4)**: Excluded from the core surface because dsregress is Stata 17-specific and not easily replicated in Python/pyfixest. The core surface covers the standard OLS specifications (Cols 1-3).

---

## Budget and Sampling Assessment

- **Budget**: 80 max core specs. Appropriate for the universe size.
- **Estimated total core specs after review changes**:
  - 2 baseline specs (Col 1 and Col 2)
  - 1 design variant (diff_in_means -- no FE)
  - 3 control sets (none, basic, extended)
  - 4 progression specs (strata_only, strata_qtr, strata_qtr_individual, strata_qtr_individual_hc)
  - 12 LOO specs (drop each control from extended set one at a time)
  - 10 random subset specs (seeded draws)
  - 2 FE variants (add/drop qtr)
  - 1 sample restriction (exclude multiple births)
  - 1 outcome variant (mort30 -- borderline RC, see below)
  - 1 treatment definition (doctor_only)
  - Total: ~37 core specs
- **3 exploration specs** (high_dose, low_dose, doctor_arm_only) -- separate from core
- **Control subset sampling**: 10 budgeted draws with seed 181581, stratified by size. Appropriate.
- Well within budget. No additional sampling needed.

---

## What's Missing (minor, non-blocking)

1. **Table A.5 (30-day mortality)**: Already included as `rc/data/outcome/mort30`. This is a borderline case: 30-day mortality includes deaths from days 8-30 which have different etiologies than day 0-7 deaths. The surface markdown argues "same concept (neonatal mortality) with different cutoff." This is defensible but the verifier notes that stricter classification would put this in `explore/variable_definitions`. Retained as RC with this caveat.

2. **Table A.7 (7-day mortality per pregnancy, woman-level)**: Uses `woman.dta` instead of `child.dta`, changing the unit of analysis from child to pregnancy. This is a different statistical object (pregnancy-level rather than child-level) and would belong in `explore/population/pregnancy_level` if desired. Not included and correctly so.

3. **Table A.8 (birthweight)**: Different outcome concept. Correctly excluded.

4. **Table A.11 (obstetric care quality: uterotonic, cord traction)**: Different outcomes, mechanism analysis. Correctly excluded.

5. **ANCOVA design variant**: Not feasible since `mort7` is not measured at baseline (it is a post-delivery outcome). Correctly omitted.

6. **Randomization inference / wild cluster bootstrap**: With facility-level clustering and potentially a small number of clusters, wild cluster bootstrap would be a valuable inference variant. Not included. Non-blocking since the paper does not use it.

7. **Strata-level clustering caveat**: The inference variant `infer/se/cluster/strata` clusters at the strata level. If the number of strata is small, this will produce unreliable SEs. Added a note to the surface.

---

## Final Assessment

**APPROVED TO RUN.**

The surface is well-constructed and faithful to the paper. The single baseline group is appropriate for the mortality claim. The main changes were: (a) removing redundant design spec IDs that duplicated RC specs, (b) reclassifying dosage/arm-restriction specs from RC to exploration (per CLAIM_GROUPING.md rules on population changes), (c) adding missing design_audit fields, and (d) documenting the exact control variable definitions from the Stata code. No blocking issues identified.
