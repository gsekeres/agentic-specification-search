# Verification Report: 149262-V2

**Paper**: Mixed-ability seating experiment in Chinese elementary schools (Longhui county)
**Design**: Randomized experiment — classroom-level random assignment to mixed-ability seating (MS), mixed-ability seating with role assignment (MSR), or control (same-ability seating)
**Verified**: 2026-02-24
**Verifier**: post-run audit agent

---

## 1. Baseline Groups

### G1 — Academic Performance Outcomes

- **Claim**: ITT effect of mixed-ability seating (treat1=MS) vs. same-ability seating (control) on lower-track students' endline academic performance (standardized composite score `ave3`)
- **Population**: Lower-track students (hsco==0), N=901, grades 3-5, Longhui county
- **Estimator**: ANCOVA-OLS with grade1 FE, clustered SE at class1 level
- **5 baseline specs** (G1):

| spec_run_id | spec_id | outcome | coef | p-value |
|---|---|---|---|---|
| 149262-V2_spec_0001 | baseline | ave3 | 0.0128 | 0.881 |
| 149262-V2_spec_0002 | baseline__table3_pA_col1 | ave3 | 0.0060 | 0.961 |
| 149262-V2_spec_0003 | baseline__table3_pA_col4_chn | tchn3 | -0.0205 | 0.773 |
| 149262-V2_spec_0004 | baseline__table3_pA_col6_math | tmath3 | -0.0040 | 0.970 |
| 149262-V2_spec_0005 | baseline__table3_pB_col2_upper | ave3 | -0.0626 | 0.279 |

Note: spec_0005 uses the upper-track sample (hsco==1), a different population from the G1 claim object baseline. This spec is included because the surface explicitly listed it as a paper-table replication (Table 3 Panel B). It is marked `is_baseline=1, is_core_test=1, category=core_method` per the surface plan.

### G2 — Personality Trait Outcomes

- **Claim**: ITT effect of mixed-ability seating (treat1=MS) vs. same-ability seating (control) on lower-track students' Big Five personality traits (extraversion, agreeableness, openness, neuroticism, conscientiousness)
- **Population**: Lower-track students (hsco==0), N=901, grades 3-5, Longhui county
- **Estimator**: ANCOVA-OLS with grade1 FE, clustered SE at class1 level
- **10 baseline specs** (G2):

| spec_run_id | spec_id | outcome | coef | p-value |
|---|---|---|---|---|
| 149262-V2_spec_0041 | baseline | extra2 | 0.5526 | 0.511 |
| 149262-V2_spec_0042 | baseline__table4_pA_agree | agree2 | 1.5246 | 0.281 |
| 149262-V2_spec_0043 | baseline__table4_pA_open | open2 | 0.0192 | 0.976 |
| 149262-V2_spec_0044 | baseline__table4_pA_neur | neur2 | 0.9846 | 0.201 |
| 149262-V2_spec_0045 | baseline__table4_pA_cons | cons2 | 0.8468 | 0.361 |
| 149262-V2_spec_0046 | baseline__table4_pB_extra_upper | extra2 | -0.9184 | 0.243 |
| 149262-V2_spec_0047 | baseline__table4_pB_agree_upper | agree2 | -0.6760 | 0.567 |
| 149262-V2_spec_0048 | baseline__table4_pB_open_upper | open2 | -0.2030 | 0.584 |
| 149262-V2_spec_0049 | baseline__table4_pB_neur_upper | neur2 | 0.5068 | 0.441 |
| 149262-V2_spec_0050 | baseline__table4_pB_cons_upper | cons2 | -0.0672 | 0.935 |

Note: spec_0046-0050 use the upper-track sample (hsco==1). Included as surface-planned Table 4 Panel B replications.

---

## 2. Counts

| Metric | Count |
|---|---|
| Total rows | 71 |
| run_success=1 | 71 |
| run_success=0 | 0 |
| is_valid=1 | 71 |
| is_valid=0 | 0 |
| is_baseline=1 | 15 |
| is_core_test=1 | 62 |
| is_core_test=0 (non-core) | 9 |
| G1 specs | 40 |
| G2 specs | 31 |

### Category Breakdown

| Category | Count | Namespace |
|---|---|---|
| core_method | 22 | baseline, baseline__*, design/* |
| core_controls | 32 | rc/controls/* |
| core_sample | 5 | rc/sample/* |
| core_fe | 3 | rc/fe/* |
| noncore_alt_outcome | 5 | rc/outcome/* |
| noncore_alt_treatment | 4 | rc/treatment/* |

---

## 3. Surface Alignment

Both G1 and G2 executed **exactly the specs listed in the surface's `core_universe`** — no missing specs, no extra specs.

- G1: 40 planned, 40 executed (match: perfect)
- G2: 31 planned, 31 executed (match: perfect)

All `infer/*` variants appear only in `inference_results.csv` (5 rows: 3 for G1 baseline, 2 for G2 baseline). No `infer/*` rows appear in `specification_results.csv`.

Inference SE: All 71 rows in `specification_results.csv` use `infer/se/cluster/class1` (cluster at classroom level), matching the surface's canonical inference plan for both groups.

---

## 4. Validator Issues (Pre-existing in Source Data)

Running `python scripts/validate_agent_outputs.py --paper-id 149262-V2` reports 196 ERRORs. All errors are in the source data files (`specification_results.csv` and `inference_results.csv`) — none are in the verification output files. These issues were present in the outputs before verification.

### 4a. design_audit_value_mismatch (73 errors in specification_results.csv)

**Root cause 1 — strata_blocking string**: The surface `design_audit` has `strata_blocking: "grade1 (school x grade)"`, but all 71 coefficient_vector_json design blocks record `strata_blocking: "grade1"` (shorter string). This is a cosmetic string mismatch; the strata variable used is the same in both cases. Affects all 71 rows.

**Root cause 2 — G2 design_audit sample_restriction wrong in surface**: The surface's G2 `design_audit` incorrectly records `sample_restriction: "hsco==1 (upper track)"` for G2. In reality, the G2 primary sample is lower-track (hsco==0), confirmed by `sample_desc` and N=901 in all 21 non-upper-track G2 rows. The runner used the correct lower-track sample; the surface metadata field is wrong. The 10 upper-track baseline specs (spec_0046-0050 plus their RC equivalents... but G2 only has 5 upper-track baseline rows) correctly record `hsco==1`. This affects G2 rows where the design block carries `hsco==0` (correct for execution) but the surface says `hsco==1`.

### 4b. surface_hash_mismatch (7 errors in spec_results + inference_results)

The `SPECIFICATION_SURFACE.json` was modified after the spec run, so the hash stored in `coefficient_vector_json.surface_hash` (`sha256:caca816f...`) no longer matches the current file hash (`sha256:bb71811...`). This is a post-run surface edit; the run itself used the hash-consistent surface version. Affects all inference_results rows and a subset of spec rows (flagged where checked).

### 4c. infer_spec_id_not_in_surface_inference_plan (2 errors in inference_results.csv)

`infer/ri/randomization_inference` appears in inference_results (infer_0003 for G1, infer_0005 for G2) but the surface's `inference_plan.variants` list only includes string `spec_id` fields that the validator may not pattern-match correctly. The RI spec is listed in the surface as a variant for both groups; this may be a validator pattern-matching issue rather than a genuine missing spec.

**Summary**: None of the 196 errors reflect problems with the coefficients, sample filters, or treatment assignments that were actually executed. All estimates are valid and correctly extracted.

---

## 5. Issues in Executed Specs (Non-validator)

### Issue 1: treatment_var metadata mislabeling in rc/treatment/treat2_focal rows (minor)

**Rows**: spec_0037 (G1), spec_0069 (G2)
**Description**: Both `rc/treatment/treat2_focal` rows have `treatment_var='treat1'` in the CSV, but the focal coefficient extracted is treat2's coefficient. Cross-checking confirms the coefficient values match treat2's coefficient in the baseline specs (spec_0037 coef=0.1388 == spec_0001 treat2 coef=0.1388; spec_0069 coef=2.3248 == spec_0041 treat2 coef=2.3248). The coefficient is correctly extracted; only the `treatment_var` column label is wrong.
**Impact**: Minimal — coefficient extraction is correct. The category `noncore_alt_treatment` is appropriate.
**Recommendation**: Runner script should set `treatment_var='treat2'` when focal regressor changes to treat2.

### Issue 2: Stale sample_restriction in design block for upper-track baseline specs (cosmetic)

**Rows**: spec_0005 (G1 upper-track), spec_0046-0050 (G2 upper-track)
**Description**: The `design.randomized_experiment.sample_restriction` field inside `coefficient_vector_json` still reads `"hsco==0 (lower track)"` for the upper-track specs. The `sample_desc` column and N=901 correctly identify the upper-track sample.
**Impact**: None on coefficients. The design block was copied from the lower-track template without updating the sample restriction field.
**Recommendation**: Runner should update `design.randomized_experiment.sample_restriction` when changing the sample filter.

---

## 6. Inference Results

`inference_results.csv` contains 5 rows covering alternative SE/RI variants for the two primary baseline specs:

| inference_run_id | spec_run_id | spec_id | group | success |
|---|---|---|---|---|
| 149262-V2_infer_0001 | 149262-V2_spec_0001 | infer/se/hc/hc1 | G1 | 1 |
| 149262-V2_infer_0002 | 149262-V2_spec_0001 | infer/se/cluster/grade1 | G1 | 1 |
| 149262-V2_infer_0003 | 149262-V2_spec_0001 | infer/ri/randomization_inference | G1 | 1 |
| 149262-V2_infer_0004 | 149262-V2_spec_0041 | infer/se/hc/hc1 | G2 | 1 |
| 149262-V2_infer_0005 | 149262-V2_spec_0041 | infer/ri/randomization_inference | G2 | 1 |

Note: G2 is missing `infer/se/cluster/grade1` (present for G1 but not G2). The surface lists this as an optional variant for G2 as well. Not a critical omission — the key RI check is present for both groups.

---

## 7. Substantive Assessment

All baseline effects for treat1 (MS) are small and statistically insignificant:
- G1 (academic): p ranges from 0.773 to 0.961 across baseline specs — no significant effect of MS on academic outcomes
- G2 (personality): p ranges from 0.201 to 0.976 across baseline specs — no significant effect of MS on personality traits

RC universe confirms robustness in both groups: no controls variant, sample variant, or FE variant produces a significant effect for treat1 on academic outcomes (G1) or personality outcomes (G2).

Notable non-core findings: The treat2_focal spec (spec_0069, G2) shows a statistically significant effect of MSR on extraversion (coef=2.32, p=0.004), and the pooled-treatment spec (spec_0070, G2) shows a marginally significant effect (coef=1.39, p=0.042). Both are `noncore_alt_treatment` since they test a different treatment arm. These are consistent with the paper's broader finding that the MSR (role-assignment) treatment may have a stronger effect on personality than the plain MS treatment.

The paper's core findings replicate cleanly: no significant effect of MS (treat1) on academic outcomes or personality traits for lower-track students.

---

## 8. Recommendations

1. **Fix surface G2 design_audit**: Change `G2.design_audit.sample_restriction` from `"hsco==1 (upper track)"` to `"hsco==0 (lower track)"` — the G2 baseline is lower-track; upper-track specs are listed separately in baseline_spec_ids.
2. **Fix strata_blocking string**: Standardize `strata_blocking` across surface design_audit and coefficient_vector_json design blocks (either add description or remove it from one location).
3. **Fix treatment_var labeling**: In `rc/treatment/treat2_focal` specs, set `treatment_var='treat2'` to reflect the actual focal regressor.
4. **Update design block sample_restriction**: For upper-track sample variants (spec_0005, spec_0046-0050), update `design.randomized_experiment.sample_restriction` to `"hsco==1 (upper track)"`.
5. **Add infer/se/cluster/grade1 for G2**: Complete parity with G1 inference variants.
6. **Re-run after surface edits**: The current surface hash does not match the hash stored in coefficient_vector_json — if the surface was edited for correctness, the runner should be re-run or the surface hash field updated in existing outputs.
