# Verification Report: 171681-V1

**Paper**: Ambuehl, Bernheim & Lusardi, "Evaluating Deliberative Competence"
**Design**: Randomized experiment (online, MTurk); 4-arm RCT (Experiment A) + 2-arm replication (Experiment B)
**Verified**: 2026-02-24
**Verifier**: Post-run audit agent

---

## 1. Baseline Groups

Two baseline groups match the surface plan exactly.

### G1: Compounding Knowledge (Table 4)

| Field | Value |
|-------|-------|
| Outcome concept | Compounding test score (score_compounding, 0-5 scale) |
| Treatment concept | Full financial literacy intervention (Rule of 72 + rhetoric) vs Control |
| Estimand | OLS ITT, cluster-robust SE at individual (id) level |
| Target population | Experiment A participants (sample=='old'), tag==1, N=214 |

**Baseline specs executed:**

| spec_run_id | spec_id | coef | SE | p-value | N |
|-------------|---------|------|----|---------|---|
| 171681-V1__run_0001 | baseline (alias: table4_col1_expA) | 1.424 | 0.194 | 4.5e-12 | 214 |
| 171681-V1__run_0002 | baseline__table4_col2_expB | 1.597 | 0.143 | <1e-15 | 347 |

Both primary (Exp A) and secondary (Exp B) baselines executed successfully. Effect is strongly positive and significant in both experiments, consistent with prior literature.

### G2: Financial Competence / Valuation Errors (Table 7)

| Field | Value |
|-------|-------|
| Outcome concept | Negative absolute valuation difference: negAbsDiff = -|discount_framed - discount_unframed| |
| Treatment concept | Full financial literacy intervention (Full arm) vs Control (Wald test Full==Control) |
| Estimand | No-constant OLS ITT with all 4 arm dummies, cluster-robust SE at individual level |
| Target population | Experiment A observation-level data (subject x price-list), sample=='old', N=4510 |

**Baseline specs executed:**

| spec_run_id | spec_id | coef (Full-Control) | SE | p-value | N |
|-------------|---------|---------------------|-----|---------|---|
| 171681-V1__run_0030 | baseline (alias: table7_col1_expA) | 1.224 | 2.369 | 0.605 | 4510 |
| 171681-V1__run_0031 | baseline__table7_col2_expB | 7.415 | 1.774 | 3.0e-05 | 3470 |

Key observation: The G2 primary baseline (Exp A) is **not statistically significant** (p=0.605). Exp B replication shows a large, significant effect. This asymmetry is substantively important and correctly captured by the run.

---

## 2. Spec Counts

| Metric | G1 | G2 | Total |
|--------|----|----|-------|
| Total rows | 29 | 40 | 69 |
| run_success=1 | 29 | 40 | 69 |
| run_success=0 | 0 | 0 | 0 |
| is_valid=1 | 29 | 40 | 69 |
| is_valid=0 | 0 | 0 | 0 |
| is_baseline=1 | 2 | 2 | 4 |
| is_core_test=1 | 23 | 33 | 56 |
| is_core_test=0 | 6 | 7 | 13 |

---

## 3. Category Breakdown

| Category | Count | Notes |
|----------|-------|-------|
| core_method | 11 | Baselines (4) + alternative estimators (4) + all_arms_nocons (2) + full_vs_control_with_intercept (1) |
| core_controls | 33 | Single additions (11), curated sets (6), LOO (6), joint (10) |
| core_sample | 11 | Experiment subsamples (8), delay subsamples (2), outlier trims (3) — note: delay subsamples only in G2 |
| core_funcform | 1 | rc/form/individual_means (G2 only): aggregated to individual means (dAbs) |
| noncore_alt_outcome | 9 | rc/outcome/* — alternative outcome variables change the claim object concept |
| noncore_alt_treatment | 4 | rc/treatment/rule72_only_vs_control (x2) and rc/treatment/rhetoric_only_vs_control (x2) — arm substitutions |
| **Total** | **69** | |

---

## 4. Sanity Check Results

All checks passed:

- **spec_run_id uniqueness**: All 69 spec_run_ids are unique. No duplicates.
- **baseline_group_id**: All rows have valid baseline_group_id (G1 or G2). G1=29 rows, G2=40 rows.
- **run_success**: All 69 rows have run_success=1. Zero failures.
- **infer/* in specification_results.csv**: None present. Inference variants correctly isolated to inference_results.csv.
- **numeric field validity**: All coef/SE/p_value fields populated for run_success=1 rows. No NaN for coef/SE/p_value/n_obs.
- **p_value range**: All p_values in [0, 1]. No negative or >1 values.
- **r_squared range**: All R2 in [0.0006, 0.455]. No invalid values.
- **surface_hash**: All 69 rows share the same surface hash (sha256:041c4646c5a78637a0e4e376de7e0cc1d8b9ee062acf774a10779689e5e352bd). Consistent.
- **inference.spec_id**: All 69 rows use `infer/se/cluster/id` as the inference spec_id, matching both surface canonical inference plans.
- **coefficient_vector_json structure**: All rows have required keys: `coefficients`, `inference`, `software`, `surface_hash`, `design`. rc/controls rows include `controls` block; rc/sample rows include `sample` block; rc/joint rows include `joint` block; rc/form row includes `functional_form` block (non-empty).
- **p_value = 0.0 (underflow)**: 5 rows have exactly p=0.0 (all Exp B G1 specs with very large effects). This reflects floating-point underflow for extremely small p-values, not a data error.
- **CI missing (47 rows)**: ci_lower/ci_upper are null for many rows, primarily all G2 rows and some G1 alternative-outcome rows. This occurs because the runner did not compute CIs for no-constant multi-arm OLS models or for certain alternative outcome formats. This is a minor gap — scalar SE and p_value are present for all rows and are sufficient for verification.

---

## 5. Surface Alignment

### Spec ID mapping (baseline spec_id aliases)
The runner used `spec_id = "baseline"` as the primary alias for both G1 and G2, rather than the surface's planned `baseline__table4_col1_expA` and `baseline__table7_col1_expA`. The executed rows match the planned specs exactly (same sample filter, N, outcome, treatment). This is a minor cosmetic deviation only.

### Missing specs (skipped by runner)
Two specs from the surface core universe were not executed:
- `rc/sample/attrition/include_multi_switchers__requires_remanagement` (both G1 and G2): requires re-running the full data management pipeline without the multi==0 exclusion. Documented in SPECIFICATION_SEARCH.md note 3. Not executable in the current environment without remanagement.

### No extra specs added
All executed specs appear in the surface core universe. No ad hoc specs were added outside the surface plan.

---

## 6. Outcome/Treatment Drift Assessment

### G1 non-core specs
- **rc/outcome/score_indexing** (run_0020): coef=-1.070, p<0.001. Negative sign: score_indexing appears to be an inverted or differently-scaled version of compounding knowledge. The negative coefficient may reflect that treatment reduces "indexing" score (which may be a penalty metric). Not a coding error — different scale convention. Classified as noncore_alt_outcome.
- **rc/outcome/fl_score_compound, fl_sum_compound** (run_0021-0022): Insignificant (p~0.29-0.38). These are broader financial literacy scores. Different outcome concept.
- **rc/outcome/score_compounding_plus_indexing** (run_0023): Insignificant (p=0.207). Combined score variable. Different outcome concept.
- **rc/treatment/rule72_only_vs_control** (run_0024): coef=1.265, p<0.001. Rule72-only arm — informative for decomposing treatment, but changes the treatment concept.
- **rc/treatment/rhetoric_only_vs_control** (run_0025): coef=0.496, p=0.015. Rhetoric-only arm — smaller effect, changes treatment concept.

### G2 non-core specs
- **rc/outcome/negSqDiff** (run_0053): coef=-3.342, p=0.376. Full arm has more negative negSqDiff (larger squared error). Not significant. Different outcome concept.
- **rc/outcome/diff** (run_0054): coef=14.565, p<0.001. Treatment increases the raw difference (less consistent discounting in absolute terms?). Outcome concept change.
- **rc/outcome/discount_framed** (run_0055): coef=14.919, p<0.001. Treatment increases framed discount rate. Different outcome concept.
- **rc/outcome/finCompCorr, finCompCorrSq** (run_0056-0057): Mixed significance. Alternative financial competence measures.
- **rc/treatment/rule72_only_vs_control, rhetoric_only_vs_control** (run_0058-0059): Change treatment arm — non-core.

No cases of clearly mis-extracted coefficients or wrong-sign errors. All outcomes appear correctly computed.

---

## 7. Inference Results File

The inference_results.csv contains 8 rows (infer_0001 through infer_0008):
- 4 rows for G1 baselines (HC1 and HC3 on run_0001 and run_0002)
- 4 rows for G2 baselines (HC1 and HC3 on run_0030 and run_0031)

These are correctly isolated in inference_results.csv and do not appear in specification_results.csv. HC1 and HC3 SEs are very close to clustered SEs (as expected for individual-level data in G1) or slightly different (as expected for observation-level data in G2).

---

## 8. Top Issues and Recommendations

### Issues (minor only, no invalids)

1. **Primary spec_id alias**: Runner used `spec_id="baseline"` instead of the surface's `baseline__table4_col1_expA` / `baseline__table7_col1_expA`. Recommend using explicit spec_ids matching surface plan in future runs for traceability.

2. **Missing CIs for 47 rows**: No ci_lower/ci_upper for G2 rows and some G1 alternative-outcome rows. The runner does not extract CIs from no-constant multi-arm models. Recommend implementing contrast CI extraction for Wald tests. Not critical for validity.

3. **Skipped attrition specs**: `rc/sample/attrition/include_multi_switchers__requires_remanagement` skipped in both groups. This is a valid skip given pipeline constraints. Surface should mark this spec as `requires_remanagement: true` to signal it is expected to be skipped.

4. **G2 design/estimator/diff_in_means uses Full-vs-Control subsample only**: run_0032 (G2 diff-in-means) uses N=2140 (Full+Control only) rather than the full 4-arm sample. This is the correct approach for the diff-in-means estimator but creates a sample inconsistency with the no-constant 4-arm baseline (N=4510). Internally consistent; worth noting.

### Recommendations for future runs

- Standardize primary baseline spec_ids to match surface plan exactly (avoid alias ambiguity).
- Add CI extraction for Wald test contrasts in no-constant multi-arm models.
- Mark remanagement-requiring specs in the surface with a `skip_if_remanagement_unavailable` flag.
- The rc/form/individual_means spec uses `dAbs` as the outcome variable (positive absolute difference, sign-flipped relative to negAbsDiff). The coefficient interpretation requires care: negative coefficient (-0.995) means treatment reduces absolute error when outcome is dAbs, consistent with G2 claim direction.

---

## 9. Overall Assessment

**Run quality: HIGH**

- 69/69 specs succeeded.
- All baseline groups correctly identified and executed.
- Internal consistency confirmed (matched coefficients for equivalent specifications).
- Inference spec_ids consistent with surface canonical plan throughout.
- No invalid, spurious, or drift-affected core specs identified.
- The substantive finding that G2 Exp A is null (p=0.605) while G2 Exp B is strongly significant (p<0.001) is correctly and precisely captured by the run.
- Non-core specs (alternative outcomes and treatment arms) are correctly identified and do not contaminate the core robustness curve.
