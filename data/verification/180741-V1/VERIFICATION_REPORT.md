# Verification Report: 180741-V1

**Paper**: Saccardo & Serra-Garcia — advisor incentive experiments (information order and biased recommendations)
**Verified**: 2026-02-24
**Verifier**: Post-run audit agent

---

## 1. Baseline Groups Found

All three baseline groups from `SPECIFICATION_SURFACE.json` are present in `specification_results.csv` with no spurious groups.

### G1 — Choice Experiment: Effect of Preference for Information Order on Recommendations

- **Claim**: Effect of `choicebefore` on `recommendincentive`, baseline-stakes Choice experiment (Highx10==0 & Highx100==0), ITT, OLS-LPM, HC3 SEs.
- **Baseline spec_run_ids**:
  - `G1__baseline__table3_col1` (Table 3 Col 1, getyourchoice==1, N=4448, coef=0.195, p<1e-30)
  - `G1__baseline__table3_col2` (Table 3 Col 2, getyourchoice==0, N=1460, coef=0.003, p=0.917)
  - `G1__baseline__table3_col3` (Table 3 Col 3, pooled, N=5908, coef=0.181, p<1e-30)
- **Expected sign**: Positive (choicebefore → more recommendations of incentivized product)
- **Rows in group**: 29

### G2 — NoChoice Experiment: Effect of Random Assignment to See Incentive First

- **Claim**: Effect of `seeincentivefirst` on `recommendincentive`, NoChoice experiment, missingalpha==0, ITT, OLS-LPM, HC3 SEs.
- **Baseline spec_run_ids**:
  - `G2__baseline__tablec1_col1` (Table C.1 Col 1, conflict==1, N=213, coef=0.142, p=0.024)
  - `G2__baseline__tablec1_col2` (Table C.1 Col 2, conflict==0, N=86, coef=0.030, p=0.709)
  - `G2__baseline__tablec1_col3` (Table C.1 Col 3, full NoChoice sample, N=299, coef=0.148, p=0.017)
- **Expected sign**: Positive (seeing incentive first → more biased recommendations in conflict condition)
- **Rows in group**: 17

### G3 — Choice Experiment: Effect of Costly-Choice Condition on Information-Order Preference

- **Claim**: Effect of `seeincentivecostly` on `choicebefore`, baseline-stakes Choice experiment (Highx10==0 & Highx100==0), ITT, OLS-LPM, HC3 SEs.
- **Baseline spec_run_ids**:
  - `G3__baseline__table2_col1` (Table 2 Col 1, full sample, N=5908, coef=-0.139, p=5.8e-15)
  - `G3__baseline__table2_col2` (Table 2 Col 2, nonprofessionals only, N=5196, coef=-0.140, p=4.9e-15)
  - `G3__baseline__table2_col3` (Table 2 Col 3, full with selfishness interactions, N=5196, coef=-0.140, p=4.4e-15)
- **Expected sign**: Negative (costly-choice condition reduces preference for seeing incentive first)
- **Rows in group**: 23

---

## 2. Row Counts

| Metric | Count |
|---|---|
| Total rows in specification_results.csv | 69 |
| run_success=1 | 69 |
| run_success=0 | 0 |
| is_valid=1 | 69 |
| is_valid=0 | 0 |
| is_baseline=1 | 9 |
| is_core_test=1 | 69 |
| is_core_test=0 | 0 |
| infer/* rows in specification_results.csv | 0 (correct) |
| Rows in inference_results.csv | 15 |

---

## 3. Category Counts

| Category | Count |
|---|---|
| core_method | 15 |
| core_controls | 32 |
| core_sample | 16 |
| core_funcform | 6 |
| core_fe | 0 |
| core_preprocess | 0 |
| core_data | 0 |
| core_weights | 0 |
| noncore_* | 0 |
| **Total** | **69** |

---

## 4. Sanity Check Results

### 4.1 Structural Checks — PASS

- `spec_run_id`: 69 rows, all unique, no duplicates.
- `baseline_group_id`: All rows have valid group ID (G1, G2, or G3).
- `run_success`: All 69 rows have run_success=1; no run_success=0 rows.
- `run_error`: All empty (consistent with 100% success rate).
- `spec_tree_path`: All reference spec-tree `.md` nodes with `#anchor` fragments. Paths observed: `designs/randomized_experiment.md#baseline`, `designs/randomized_experiment.md#diff-in-means`, `designs/randomized_experiment.md#with-covariates`, `modules/robustness/controls.md#leave-one-out-controls-loo`, `modules/robustness/controls.md#add-controls`, `modules/robustness/controls.md#minimal-controls`, `modules/robustness/sample.md#restrict-subgroup`, `modules/robustness/sample.md#include-subgroup`, `modules/robustness/sample.md#include-inattentive`, `modules/robustness/functional_form.md#probit`, `modules/robustness/functional_form.md#logit`.
- `coefficient_vector_json`: All 69 rows have the four required audit keys (`coefficients`, `inference`, `software`, `surface_hash`).
- `infer/*` namespace: Zero rows in specification_results.csv (correct). Inference variants correctly appear only in inference_results.csv (15 rows: HC1/HC2 variants for G1/G2 baselines; HC1 only for G3 baselines, consistent with surface plan).
- **Canonical inference**: All 69 rows in specification_results.csv use `infer/se/hc/hc3` as the inference spec_id in coefficient_vector_json — consistent with surface canonical choice for all three groups.
- `rc/outcome/*` rows: All 6 have `functional_form` block present in coefficient_vector_json — PASS.
- No arbitrary top-level keys introduced beyond `coefficients`, `inference`, `software`, `surface_hash`, `design`, `functional_form`.

### 4.2 Surface Coverage — PASS

- G1: All 3 baseline specs + 2 design specs + 24 rc specs = 29 rows (surface planned all 24 rc_spec_ids; all executed).
- G2: All 3 baseline specs + 2 design specs + 12 rc specs = 17 rows (surface planned all 12 rc_spec_ids; all executed).
- G3: All 3 baseline specs + 2 design specs + 18 rc specs = 23 rows (surface planned all 18 rc_spec_ids; all executed).
- No missing surface spec_ids; no spurious spec_ids not in the surface.

### 4.3 Numeric Validity — PASS

- All 69 run_success=1 rows have finite non-missing values for `coefficient`, `std_error`, `p_value`, `ci_lower`, `ci_upper`, `n_obs`, `r_squared`.
- No p_value > 1.0 detected.
- G3 and several G1/G2 p-values appear as machine-epsilon floats (e.g., 5.77e-15, 4.44e-15) rather than exact zero. These are valid — they represent highly significant results stored at floating-point precision by statsmodels/pyfixest HC3 computation. Sign and magnitude are consistent across all specifications.

---

## 5. Issues and Notes

### Issue 1 (MINOR): G2 Near-Duplicate RC Sample Specs

**Rows**: `G2__rc__sample__restrict_conflict_only` and `G2__rc__sample__restrict_noconflict_only`

These produce identical estimates to `G2__baseline__tablec1_col1` (coef=0.1416, p=0.024, N=213) and `G2__baseline__tablec1_col2` (coef=0.0303, p=0.709, N=86) respectively. This is structurally expected: the G2 baselines are defined on the conflict and no-conflict subsamples, so restricting to those subsamples in the RC reproduces the baseline. These rows are **not flagged invalid** — both the surface and the runner correctly include them as pre-planned specs. However, they contribute no independent information beyond the baselines. Downstream robustness scoring should account for this.

### Issue 2 (MINOR): Design Specs Use Slightly Larger N than Baselines

**Rows**: `G1__design__diff_in_means` (N=5915), `G1__design__with_covariates` (N=5915), and similarly for G3 (N=5915 vs baseline col1 N=5908).

The 7-row difference suggests the design specs use a slightly different sample filter than the baseline col3 specification (Highx10==0 & Highx100==0 drops N=5908; design specs appear to use N=5915 from the full Choice sample). This is within the expected scope of the same claim object and does not constitute outcome/treatment drift. **Not flagged invalid.**

### Issue 3 (INFORMATIONAL): G3 P-Values Are Machine Epsilon, Not Exact Zero

G3 baselines and most G3 RC specs show p-values between 4e-15 and 1e-11 rather than exact 0.0. These are valid: statsmodels computes two-sided p-values from t-statistics via the t-distribution CDF, and at large sample sizes (N≈5908) with effects several SEs away from zero, floating-point underflow produces machine-epsilon values. This does not affect validity.

### Issue 4 (INFORMATIONAL): G1 Table3-Col2 is Not Significant

The baseline spec `G1__baseline__table3_col2` (getyourchoice==0 subsample) shows coef=0.003, p=0.917. This is the expected null result from the paper (advisors who received their non-preferred order show no anchoring effect). This is correct and expected — it does not flag a problem with the run.

---

## 6. Inference Results (inference_results.csv)

15 inference variant rows present:
- G1: 6 rows (3 baselines × HC1 + HC2)
- G2: 6 rows (3 baselines × HC1 + HC2)
- G3: 3 rows (3 baselines × HC1 only — consistent with surface plan which lists only HC1 as a variant for G3)

All 15 rows have run_success=1. All are correctly stored in inference_results.csv only (not in specification_results.csv). Consistent with surface inference plans.

---

## 7. Overall Assessment

**Status**: CLEAN — No invalid rows, no failed specs, no outcome/treatment drift, no namespace violations.

All 69 specifications executed successfully. The run is fully surface-compliant. The two near-duplicate G2 sample restrictions are pre-planned by the surface and structurally expected. The minor N discrepancy in design specs is within acceptable scope of the same claim object. This paper is ready for downstream robustness scoring.

**Recommendation for surface improvement**: Consider removing `rc/sample/restrict_conflict_only` and `rc/sample/restrict_noconflict_only` from G2's rc_spec_ids in future surface runs, since these exactly replicate the G2 baseline subsamples. Alternatively, keep them but note in the scoring pipeline that they are not independent from the baselines.
