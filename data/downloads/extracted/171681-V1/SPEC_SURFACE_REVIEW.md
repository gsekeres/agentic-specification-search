# Spec Surface Review: 171681-V1

**Paper**: Ambuehl, Bernheim & Lusardi, "Evaluating Deliberative Competence"
**Audit date**: 2026-02-24
**Reviewer**: Spec Surface Verifier Agent
**Files audited**: `SPECIFICATION_SURFACE.json`, `SPECIFICATION_SURFACE.md`, `doFiles/analysis.do`, `doFiles/dataManagementExpA.do`, `doFiles/dataManagementExpB.do`

---

## Summary of Baseline Groups

Two baseline groups are confirmed as appropriate and cover the paper's two main claims:

- **G1**: Compounding knowledge (`score_compounding`) — individual-level, one obs per subject (`tag == 1`), Exp A, Full vs Control. Table 4, Col 1.
- **G2**: Financial competence (`negAbsDiff = -absDiff`) — observation-level (subject × price-list), Exp A, no-constant regression with all four arm dummies, focal test `Full == Control`. Table 7, Col 1.

Both groups use cluster-robust SE at the `id` level, matching the paper's `cl(id)` throughout.

---

## Issues Found and Changes Made

### Issue 1 (MATERIAL): `multi == 0` filter absorbed globally before table regressions

**Problem**: The surface represents `multi == 0` as a sample filter in the baseline specs and as an RC axis to relax. However, `analysis.do` applies `keep if multi == 0` globally at line 107, before any table regressions are run. All regressions in the tables section therefore operate on a dataset already restricted to single-switchers.

Consequently:
- G1 baseline filter `tag == 1 & (Full == 1 | Control == 1) & sample == "old"` is correct for the actual regression (line 565), but `multi == 0` is implicit, not explicit.
- G2 baseline filter `sample == "old" & multi == 0` should be `sample == "old"` — the `multi == 0` clause is redundant.
- **RC: `rc/sample/attrition/include_multi_switchers`** cannot be implemented by relaxing a filter in the analysis script, because the multi-switchers are dropped from the managed dataset before the table section. This RC would require rebuilding the data from scratch. This axis is kept in the surface as an aspirational RC but should be noted as requiring full data re-management.

**Change**: Updated `sample_filter` for G2 baseline to `sample == 'old'`. Added note to `rc/sample/attrition/include_multi_switchers` about data-management dependency.

---

### Issue 2 (MATERIAL): `sqDiff` uses `abs()` — outcome definition in surface is wrong

**Problem**: The surface documents `negSqDiff = -sqDiff = -(discount_framed^2 - discount_unframed^2)/100` (signed). But `analysis.do` line 30 shows:

```stata
gen sqDiff = abs(discount_framed^2 - discount_unframed^2) / 100
```

`sqDiff` is the **absolute value** of the squared difference, not the signed difference. Therefore `negSqDiff = -sqDiff` is always `≤ 0` by construction (it is a loss measure, like `negAbsDiff`). The surface's description was directionally correct (it is a negative loss measure) but the formula was wrong: it should read `negSqDiff = -abs(discount_framed^2 - discount_unframed^2)/100`.

**Change**: Updated the variable definition in the surface. The `rc/outcome/negSqDiff` axis remains valid.

---

### Issue 3 (MATERIAL): Outcome variable `diff_raw` does not exist — code uses `diff`

**Problem**: G2's RC list includes `rc/outcome/diff_raw`. No variable called `diff_raw` exists in the managed dataset. The actual variable is `diff` (defined at line 505 of `dataManagementExpA.do`: `gen diff = discount_framed - discount_unframed`).

**Change**: Renamed RC spec ID from `rc/outcome/diff_raw` to `rc/outcome/diff` throughout G2. Updated variable definition table accordingly.

---

### Issue 4 (MINOR): G1 `core_universe.baseline_spec_ids` omits the primary baseline

**Problem**: G1's `core_universe.baseline_spec_ids` lists only `["baseline__table4_col2_expB"]`. The primary baseline (Table 4, Col 1, Exp A) has label `"Table4-Col1-ExpA"` and is defined in `baseline_specs`, but has no corresponding entry in `baseline_spec_ids`. The field should list the primary baseline spec as well.

**Change**: Added `"baseline__table4_col1_expA"` to G1's `core_universe.baseline_spec_ids`. The primary baseline is the Exp A spec; the Exp B spec is an additional baseline for cross-experiment replication.

---

### Issue 5 (MINOR): G2 `design_audit.n_arms` field is ambiguous

**Problem**: G2's `design_audit` lists `n_arms: 4` and the four arms of Exp A. This is correct for the primary G2 baseline (Exp A). However, the additional Exp B baseline for G2 uses only 2 arms (`contNew`, `fullNew`). The `design_audit` block describes Exp A correctly, but should clarify that the additional baseline for Exp B uses 2 arms.

**Change**: Added clarifying note to G2's `design_audit.notes` that the Exp B additional baseline uses 2 arms.

---

### Issue 6 (MINOR): `negAbsDiff` is constructed inline, not in the managed dataset

**Observation**: `negAbsDiff` does not exist in `data/managedData.dta`. It is constructed ad-hoc (`cap gen negAbsDiff = -absDiff`) in the analysis blocks. Similarly, `finCompCorr` and `finCompCorrSq` are generated inline from `absDiff` and `sqDiff`.

For the Python specification search script, these variables must be constructed from `absDiff` (and `sqDiff`) before running regressions. This is a coding note, not a surface error.

**Change**: Added a construction note to the relevant outcome variable entries in the surface.

---

### Issue 7 (OBSERVATION): `meanVsimple` control is potentially post-treatment

**Observation**: The RC axis `rc/controls/single/add_meanVsimple` adds `meanVsimple = 100 * (meanVsimple36 + meanVsimple72)/2`, which is the within-individual mean of simple-frame discount rates averaged over both delay horizons. This is measured concurrently with the MPL task, after treatment. Because treatment could in principle shift simple-frame valuations (the paper tests this in Table 9 and finds no significant effect), this control is a potential collider/post-treatment covariate.

**Decision**: Retained in G2 surface as an RC, consistent with the paper's own use of simple-frame valuations as a conditioning variable in its heterogeneity analyses (Table 11). Added a note flagging the potential post-treatment nature of this control. It should be reported as a sensitivity check, not a robustness check.

---

### Issue 8 (OBSERVATION): `demoControl` spec covers pooled Exp A + Exp B

**Observation**: The demographic-controls table (`demoControl.tex`) at line 1864 pools Exp A and Exp B (no sample restriction), using `fullNew` (abbreviated as `fullN` in the Stata variable abbreviation system) alongside `Full`, `Control`, `Rule72`, `Rhetoric`. The no-constant regression with all five treatment dummies (`fullNew Full Control Rule72 Rhetoric`) covers both experiments simultaneously.

For Python implementation, the `rc/controls/sets/demographics_full` axis for G1 and G2 should implement within-experiment regressions (with `sample == "old"` for G1 and the equivalent for G2), not the pooled version. The pooled version with full controls is a separate, exploratory specification.

**Change**: Added a note to the `demographics_full` control set clarifying that baseline G1/G2 regressions with controls remain within-experiment.

---

## Controls Count Verification

The paper's full control set (`demoControl`, line 1861) contains:

| Category | Variables | Count |
|---|---|---|
| Financial literacy | fl1 fl2 fl3 fl4 fl5 | 5 |
| Demographics | gender age income | 3 |
| Race/ethnicity | afram asian caucasian hispanic other | 5 |
| Education | lessThanHighSchool highSchool voc someCollege college graduate | 6 |
| Employment | fullTime partTime | 2 |
| Marital | married widowed divorced never_married | 4 |
| Urban/rural | rural urbanSuburban | 2 |
| Household size | hh1 hh2 hh3 hh4 | 4 |
| Financial | ownStocks | 1 |
| **Total** | | **32** |

This confirms `controls_count_max = 32` is correct in the surface.

---

## Budget and Sampling Assessment

- G1 budget: 35 specs — adequate. Curated control sets are correct approach given the paper's own revealed search.
- G2 budget: 45 specs — adequate. The observation-level data with multiple outcomes and sample splits justifies the larger budget.
- Seed: 171681 (G1), 171682 (G2) — reasonable, distinct.
- No combinatorial sampling needed; curated control sets are appropriate.

---

## Inference Plan Assessment

- Canonical: cluster-robust at `id` — correct and essential for G2 (multiple obs per subject).
- Variants (HC1, HC3): Appropriate comparisons. For G1 (individual-level), clustering at id is equivalent to HC (one cluster per observation), so HC1/HC3 variants are meaningful for comparison.

---

## RC Axes Assessment

### G1 RC axes: All confirmed appropriate

- Control axes (single, sets, LOO): Pre-treatment demographics, confirmed correct.
- Sample axes: Exp A only / Exp B only / pooled — appropriate given paper structure.
- Outcome axes (`score_indexing`, `fl_score_compound`, `fl_sum_compound`): All verified in code.
- Treatment arms (`rule72_only_vs_control`, `rhetoric_only_vs_control`): Correct; these decompose the full-vs-control effect.
- `rc/sample/attrition/include_multi_switchers`: Requires data re-management (see Issue 1).

### G2 RC axes: One variable name corrected

- `rc/outcome/diff_raw` → `rc/outcome/diff` (see Issue 3).
- `rc/outcome/negSqDiff`: Definition corrected (see Issue 2).
- `rc/form/individual_means`: Collapses to `dAbs` (individual-level mean absolute difference). Confirmed in code at lines 88-90. This changes the unit of observation and thus requires a separate note on interpretation.
- All other axes confirmed appropriate.

---

## What Is Excluded (Confirmed Correct)

The following are correctly excluded from the core universe:

- Quartile heterogeneity (Table 11): Exploration, not a main claim.
- `finCompCorrAltApprox`: Requires per-individual first-stage regression (constructed regressor problem). Correctly excluded.
- Individual test questions (Table D.3): Decompositions, not main claims.
- Self-report outcomes (Table D.4): Manipulation checks.
- Simple-frame discount rates (Table 9): Placebo-type test, not main competence claim.

---

## Final Assessment

**Status**: APPROVED TO RUN with the corrections noted above.

Blocking issues resolved: variable name `diff_raw` → `diff`; `sqDiff` formula corrected; `multi == 0` filter noted as globally absorbed.

Non-blocking notes: `meanVsimple` flagged as potential post-treatment control; `include_multi_switchers` RC requires data re-management; pooled demoControl spec is exploratory rather than core.
