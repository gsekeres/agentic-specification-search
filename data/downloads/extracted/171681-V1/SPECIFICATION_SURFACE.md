# Specification Surface: 171681-V1

**Paper**: Ambuehl, Bernheim & Lusardi, "Evaluating Deliberative Competence: A Simple Method with an Application to Financial Choice"

**Design**: Randomized Experiment (online, MTurk)

**Created**: 2026-02-24

---

## Paper Summary

This paper studies whether a financial literacy intervention (teaching the Rule of 72 heuristic plus framing rhetoric) improves participants' ability to evaluate compound-interest investment products. Two online experiments are conducted:

- **Experiment A** (8 batches, `sample == "old"`): 4 arms -- Control, Full Treatment, Rule of 72 only (substance), Rhetoric only (framing)
- **Experiment B** (2 batches, `sample == "new"`): 2 arms -- Control, Full Treatment

Participants face multiple price lists (MPLs) eliciting willingness-to-pay for delayed payments, presented in both a "simple" frame (plain future amount) and a "complex" frame (interest-bearing investment). The key outcome measures are:

1. **Compounding test scores** (`score_compounding`): 5-question quiz on compounding, measured once per subject
2. **Financial competence** (`negAbsDiff` = `-abs(discount_framed - discount_unframed)`): how close the valuation of the complex frame is to the simple frame, measured at the observation level (subject x price-list)

Treatment assignment is via the `fined` variable: 0=Control, 1=Full, 2=Rule72-only, 3=Rhetoric-only. Data management creates dummy variables: `Control`, `Full`, `Rule72`, `Rhetoric` (Exp A) and `contNew`, `fullNew` (Exp B).

---

## Baseline Groups

### G1: Compounding Knowledge (Test Scores)

**Claim**: The full treatment significantly improves compounding knowledge.

| Field | Value |
|---|---|
| Outcome concept | Compounding test score (0-5 scale) |
| Treatment concept | Full financial literacy intervention (Rule of 72 + rhetoric) |
| Estimand | ITT: Full vs Control |
| Target population | Online participants, Experiment A |
| Baseline spec | `reg score_compounding Full if tag == 1 & (Full == 1 | Control == 1) & sample == "old", cl(id)` |
| Table | Table 4, Column 1 |
| Unit of observation | Individual (one obs per subject, `tag == 1`) |
| N controls (baseline) | 0 |

**Additional baseline**: Table 4, Column 2 runs the same regression for Experiment B (`sample == "new"`, `fullNew` treatment dummy).

### G2: Financial Competence (Valuation Errors)

**Claim**: The full treatment reduces the gap between complex-frame and simple-frame valuations, improving financial competence.

| Field | Value |
|---|---|
| Outcome concept | Negative absolute valuation difference: `-abs(discount_framed - discount_unframed)` |
| Treatment concept | Full financial literacy intervention (Rule of 72 + rhetoric) |
| Estimand | ITT: Full vs Control (tested via `test Full == Control` after no-constant regression) |
| Target population | Online participants, Experiment A |
| Baseline spec | `reg negAbsDiff Control Full Rule72 Rhetoric if sample == "old", nocons cl(id)` |
| Table | Table 7, Column 1 |
| Unit of observation | Observation-level (subject x price-list pair, multiple per subject) |
| N controls (baseline) | 0 |

**Additional baseline**: Table 7, Column 2 runs for Experiment B (`sample == "new"`, `contNew` and `fullNew` dummies, no-constant).

**Note on model form**: The paper's main financial competence regressions use a no-constant (`nocons`) specification with separate treatment-arm dummies (Control, Full, Rule72, Rhetoric), then test equality of coefficients (`test Full == Control`). This is algebraically equivalent to a difference-in-means with intercept, but the `nocons` form directly reports group means. We preserve this convention for baselines and use both forms in the surface.

---

## Core Universe Design

### G1 Specifications (~35 total)

| Category | Spec IDs | Count | Rationale |
|---|---|---|---|
| **Baseline** | `baseline` (Table 4 Col 1, Exp A) | 1 | Paper's canonical knowledge result |
| **Additional baseline** | `baseline__table4_col2_expB` | 1 | Replication in Experiment B |
| **Design estimator** | `design/randomized_experiment/estimator/diff_in_means` | 1 | Minimal difference-in-means |
| **Design estimator** | `design/randomized_experiment/estimator/with_covariates` | 1 | With paper's demographic controls |
| **RC: Add single control** | `rc/controls/single/add_{gender,age,income,fl_high,ownStocks}` | 5 | Key pre-treatment covariates |
| **RC: Control sets** | `rc/controls/sets/demographics_{minimal,extended,full}` | 3 | Curated demographic bundles |
| **RC: LOO from full** | `rc/controls/loo/drop_{gender,age,income}` | 3 | Leave-one-out from full control set |
| **RC: Sample - experiment** | `rc/sample/experiment/{expA_only,expB_only,pooled_AB}` | 3 | Experiment subsamples and pooling |
| **RC: Sample - inclusion** | `rc/sample/attrition/include_multi_switchers` | 1 | Relax `multi == 0` exclusion |
| **RC: Sample - outliers** | `rc/sample/outliers/trim_y_1_99` | 1 | Trim extreme scores (edge case) |
| **RC: Alternative outcomes** | `rc/outcome/{score_indexing, score_compounding_plus_indexing, fl_score_compound, fl_sum_compound}` | 4 | Alternative knowledge measures from the paper |
| **RC: Treatment arm** | `rc/treatment/{rule72_only_vs_control, rhetoric_only_vs_control, all_arms_nocons}` | 3 | Alternative treatment contrasts (Exp A arms) |
| **RC: Joint** | `rc/joint/{expA_with_controls, expB_with_controls, pooled_with_controls}` | 3 | Experiment x controls combinations |
| **Total** | | **~31** | |

### G2 Specifications (~45 total)

| Category | Spec IDs | Count | Rationale |
|---|---|---|---|
| **Baseline** | `baseline` (Table 7 Col 1, Exp A) | 1 | Paper's canonical competence result |
| **Additional baseline** | `baseline__table7_col2_expB` | 1 | Replication in Experiment B |
| **Design estimator** | `design/randomized_experiment/estimator/diff_in_means` | 1 | Standard diff-in-means with intercept |
| **Design estimator** | `design/randomized_experiment/estimator/with_covariates` | 1 | With demographic controls |
| **RC: Add single control** | `rc/controls/single/add_{gender,age,income,fl_high,ownStocks,meanVsimple}` | 6 | Key pre-treatment covariates + simple-frame mean |
| **RC: Control sets** | `rc/controls/sets/demographics_{minimal,extended,full}` | 3 | Curated demographic bundles |
| **RC: LOO from full** | `rc/controls/loo/drop_{gender,age,income}_from_full` | 3 | Leave-one-out from full set |
| **RC: Sample - experiment** | `rc/sample/experiment/{expA_only,expB_only,pooled_AB}` | 3 | Experiment subsamples and pooling |
| **RC: Sample - delay** | `rc/sample/delay/{delay_72_only,delay_36_only}` | 2 | Paper splits by delay horizon (Table 7 Cols 3-6) |
| **RC: Sample - inclusion** | `rc/sample/attrition/include_multi_switchers` | 1 | Relax multi-switcher exclusion |
| **RC: Sample - outliers** | `rc/sample/outliers/trim_absDiff_{1_99,5_95}` | 2 | Trim extreme valuation errors |
| **RC: Alternative outcomes** | `rc/outcome/{negSqDiff, diff_raw, discount_framed, finCompCorr, finCompCorrSq}` | 5 | Alternative competence measures from Tables 5, D.5, 10, D.5-Sq |
| **RC: Treatment arm** | `rc/treatment/{rule72_only_vs_control, rhetoric_only_vs_control, full_vs_control_with_intercept, all_arms_nocons}` | 4 | Different treatment contrasts |
| **RC: Joint (experiment x delay)** | `rc/joint/{expA_delay72, expA_delay36, expB_delay72, expB_delay36}` | 4 | Paper's own subgroup columns |
| **RC: Joint (experiment x controls)** | `rc/joint/{pooled_with_controls, expA_with_controls, expB_with_controls}` | 3 | Controls within experiment subsamples |
| **RC: Aggregation** | `rc/form/individual_means` | 1 | Collapse to individual-level means (dAbs variable) |
| **Total** | | **~41** | |

**Combined total across both groups: ~72 specifications** (exceeds 50 target).

---

## Variable Definitions

### Outcome Variables

| Variable | Definition | Used in |
|---|---|---|
| `score_compounding` | Correct answers to 5 compounding questions (t1-t5) | G1 baseline |
| `score_indexing` | Total score minus compounding score | G1 alternative |
| `fl_score_compound` | fl1 * fl2 * fl3 (all 3 financial literacy compounding Qs correct) | G1 alternative |
| `fl_sum_compound` | fl1 + fl2 + fl3 (sum of financial literacy compounding Qs) | G1 alternative |
| `negAbsDiff` | `-abs(discount_framed - discount_unframed)` = `-absDiff` | G2 baseline |
| `negSqDiff` | `-sqDiff` = `-(discount_framed^2 - discount_unframed^2)/100` | G2 alternative (Table D.5) |
| `diff` | `discount_framed - discount_unframed` (signed difference) | G2 alternative |
| `discount_framed` | `v_framed / amount_precise * 100` (WTP as % of precise FV) | G2 alternative (Table 5) |
| `finCompCorr` | `-correctionFactor * absDiff * 100` (corrected for simple-frame changes) | G2 alternative (Table 10) |
| `finCompCorrSq` | `-correctionFactorSq * sqDiff` (squared, corrected) | G2 alternative (Table D.5) |

### Treatment Variables

| Variable | Definition | Experiment |
|---|---|---|
| `Full` | `fined == 1` (Full Treatment, Exp A) | A |
| `Control` | `fined == 0` (Control, Exp A) | A |
| `Rule72` | `fined == 2` (Rule of 72 substance only, Exp A) | A |
| `Rhetoric` | `fined == 3` (Rhetoric framing only, Exp A) | A |
| `fullNew` | `fined == 1` (Full Treatment, Exp B) | B |
| `contNew` | `fined == 0` (Control, Exp B) | B |

### Demographic Controls (Pre-Treatment)

Available controls (all binary except `age` and `income`):

- **Financial literacy**: `fl1`, `fl2`, `fl3`, `fl4`, `fl5`, `fl_high` (= `fl_score_compound`)
- **Demographics**: `gender`, `age`, `income`
- **Race/ethnicity**: `afram`, `asian`, `caucasian`, `hispanic`, `other`
- **Education**: `lessThanHighSchool`, `highSchool`, `voc`, `someCollege`, `college`, `graduate`
- **Employment**: `fullTime`, `partTime`
- **Marital status**: `married`, `widowed`, `divorced`, `never_married`
- **Geography**: `rural`, `urbanSuburban`
- **Household**: `hh1`, `hh2`, `hh3`, `hh4`
- **Financial**: `ownStocks`

**Curated control sets**:
- **Minimal** (~5): `gender`, `age`, `income`, `caucasian`, `college`
- **Extended** (~12): minimal + `fl1`, `fl2`, `fl3`, `married`, `fullTime`, `ownStocks`, `urbanSuburban`
- **Full** (~32): all demographic controls (matching Table D.2)

### Sample Filters

| Filter | Definition |
|---|---|
| `sample == "old"` | Experiment A |
| `sample == "new"` | Experiment B |
| `multi == 0` | Single-switchers only (no multiple switching in MPL) |
| `tag == 1` | One observation per subject (for individual-level outcomes) |
| `delay == 72` | 72-month delay horizon only |
| `delay == 36` | 36-month delay horizon only |

---

## Inference Plan

**Canonical**: Cluster-robust standard errors at the individual (`id`) level, matching the paper's `cl(id)` throughout.

**Rationale**: Clustering at `id` is essential for G2 (observation-level data with multiple obs per subject). For G1 (individual-level, `tag == 1`), clustering at `id` is equivalent to robust SE, but we maintain consistency.

**Variants**:
- HC1 (heteroskedasticity-robust, no clustering): for comparison
- HC3 (jackknife-based): conservative small-sample correction

---

## Constraints and Guardrails

1. **Control-count envelope**: 0 to 32 controls. The paper reports both no-controls (baseline) and full-controls (Table D.2). No intermediate specifications appear in the paper, but the curated sets (minimal, extended, full) provide structured progression.

2. **No linked adjustment**: All specifications use single-equation OLS; no bundled estimators.

3. **Multi-switcher exclusion**: The paper's main analyses restrict to `multi == 0` (single-switchers only). One RC variant relaxes this.

4. **Experiment separation**: Exp A and Exp B are analyzed separately in the paper, with pooled analysis as an additional RC.

5. **No-constant convention**: The paper's financial competence regressions (G2) use `nocons` with all arm dummies. We preserve this for baselines and include an intercept-form variant (`full_vs_control_with_intercept`) as an RC.

---

## What Is Excluded (and Why)

- **Explore/Sens/Post/Diag axes**: Not part of the core universe per protocol. Balance checks (Table D.1) and attrition are listed in diagnostics_plan.
- **Quartile heterogeneity analysis** (Table 11): Interaction of treatment with quartiles of simple-frame valuation. This is a heterogeneity/exploration analysis, not a main claim.
- **Alternative correction methods** (Tables 10, D.5, D.7): The correction-for-simple-frame-changes outcomes (`finCompCorr`, `finCompCorrSq`, `finCompCorrAltApprox`) are included as outcome variants in G2, but the individual-level rho estimation (`finCompCorrAltApprox`) is excluded because it requires a per-individual first-stage regression that creates a constructed regressor problem.
- **Individual test question regressions** (Table D.3): These are 5 separate regressions for t1-t5 individually. Not a main claim; they decompose the total score.
- **Self-report outcomes** (Table D.4): `calculate`, `r72report`, `r72invReport`, `help_test`. These are secondary manipulation-check outcomes, not main claims.
- **Simple-frame discount rates** (Table 9): `discount_unframed` as outcome. This tests whether treatment changes simple-frame valuations (a placebo-like check), not the main competence claim.

---

## Budget and Sampling

- **G1**: ~35 specs (full enumeration, no sampling needed)
- **G2**: ~45 specs (full enumeration, no sampling needed)
- **Total**: ~72-80 specs across both groups
- **Seed**: 171681 (G1), 171682 (G2)
- **Controls**: Curated sets rather than combinatorial sampling, because the paper's own revealed search space uses only 0-control and full-control specifications.
