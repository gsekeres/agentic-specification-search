# Specification Surface Review: 149481-V1

**Paper**: "Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence"
**Authors**: Anya Samek and Chuck Longfield
**Reviewer date**: 2026-02-24

---

## Summary of Baseline Groups

- **G1**: Thank-you call -> Donation behavior, Experiment 1 (Public Television Stations)
  - Well-defined claim object: ITT of assignment to receive a thank-you call on donation outcomes
  - Randomization at individual level within station x quarter strata
  - 7 baseline specs: 5 from Table 2 (nonparametric diff-in-means) + 2 from Table A1 (OLS with controls/FE)
  - Verified against code: Table2_treatment_effects.do confirms ranksum for continuous, chi2 for binary; TableA1 confirms xtreg with `ii` as panel variable (strata FE) and 9 controls

- **G2**: Thank-you call -> Donation behavior, Experiment 2 (National Non-Profit)
  - Distinct population (national non-profit donors vs TV station donors) -- correctly a separate baseline group
  - No strata, no demographic covariates -- only 2 baseline giving controls available
  - 7 baseline specs: 5 from Table 2 + 2 from Table A1 (OLS without FE)
  - Verified against code: Exp 2 uses `reg` (not `xtreg`) with no FE, and `$controls` is set to empty

---

## Changes Made

### 1. SE type correction (G1 design_audit and inference_plan)
- **Issue**: The original surface described the Table A1 OLS standard errors as "conventional panel-robust SE." This is incorrect. The Stata code uses `xtreg donated treat ... , fe` without any `,robust` or `,cluster()` option. Default `xtreg, fe` produces classical (non-robust) standard errors.
- **Change**: Updated `se_type` in G1's `design_audit` and `canonical` inference notes to read "classical OLS SE (xtreg, fe without robust/cluster option)." Same correction applied to G2's canonical inference notes (which uses `reg` without robust).

### 2. Sample filter made explicit (G1 design_audit)
- **Issue**: The sample filter description omitted key data construction drops.
- **Change**: Updated G1's `sample_filter` to explicitly list: `payment_amount2==0` dropped (step2 line 14), sustaining donors dropped (step1 line 369), big donors >=10k dropped (step1 line 347), exec_date 1610/1701 excluded, stations 24/55/64/61 dropped.

### 3. G2 control set naming consistency
- **Issue**: G2 used `rc/controls/sets/baseline_giving` while G1 used `rc/controls/sets/baseline_giving_only` for the analogous set.
- **Change**: Renamed G2's set to `rc/controls/sets/baseline_giving_only` for consistency.

### 4. Functional form interpretation notes added
- **Issue**: The `rc/form/outcome/asinh` and `rc/form/outcome/log1p` RC variants change the coefficient interpretation from level effect to approximately semi-elasticity/percent change. The functional_form module requires explicit interpretation notes.
- **Change**: Added interpretation notes to both G1 and G2 rc_notes for asinh and log1p explaining the coefficient interpretation and why these remain RC (binary treatment makes the coefficient still directly interpretable as a transformed-unit treatment effect).

### 5. Sustaining donors RC feasibility note
- **Issue**: The `rc/sample/quality/include_sustaining_donors` RC requires re-running the data construction pipeline from step1. The sustaining donor drop happens early in data preparation (step1 line 369), before the collapse and reshape operations.
- **Change**: Added a note to the RC description clarifying that this spec requires data reconstruction from step1.

### 6. G2 sample filter clarified
- **Issue**: The G2 sample filter description was vague.
- **Change**: Updated to note the data source (gift.dta) and the key filter (existing donors dropped via `payment_amount1!=.`).

---

## Key Constraints and Linkage Rules

- **No bundled estimator**: All specifications are single-equation (difference-in-means or OLS). No linked adjustment is needed.
- **Outcome-type restrictions**: Functional form transforms (asinh, log1p) and outlier trimming apply only to continuous monetary outcomes (payment_amount3, gift_cond, retention), NOT to binary outcomes (renewing, donated). This is correctly documented.
- **Conditional sample**: The `gift_cond` outcome restricts to `renewing==1` (donors who gave). This is a conditional sample, not a full-population estimate. This is correctly documented in the baseline spec notes.
- **FE variation (G1 only)**: Station-only FE and no-FE variants are correctly restricted to OLS specs. G2 has no FE variation (no strata).
- **LOO controls (OLS specs only)**: Leave-one-out from the full control set is correctly restricted to the OLS baseline specs (Table A1 columns), not the diff-in-means specs (Table 2).

---

## Budget/Sampling Assessment

### G1: 80 specs (max)
- 7 baseline specs
- Design variants: 3 (diff_in_means, strata_fe, with_covariates) x focal outcomes -- many overlap with baselines
- RC axes: 9 LOO + 4 control sets + 5 sample + 2 form + 2 FE = 22 RC spec_ids, but each applies only to compatible outcomes/estimators
- Full enumeration is feasible with the small control pool (9 controls)
- Budget of 80 is adequate with headroom
- Seed: 149481

### G2: 40 specs (max)
- 7 baseline specs
- Much smaller RC space: 2 LOO + 2 control sets + 2 sample + 2 form = 8 RC spec_ids
- Budget of 40 is generous for the small RC space
- Seed: 149482

---

## What's Missing (minor, non-blocking)

1. **Missing data indicators as controls**: G1 has `female_missing` and `age_income_missing` available but not used in the paper. Adding these as an additional control set RC (`rc/controls/sets/with_missing_indicators`) would test sensitivity to missing data handling. Not added because the paper never uses them and they are a minor axis.

2. **Table A4 as explicit OLS baseline for payment_amount3**: Table A4 runs OLS on unconditional gift amount with controls for Exp 1. The surface notes this is implicitly covered via the design variant `with_covariates` applied to the `payment_amount3` outcome. This is correct -- no change needed.

3. **Randomization inference / permutation test**: The spec tree lists `infer/resampling/randomization_inference` as a potential inference variant for RCTs. This could be added as an inference variant for the diff-in-means specs. Not added because the paper uses standard nonparametric tests and this would be an unusual addition.

4. **Noncompliance diagnostic**: The paper's Table A2 IV/LATE results suggest meaningful noncompliance (not all treatment-assigned donors were reached). A `diag/randomized_experiment/noncompliance/first_stage` diagnostic could be added. However, since the baseline claim is ITT (not LATE), this is informational only. The surface already excludes LATE from the core universe.

5. **Data construction sensitivity (payment_amount2==0 drop)**: Step 2 drops observations where `payment_amount2==0` for Exp 1. This could be an RC variant (include these obs). Not added because it changes the population definition (these are donors who did not give in the baseline year, which is outside the paper's intended population).

---

## Verification Against Code

### Table 2 (Table2_treatment_effects.do)
- Confirmed: `renewing` uses `tabulate renewing treat, chi2` (chi-square test)
- Confirmed: `payment_amount3`, `var13`, `retention` use `ranksum var, by(treat) porder` (Wilcoxon rank-sum)
- Confirmed: Conditional outcomes computed `if renewing==1`
- Confirmed: Variables match surface specification exactly

### Table A1 (TableA1_factors_associated_with_giving.do)
- Confirmed: Exp 1 uses `xtreg donated treat payment_amount2 var12 $controls, fe` where `$controls = female age_display2 age_display3 inc_display3 inc_display1 inc_display2 lor_display2` (7 demographic vars)
- Confirmed: Exp 2 uses `reg donated treat payment_amount2 var12 $controls` where `$controls` is empty
- Confirmed: No robust/cluster SE option used in either case
- Confirmed: `xtset ii` sets strata FE for Exp 1; no panel set for Exp 2

### Data construction (step1, step2)
- Confirmed: Big donors (>=10k any transaction) dropped in step1
- Confirmed: Sustaining donors dropped in step1
- Confirmed: `payment_amount2==0` dropped in step2
- Confirmed: Stations 24, 55, 64, 61 dropped in step2
- Confirmed: exec_date 1610, 1701 excluded in step2
- Confirmed: `ii = group(station_id, edate_dummy)` is the strata FE variable
- Confirmed: `donated` and `renewing` constructed identically from `var13`

### Exclusions verified
- Experiment 3: Correctly excluded (tests new_script vs original_script, different treatment concept)
- LATE (Table A2): Correctly excluded (different estimand)
- Future years (Table A3): Correctly excluded (different outcome timing)
- Interactions (Table A6): Correctly excluded (heterogeneity analysis)
- Expert forecasts (Figure 2-3, Table A5): Correctly excluded (different research question)

---

## Final Assessment

**APPROVED TO RUN.** The specification surface is conceptually coherent, faithful to the paper's revealed analysis space, and verified against the replication code. The changes made are minor corrections (SE type description, naming consistency, interpretation notes, feasibility notes) that do not alter the structure of the surface. The two baseline groups (Experiment 1 and Experiment 2) are well-defined with distinct populations. The exclusion of Experiment 3, LATE estimates, future years, and interactions is well-justified. Budgets are feasible and full enumeration is appropriate given the small control pool.
