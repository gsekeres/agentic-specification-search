# Specification Surface: 149481-V1

**Paper**: "Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence"
**Authors**: Anya Samek and Chuck Longfield
**Design**: Randomized Experiment (Field Experiment)
**Created**: 2026-02-24

---

## 1. Paper Overview

This paper reports results from three field experiments testing whether thank-you phone calls to existing charitable donors increase subsequent giving. The paper also collects expert forecasts of the treatment effects and compares them to observed outcomes.

- **Experiment 1**: Conducted at public television stations. Donors were randomly assigned to receive a thank-you call (treatment) or not (control), randomized within station x quarter strata. This is the largest experiment with the richest covariate set.
- **Experiment 2**: Conducted at a national non-profit. Individual-level randomization with no strata and no demographic covariates.
- **Experiment 3**: Conducted at public television stations, same context as Experiment 1 but with two treatment arms: original call script and a new call script, plus a control group.

The main finding is that thank-you calls do not significantly increase giving, contrary to expert predictions.

---

## 2. Baseline Groups

### G1: Experiment 1 -- Public Television Stations

**Claim object**:
- **Outcome concept**: Charitable donation behavior in the year following the thank-you call (whether donated, amount donated, number of gifts, conditional gift amount, retention rate)
- **Treatment concept**: Assignment to receive a thank-you phone call (treat=1 vs control=0)
- **Estimand concept**: Intent-to-Treat (ITT) average effect
- **Target population**: Existing donors at public television stations (excluding sustaining donors and big donors >= $10,000)

**Why this is a baseline group**: Experiment 1 is the largest experiment (multiple TV stations over multiple quarters), with the richest covariate set (demographics, baseline giving history), and station x quarter strata FE. It is the primary experiment discussed in the paper.

**Baseline specifications**:
- Table 2, Experiment 1: Five outcome variables (renewing, payment_amount3, var13, gift_cond, retention) tested via nonparametric methods (Wilcoxon rank-sum for continuous, chi-square for binary). No controls.
- Table A1, Experiment 1 columns: Two OLS regressions (donated, gift_cond) with strata FE and 9 controls (2 baseline giving + 7 demographic indicators).

### G2: Experiment 2 -- National Non-Profit

**Claim object**:
- **Outcome concept**: Charitable donation behavior at a national non-profit
- **Treatment concept**: Same thank-you phone call treatment
- **Estimand concept**: ITT
- **Target population**: Existing donors at a national non-profit (distinct population from Exp 1)

**Why this is a separate baseline group**: Experiment 2 involves a different organization, a different donor population, and has no demographic covariates. It provides an independent test of the same treatment concept.

**Baseline specifications**:
- Table 2, Experiment 2: Same five outcomes, nonparametric tests.
- Table A1, Experiment 2 columns: OLS with only baseline giving controls (payment_amount2, var12). No FE, no demographics.

---

## 3. What Is Excluded from Core (and Why)

| Excluded element | Reason | Tables/Figures |
|---|---|---|
| Experiment 3 (new script) | Different treatment concept: tests new vs. original call script, not simply "call vs. no call" | Table 2 Panel C, Table A1 Cols 3/6 |
| LATE estimates | Changes estimand from ITT to LATE (effect of actually being reached). IV with treat as instrument for reached. | Table A2 |
| Future years outcomes | Changes outcome timing concept (2-5 years after randomization vs. 1 year) | Table A3 |
| Interaction effects | Heterogeneity analysis (treatment x pledge drive, x demographics). Not estimand-preserving. | Table A6 |
| Expert forecasts analysis | Concerns forecast accuracy, not the experimental treatment effect itself | Figures 2-3, Table A5 |
| Unconditional gift OLS (Table A4) | Implicitly covered by RC variant (OLS with controls on payment_amount3) | Table A4 |

---

## 4. Core Universe

### G1 Design Variants

| spec_id | Description |
|---|---|
| `design/randomized_experiment/estimator/diff_in_means` | Pure difference-in-means (OLS without any controls or FE) |
| `design/randomized_experiment/estimator/strata_fe` | OLS with strata FE only (no additional controls) |
| `design/randomized_experiment/estimator/with_covariates` | OLS with strata FE and full control set (replicates Table A1) |

### G1 Robustness Checks

**Controls axis** (for OLS specs only):
- `rc/controls/loo/drop_*`: Leave-one-out from the 9-control set (9 specs)
- `rc/controls/sets/none`: No controls, no FE (pure diff-in-means)
- `rc/controls/sets/baseline_giving_only`: Only payment_amount2 and var12, with strata FE
- `rc/controls/sets/demographics_only`: Only 7 demographic controls, with strata FE
- `rc/controls/sets/full_with_fe`: All 9 controls with strata FE (matches Table A1)

**Sample axis**:
- `rc/sample/outliers/trim_y_1_99`: Trim top 1% of continuous monetary outcomes
- `rc/sample/outliers/trim_y_5_95`: Trim top 5% of continuous monetary outcomes
- `rc/sample/quality/drop_big_donors_5k`: Stricter big-donor threshold ($5,000 vs paper's $10,000)
- `rc/sample/quality/include_sustaining_donors`: Include sustaining donors (excluded in data prep)
- `rc/sample/quality/drop_stations_with_few_obs`: Drop thin strata

**Functional form axis** (continuous monetary outcomes only):
- `rc/form/outcome/asinh`: Inverse hyperbolic sine transform
- `rc/form/outcome/log1p`: Log(1+y) transform

**Fixed effects axis** (OLS specs only):
- `rc/fe/strata/station_only`: Station FE instead of station x quarter FE
- `rc/fe/strata/none`: No fixed effects

### G2 Robustness Checks

Same structure as G1 but with a smaller control space:
- Only 2 controls available (payment_amount2, var12)
- No FE variation (no strata in Experiment 2)
- Same sample and functional form RC axes

---

## 5. Inference Plan

### Canonical inference

For difference-in-means specifications: Wilcoxon rank-sum test (continuous outcomes) or chi-square test (binary outcomes), matching the paper's Table 2.

For OLS specifications: default standard errors from Stata `xtreg, fe` (conventional panel-robust SE for Exp 1) or `reg` (classical OLS SE for Exp 2).

### Inference variants

| spec_id | Description |
|---|---|
| `infer/se/hc/hc1` | HC1 heteroskedasticity-robust SE (OLS specs) |
| `infer/se/cluster/station` | Cluster at station level (G1 only; stress test for within-station correlation) |
| `infer/test/ttest` | Two-sample t-test as alternative to rank-sum (diff-in-means specs) |

---

## 6. Constraints

### G1 Constraints
- **Controls count envelope**: 0 (Table 2 diff-in-means) to 9 (Table A1 OLS with all controls)
- **Linked adjustment**: No (single-equation OLS, no bundled estimator)
- **Functional form**: asinh and log1p apply only to continuous monetary outcomes (payment_amount3, gift_cond, retention), NOT to binary outcomes (renewing, donated)
- **Outlier trimming**: Applies only to continuous outcomes, not binary

### G2 Constraints
- **Controls count envelope**: 0 to 2
- **No FE variation**: Experiment 2 has no strata
- **Same outcome-type restrictions** for functional form and outlier trimming

---

## 7. Budgets and Sampling

### G1 Budget: 80 specs (max)

Enumeration plan:
- 7 baseline specs (5 diff-in-means + 2 OLS)
- 3 design variants x 3 focal outcomes (renewing, payment_amount3, donated) = 9
- 13 control RC x 2 focal OLS outcomes = 26 (but many overlap with design/baselines)
- 5 sample RC x 3 focal outcomes = 15
- 2 form RC x 2 continuous focal outcomes = 4
- 2 FE RC x 2 focal OLS outcomes = 4

Full enumeration is feasible. Seed: 149481.

### G2 Budget: 40 specs (max)

Smaller RC space due to fewer controls and no FE variation. Full enumeration is feasible. Seed: 149482.

---

## 8. Diagnostics Plan

| Diagnostic | Scope | Notes |
|---|---|---|
| `diag/randomized_experiment/balance/covariates` | baseline_group | Balance test on pre-treatment covariates (replicates Table 1) |
| `diag/randomized_experiment/attrition/attrition_diff` | baseline_group | Differential attrition test (G1 only) |

---

## 9. Key Data Notes

- **Data files**: `exp1_analysis_data.dta`, `exp2_analysis_data.dta`, `exp3_analysis_data.dta` (created by data prep scripts)
- **Strata variable (Exp 1)**: `ii` = group(station_id, edate_dummy), used for xtreg FE
- **Treatment variable (Exp 1, 2)**: `treat` (0=control, 1=treatment)
- **Treatment variable (Exp 3)**: `exp3_treat` (0=control, 1=original script, 2=new script)
- **Key outcome variables**:
  - `renewing`: Binary, whether donor gave in year after (derived from var13 != 0)
  - `payment_amount3`: Total gift amount in year after (includes 0 for non-donors)
  - `var13`: Number of gifts in year after
  - `gift_cond`: Gift amount conditional on donating (payment_amount3 if > 0)
  - `retention`: payment_amount3 / payment_amount2 (ratio)
  - `donated`: Binary, 1 if var13 > 0 (same as renewing but constructed differently in the code)
- **Baseline covariates**: `payment_amount2` (baseline gift amount), `var12` (baseline number of gifts)
- **Demographic covariates (Exp 1 only)**: `female`, `age_display2`, `age_display3`, `inc_display1`-`inc_display4`, `lor_display2`
- **Big donors**: Already excluded (>= $10,000 in any single transaction)
- **Sustaining donors**: Already excluded in data preparation
- **Missing data indicators**: `female_missing`, `age_income_missing` available but not used in main regressions

---

## 10. Statistical Methods in the Paper

The paper's Table 2 is notably non-regression-based: it reports group means and nonparametric p-values (Wilcoxon rank-sum for continuous, chi-square for binary). The regression-based results appear in the appendix (Table A1 OLS, Table A2 IV). This means the specification surface must accommodate both nonparametric and parametric approaches.

For the core specification surface, we implement both:
1. **Difference-in-means** (corresponding to Table 2): simple treatment-control comparison, with p-values from nonparametric tests or t-tests
2. **OLS with controls** (corresponding to Table A1): regression-based ITT with strata FE and covariates

Both approaches estimate the same ITT estimand under random assignment, making them estimand-preserving alternatives.
