# Specification Surface: 140161-V1

**Paper**: "Checking and Sharing Alt-Facts" by Emeric Henry, Ekaterina Zhuravskaya, and Sergei Guriev (AEJ: Policy)

**Design**: Randomized Experiment (online survey experiment)

**Created**: 2026-02-24

---

## 1. Paper Overview

This paper studies how fact-checking interventions affect the sharing of fake news ("alt-facts") on Facebook. The experiment was conducted during the 2019 French European Parliament election period using an online survey company. Approximately 2,537 respondents were randomly assigned to one of three treatment arms:

1. **Alt-Facts** (survey==1): Respondents see only the alt-facts article
2. **Imposed Fact-Check** (survey==2): Respondents see the alt-facts article and are then forced to view a fact-check
3. **Voluntary Fact-Check** (survey==3): Respondents see the alt-facts article and are offered the option to view a fact-check

The main outcomes are measured at three levels of commitment:
- **Click 1** (sharing intent): Whether the respondent says they want to share on Facebook
- **Click 2** (sharing action): Whether the respondent actually navigates to the sharing page (tracked via Google Analytics)
- **Click 3** (reconfirm sharing): Whether the respondent completes the sharing action on the page

Randomization was stratified by gender (`male`) and education level (`low_educ`, `mid_educ`).

---

## 2. Baseline Groups

### G1: Effect of Fact-Checking on Alt-Facts Sharing (Table 2, Panel A)

**Claim object**: Fact-checking interventions (both imposed and voluntary) reduce the desire/action to share alt-facts on Facebook.

**Baseline specification** (Table 2, Panel A, Column 2):
- Outcome: `want_share_fb` (binary: intent to share alt-facts on Facebook)
- Treatment: `i.survey` (dummies for survey==2 and survey==3 vs. survey==1)
- Controls: `male`, `low_educ`, `mid_educ` (stratification controls)
- Sample: `survey < 4` (main experiment only, N~2,537)
- Inference: HC1 robust standard errors

**Why Column 2**: The paper's preferred specification includes stratification controls (as is standard for stratified RCTs). Column 1 (no controls) and Columns 3-4 (progressively more controls) are robustness checks shown in Table A3.

**Additional baseline**: `share_click2` (sharing action, 2 clicks) -- this is the behavioral measure tracked via Google Analytics, reported in Columns 5-8 of Table 2 Panel A. It is equally prominent as the intent measure.

### G2: Effect of Voluntary vs. Imposed Fact-Checking on Fact-Check Sharing (Table 2, Panel B)

**Claim object**: When the fact-check is voluntary rather than imposed, respondents are less willing to share the fact-check itself on Facebook.

**Baseline specification** (Table 2, Panel B, Column 2):
- Outcome: `want_share_facts` (binary: intent to share fact-check on Facebook)
- Treatment: `i.survey` with sample restricted to surveys 2 and 3 (coefficient on 3.survey is the effect of voluntary vs. imposed)
- Controls: `male`, `low_educ`, `mid_educ`
- Sample: `survey > 1 & survey < 4` (fact-check arms only)
- Inference: HC1 robust standard errors

**Additional baseline**: `share_facts_click2` (action of sharing fact-check, 2 clicks).

---

## 3. Why Two Baseline Groups

The paper makes two distinct headline claims:

1. **G1**: Fact-checking reduces sharing of alt-facts (comparing treatment arms 2 and 3 against the alt-facts-only arm 1)
2. **G2**: The mode of fact-checking (voluntary vs. imposed) affects sharing of the fact-check itself (comparing arm 3 against arm 2)

These use different samples (G1: all three arms; G2: only the two fact-check arms) and different outcomes (G1: alt-facts sharing; G2: fact-check sharing). They are conceptually distinct claims.

Table 3 (within-respondent FB vs. other sharing comparison using panel FE) is a secondary analysis and is not included as a baseline group.

---

## 4. Control Variable Pool

The paper defines control sets in explicit Stata globals:

| Set | Variables | Count |
|-----|-----------|-------|
| `strata` | male | 1 |
| `strata1` | low_educ, mid_educ | 2 |
| `socio` | age, age_sqrd, income, married, single, village, town, children, catholic, muslim, no_religion, educ, religious | 13 |
| `fb` | use_FB, often_share_fb, log_nb_friends_fb | 3 |
| `vote` | second_mlp, negative_image_UE | 2 |
| `behavioral` | altruism, reciprocity, image | 3 |
| `reported` | share_interest, share_influence, share_image, share_reciprocity | 4 |

The paper's revealed control progressions (Table 2/A3/A4 columns):
1. No controls (0 controls)
2. Strata only: male + low_educ + mid_educ (3 controls)
3. Strata + socio + vote + fb + i.educ (~21 controls + education dummies)
4. Full: strata + socio + vote + fb + behavioral + reported + all pre_treatment_vars (~30+ controls)

---

## 5. Core Universe

### G1 Specifications (~55 total)

**Baseline + additional baselines** (2):
- `baseline` (want_share_fb with strata controls)
- `baseline__share_action` (share_click2 with strata controls)

**Design variants** (3):
- `design/randomized_experiment/estimator/diff_in_means`: No controls at all (pure randomization)
- `design/randomized_experiment/estimator/with_covariates`: Add pre-treatment covariates for precision
- `design/randomized_experiment/estimator/strata_fe`: Include stratification vars as fixed effects rather than linear controls

**RC: Control progressions** (6):
- Strata only, strata+socio, strata+socio+vote, strata+socio+vote+fb, +behavioral, +reported
- These mirror the paper's own progressive control additions

**RC: Control sets** (4):
- None, strata_only, strata+socio+vote+fb, full

**RC: Leave-one-out from strata+socio+vote+fb set** (22):
- Drop each of: male, low_educ, mid_educ, age (drops age+age_sqrd), income, married, single, village, town, children, catholic, muslim, no_religion, religious, use_FB, often_share_fb, log_nb_friends_fb, second_mlp, negative_image_UE, altruism, reciprocity, image

**RC: Random control subsets** (10):
- 10 random draws from the full control pool, stratified by subset size

**RC: Sample** (2):
- Trim on survey duration threshold (300s, 200s vs. baseline 250s)

**RC: Treatment coding** (4):
- Pairwise: imposed vs. control only (survey 1 vs 2)
- Pairwise: voluntary vs. control only (survey 1 vs 3)
- Pairwise: imposed vs. voluntary (survey 2 vs 3)
- Binary: any fact-check (survey 2 or 3) vs. control (survey 1)

**RC: Outcome variants** (2):
- share_click2 (action of sharing, 2 clicks) as outcome with strata controls
- share_click3 (reconfirm sharing, 3 clicks) as outcome with strata controls

### G2 Specifications (~27 total)

**Baseline + additional baselines** (2):
- `baseline` (want_share_facts with strata controls, sample: survey>1 & survey<4)
- `baseline__factcheck_action` (share_facts_click2 with strata controls)

**Design variants** (2):
- diff_in_means, with_covariates

**RC: Control progressions** (4):
- Strata, strata+socio, strata+socio+vote, strata+socio+vote+fb

**RC: Control sets** (4):
- None, strata_only, strata+socio+vote+fb, full

**RC: Leave-one-out** (8):
- Drop each of key controls from strata+socio+vote+fb set

**RC: Random subsets** (5):
- 5 random draws from control pool

**RC: Outcome variants** (2):
- share_facts_click2 (action), share_fact_click3 (reconfirm)

---

## 6. Excluded from Core Universe

- **Table 2, Panel C** (predicted vs. actual sharing): This uses LASSO-predicted outcomes as the dependent variable. It is a methodological test of selection bias, not a standard ATE specification. Excluded.
- **Table 3** (within-respondent FB vs. others comparison): Uses panel FE (xtreg) on reshaped data comparing sharing on FB vs. with other participants. This is a secondary design (panel FE) and conceptually different from the main claim. Excluded.
- **Table A8** (heterogeneity by social desirability score): Interaction effects, not main ATE. Excluded.
- **Figures 5-6** (LASSO prediction comparisons): Diagnostic/predictive exercises. Excluded.
- **Follow-up experiment** (surveys 5-6): Not used in the paper's main analysis. Excluded.
- **Sensitivity/exploration**: Not part of the core universe per the specification tree protocol.

---

## 7. Inference Plan

**Canonical**: HC1 robust standard errors (matching the paper's `robust` option in Stata throughout).

**Variant**: HC3 standard errors as a small-sample correction stress test. The paper uses individual-level randomization with no clustering, so cluster-robust SEs are not applicable.

---

## 8. Constraints

- **Controls count envelope**: [0, 30] for both groups. The paper shows progressions from 0 to ~30 controls.
- **Linked adjustment**: Not applicable (single-equation OLS).
- **Mandatory strata controls**: The stratification variables (male, low_educ, mid_educ) should be included in all specifications except the pure diff-in-means design variant, as is standard for stratified RCTs.
- **Sample filter**: `survey < 4` for G1, `survey > 1 & survey < 4` for G2 (these are fixed).

---

## 9. Budget and Sampling

**G1**: 80 max core specs, 10 random control subsets (seed: 140161)
**G2**: 40 max core specs, 5 random control subsets (seed: 140162)

Random control subsets are drawn stratified by subset size from the full control pool (strata + socio + fb + vote + behavioral + reported = ~28 individual controls).

**Total estimated specs**: ~82 across both groups, well above the 50-spec target.
