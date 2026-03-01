# Specification Surface: 128143-V1

**Paper**: "Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion" (Douenne and Fabre, AEJ: EP)
**Design**: Randomized experiment (survey RCT) with IV/2SLS as primary estimator

---

## Overview

This paper studies why French citizens overwhelmingly oppose a carbon tax despite most being net beneficiaries of the proposed tax+dividend policy. Using a large representative online survey (~3,000 respondents), the authors randomly assign information treatments that correct pessimistic beliefs about (1) self-interest (personal gains/losses from the tax) and (2) environmental effectiveness of carbon taxation. Random treatment assignment is used as an instrument for beliefs in a 2SLS/IV framework, estimating the causal effect of corrected beliefs on tax acceptance.

The identification is stratified RCT: respondents are randomly assigned to different information conditions, and these random assignments instrument for belief changes (imperfect compliance). The paper uses manual 2SLS (first-stage LPM, then second-stage LPM with fitted values) because the R `ivreg` package does not support survey weights. Standard `ivreg` is used separately to obtain effective F-statistics.

---

## Baseline Groups

### G1: Self-Interest Beliefs and Tax Acceptance (Section 5.1, Tables 5.1-5.2)

**Claim object**: Beliefs about personal gains/losses from the carbon tax causally affect tax acceptance. Respondents who believe they do not lose from the policy are significantly more likely to accept it.

**Identification**: Random assignment of the income threshold for dividend eligibility (`traite_cible`) creates exogenous variation in self-interest beliefs. The instrument is `traite_cible * traite_cible_conjoint` (treatment for respondent and spouse). The main sample restriction is to respondents near the income threshold (p10-p60) where the instrument has the most power.

**Baseline specs**:
- Col 1 (Table 5.2): IV on sub-sample p10-p60 -- 53 p.p. effect
- Col 2 (Table 5.2): IV on full sample -- 46 p.p. effect

**Control set** (`variables_reg_self_interest`): ~20 variables including progressivity belief (prog_na), estimated gains (Simule_gain, quadratic), tax effectiveness belief (taxe_efficace), household composition (single), expense increase (hausse_depenses_par_uc), demographics (sex, age groups, education, employment status, region, town size, household composition, income), and piecewise-linear income percentiles (knots at p20, p70).

### G2: Environmental Effectiveness Beliefs and Tax Approval (Section 5.2, Tables 5.3-5.4)

**Claim object**: Beliefs about the environmental effectiveness of carbon taxation causally affect tax approval. Respondents who believe the tax is effective are significantly more likely to approve it.

**Identification**: Random assignment of information about environmental effectiveness (`apres_modifs`) and climate change (`info_CC`) creates exogenous variation in effectiveness beliefs. Two instruments for one endogenous variable, with Sargan overid test not rejecting (p=0.93).

**Baseline specs**:
- Col 1 (Table 5.4): IV, approval(Yes) ~ effectiveness(Yes) -- 42 p.p.
- Col 3 (Table 5.4): IV, acceptance(not No) ~ effectiveness(Yes) -- 50 p.p.

**Control set** (`variables_reg_ee`): ~15 variables including income (quadratic for respondent and spouse), household composition, estimated gains, self-interest belief category (gagnant_categorie), and demographics.

---

## What is Included (Core Universe)

### RC axes for G1:

**Functional form / outcome definition**:
- `rc/form/outcome_approval_yes`: Change outcome from acceptance (not No) to approval (Yes)
- `rc/form/ols_instead_of_iv`: Simple OLS with non_perdant as explanatory variable (endogeneity caveat, Col 3 of Table 5.2)

**Sample**:
- `rc/sample/feedback_subsample`: Use feedback sub-experiment (variante_taxe_info=='f', |simule_gain| < 50) with simule_gagnant as instrument instead of traite_cible (Col 4 of Table 5.2)

**Controls (LOO by conceptual block)**:
- Drop taxe_efficace (effectiveness belief)
- Drop tax_acceptance (initial acceptance)
- Drop piecewise income percentiles
- Drop demographics block

**Weights**:
- Unweighted (no survey weights)

### RC axes for G2:

**Functional form / outcome definition**:
- `rc/form/outcome_approval_yes`: Outcome is strict approval (Yes) rather than acceptance (not No)
- `rc/form/ols_instead_of_iv`: OLS with taxe_efficace as explanatory variable (Col 2 of Table 5.4)
- `rc/form/endogenous_not_no`: Endogenous variable is taxe_efficace!='Non' instead of taxe_efficace=='Oui' (Table E, Appendix A4)

**Controls (LOO by conceptual block)**:
- Drop income quadratic
- Drop estimated gains (Simule_gain)
- Drop self-interest belief (gagnant_categorie)
- Drop demographics

**Weights**:
- Unweighted

---

## What is Excluded (and Why)

- **Section 3 (Pessimistic beliefs descriptives)**: Descriptive statistics, CDFs, and summary comparisons of subjective vs objective gains. Not regression estimates.
- **Table 3.1 (Heterogeneity in bias)**: OLS/logit predicting whether respondents have large bias. This examines determinants of bias, not the causal effect of beliefs on acceptance. Classified as `explore/*`.
- **Table 4.2 (Effect of feedback on beliefs)**: Reduced-form first-stage result showing feedback shifts beliefs. This is a diagnostic/first-stage object, not the causal effect of beliefs on acceptance.
- **Section 5.3 / Appendix D.4 (Progressivity)**: OLS of progressivity beliefs on information treatment. No IV/2SLS. Weak first stage (info_progressivite has near-zero correlation with beliefs). Not a main claim.
- **Table D.1 (Elasticities)**: Relationship between subjective elasticities and effectiveness beliefs. Different outcome/question. `explore/*`.
- **Appendix E (Alternative specifications)**: Alternative IV definitions (wins vs does not lose, approval vs acceptance). Some are already captured in rc/form variants. Others with further variations are `explore/*`.

---

## Inference Plan

**Canonical**: Default OLS standard errors from `lm()` in R with survey weights. The paper does not use explicit HC or clustered SEs.

**Variants**:
- HC2 robust SEs: a stress test since the default lm() SEs assume homoskedasticity conditional on covariates.

**Note**: The paper also reports `ivreg` results (without weights) for effective F-statistics. These are recorded as diagnostics, not as alternative inference for the main estimates.

---

## Budgets and Sampling

| Group | Max Core Specs | Max Controls Subset | Sampling |
|---|---|---|---|
| G1 | 60 | 30 | Stratified by block size |
| G2 | 50 | 25 | Stratified by block size |

Control sets are moderately large (~15-20 variables) but organized into clear conceptual blocks (income, gains, demographics, political, tax-related). LOO by block is the primary strategy, keeping combinatorics manageable.

---

## Key Linkage Constraints

1. **Bundled IV estimator**: The same control set must appear in both first and second stage. Control variation is joint, not independent across stages.
2. **Instrument set is experimentally fixed**: The instruments (traite_cible for G1, info_CC + apres_modifs for G2) are determined by the experimental design. They cannot be varied as a specification choice.
3. **Survey weights**: All regressions use survey weights (s$weight). The unweighted variant is a robustness check, not the baseline.
4. **Sub-sample restriction in G1**: The main specification restricts to p10-p60 income percentiles where the instrument has the most power. The full-sample variant is included as a baseline.
5. **Manual 2SLS**: The paper uses manual 2SLS (fitted values from first stage plugged into second stage) rather than system-estimated 2SLS, because R's `ivreg` does not support survey weights. This means standard errors are not corrected for generated-regressor uncertainty (though the paper also reports ivreg SEs without weights for comparison).

---

## Special Notes on This Paper

- The paper is a survey experiment, not a field experiment. The "treatment" is information provision, and the "outcome" is a stated preference (policy support). This means treatment effects are on beliefs and attitudes, not on behavior.
- The design is fundamentally cross-sectional (one survey wave), though the information treatments create within-survey exogenous variation.
- The classification as `randomized_experiment` with `secondary_design_codes: ["instrumental_variables"]` reflects the dual nature: random assignment provides the experiment, but the estimand requires IV because of imperfect compliance (information does not perfectly shift beliefs).
- The paper uses linear probability models (LPM) throughout. Logit/probit alternatives are shown in some appendix tables but are not the main specification.
