# Specification Surface: 180741-V1

**Paper**: "Enabling or Limiting Cognitive Flexibility? Evidence of Demand for Moral Commitment"
**Authors**: Saccardo & Serra-Garcia (AER, forthcoming)
**Design**: Randomized experiment (online experiments on advisor incentives and financial recommendations)

---

## Paper Summary

This paper runs several online experiments studying whether financial advisors prefer to see their incentive information before or after assessing product quality, and how this preference (and its interaction with random assignment) affects biased recommendations. The key design feature is that in "Choice" experiments, advisors first state whether they prefer to see incentive information first or quality information first (`choicebefore`), and are then randomly assigned to one order (`getbefore`). The paper finds that advisors who prefer to see incentives first are significantly more likely to recommend the incentivized product, and that being assigned one's preferred order amplifies this effect.

---

## Baseline Groups

### G1: Choice Experiment -- Recommendations by Preference and Assignment (Table 3)

This is the paper's central result. It examines how an advisor's *preference* for information order (`choicebefore`) predicts their recommendation of the incentivized product (`recommendincentive`), interacted with whether the advisor was assigned their preferred order (`getyourchoice` / `notgetyourchoice`) and whether there is a conflict of interest (`noconflict`).

**Claim object**:
- **Outcome**: `recommendincentive` (binary: advisor recommends the incentivized product)
- **Treatment**: `choicebefore` (binary: advisor prefers to see incentive first) and its interactions with `noconflict` and `notgetyourchoice`
- **Estimand**: ITT effect of preference for information order on biased recommendations
- **Population**: Advisors in Choice experiment (ChoiceFree, ChoiceFree_Professionals, PayBefore, PayAfter), baseline stakes (Highx10==0 & Highx100==0)

**Baseline specifications** (Table 3):
- **Col 1**: `getyourchoice==1` subsample, 14 covariates (treatment dummies + design parameters + demographics)
- **Col 2**: `getyourchoice==0` subsample, same covariates
- **Col 3**: Full sample with `notgetyourchoice` and its interactions, 17 covariates

**Covariates (global `covariates2`)**: `professionalsfree`, `seeincentivecostly`, `seequalitycostly`, `wave2`, `wave3`, `professionalscloudresearch`, `incentiveshigh`, `incentiveleft`, `incentiveshigh_incentiveleft`, `age`, `female`

**Structural regressors** (always included): `choicebeforenoconflict`, `noconflict`, `incentiveB`

### G2: NoChoice Experiment -- Effect of Random Assignment on Recommendations (Table C.1)

This is the foundational experiment where advisors have *no choice* over information order and are purely randomly assigned to see incentives first or quality first. The treatment is `seeincentivefirst`.

**Claim object**:
- **Outcome**: `recommendincentive`
- **Treatment**: `seeincentivefirst` (binary: randomly assigned to see incentive first)
- **Estimand**: ITT effect of seeing incentive information first on biased recommendations
- **Population**: Advisors in NoChoice experiment (N~300, attentive participants)

**Baseline specifications** (Table C.1):
- **Col 1**: Conflict subsample (`conflict==1`), 5 covariates
- **Col 2**: No-conflict subsample (`conflict==0`), 5 covariates
- **Col 3**: Full sample with `seeincentivefirst_noconflict` interaction, 6 covariates

**Covariates**: `noconflict`, `incentiveB`, `female`, `age`, `stdalpha`

### G3: Choice Experiment -- Preferences for Information Order (Table 2)

This baseline group examines the demand side: what fraction of advisors *choose* to see the incentive first, and how this varies across costly-choice vs. free-choice conditions.

**Claim object**:
- **Outcome**: `choicebefore` (binary: advisor prefers to see incentive first)
- **Treatment**: `seeincentivecostly` (indicator for costly incentive-first treatment) with `seequalitycostly` (indicator for costly quality-first treatment) as additional treatment arm
- **Estimand**: Treatment effect of costly-choice conditions on preference to see incentive first
- **Population**: Advisors in Choice experiment, baseline stakes

**Baseline specifications** (Table 2):
- **Col 1**: Full sample, 10 covariates
- **Col 2**: Non-professionals only, adds `stdalpha`, 11 covariates
- **Col 3**: Full sample, adds `stdalpha` and selfishness interactions, 13 covariates

---

## Core Universe Design

### Design variants (`design/randomized_experiment/*`)

For all three baseline groups:
- **`design/randomized_experiment/estimator/diff_in_means`**: Simple difference in means without covariates (minimal adjustment). Since randomization is at the individual level, this is valid and provides a useful benchmark.
- **`design/randomized_experiment/estimator/with_covariates`**: Paper's approach -- OLS with pre-treatment covariates for precision.

### Robustness checks (`rc/*`)

**Controls (LOO)**: Leave-one-out of each optional covariate. The paper's control sets include:
- Design parameters: `incentiveshigh`, `incentiveleft`, `incentiveshigh_incentiveleft`, `professionalscloudresearch`, `wave2`, `wave3`
- Treatment arm indicators: `professionalsfree`, `seeincentivecostly`, `seequalitycostly` (these are structural for G3 but optional for G1)
- Demographics: `age`, `female`
- Structural interaction terms (`choicebeforenoconflict`, `noconflict`, `seeincentivefirst_noconflict`) are NOT dropped in LOO since they define the estimand.

**Controls (additions)**:
- `stdalpha` (standardized selfishness measure from MPL task)
- `selfishseeincentivecostly` (selfishness x treatment interaction)
- `selfishseequalitycostly` (selfishness x treatment interaction)

**Sample variants**:
- Include inattentive participants (drop the `alphavaluefinal!=.` filter) -- paper reports these in Appendix Tables C.2, C.20, C.21
- Include high-stakes treatments (`Highx10==1` or `Highx100==1`) -- paper reports in Appendix Tables C.18, C.19
- Restrict to ChoiceFree only (treatment==0) -- most comparable to NoChoice
- Restrict to professionals only (study==1)
- Restrict to incentiveA or incentiveB only -- paper reports separately in Appendix Tables C.7, C.8

**Outcome variants**:
- Probit model (since outcome is binary)
- Logit model (since outcome is binary)

---

## Inference Plan

**Canonical**: HC3 robust standard errors (`vce(hc3)`), matching the paper throughout.

**Variants**:
- HC1 robust SEs (Stata's default `robust`)
- HC2 robust SEs

No clustering is needed because randomization is at the individual level. The paper does not cluster.

---

## Constraints and Guardrails

1. **Controls-count envelope**: G1 [14, 18], G2 [3, 6], G3 [10, 13]. These reflect the range across main table columns.

2. **Structural interaction terms**: The following are NOT optional controls but define the estimand:
   - G1: `choicebeforenoconflict`, `noconflict` (and in Col 3: `notgetyourchoice`, `choicebeforenotgetyourchoice`, `notgetyourchoicenoconflict`)
   - G2: `noconflict`, `seeincentivefirst_noconflict` (in full-sample spec)
   - G3: `seequalitycostly` (second treatment arm indicator)

3. **Sample filters**: The baseline sample excludes high-stakes treatments (`Highx10==0 & Highx100==0`) and inattentive participants (`alphavaluefinal!=.`). Relaxing these are explicit RC variants.

4. **Cross-experiment comparison**: The paper compares NoChoice vs. Choice results (Table C.16/C.17), but we keep these as separate baseline groups since they use different datasets and different treatment variables.

---

## Budget and Sampling

**G1 (Choice -- Recommendations)**: Target ~50-60 specs
- 3 baseline specs (Table 3 Cols 1-3)
- 2 design variants (diff-in-means, with-covariates) x 2 subsamples = 4
- 12 LOO specs (one per optional covariate)
- 3 addition specs (stdalpha, selfishness interactions)
- 7 sample restriction specs
- 2 functional-form specs (probit, logit)
- Subtotal: ~31 core specs

**G2 (NoChoice -- Recommendations)**: Target ~20-30 specs
- 3 baseline specs (Table C.1 Cols 1-3)
- 2 design variants
- 4 LOO specs
- 1 addition spec (drop stdalpha filter)
- 5 sample restriction specs
- 2 functional-form specs
- Subtotal: ~17 core specs

**G3 (Choice -- Preferences)**: Target ~20-30 specs
- 3 baseline specs (Table 2 Cols 1-3)
- 2 design variants
- 9 LOO specs
- 3 addition specs
- 3 sample restriction specs
- 2 functional-form specs
- Subtotal: ~22 core specs

**Total across all groups**: ~70 core specifications (within budget of 50+ target).

Full enumeration is feasible for all groups given moderate control pools.

---

## What is Excluded and Why

1. **Belief updating regressions (Table 4)**: These use a different outcome (`logitbelief`) and a fundamentally different regression structure (no-constant regressions of log-odds beliefs on log-likelihood ratios `bad` and `good`). This is a separate claim object with a different estimand (Bayesian updating coefficients) that would require its own specification surface. Excluded from scope.

2. **Stakes experiment (ChoiceStakes)**: Uses a different dataset (`stakes.dta`) with different treatment structure (`commissionlow`, `commissionhigh`). The paper treats this as a separate experiment (Section 8). Could be its own baseline group but is lower priority.

3. **Information Architect experiment**: Uses a different dataset (`InformationArchitect.dta`) with different treatment (`IAAdvisor`). The paper treats this as a separate experiment (Section 8.2). Excluded.

4. **Choice Deterministic experiment**: Uses a different dataset (`Choice_Deterministic.dta`) with a `Deterministic` treatment indicator. Appendix-only results. Excluded.

5. **NoChoice Simultaneous experiment**: Uses merged `nochoice1_2_merged.dta` with a `Together` indicator. Appendix-only results. Excluded.

6. **Client follow-up studies**: Descriptive statistics only (tab follow), no regression analysis.

7. **Predictions study**: Separate study design, not a core experimental claim.

8. **Sensitivity and exploration**: `sens/*`, `explore/*`, `post/*`, `diag/*` are not in the core universe per protocol.
