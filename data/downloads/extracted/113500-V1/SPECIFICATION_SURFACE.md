# Specification Surface: 113500-V1

## Paper Overview
- **Title**: Gender Differences in the Allocation of Low-Promotability Tasks: The Role of Backlash (Babcock, Recalde, Vesterlund, 2017, AER P&P)
- **Design**: Randomized experiment (lab experiment)
- **Data**: Subject-period level data from a lab experiment with two treatment conditions (no-penalty and backlash/penalty). Subjects are "green players" who decide whether to invest (volunteer for a low-promotability task) in each period, and may or may not be solicited (asked) by another group member.

## Baseline Groups

### G1: Gender Gap in Response to Solicitation -- No-Penalty Condition (Table 1, Col 1)

**Claim object**: In the no-penalty baseline condition, being asked (solicited) to volunteer for a low-promotability task differentially increases women's probability of volunteering compared to men.

**Why a separate group**: The paper's Table 1 reports three probit regressions -- Columns 1 and 2 are separate by treatment condition, and Column 3 pools both. Column 1 (treatment==1, no penalty) is the baseline finding about gender differences in response to solicitation. This is the core claim about how social pressure (requests) differentially affects women.

**Baseline specification**:
- Model: Probit with marginal effects
- Formula: `decision ~ solicited + female + femaleXsolicited + period + risk_seeking1 + social1 + age + non_caucasian + student + usborn + business + other`
- Sample: `invest_group > 0 & treatment==1`
- Clustering: `session_id` (randomization unit)
- Focal coefficient: Interaction effect of `femaleXsolicited` (corrected for probit nonlinearity using `inteff`)
- Note: The paper uses `cgmwildboot` for wild cluster bootstrap inference because of the small number of session clusters.

### G2: Gender Gap in Response to Solicitation -- Backlash/Penalty Condition (Table 1, Col 2)

**Claim object**: Under the backlash condition (where a penalty can be imposed for group failure), the gender gap in response to solicitation is examined. The paper asks whether the threat of punishment changes the gender dynamics of volunteering.

**Why a separate group**: Treatment==2 introduces the possibility of penalties. The same probit model is run on this subsample. This is a distinct claim about how backlash threats interact with gender and solicitation.

**Baseline specification**: Same model structure as G1 but restricted to `treatment==2`.

### G3: Difference Across Treatments -- Pooled Triple Interaction (Table 1, Col 3)

**Claim object**: Whether the gender difference in response to solicitation changes when the backlash/penalty treatment is introduced (the "difference-in-difference-in-differences" across treatments).

**Why a separate group**: Column 3 pools both treatment conditions and estimates a triple interaction (`femaleXbacklashXsol`). This directly tests whether the penalty condition differentially changes how women respond to solicitation. This is the paper's main between-treatment comparison.

**Baseline specification**:
- Model: Probit with marginal effects
- Formula: `decision ~ solicited + female + backlash + femaleXsolicited + backlashXsolicited + femaleXbacklash + femaleXbacklashXsol + [controls]`
- Sample: `invest_group > 0` (both treatments pooled)
- Focal coefficient: Triple interaction `femaleXbacklashXsol` (corrected using `inteff3`)

## RC Axes Included (Same Structure for All Three Groups)

### Controls (9 individual-level covariates)
- **Leave-one-out drops** (9 specs): Drop each control individually (period, risk_seeking1, social1, age, non_caucasian, student, usborn, business, other)
- **No controls**: Simple probit with only the core treatment/gender/interaction terms
- **Demographics only**: age, non_caucasian, student, usborn, business, other (6 controls)
- **Preferences only**: risk_seeking1, social1 (2 controls)

### Functional form / Estimator
- **LPM (linear probability model)**: OLS regression instead of probit. Interaction coefficients are directly interpretable without inteff correction.
- **Logit**: Alternative binary choice model

### Sample / Period restrictions
- **First half only** (periods 1-5): Tests whether the effect is present early
- **Second half only** (periods 6-10): Tests whether the effect persists or changes with learning
- **First period only** (period==1): No learning effects, cleanest experimental test

### Design estimator variants
- **Difference-in-means**: Raw mean comparison of decision rates by solicited x female cells
- **With covariates**: OLS with all controls (equivalent to LPM + controls)

## What Is Excluded and Why

- **Penalty choice analysis** (Section 6 of the paper): Analysis of penalty sizes chosen by red players in treatment==2 is a different outcome and population (red players, not green players). This is a secondary analysis.
- **Requests received analysis** (Table B2, Appendix): OLS of number of times asked (n_asked_session) on gender and controls. This is about the allocation mechanism, not the response to solicitation.
- **Investment time analysis** (Table B1): Descriptive analysis of time spent investing. Not the main claim.
- **Exploration/heterogeneity**: No formal subgroup analysis beyond the gender x solicited interaction.
- **Penalty observations** (Section 6, descriptive): Descriptive statistics on observed penalties. Not a regression claim.

## Budgets and Sampling

- **Each group max core specs**: 55
  - 1 baseline + 9 LOO + 3 control sets + 2 functional form + 3 period splits + 2 design variants = ~20 per treatment condition
  - With 3 groups: ~60 total across all groups (165 nominal, but many overlap)
- **Control subsets**: 20 max (exhaustive: 9 LOO + 3 grouped sets + no-controls variant)
- **Seed**: 113500
- **Enumeration**: Full (small control pool, structured variations)

## Inference Plan

- **Canonical**: Cluster at session_id (the randomization unit). This is critical because treatment is assigned at the session level and within-session observations are correlated.
- **Variants**:
  - Cluster at individual (unique_subjectid) -- accounts for within-subject correlation across periods
  - Robust SEs (HC1) without clustering
  - Wild cluster bootstrap at session level (matching the paper's cgmwildboot approach, seed=999)
- **Note on small clusters**: The paper has a small number of sessions per treatment condition. Standard clustered SEs may perform poorly; wild cluster bootstrap is preferred for inference. This is flagged in the inference plan.

## Key Linkage Constraints

- The interaction terms (femaleXsolicited for G1/G2; the triple interaction set for G3) are mandatory for the estimand. They cannot be dropped.
- The `period` control is particularly relevant because subjects play 10 rounds, and learning effects may change behavior over time. It is in the paper's specification but its role is purely for precision.
- The `solicited` variable is within-subject and varies by period (who gets asked each round). The `female` variable is between-subject. The experimental variation comes from random assignment to treatment conditions (session-level) and the within-session solicitation mechanism.
- For the LPM variant, the standard Stata `inteff` correction is not needed, and the interaction coefficient is directly interpretable. This is a meaningful advantage for specification analysis.
