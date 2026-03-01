# Specification Surface: 113561-V1

## Paper
"What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
Fong & Luttmer, AEJ: Applied Economics, 2009

## Baseline Groups

### G1: Effect of racial picture priming on giving

**Claim object:**
- **Outcome concept**: Charitable giving to Hurricane Katrina victims via dictator game ($0-$100)
- **Treatment concept**: Showing pictures of black vs white Katrina victims (picshowblack)
- **Estimand concept**: ITT effect of racial picture priming on giving, controlling for other experimental manipulations and demographics
- **Target population**: US adult internet panel respondents (all races)

**Baseline spec**: Table 3, Col 2 (and equivalently Table 4, Panel 1, Col 1)
- `giving ~ picshowblack + picraceb + picobscur + [manipulation controls] + [demographic controls]`
- Weighted by `tweight` (oversampling correction)
- Robust (HC1) standard errors
- N = 1343

**Why one baseline group**: The paper's headline result is the effect of picshowblack on giving for all respondents. Table 4 extends this to multiple outcomes (hypothetical giving, support for charity/govt spending) and subgroups (white, black), but these are presented as extensions/heterogeneity, not separate main claims. Table 5 provides the paper's own robustness surface. Table 6 explores interactions with racial attitudes. We treat these as robustness checks and explorations around the single baseline claim.

## Revealed Search Space

The paper reveals the following specification dimensions in its robustness checks (Table 5):

1. **Sample restrictions**: main survey only (mweight), Slidell only, Biloxi only, race-shown only
2. **Control sets**: no demographics (manipulation vars only), baseline demographics, extra controls (hfh_effective, lifepriorities)
3. **Functional form**: censored regression (cnreg) for continuous outcomes, ordered probit (oprob) for ordinal outcomes
4. **Weights**: tweight vs mweight (for main-survey-only subsample)

We extend beyond the paper's own revealed space with:
- Leave-one-out (LOO) control dropping
- Control progression (building up from bivariate)
- Outcome transformations (log(1+y), asinh, binary)
- Unweighted regressions
- Outlier trimming

## Core Universe

### Baseline specs (2)
- `baseline`: Table 3 Col 2 (all respondents, full manipulation + demographic controls, tweight, HC1)
- `baseline__table4_panel1_all`: Table 4 Panel 1 (uses nraudworthy composite instead of individual worthiness dummies)

### Design specs (1)
- `design/randomized_experiment/estimator/diff_in_means`: Simple difference in means (no controls)

### RC: Controls (29)
- **Sets**: none, minimal (manip only), baseline, extended (+ extra controls)
- **LOO**: Drop each of 20 demographic control variables one at a time
- **Progression**: Bivariate, manipulation only, + demographics, + giving history, full

### RC: Sample (5)
- Main survey only (surveyvariant==1, use mweight)
- Slidell only
- Biloxi only
- Race-shown only (exclude picobscur==1)
- Trim giving outliers (1/99 percentile)
- Topcode giving at 90 (alternative topcode)

### RC: Functional form (4)
- log(1+giving)
- asinh(giving)
- Binary: any giving (giving > 0)
- nraudworthy composite (aggregate worthiness manipulations)

### RC: Weights (2)
- Unweighted (no survey weights)
- mweight (main survey weight)

## Total budget: ~55 core specs

## Inference Plan

**Canonical**: HC1 (robust) standard errors, matching the paper's approach. Randomization is at the individual level, so clustering is not necessary.

**Variants** (recorded separately in inference_results.csv):
- Classical (homoskedastic) SE
- HC2 (leverage-corrected)
- HC3 (small-sample)

## What Is Excluded (and why)

1. **Table 6 interactions**: These change the estimand (heterogeneity by racial attitudes). They are not estimand-preserving robustness checks.
2. **Table 3 Cols 3-6**: These involve interactions with respondent race and subjective identification, changing the estimand.
3. **Table 4 Panels 2-4**: Alternative outcomes (hypothetical giving, charity/govt support). These change the outcome concept.
4. **Table 5 Panels 2-4**: Same alternative outcomes.
5. **Table 5 rows for black respondents**: Changes the target population.
6. **cnreg/oprob**: These are nonlinear functional form changes. We approximate with log/asinh/binary instead, as cnreg and oprob require specialized estimation.
7. **Subgroup analyses by race**: These change the target population (exploration, not core RC).

## Linkage Constraints

Not applicable (no bundled estimator). The baseline is a simple OLS/WLS regression.

## Control Blocks (for progression)

1. **Treatment dummies**: picshowblack, picraceb, picobscur (always included, these are treatment arms)
2. **Manipulation controls**: aud_* variables, cityslidell, var_fullstakes, var_racesalient
3. **Demographics**: age, age2, black, other, edudo, edusc, educp, lnhhinc, dualin, married, male, singlemale, south
4. **Labor force status**: work, disabled, retired
5. **Prior giving**: dcharkatrina, lcharkatrina, dchartot2005, lchartot2005
6. **Extra controls**: hfh_effective, lifepriorities_help, lifepriorities_mony (potentially endogenous)
