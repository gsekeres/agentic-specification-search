# Specification Surface: 112805-V1

## Paper: Using School Choice Lotteries to Test Measures of School Effectiveness (Deming, 2014)

## Baseline Groups

### G1: Validity of value-added models (VAM coefficient = 1 under lottery-based IV)
- **Outcome**: testz2003 (average of standardized math and reading scores in 2003)
- **Treatment**: VA (value-added of assigned school, constructed from pre-lottery data)
- **Instrument**: lott_VA (VAM of lottery-determined school: home school for losers, first choice for winners)
- **Estimand**: IV coefficient on school VAM; null hypothesis is coefficient = 1 (unbiased VAM)
- **Population**: Students on the margin in CMS lottery (onmargin==1), grades 4-8, Spring 2002 lottery
- **Baseline spec**: Table 1: xtivreg2 testz2003 lagged_scores (VA=lott_VA), fe(lottery_FE) bootstrap(100) cluster(lottery_FE)

### Additional baselines (same claim, different VAM model types)
- **Model 1 levels**: VAM constructed from mean outcomes only (no lagged score controls)
- **Model 2 gains (mix)**: VAM from mixed-effects model with lagged scores
- **Model 2 gains (FE)**: VAM from school fixed effects model with lagged scores

## Key structural features

This paper is unusual because the main "specification dimensions" are the VAM construction choices (Model 1 vs 2, estimation method, pre-lottery sample window), not standard control variation. The VAM is constructed in a first stage on non-lottery students, then used as the endogenous variable in the lottery-based IV.

### VAM construction dimensions (the paper's main specification grid)
1. **Model type**: Model 1 (levels only) vs Model 2 (gains/lagged scores)
2. **Estimation method**: Average residual (ar) vs Mixed effects (mix) vs School FE (FE)
3. **Training sample**: Single year (02) vs 2 years (2yr) vs all pre-lottery years (all)
4. **Counterfactual school**: Home/neighborhood school (hm) vs weighted alternative (alt)

This gives 2 x 3 x 3 x 2 = 36 combinations, which IS the paper's main specification grid (Table 1).

## Core Universe

### VAM variant axes (18 specs per counterfactual, 2 counterfactuals = 36 total from VAM grid)
- All 18 model x estimation x sample combinations for home-school counterfactual
- All 18 for weighted-alternative counterfactual

### Additional axes
- Drop lagged score controls from the IV second stage
- No controls in the IV second stage (diff-in-means style)
- Restrict to grades 4-5 (elementary) vs grades 6-8 (middle school)
- Math-only and reading-only outcomes (instead of average)

## Inference Plan
- **Canonical**: Bootstrap(100) clustered at lottery_FE (matching the paper)
- **Variant**: Analytic clustered SEs at lottery_FE (faster, asymptotic)

## Constraints
- Linked adjustment: VAM controls and IV controls are structurally linked
- lottery_FE fixed effects are required in all specifications (unit of randomization)
- onmargin==1 sample restriction is maintained (only students in non-degenerate lotteries)
- Demographic controls (race, lunch) are not available in public-use data

## Budget
- Max core specs: 60
- Total planned: ~45-55
- Seed: 112805

## What is excluded and why
- Model 3 and Model 4 (with demographics and twice-lagged scores): excluded from public-use data due to confidentiality restrictions
- Table A2 (2004 outcomes with lead VAM): different outcome horizon, could be explore/*
- Table A3 (first stage and reduced form): diagnostics, not the claim
- VAM weighted by year correlations (commented-out code): not available in public data
