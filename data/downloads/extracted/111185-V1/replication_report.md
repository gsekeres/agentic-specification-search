# Replication Report: 111185-V1

## Summary
- **Paper**: Optimal Climate Policy When Damages are Unknown
- **Author**: Ivan Rudik
- **Journal**: American Economic Journal: Economic Policy, 2020
- **Replication status**: full
- **Total regressions in original package**: 1
- **Regressions in scope (main + key robustness)**: 1
- **Successfully replicated**: 1
- **Match breakdown**: 0 exact, 1 close, 0 discrepant, 0 failed

## Data Description
- Files used: `estimate_damage_parameters/10640_2017_166_MOESM10_ESM.dta` (Howard & Sterner 2017 replication data)
- Key variables:
  - `D_new`: GDP loss as percentage, transformed to damage function form `correct_d = (D_new/100)/(1 - D_new/100)`
  - `t`: temperature change
  - `log_correct`: log of transformed damages (dependent variable)
  - `logt`: log of temperature (independent variable)
- Sample sizes: 49 observations in dataset, 43 used in regression (6 dropped due to non-positive damage values producing missing log values)
- No discrepancies with published N

## Replication Details

### Table 1: Damage Parameter Estimation
- **What was replicated**: OLS regression of log damages on log temperature (`reg log_correct logt`) to estimate the power-law damage function parameters used as inputs to the structural model.
- **Estimates match**: Yes. The regression coefficient on `logt` (the damage function exponent d2) is 1.882038, which rounds to 1.88 matching the original exactly. The standard error is 0.450527, rounding to 0.45 (matching `sqrt(0.203) = 0.4506`). All six derived parameters in Table 1 match at reported precision.
- **Match status**: `close` -- the original table_1.csv stores the coefficient rounded to 2 decimal places (1.88), while our unrounded coefficient is 1.882038. The relative difference of 0.1% is well within the 1% threshold. The coefficients are numerically identical; the difference is purely due to the original storing rounded values.

### Tables 2, 3, A1, A4 and Figures 5-8
- **Not replicated**: These results are produced by Julia code that solves dynamic programming models via value function iteration (>20,000 core-hours of computation). They are NOT regression-type results. They involve solving Bellman equations, computing optimal carbon taxes, simulating model trajectories, and computing welfare gains. These are structural model outputs, not statistical estimations that produce coefficients and standard errors.

## Nature of the Paper
This is primarily a structural/calibration paper in environmental economics. The economic model is a DICE-type integrated assessment model extended with:
1. Unknown damage function parameters (Bayesian learning)
2. Robust control (ambiguity aversion)
3. Combined learning + robust control

The single regression (Table 1) is a calibration step that estimates damage function parameters from the meta-analysis data of Howard & Sterner (2017). These parameters feed into the structural model. All other results in the paper come from numerically solving the dynamic programming problems in Julia.

## Translation Notes
- Original language: Stata (for Table 1 regression), Julia (for structural model), R (for plots)
- Translation approach: Direct translation of the single Stata OLS regression to Python statsmodels. The Stata `reg` command with no options uses classical (non-robust) standard errors, which is the statsmodels default.
- Known limitations: None for the regression component. The Julia structural model code (the vast majority of the package) is not translatable to a simple regression framework -- it solves continuous-time dynamic programming problems via collocation methods and requires >20,000 core-hours to run.

## Software Stack
- Language: Python 3.12
- Key packages: statsmodels 0.14.6, pandas 2.x, numpy 1.x
