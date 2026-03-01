# Specification Surface: 111185-V1

## Paper Overview
- **Title**: Optimal Climate Policy When Damages are Unknown (Rudik, 2020)
- **Design**: Structural calibration (primary); Cross-sectional OLS (secondary, Table 1 only)
- **Key limitation**: This paper is overwhelmingly a structural/computational model. The only regression-type content is a single bivariate OLS used to estimate damage function parameters (Table 1). The specification surface covers only this OLS regression.

## Baseline Groups

### G1: Damage Function Parameter Estimation (Table 1)

**Claim object**: The damage exponent d2 in the power-law damage function D(T) = d1 * T^d2, estimated by regressing log(damages) on log(temperature) using meta-analysis data from Howard & Sterner (2017).

**Baseline specification**:
- Formula: `log_correct ~ logt`
- Outcome: `log_correct` (log of corrected damage = log(D_new/100 / (1 - D_new/100)))
- Treatment: `logt` (log of temperature increase)
- Controls: None (bivariate regression)
- N = 43 (6 observations dropped from 49 due to non-positive damages)
- Focal coefficient: 1.882 (= d2 location parameter, rounded to 1.88)
- Inference: Classical OLS standard errors (matching Stata `reg` default)

## RC Axes Included

### Controls (7 optional variables from Howard & Sterner 2017)
- **Single additions**: Each of 7 controls added individually
- **Standard sets**: Minimal (Grey, Based_On_Other), Extended (+ Repeat_Obs, Market, cat), Full (all 7)
- **Progression**: Build-up from bivariate to full
- **Block combinations**: 3 blocks (study_quality, damage_type, study_design), 7 exhaustive block combos
- **Random variable subsets**: 15 random draws (seed=111185), subset sizes 2-6

### Sample restrictions
- **Outlier trimming**: Outcome at [1,99] and [5,95]; treatment at [1,99]
- **Cook's D**: Drop observations with Cook's D > 4/N
- **Quality filters**: Drop repeat observations, based-on-other studies, grey literature, catastrophic damage estimates
- **Temporal split**: Early vs late studies (median year cutoff)
- **Independence filter**: Drop both repeat and based-on-other simultaneously

### Functional form
- **Outcome transforms**: Levels, asinh (both preserve damage concept)
- **Treatment transforms**: Level temperature (semi-elasticity interpretation)
- **Nonlinear models**: Quadratic in log-temp; quadratic polynomial in levels

### Preprocessing
- **Winsorization**: Outcome [1,99]; treatment [1,99]

### Joint axis variations
- Combined sample restriction + controls (8 combinations)

## What Is Excluded and Why

- **Structural model computations**: Tables 2-3, A1, A4, all figures -- these are outputs of the calibrated dynamic programming model, not regression-type estimates
- **Design variants**: No `design/*` variants since there is only one baseline OLS and no within-design alternatives are meaningful (no FE to vary, no instruments, no panel structure)
- **Exploration**: No alternative outcome/treatment concepts beyond what is covered by functional form RC
- **Diagnostics**: No standard diagnostics (no endogeneity concern, no panel structure)

## Budgets and Sampling

- **Max core specs**: 60
- **Max control subsets**: 25 (7 block combos + 15 random + 3 standard sets)
- **Seed**: 111185
- **Full enumeration**: For block combinations; random sampling for variable-level subsets

## Inference Plan

- **Canonical**: Classical OLS standard errors (matches paper)
- **Variants**: HC1, HC2, HC3 (heteroskedasticity-robust alternatives)
- Inference variants are recorded in `inference_results.csv`, not in `specification_results.csv`
