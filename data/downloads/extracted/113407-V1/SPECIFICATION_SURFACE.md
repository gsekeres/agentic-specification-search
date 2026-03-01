# Specification Surface: 113407-V1

## Paper Overview
- **Title**: Living Arrangements, Doubling Up, and the Great Recession: Was This Time Different? (Hoynes, 2014 AER Papers & Proceedings)
- **Design**: Panel fixed effects
- **Data**: State-year panel from March CPS (1980-2013), collapsed by state and calendar year
- **Key finding**: State unemployment rate increases lead to more doubling up (lower fraction living alone) among young adults, with the Great Recession showing similar cyclicality to the 1980s recession

## Baseline Groups

### G1: Young Adult Living Alone (Table 1 Panel A, Col 3)

**Claim object**: The effect of a 1-percentage-point increase in the state unemployment rate on the fraction of young adults (18-30) living alone (not doubled up), conditional on state and year fixed effects.

**Baseline specification**:
- Formula: `myadult_aloneall_1830 ~ urate | stfips + calyear`
- Outcome: `myadult_aloneall_1830` (fraction of young adults 18-30 living independently)
- Treatment: `urate` (state unemployment rate)
- Controls: None beyond state and year FE
- Weights: Analytic weights (`weight_1830`, CPS population weights)
- Clustering: State level (`stfips`)
- The paper reports a negative coefficient, indicating that higher unemployment leads to more doubling up

**Additional baseline specs**:
- `baseline__table1_panA_col4`: Outcome = `myadult_aloneall_1824` (ages 18-24)
- `baseline__table1_panA_col5`: Outcome = `myadult_aloneall_2530` (ages 25-30)

### G2: Household Size (Table 1 Panel A, Col 1)

**Claim object**: The effect of a 1-percentage-point increase in the state unemployment rate on average household size for non-elderly population.

**Baseline specification**:
- Formula: `h_numpers_noneld ~ urate | stfips + calyear`
- Outcome: `h_numpers_noneld` (average number of persons per household, non-elderly)
- Treatment: `urate` (state unemployment rate)
- Controls: None beyond state and year FE
- Weights: Analytic weights (`weight_noneld`)
- Clustering: State level (`stfips`)

**Additional baseline spec**:
- `baseline__table1_panA_col2`: Outcome = `h_numfams_noneld` (number of families per household)

## RC Axes Included

### Outcome definitions (via age group subsamples)
- **18-24 only**: Younger subset of the 18-30 baseline sample
- **25-30 only**: Older subset of the 18-30 baseline sample
- **Number of families per HH**: Alternative household composition measure (Col 2)

### Functional form
- **Log outcome**: Log-transform the fraction living alone (or household size)
- **Level outcome**: Ensure level is the baseline (it is)
- **Log unemployment**: Use log of state unemployment rate
- **Recession-specific treatment**: Panel B specification with period-specific unemployment effects (`urate_80`, `urate_rest`, `urate_07`)

### Fixed effects structure
- **Drop state FE**: Year FE only (tests whether within-state variation drives results)
- **Add state-specific linear trend**: State + year FE + state-specific trends
- **Add region-by-year FE**: Absorbs region-level shocks (replaces year FE with region*year)
- **Add division-by-year FE**: Finer geographic time trends

### Weighting
- **Unweighted**: Drop analytic weights (robustness to population-size weighting)

### Sample restrictions (time period)
- **Pre-2007**: Drop the Great Recession period (test with historical variation only)
- **Post-1990**: Drop the 1980s recession period
- **1990-2013**: Exclude both early recessions
- **Drop recession years**: Exclude NBER recession years only

### Sample restrictions (outliers/geography)
- **Trim extreme unemployment**: Drop state-years with unemployment in top/bottom 1%
- **Drop high-unemployment states**: Exclude states with persistently high unemployment
- **Drop small states**: Drop states below a population threshold

### Joint variations (G1)
- **Outcome x age group**: Combine outcome with different age bins (18-24, 25-30)
- **Form x time period**: Combine recession-specific treatment with time period restrictions

### Design alternatives
- **First difference**: First-difference estimator instead of within/FE estimator

## What Is Excluded and Why

- **Panel B as separate baseline groups**: The recession-period-specific coefficients (Panel B) are treated as functional-form RC variants of the pooled specification, not as separate baseline claims. The paper's main claim is about the general cyclicality of living arrangements.
- **Poverty-related outcomes**: The collapse-aeapap.do file creates poverty rate variables, but these are not used in the main regressions (Table 1). Excluded as the paper does not present poverty results as headline claims.
- **Exploration / heterogeneity**: No heterogeneity analysis is presented in the paper (AER P&P format). No explore/* specs.
- **Controls pool**: The paper uses no covariates beyond state and year FE. There is no control pool to vary, so the controls RC axis is empty.
- **Diagnostics**: Limited diagnostics given the simple panel structure. Wooldridge test for serial correlation included for G1.

## Budgets and Sampling

- **G1**: Max 55 core specs. Full enumeration feasible since the universe is generated by combining discrete axes (outcomes, forms, FE, samples, weights, design) rather than combinatorial control subsets.
- **G2**: Max 25 core specs. Secondary claim with fewer variations.
- **Combined target**: ~80 total core specs across both groups.
- **Seed**: 113407

## Inference Plan

- **Canonical**: Clustered at state level (`vce(cluster stfips)`), matching the paper
- **Variants**: HC1 robust (no clustering); Driscoll-Kraay (cross-sectional dependence); region-level clustering (coarser, few-clusters stress test for G1)
- Inference variants are recorded in `inference_results.csv`, not in `specification_results.csv`

## Key Linkage Constraints

- No bundled estimator (simple OLS with FE)
- State and year FE are always included (dropping state FE is an explicit RC variant)
- Weights and clustering are tied to the outcome/sample definition (different age groups use different population weights)
