# Specification Surface: 112370-V1

## Paper: Elections, Capital Flows and Politico Economic Equilibria (Chang, 2009)

## Baseline Groups

### G1: Effect of capital flow disruptions on leftist election outcomes
- **Outcome**: y_exe_left (strict: leftist wins AND replaces non-leftist) or _y_exec_left (broad: any leftist win)
- **Treatment**: dffo12_X_lagfdi_gdp (change in US federal funds rate x lagged net FDI/GDP)
- **Estimand**: Effect of capital flow shocks (US rate changes interacted with FDI exposure) on probability of leftist election victory
- **Population**: 18 Latin American countries, election-year observations, 1976-2004
- **Baseline spec**: Table 2 Col 2 (probit y_exe_left ~ democ lgrgdpwork dffo12_X_lagfdi_gdp, robust)
- **Baseline coefficient**: 0.117 (p=0.048)
- **N**: 101

### Additional baselines
- _y_exec_left (broader definition of leftist transition): Table 1 Col 2, coef=0.071 (p=0.045)

## Key challenge: very small sample and very few controls

This paper has N=101 observations (election-year events in 18 Latin American countries over ~30 years). The analysis uses probit with only 2 controls (democracy score and lagged GDP growth). This severely limits the scope of a specification search:

- No additional controls are available in the provided data beyond the 3 regressors
- The probit model is the native specification; LPM (OLS) is a natural design alternative
- Sample trimming is limited by the small N
- Country and time subsampling is feasible but noisy

## Core Universe

### Design alternatives
- **LPM (OLS)**: Linear probability model as alternative to probit

### Controls axes
- **LOO**: 2 specs (drop democ, drop lgrgdpwork)
- **Standard sets**: 2 specs (no controls; add dffo12 level separately)

### Sample axes
- Drop Venezuela (outlier in FDI and political instability)
- Drop Haiti (outlier in democracy scores)
- Restrict to post-1985 (Third Wave of democratization)
- Restrict to post-1990

### Treatment/functional form axes
- Use dffo12 (level of Fed funds change) instead of interaction term
- Use lagged FDI/GDP alone as treatment
- Switch between broad (_y_exec_left) and strict (y_exe_left) outcome definitions

## Inference Plan
- **Canonical**: HC1 robust SEs (matching Stata vce(robust))
- **Variant**: Cluster at country level (18 clusters - very few, may be unreliable)

## Constraints
- Control-count envelope: [0, 2] (paper uses exactly 2 controls in all specs)
- No linkage constraints (single-equation probit)
- N=101 limits all sample restriction axes

## Budget
- Max core specs: 30
- Total planned: ~15-20
- Seed: 112370

## What is excluded and why
- Marginal effects calculations: these are transforms of the baseline probit, not separate specifications
- Table 1 regressions with dffo12 (level) instead of interaction: included as rc/form variant
- No additional control variables available in the dataset
- Country fixed effects: infeasible with N=101 and 18 countries in probit
