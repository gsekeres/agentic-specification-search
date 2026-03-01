# Specification Surface: 133501-V1

## Paper: Teenage Driving, Mortality, and Risky Behaviors (Huh & Reif)

## Baseline Groups

### G1: Discontinuous change in mortality at minimum driving age
- **Outcome**: Motor vehicle accident (MVA) mortality rate per 100,000 person-years (cod_MVA)
- **Treatment**: Reaching minimum legal driving age (MDA)
- **Running variable**: agemo_mda (age in months relative to MDA cutoff)
- **Cutoff**: 0 (at MDA)
- **Estimand**: Sharp RD estimate of the discontinuity in mortality at MDA
- **Population**: US teenagers near minimum legal driving age, 1983-2014
- **Baseline spec**: rdrobust with local linear, triangular kernel, MSE-optimal bandwidth, firstmonth covariate; and OLS parametric with post#c.agemo_mda + firstmonth, triangular weights
- **Key feature**: Aggregated data at age-in-months cell level; death rates per 100,000

### Additional baselines (same claim, different mortality measures)
- cod_any (all-cause mortality)
- cod_sa_poisoning (suicide/accident poisoning -- second key finding of paper)

## Core Universe

### Design variants -- bandwidth selection (from Table A.10)
- MSE-optimal common bandwidth (CCT, baseline)
- MSE-optimal two-sided (different bandwidth left/right)
- CER-optimal common
- CER-optimal two-sided
- Half baseline bandwidth
- Double baseline bandwidth

### Design variants -- polynomial order (from Table A.11)
- Local linear (order 1, baseline)
- Local quadratic (order 2)
- Local cubic (order 3)

### Design variants -- kernel choice
- Triangular (baseline)
- Uniform
- Epanechnikov

### Design variants -- inference procedure
- Conventional (no bias correction)
- Robust bias-corrected (baseline)

### Controls axis
- Drop firstmonth indicator (only available control)

### Sample restrictions
- Males only / Females only (from Table 1)
- MDA = 192 months (16 years) states only (Table A.2)
- Non-192 MDA states only (Table A.2)
- Early period (1983-1998) / Late period (1999-2014) (from Figure 4)

### Functional form
- log(1 + rate) transformation of outcome

### Alternative outcomes (within same claim object)
- All-cause mortality (cod_any)
- External causes (cod_external)
- Suicide/accident poisoning (cod_sa_poisoning)
- Drowning (cod_sa_drowning)
- Other external (cod_extother)

## Inference Plan
- **Canonical**: HC1 robust SEs (from rdrobust bias-corrected inference). No clustering because units are age-month cells.
- No clustering variants (no natural clustering structure in aggregated data)

## Constraints
- No time-varying controls beyond firstmonth (aggregated cell-level data)
- RD design: main variation is in bandwidth, polynomial order, and kernel
- Running variable (age) cannot be manipulated -- McCrary test is formally standard but substantively not relevant
- Data is aggregated: individual-level confounders cannot be added

## Budget
- Max core specs: 80
- No control subset sampling needed
- Total planned: approximately 50-60 specs (bandwidth x polynomial x kernel grid plus sample restrictions)
- Seed: 133501

## What is excluded and why
- Add Health survey outcomes (driver's license, vehicle miles, work, enrollment): different dataset entirely, survey-based not vital statistics
- Placebo cutoff analysis (Table A.8): diagnostic/falsification, not a core estimate
- Year-bin analysis (Figure 4): time heterogeneity decomposition, not a different estimand
- Subgroup analysis by birth month (Tables A.8-A.9): requires confidential data not available
- Multiple hypothesis testing corrections: post-processing, not core
- Table A.12 (parametric OLS): included as a design variant within the RD framework
