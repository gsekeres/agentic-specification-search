# Specification Surface: 116531-V1

## Paper: Student Loan Nudges: Experimental Evidence on Borrowing and Educational Attainment (Marx & Turner)

## Baseline Groups

### G1: Effect of loan offers on borrowing (IV)
- **Outcome**: Borrowed indicator / Accepted loan amount
- **Treatment**: Offered a loan (endogenous, instrumented by random assignment)
- **Instrument**: package (random assignment indicator)
- **Estimand**: LATE of loan offer on borrowing behavior
- **Population**: Community college students eligible for federal loans, 2015-16 academic year
- **Baseline spec**: Table 4 Cols 2-3 -- xtivreg2 borrowed/AcceptedAmount (offered=package) + controls, cluster(stratum_code) fe(stratum_code)
- **Key feature**: Stratified RCT with noncompliance; IV approach where random assignment instruments for actual loan offer

### Additional baselines for G1
- AcceptedAmount (dollar amount of accepted loans)

### G2: ITT effect of loan assignment on educational attainment
- **Outcome**: Credits attempted (total), credits earned, GPA, degree completion
- **Treatment**: package (random assignment to loan offer group)
- **Estimand**: ITT (intent-to-treat) effect of loan offer assignment on attainment
- **Population**: Community college students enrolled in Fall 2015
- **Baseline spec**: Table 7 Panel A -- areg crdattm_total package + controls, cluster(stratum_code) a(stratum_code)

### Additional baselines for G2
- credits_total, gpa_total, anydeg

## Core Universe

### G1 Controls axes
- **LOO**: 7 specs (drop each covariate one at a time: EFC, GPA, earned hours, Pell eligibility, independence, outstanding debt, month packaged)
- **Standard sets**: 3 specs (none/minimal/full)
- **Progression**: 5 specs (bivariate, demographics only, academic only, financial only, all)
- **Subset search**: 10 budgeted random draws (seed=116531)

### G1 Design variants
- Difference-in-means (no covariates, no FE -- pure randomization)
- With covariates (OLS with pre-treatment controls, not IV)

### G1 Sample axes
- Trim outcome at 1st/99th percentile
- Trim outcome at 5th/95th percentile

### G1 Fixed effects
- Drop stratum FE (rely on randomization alone)

### G1 Functional form
- log(1 + AcceptedAmount) for the dollar outcome

### G2 Controls axes
- LOO: same 7 covariates as G1
- Standard sets: none, full
- Progression: bivariate, all controls

### G2 Design variants
- Difference-in-means
- Strata FE (include strata FE without partialling)

### G2 Sample axes
- Restrict to enrolled_fall==1 (baseline sample restriction)
- Trim outcome

## Inference Plan (both groups)
- **Canonical**: Cluster SEs at stratum level (matching Stata cluster(stratum_code))
- **Variant**: HC1 robust SEs (no clustering)

## Constraints
- G1 uses IV: controls are linked across first and second stage (partialled out)
- Controls are pre-treatment covariates for precision, not identification (randomized experiment)
- Strata fixed effects are important for valid inference in the stratified design

## Budget
- G1: Max 80 core specs, 10 control subset draws
- G2: Max 50 core specs
- Total planned G1: ~40, G2: ~30
- Seed: 116531

## What is excluded and why
- Table 5 heterogeneity analysis (subgroups by outstanding debt, Pell, freshman status): exploration, not baseline claims
- Table 8 (heterogeneity in attainment): same reason
- Table 9 (year 2 attainment): different time horizon, not baseline claim
- Lee bounds (Table 7): sensitivity analysis, not a core spec
- Familywise p-value corrections: post-processing, not a spec
- Table 6 (complier characteristics): descriptive, not a regression estimate
