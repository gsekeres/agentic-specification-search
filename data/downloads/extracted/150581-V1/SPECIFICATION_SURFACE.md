# Specification Surface: 150581-V1

## Paper: Wage Cyclicality and Labor Market Sorting (Figueiredo)

## Baseline Groups

### G1: Wage cyclicality, job transitions, and skill mismatch

- **Outcome**: lhrp2 (log hourly wage)
- **Treatment**: unempl (unemployment rate) interacted with job transition dummies (EE, UE) and skill mismatch
- **Estimand**: Wage semi-elasticity to unemployment rate, and how it varies by transition type and skill mismatch
- **Population**: Male workers in NLSY79 cohort (1979-2016), age >= 20, working >= 75 hours/month
- **Baseline spec**: Table 2 Col 4 (reghdfe lhrp2 unempl + unempl#dummy1 + unempl#dummy2 + mismatch1w#unempl + triple interactions + controls, absorb(ID industry#year occupation_agg#year), cluster(ID))
- **Focal parameters**: Coefficients on unempl (stayers' cyclicality), unempl#dummy1 (EE job-switcher differential), unempl#dummy2 (UE new-hire differential), and the triple interactions with mismatch

### Additional baselines (same claim, different model specifications)
- Table 2 Col 1: Job transition dummy (no EE/UE split) (baseline__table2_col1)
- Table 2 Col 2: EE/UE split, no mismatch (baseline__table2_col2)
- Table 2 Col 3: EE/UE split + mismatch level, no interactions (baseline__table2_col3)
- Table 2 Col 5: Overqualification/underqualification decomposition (baseline__table2_col5)

## Design and Identification

The paper studies wage cyclicality using the NLSY79 longitudinal panel. Identification relies on within-individual wage variation over the business cycle: individual fixed effects absorb time-invariant worker ability, while industry-year and occupation-year fixed effects absorb sector-specific cyclical patterns. The key variation comes from how wages respond to the aggregate unemployment rate differently for job stayers, employer-to-employer (EE) switchers, and unemployment-to-employment (UE) transitions, and how this cyclicality varies with skill mismatch.

Table 2 Column 4 is the main specification (symmetric mismatch interaction). Column 5 decomposes mismatch into overqualification and underqualification components. Figure 3 presents extensive robustness checks that vary: (1) the definition of job transitions (3-month vs baseline), (2) treatment of recalls, (3) skill-specific weights for mismatch, (4) occupational skill requirements, (5) occupational tenure, (6) regional unemployment rate, and (7) cumulative mismatch.

## Core Universe

### Controls axes
- **LOO**: 5 specs (drop age/agesq, education dummy, time polynomial, time trend, month dummies -- each from Table 2 Col 4)
- **Standard sets**: 5 specs (minimal / baseline col4 / extended with occ tenure / extended with cumulative mismatch / extended with occ skill requirements)
- **Progression**: 4 specs (bivariate / transition only / transition + mismatch / full with interactions)
- **Subset search**: 10 budgeted random draws (seed=150581)

### Sample axes
- Restrict to HOURSM >= 100 (stricter hours filter)
- Restrict to age >= 25 (more established workers)
- Trim log wage at 1st/99th percentile
- Trim log wage at 5th/95th percentile

### Fixed effects axes
- Drop industry-year FE (keep individual + occ-year)
- Drop occupation-year FE (keep individual + industry-year)
- Swap industry-year FE for year FE only

### Data construction axes
- Alternative transition definition: 3-month window (dummy1_alt3, dummy2_alt3) -- Figure 3 robustness
- Alternative transition definition: recalls included (dummy1_alt4, dummy2_alt4) -- Figure 3 robustness
- Alternative mismatch measure: skill-specific weights (mismatch1w_w) -- Figure 3 robustness
- Regional unemployment rate (unempl_region) instead of aggregate -- Figure 3 robustness

## Inference Plan
- **Canonical**: Cluster SEs at individual level (ID), matching paper's cluster(ID)
- **Variant 1**: HC1 robust SEs (no clustering)
- **Variant 2**: Two-way clustering by individual and year

## Constraints
- Control-count envelope: [6, 14]
- No linkage constraints (single-equation panel FE with high-dimensional FE absorbed via reghdfe)
- The interaction structure (unempl x transition type x mismatch) is the core model -- robustness variations preserve this structure while changing control variables, mismatch definitions, or transition definitions
- Sample always restricted to HOURSM >= 75 & age >= 20
- Unemployment rate rescaled to [0,1] (divided by 100) before entering regressions

## Budget
- Max core specs: 70
- Max control subset specs: 10
- Total planned: ~46
- Seed: 150581

## What is excluded and why
- **Table 2 Col 5 (overqualification/underqualification decomposition)**: Included as a baseline variant since it is a different parameterization of the same claim object
- **Figure 1 (separation hazard model)**: Different outcome (job separation, not wages) and different estimator (cloglog). Not the wage cyclicality claim.
- **Figure 2 (nonparametric wage semi-elasticity by mismatch percentile)**: These are derived from the same Table 2 Col 4 regression via `lincom` -- not separate specifications but post-estimation calculations. The figures plot the implied semi-elasticity at different mismatch percentiles.
- **Figure 3 robustness checks**: The underlying regression variations (3-month transitions, recalls, weighted mismatch, skill requirements, occupational tenure, regional unemployment, cumulative mismatch) are included as `rc/data/*` specifications.
- **Appendix tables/figures**: Supplementary analyses with different sample selection criteria (Figure D.1). Not the main claim.
