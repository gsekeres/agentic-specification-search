# Specification Surface: 114879-V1

## Paper: Davis, Fuchs & Gertler -- "Cash for Coolers: Evaluating a Large-Scale Appliance Replacement Program in Mexico" (AEJ: Economic Policy)

This paper evaluates a Mexican appliance replacement program (primarily refrigerators) using difference-in-differences with two-way fixed effects on household-level electricity billing data.

---

## Baseline Groups

### G1: Effect of refrigerator replacement on electricity consumption

- **Design**: Difference-in-differences (TWFE with double-demeaning via reg2hdfe)
- **Panel unit**: Household x calendar-month (rpu x copy x moy)
- **Panel time**: Month-of-sample (month)
- **FE structure**: Household x calendar-month FE + county x month FE
- **Outcome**: usage (electricity consumption in kWh per billing period)
- **Treatment**: rrefr (refrigerator replacement indicator, turns on post-replacement)
- **Estimand**: ATT of refrigerator replacement on electricity consumption
- **Target population**: Mexican households in the C4C program and matched control households
- **Clustering**: County level

The paper's Table 3 presents the main regressions with three columns varying the FE structure:
1. Household x calendar-month FE + month FE (simpler)
2. Household x calendar-month FE + county x month FE (preferred)
3. Same as (2) plus AC x summer interaction

The only time-varying regressors beyond the treatment are the AC replacement indicator (rAC) and its summer interaction (rAC_summer). The paper runs the same specification with three different control groups: random, location-matched, and usage-matched.

---

## Baseline Specs

- **Table3-Col2-Random**: rrefr + rAC with hhXm and CxM FE, random controls, cluster(county)
- **Table3-Col3-Random**: rrefr + rAC + rAC_summer with hhXm and CxM FE, random controls, cluster(county)

---

## Core Universe

### Design variants
- TWFE is the only feasible estimator given the double-demeaning approach and massive FE dimensions

### RC axes
- **Controls**: LOO drops (rAC, rAC_summer), control set progressions
- **FE structure**: Simple (hhXm + month) vs preferred (hhXm + CxM)
- **Data construction/control group**: Random, location-matched, usage-matched control groups (the paper's primary revealed robustness axis)
- **Sample restrictions**: Drop transition months, no control households, summer/winter only
- **Sample outliers**: Trim extreme usage values, drop usage > 200,000
- **Functional form**: Log and asinh transformations of usage

### Excluded from core
- Census-based analysis (census.do, avgefficiencies.do) -- separate analysis
- National-level energy accounting (national39.do) -- aggregate analysis, not the household DiD

---

## Constraints

- Very few time-varying controls (only rAC and rAC_summer beyond rrefr)
- Control-count envelope: 1-2
- The main robustness axis is the choice of control group (random vs matched)
- FE dimensions are very large (household x calendar-month), requiring specialized estimation (reg2hdfe)

---

## Inference Plan

- **Canonical**: Cluster at county level (matching the code)
- **Variants**: State-level clustering (coarser), HC1 robust

---

## Budget

- Total core specs: up to 70
- No controls-subset sampling needed (very few controls)
- The main combinatorial axis is control group x FE structure x sample restrictions
