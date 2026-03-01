# Specification Surface: 112629-V1

## Paper: Children's Healthcare Utilization and Parental Immigration Status (Garcia Perez)

## Baseline Groups

### G1: Effect of parental immigration status on children's healthcare utilization
- **Outcome**: visityrno_c (child visited doctor in last year, binary)
- **Treatment**: forcit_pc (foreign parent, citizen child) and forncit_pc (foreign parent, non-citizen child)
- **Estimand**: Average marginal effect of parental foreign-born/citizenship status on child healthcare use
- **Population**: Children in US households, IHIS survey data
- **Baseline spec**: Full logit model with race controls, survey-weighted, PSU-clustered
- **Analysis method**: svy: logit with marginal effects

### Additional baselines (same claim, different outcome measures)
- **goodhealth**: Perceived good/excellent health status (binary)
- **usualplace**: Has usual place of care (binary)

These are the paper's three main health outcome variables, each analyzed with the same specification structure across three separate do-files.

## Core Universe

### Design alternatives
- **LPM (OLS)**: Linear probability model instead of logit -- same controls, same sample

### Controls axes
- **LOO**: 11 specs (drop each control group one at a time: medicaid, uninsured, grandchild, nusualpl, famsize, unmarried, parent age, poverty dummies, employment dummies, education dummies, race dummies)
- **Standard sets**: 5 specs (none / basic / demographics only / insurance only / full without race)
- **Progression**: 6 specs (build up from bivariate to full, following the paper's progression)
- **Subset search**: 10 budgeted random draws (seed=112629)

### Treatment definition axes
- foreign_p only (binary foreign parent, no citizenship interaction)
- foreign_p x ncitizen_c interaction (paper's intermediate specification)
- Region of birth dummies (latinoamerica_p, africa_p, asia_p, etc.)

### Weights axes
- Unweighted logit (drop survey weights)

### Fixed effects axes
- Drop year FE
- Drop region FE

## Inference Plan
- **Canonical**: Survey-design SEs with PSU clustering, strata, and weights (svy:)
- **Variant 1**: PSU clustering without strata
- **Variant 2**: HC1 robust with weights only

## Constraints
- Control-count envelope: [2, 23]
- No linkage constraints (single-equation logit)
- Survey design requires PSU-level clustering in canonical inference
- Data not provided in package (IHIS microdata requires separate access) -- analysis must construct data from raw IHIS files

## Budget
- Max core specs: 80
- Max control subset specs: 10
- Total planned: ~55-60
- Seed: 112629

## What is excluded and why
- Race x immigration interaction models (Tables with forcit_pc##hispanic_c etc.): these are explore/* heterogeneity analyses, not the main claim
- Region-of-birth with citizenship interactions (very many parameters): explore/*
- Odds ratio reporting: same regression, different presentation
- Lincom comparisons across race groups: post-processing of interaction models
