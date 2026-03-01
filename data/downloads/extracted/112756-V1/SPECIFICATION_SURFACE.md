# Specification Surface: 112756-V1

## Paper: The Role of Local Officials in New Democracies: Evidence from Indonesia (Martinez-Bravo, 2014)

## Baseline Groups

### G1: Effect of appointed (kelurahan) vs elected (desa) village heads on Golkar vote share
- **Outcome**: GolkarFirst (Golkar party won plurality in 1999 election, binary)
- **Treatment**: kelurDum (kelurahan indicator: 1 = appointed village head, 0 = elected desa head)
- **Estimand**: Within-district difference in Golkar vote share between appointed and elected villages
- **Population**: Indonesian villages matched across PODES 1996/2000/2003, excluding districts with <5 kelurahan or <5 desa
- **Baseline spec**: Table 2 Col 5 OLS: reg GolkarFirst kelurDum $geography $religion $facilities kab_FE, cluster(kab)
- **N**: ~43,000 villages

### Additional baselines
- **Probit full model**: Table 2 Col 9 (probit with same controls and FE)

## Core Universe

### Controls axes
- **LOO**: 20 specs (drop each control variable or polynomial group one at a time)
- **Standard sets**: 4 specs (no controls / geography only / geo+religion / full)
- **Progression**: 5 specs (following Table 2 column progression: raw diff -> FE only -> geo -> geo+relig -> geo+relig+facilities)
- **Subset search**: 15 budgeted random draws (seed=112756)

### Sample axes
- Trim GolkarFirst (vote share) at 1st/99th percentile
- Trim at 5th/95th percentile

### Fixed effects axes
- Drop district FE entirely (raw cross-sectional comparison)
- Add sub-district (kecamatan) FE instead of district FE

### Functional form axes
- Use continuous Golkar vote share instead of binary indicator
- Probit instead of OLS (both reported in paper)

## Inference Plan
- **Canonical**: District-clustered SEs matching paper's vce(cluster kab)
- **Variant 1**: HC1 robust SEs (no clustering)
- **Variant 2**: Sub-district (kecamatan) clustered SEs

## Constraints
- Control-count envelope: [0, 23] (paper shows specs from bivariate to full controls)
- No linkage constraints (single-equation OLS)
- District FE are maintained in most specifications (identification relies on within-district variation)
- lpopulation_1996 variables include polynomial terms that should move together

## Budget
- Max core specs: 100
- Max control subset specs: 15
- Total planned: ~60-70
- Seed: 112756

## What is excluded and why
- Propensity score matching (Table 2 Panel C): complex procedure requiring psmatch2, treated as design alternative but may not be feasible in Python
- Table 3 (mechanisms: conflict, army/police, mining, facilities change, funds change, IDT): these are explore/* outcome changes, not the main claim
- Table 4 (alignment effects): different claim object (effect of alignment, not kelurahan status)
- Appendix tables on detailed electoral outcomes: explore/*
- Figure 1 (computed from Appendix Table 12): visualization of combined coefficients
