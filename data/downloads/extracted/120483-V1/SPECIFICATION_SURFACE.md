# Specification Surface: 120483-V1

## Paper: Malaria and the Historical Roots of Slavery (Esposito, 2021, AEJ: Applied)

## Baseline Groups

### G1: Effect of malaria ecology on slavery intensity

- **Outcome**: slaveratio (share of slaves in county population)
- **Treatment**: MAL (Malaria Stability Index from Kiszewski et al. 2004)
- **Estimand**: Conditional association of malaria ecology with slavery prevalence, controlling for geography and crop suitability, with state FE
- **Population**: US counties in 1790 and 1860
- **Design**: Cross-sectional OLS with state fixed effects and Conley spatial standard errors

### Baseline specifications

The paper's Table 1 presents the core claim across 7 columns:
- Cols 1-3: 1790 county data (all states), progressively adding controls
- Cols 4-5: 1860 county data (all states)
- Cols 6-7: Slave states only (1790, 1860)

The primary baseline is **Table 1 Col 3** (1790, all states, full controls: coefficient 0.065, Conley SE 0.030 at 100km) and **Table 1 Col 5** (1860, all states, full controls: coefficient 0.192, Conley SE 0.042 at 100km). The 1860 full-controls result (Col 5) is chosen as the main baseline because it uses the larger dataset and the most complete control set.

An additional baseline row captures Table 1 Col 5 (1860 sample) as the anchor.

### Control variables

The paper reveals two control sets that build progressively:

1. **Crop suitability** (1790): cotton, rice, sugar, tea, tobacco, indigo (6 vars)
   **Crop suitability** (1860): cotton, coffee, rice, sugar, tea, tobacco, indigo (7 vars)
2. **Geography/distances**: ELEV, prec, temp, DISTRIV, DISTSEA, lat_deg, long_deg, lat_long (8 vars)

Total controls for 1860 full specification: 15.

## Core Universe

### Controls axes
- **LOO**: 15 specs (drop each control one at a time from the 1860 full set)
- **Standard sets**: 4 specs (none / crop only / geo only / full)
- **Progression**: 3 specs (bivariate / crop suitability / crop + geography)
- **Subset search**: 15 budgeted random draws (seed=120483)

### Sample axes
- **Slave states only**: restricts to slave_state==1 (matches Table 1 Cols 6-7)
- **1790 data**: uses county_1790.dta instead of county_1860.dta
- **1860 data**: uses county_1860.dta (default baseline)
- **Trim outcome 1/99**: winsorize slaveratio at 1st and 99th percentile
- **Trim outcome 5/95**: winsorize slaveratio at 5th and 95th percentile

### Fixed effects axes
- **Drop state FE**: no absorption (bivariate-like, but different from controls/none because it removes the FE)

### Functional form axes
- **asinh(slaveratio)**: inverse hyperbolic sine handles zeros
- **log(1 + slaveratio)**: alternative zero-handling transform

## Inference Plan

- **Canonical**: Conley spatial SEs at 100km (primary inference in paper)
- **Variant 1**: Conley SEs at 250km
- **Variant 2**: Conley SEs at 500km
- **Variant 3**: State-clustered SEs

The paper reports all four SE variants for every specification in Table 1, making inference variation a revealed dimension. However, since spatial SEs are non-standard in pyfixest, the runner may fall back to state-clustered SEs as the canonical choice with HC robust as a variant.

## Constraints

- **Control-count envelope**: [0, 15] (paper ranges from 0 to 15 controls)
- **No linkage constraints**: single-equation OLS
- **All specifications maintain the same treatment concept** (Malaria Stability Index)
- **Functional form**: slaveratio is a share bounded [0,1]; log/asinh transforms are feasible but change coefficient interpretation

## Budget

- Max core specs: 80
- Max control subset specs: 15
- Seed: 120483

## What is excluded and why

- **Table 2** (crop interactions with malaria): changes the estimating equation to include interaction terms, making it a different claim about moderating effects of crop suitability
- **Table 3** (pro-slavery convention votes and presidential elections): different outcome concept (political attitudes, not slavery intensity)
- **Tables 5-7** (state-level panel with DiD identification): different design family (difference-in-differences), different unit of observation (states), different time period (1630-1750 colonial era). Would require a separate baseline group with design_code="difference_in_differences"
- **Tables 8-10** (slave prices): different outcome concept (price levels, not slavery prevalence)
- **Americas comparison** (Table 2, Panel B: US + Brazil + Cuba regions): different target population and unit of observation
- **IV estimates** (Table 7 Panel C, falciparum instrumented): different design family
- **Lasso/post-lasso** specifications in appendix: alternative variable selection procedure, would be explore/estimation variant
