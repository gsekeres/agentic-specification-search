# Specification Surface: 112749-V1

## Paper
Hornbeck, R. & Naidu, S. (2014). "When the Levee Breaks: Black Migration and Economic Development in the American South." *AER*, 104(3): 963-990.

## Design Classification
**Primary**: Difference-in-differences (continuous treatment intensity)
**Secondary**: Panel fixed effects

## Baseline Groups

### G1: Black Population Share (Table 2)
- **Outcome**: lnfrac_black (log Black population share)
- **Treatment**: f_int_{year} (flood intensity x post-flood year dummies)
- **Estimand**: Within-county change in Black population share caused by 1927 flood, at each post-flood decade
- **Population**: Southern cotton-belt counties (>=10% Black, >=15% cotton in 1920)
- **Baseline spec**: Table 2 Col 1, coef(f_int_1930) = -0.156, se = 0.032, N = 2604
- **Focal parameter**: f_int_1930 coefficient (immediate post-flood decade effect)

### G2: Farm Equipment Value (Table 4)
- **Outcome**: lnvalue_equipment (log value of farm equipment)
- **Treatment**: f_int_{year} (flood intensity x post-flood year dummies)
- **Estimand**: Within-county change in farm capital intensity caused by flood-induced Black out-migration
- **Population**: Same cotton-belt sample
- **Baseline spec**: Table 4 Col 2, coef(f_int_1940) = 0.440, se = 0.099, N = 2170
- **Focal parameter**: f_int_1940 coefficient (expected peak mechanization response)

## Why Two Baseline Groups
The paper makes two distinct headline claims:
1. The flood caused persistent Black out-migration (G1, Table 2)
2. The resulting labor scarcity led to agricultural mechanization (G2, Table 4)

Tables 3 and 5 are secondary analyses (alternative outcomes like tractors, mules/horses, farmland) that we treat as exploration, not separate baseline groups.

## What Is Included

### Control Progression (both groups)
The paper's control structure has clear semantic blocks:
- **Bivariate**: Treatment only (+ county FE)
- **Geography**: State-year FE + geographic controls (crop suitability x year, distance to MS river x year, coordinates x year, ruggedness x year)
- **Lags**: + lagged outcome variables (up to 4 lags)
- **New Deal**: + New Deal spending controls
- **Full**: Geography + Lags + New Deal (= paper's extended specification)

We run: bivariate (no controls), geography only, lags only, new deal, baseline (geo+lags), extended (geo+lags+ND).

### Fixed Effects Variations
- Baseline: county FE + state-year FE (via dummies in controls)
- Drop state-year FE: county FE only

### Sample Variations
- Drop first post-flood period (1930): tests whether immediate effect drives results
- Drop last period (1970): tests sensitivity to long-run extrapolation
- Short window (1930-1950 for G1): concentration on immediate post-flood decades
- Outcome trimming at 1/99 percentiles

### Functional Form
- Level outcome (instead of log): tests whether log transformation drives results
- Binary flood indicator (instead of continuous intensity): tests dose-response linearity

### Weights
- Unweighted (baseline uses county area weights)
- Population-weighted (1920 population as alternative to area)

### Inference Variants (separate table)
- HC1 robust SE (no clustering)
- State-level clustering (coarser than county)

## What Is Excluded (and Why)
- **Tables 3, 5**: Alternative outcomes (tractors, mules/horses, farmland, land values). These change the outcome concept and are exploration, not core robustness for G1 or G2.
- **Table 1**: Cross-sectional pre-differences. These are balance checks / first-stage evidence, not DiD estimates.
- **Conley spatial SE**: The paper uses Conley (1999) spatial SE in some specifications. We do not have the spatial SE package available, so these are omitted.
- **DML/DR-DiD**: Not used in the paper; not applicable here.
- **Staggered DiD estimators**: Treatment timing is not staggered (all treatment from a single event in 1927), so modern staggered-adoption estimators are not relevant.

## Budgets and Sampling
- Target: 50+ specifications across both groups
- G1: ~30 core specs (baseline + ~14 RC variants x 1-2 focal params)
- G2: ~30 core specs (baseline + ~14 RC variants x 1-2 focal params)
- Full enumeration: feasible (no combinatorial control-subset search needed)
- Seed: 112749

## Key Linkage Constraints
- County FE are always included (defining feature of within-county DiD)
- When state-year FE are included, they are via dummy variables in the control set (not absorbed separately)
- Treatment variables (f_int_{year}) are always included as a block for the relevant years
- Weights are area weights by default (county_w = 1920 county area in acres)
