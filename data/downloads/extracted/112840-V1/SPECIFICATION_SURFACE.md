# Specification Surface: 112840-V1

## Paper Overview
- **Title**: Inequality, Leverage, and Crises (Kumhof, Ranciere, & Winant, 2015, AER)
- **Design**: Panel fixed effects
- **Data**: 22 OECD countries, 1975-2005 annual panel
- **Key tables**: Table O1 (Index 1 regressions), Table O2 (Index 2 regressions), Table O3 (interest rate elasticity)
- **Note**: Paper also includes a Fortran structural calibration model, which is excluded from the specification surface.

## Baseline Groups

### G1: Financial Liberalization and Public Debt Growth

**Claim object**: Financial liberalization (measured by GDP-weighted average of the Abiad-Detragiache-Tressel liberalization index) is associated with growth in real public debt across OECD countries. All regressions are in first differences of logs.

**Baseline specifications**:
- Table O1, Col 1 (minimal): `changerealdebt ~ lag_d_ave_finindex1 + lag_debtgdp + lagchangerealgdp | country FE`
- Table O1, Col 5 (extended): adds `lag_emu_dum + size_product1 + d_dep_ratio_old + changetop1incomeshare`
- Table O1, Col 6 (Gini variant): replaces `changetop1incomeshare` with `lag_changeave_gini_gross`, drops Korea
- Country FE, robust SE throughout
- Focal coefficient: `lag_d_ave_finindex1` (lagged change in financial liberalization index)

## RC Axes Included

### Controls
- **Progressive build-up**: Matching paper's Column 1 through Column 5/6
- **LOO**: Drop each optional control (EMU dummy, size-finlib interaction, dependency ratio, inequality measure)
- **Swap inequality measure**: Replace top 1% income share with gross Gini coefficient
- **Random control subsets**: Stratified by subset size from the optional pool

### Sample restrictions
- **Outlier trimming**: y at [1,99], [5,95]; Cook's D
- **Country exclusions**: Drop Korea (no Gini), drop Greece (debt crisis outlier), drop small countries
- **Period restrictions**: 1980-2005, 1975-2000 (end-of-sample sensitivity)

### Functional form / treatment
- **Alternative liberalization index**: Use Index 2 (Chinn-Ito) instead of Index 1 (Table O2 regressions)
- **Level change in debt**: Instead of log change, use level change

### Fixed effects
- **Add year FE**: Country FE only in baseline; adding year FE is important for macro panels
- **Add region FE**: Regional groupings instead of or in addition to country FE

### Preprocessing / data construction
- **Unweighted index**: Use unweighted average of liberalization index instead of GDP-weighted
- **Alternative debt measure**: Explore different debt series if available

### Joint variations
- Index choice x control set combinations
- Sample period x control set combinations

## What Is Excluded and Why

- **Table O3 (interest rate elasticity)**: Different outcome (percent change in interest rate) and a fundamentally different regression with interaction terms. Could be a separate baseline group but is supplementary to the main debt-growth claim.
- **Fortran structural model**: Not amenable to specification search.
- **Figure 2 (fitted values plot)**: Descriptive, not a regression specification.

## Budgets and Sampling

- **Max core specs**: 70
- **Max control subsets**: 25
- **Seed**: 112840
- **Sampling**: Stratified by subset size. 5 optional controls means 2^5 - 1 = 31 possible subsets, but with the mandatory pair and mutual exclusion of inequality measures, the effective space is smaller. Near-exhaustive enumeration plus random draws.

## Inference Plan

- **Canonical**: HC1 robust SE (Stata vce(robust) in xtreg fe)
- **Variants**: Cluster by country (22 clusters -- small, may be unreliable), HC3 (small-sample), Driscoll-Kraay (cross-sectional dependence, important for macro panels with T > N)

## Key Constraints

- `lag_debtgdp` and `lagchangerealgdp` are mandatory controls (included in all paper columns)
- `changetop1incomeshare` and `lag_changeave_gini_gross` are mutually exclusive (different inequality measures, not used simultaneously)
- Small N (22 countries) limits the statistical power and reliability of clustering
