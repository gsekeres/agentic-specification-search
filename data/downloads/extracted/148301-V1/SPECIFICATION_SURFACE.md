# Specification Surface: 148301-V1

## Paper Overview

This paper studies how foreign market access and tax rates affect the location of export platforms by US multinational enterprises (MNEs). The data is a country-by-sector-by-year panel of US MNE activities abroad from the Bureau of Economic Analysis (BEA), covering 1999-2013. The primary estimator is GLM fractional logit with sector-by-year fixed effects and clustering at the country level. A secondary analysis examines the profit equation using OLS (reghdfe).

## Baseline Groups

### G1: Foreign Market Access and Export Platform Location

**Claim object**: Foreign market access (lfma) increases the share of MNE affiliate sales that are export platforms (foreign + US sales / total sales). This is the paper's main finding (Table 2).

**Baseline spec**: Table 2, Column 4 -- GLM fractional logit of ep on lfma, taxr, haven, eoi_enf, dtc_enf, lrgdp, dtc_num with sector-year FE (i.kt), clustered at country (i). Marginal effects (dydx) are the reported coefficients.

**Why GLM fractional logit**: The outcome (export platform share) is bounded in [0,1]. The paper uses GLM with logit link and binomial family as its primary estimator. OLS via reghdfe appears in columns 7-8 as a robustness check.

### G2: Profit Shifting Through Export Platforms

**Claim object**: MNE affiliates in tax havens that have higher export platform shares earn disproportionately higher profits, consistent with profit shifting. Table 4.

**Baseline spec**: Table 4, Column 1 -- OLS (reghdfe) of log(profit) on ep_haven (export platform share x haven indicator), with controls for lfma, ep, taxr, haven, treaty variables, lrgdp, lemp, leqpmt, and sector-year FE.

## What Is Included and Why

### G1 core universe:

1. **Control progression** (Table 2 columns 1-4): The paper reveals 4 control configurations progressing from lrgdp only to the full set with haven.
2. **Leave-one-out controls**: Drop each of the 6 controls individually.
3. **Random control subsets**: 10 random draws from the 6-control pool.
4. **Sample splits**: Manufacturing vs. services (Table 3), haven vs. non-haven (Table 2 cols 5-6), big-5 havens vs. Caribbean havens.
5. **Estimator switch**: OLS (reghdfe) as alternative to GLM fractional logit (Table 2 cols 7-8).
6. **FE variants**: Sector only, year only, country + sector-year, country-year.
7. **Additional controls**: lemp, leqpmt (used in profit equation but not in baseline ep equation).
8. **Outlier trimming**: Trim ep at 1/99 and 5/95 percentiles.
9. **Sample restrictions**: Drop imputed ep values.

### G2 core universe:

1. **LOO controls**: Drop each non-essential control individually.
2. **Outcome transforms**: Log profit (baseline), GPML Poisson (Table 4 col 2), cube root transform (Table 4 col 3).
3. **Sample restrictions**: Positive-profit only, drop imputed profits.
4. **FE variants**: Add country FE.

## What Is Excluded and Why

- **IPW/AIPW/matching**: Not applicable -- treatment (lfma) is continuous, not binary.
- **First-difference/long-difference**: The treatment variable (lfma) has primarily cross-sectional variation. Within-country variation over time is limited, making differencing estimators less informative.
- **Correlated random effects**: Not a natural comparison given the fractional outcome.
- **Diagnostics**: Not in core. Standard panel diagnostics (Hausman, serial correlation) are less relevant for GLM fractional logit.
- **Sensitivity/exploration**: Not in core universe.

## Inference Plan

- **Canonical**: Cluster at country level (matching the paper throughout).
- **Variants**: Two-way clustering (country x sector), heteroskedasticity-robust only (no clustering).

## Budgets and Sampling

- **G1**: ~80 specifications. The control pool is small (6 controls), so LOO and progression cover most of the space. Random subset draws supplement structured specs.
- **G2**: ~30 specifications. The profit equation has a fixed control set; variation comes from LOO, outcome transforms, and sample restrictions.
- Combined target: ~80+ specifications across both groups.

## Key Linkage Constraints

- The GLM fractional logit reports marginal effects (dydx), which differ from raw index coefficients. The runner must compute and store marginal effects for comparability with the paper.
- The OLS alternative (reghdfe) is a genuine estimator switch, not just a robustness check on inference. It changes the functional form assumption.
- For G2, ep_haven = ep * haven is the focal parameter. When haven is dropped from controls, ep_haven becomes undefined. The runner should only drop haven when also dropping ep_haven and switching to a specification without the interaction.
