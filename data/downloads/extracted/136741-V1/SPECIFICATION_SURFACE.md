# Specification Surface: 136741-V1

## Paper: Historical Lynchings and the Contemporary Voting Behavior of Blacks (Williams)

## Baseline Groups

### G1: Effect of historical black lynching rate on contemporary black voter registration

- **Outcome**: Blackrate_regvoters (black registered voters as percentage of black voting-age population)
- **Treatment**: lynchcapitamob (black lynchings 1882-1930 per 10,000 black population in 1900)
- **Estimand**: Conditional association interpreted as persistent causal effect of historical racial violence on contemporary political participation, under selection-on-observables
- **Population**: Counties in six Southern US states (AL, FL, GA, LA, NC, SC) with complete covariate data (N=267 after sample restriction)
- **Baseline spec**: Table 2 Col 1 / Table 3 Col 1 (regress Blackrate_regvoters lynchcapitamob + 7 historical controls + state FE)
- **Baseline coefficient**: -0.469 (SE = 0.144)

### Why one baseline group

The paper has one central claim: that historical black lynching rates are negatively associated with contemporary black voter registration. Tables 2-5 all test variants of this same relationship. Table 2 contains falsification exercises (using white lynchings and white voter registration as placebos), Table 3 adds slavery controls, Table 4 uses count outcomes, and Table 5 adds contemporary controls. Table 8 explores heterogeneity by education/earnings/church membership. The falsification and heterogeneity analyses are not separate main claims -- they support the single main claim.

## Core Universe

### Controls axes

The paper reveals a rich control structure through its table progression:

- **Historical controls (baseline)**: Black_share_illiterate, initial (county formation year), newscapita (newspapers per capita 1840), farmvalue (average farm value 1860), sfarmprop1860 (proportion small farms), landineq1860 (farmland inequality), fbprop1860 (proportion free blacks 1860). These 7 controls form the baseline set.
- **Extended controls revealed**: share_slaves (Table 3), contemporary controls (Table 5): Black_beyondhs, Black_avgage, Black_Earnings, share_maritalblacks, incarceration_2010, pollscapita.
- **Full set**: historical + contemporary + slaves = up to ~15 controls.
- **Control pool**: Black_share_illiterate, initial, newscapita, farmvalue, sfarmprop1860, landineq1860, fbprop1860, share_slaves, Black_beyondhs, Black_avgage, Black_Earnings, share_maritalblacks, incarceration_2010, pollscapita.

Axes:

- **LOO**: 7 specs (drop each baseline historical control one at a time)
- **Standard sets**: 4 specs (no controls; historical + slaves; historical + contemporary; full kitchen sink)
- **Progression**: 9 specs (build up controls from bivariate through full set, following paper's table structure)
- **Subset search**: 20 budgeted random draws from the 14-control pool (seed=136741, stratified by subset size in range [0, 15])

### Sample axes

- Trim Blackrate_regvoters at 1st/99th percentile
- Trim Blackrate_regvoters at 5th/95th percentile
- Drop observations where Blackrate_regvoters > 100 (registration rates above 100% are implausible; paper addresses this in Appendix Table B3)
- Cap Blackrate_regvoters at 100 (paper's alternative treatment of outlier rates in Table B3)

### Fixed effects axes

- Drop state FE (no absorption)

### Treatment variable construction axes (data construction RC)

The paper tests alternative lynching rate constructions in Appendix Table B2 and B4:
- lynchcapitamob1910: black lynching rate using 1910 population denominator
- lynchcapitamob1920: using 1920 population denominator
- lynchcapitamob1930: using 1930 population denominator
- lynchcapitasteve: alternative lynching data source (Equal Justice Initiative / Stevenson data)

These all estimate the same conceptual parameter (effect of historical black lynching intensity) with different measurement choices.

### Functional form axes

- asinh(Blackrate_regvoters): inverse hyperbolic sine transform handles zeros/outliers
- log(1 + Blackrate_regvoters): alternative zero-handling transform

## Inference Plan

- **Canonical**: IID/default OLS standard errors. The paper's Maindo.do uses plain `regress` without robust or cluster options for the main tables (Tables 2-5, 8). Note: Createmain.do runs one regression with `cluster(fips)` before collapsing data to cross-section, but the main analysis tables do not use clustered SEs.
- **Variant 1**: HC1 robust SEs (heteroskedasticity-robust)
- **Variant 2**: Cluster at state level (State_FIPS) -- only 6 clusters, so interpret with caution

## Constraints

- Control-count envelope: [0, 15] (paper shows bivariate through full set of ~15 controls)
- No linkage constraints (single-equation OLS)
- Treatment variable: lynchcapitamob in baseline; alternative constructions are data-construction RC
- All specifications maintain the same outcome concept (black voter registration rate as percentage)

## Budget

- Max core specs: 80
- Max control subset specs: 20
- Estimated total core specs: ~53 (7 LOO + 4 sets + 9 progression + 20 subset + 4 sample + 1 FE + 4 data/treatment + 2 functional form + 1 baseline = ~52)
- Seed: 136741

## What is excluded and why

- **Table 4 (register_black as level outcome)**: Different outcome definition (number rather than rate). The claim object is the rate, not the count. This would be an explore/variable_definitions variant.
- **Table 8 (heterogeneity interactions)**: Interaction terms change the estimand from main effect to conditional effect. These belong in explore/heterogeneity.
- **White voter registration outcome (Table 2 cols 3-4)**: Falsification exercise, not main claim. Belongs in diag/falsification.
- **White lynching rate treatment (Table 2 col 2)**: Falsification exercise.
- **Table 6 (Migration)**: Uses different dataset and mechanism analysis.
- **Table 7 (Polling locations)**: Uses different dataset (polllocation.dta), different unit of analysis (census tract level).
- **Oster psacalc bounds (Table B5)**: Sensitivity analysis, belongs in sens/unobserved_confounding.
- **Historical voter registration (Table B1)**: Historical outcome, different time period.
- **Southern Focus Poll survey (Figure 10)**: Different data source and outcome.
