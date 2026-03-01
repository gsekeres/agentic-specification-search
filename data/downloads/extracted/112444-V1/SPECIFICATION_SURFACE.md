# Specification Surface: 112444-V1

## Paper Overview
- **Title**: From Financial Crash to Debt Crisis (Reinhart & Rogoff, 2011, AER)
- **Design**: Panel fixed effects (classified), but actual implementation is pooled OLS with robust SE on stacked country-year panels
- **Software**: TSP (Time Series Processor)
- **Data**: 70 countries, up to 200+ years (1824-2009, 1900-2009, 1946-2009 subperiods)

## Baseline Groups

### G1: Banking Crisis Contagion (Equations 1 and 5)

**Claim object**: Cross-country contagion predicts banking crises. The paper regresses banking crisis incidence on 3-year moving averages of banking and external debt crises in other countries, plus a financial-center crisis indicator (center). Development-status dummies serve as group controls.

**Baseline specifications**:
- Eq1: `bank ~ develop1 + develop2 + bank_move + debt_move + center` (robust SE)
- Eq5: Same with `public` debt ratio added
- Focal coefficient: `center` (financial center contagion effect)
- N = 7810 (71 countries x 110 years for 1900-2009 period)
- Inference: HC2 robust SE (TSP HCTYPE=2)

### G2: External Debt Crisis Contagion (Equations 2 and 6)

**Claim object**: Cross-country contagion predicts external debt crises. Same regressors but with external debt crisis as the outcome.

**Baseline specifications**:
- Eq2: `debt ~ C + bank_move + debt_move + center` (note: development dummies dropped in 1900-2009 version for debt)
- Eq6: `debt ~ develop1 + develop2 + bank_move + debt_move + center + public`
- Focal coefficient: `debt_move` (debt crisis contagion)

## RC Axes Included

### Controls
- **Add public debt**: The paper itself varies this (equations without/with `public`)
- **Leave-one-out**: Drop each of develop1, develop2, bank_move, debt_move, center individually
- Small control pool makes full enumeration feasible

### Sample restrictions
- **Period variation**: 1824-2009, 1900-2009, 1946-2009 (all three in the paper's code)
- **Subsample by development**: Advanced economies only, emerging markets only
- **Drop financial centers**: Exclude UK/US (the center variable sources)
- **Outlier trimming**: Trim extreme values

### Functional form
- **Logit**: Paper already estimates logit alongside OLS (Equations 3-4, 7-8)
- **Probit**: Standard alternative binary-choice model

### Fixed effects
- **Add country FE**: The paper uses no country FE (development dummies instead); adding country FE is a key robustness variation
- **Add year FE**: Time dummies to absorb common shocks
- **Region FE**: Regional fixed effects instead of development dummies

### Joint variations
- Period x public debt inclusion
- FE structure x period

## What Is Excluded and Why

- **Logit as separate baseline group**: The paper frames OLS as the main result; logit is an alternative estimator for the same claim object, included as `rc/form/model/logit`
- **Region dummies as alternative grouping**: These are explored as FE variations rather than separate baseline groups
- **Welfare analysis / structural interpretation**: The paper includes historical narrative which is not amenable to specification search

## Budgets and Sampling

- **G1**: Max 60 core specs, 15 control subsets (small pool)
- **G2**: Max 40 core specs, 10 control subsets
- **Seed**: 112444
- **Full enumeration** of LOO controls; random sampling not needed

## Inference Plan

- **Canonical**: HC2 robust SE (matches TSP HCTYPE=2)
- **Variants**: HC1, cluster by country (the paper does not cluster, but panel structure suggests clustering is a natural robustness check)
