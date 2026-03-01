# Specification Surface: 114747-V1

## Paper Overview
- **Title**: Incentives and Promotion for Adverse Drug Reactions (Dranove, Hughes, and Meltzer)
- **Design**: Panel fixed effects (Poisson count models, Logit)
- **Data**: Drug-month panel (1456 obs, 63 drugs, 24 months, 4 conditions). The provided .dta contains only ADR counts, FDA labeling changes, drug approval dummies, and basic identifiers. Key treatment variables (promotion expenditures) and most controls (patient demographics) are NOT in the provided data.
- **Key limitation**: Major data completeness issue -- the do-file references many variables not in the provided dataset.

## Baseline Groups

### G1: ADR Counts and Promotion (Tables 3-4)

**Claim object**: Pharmaceutical promotion and advertising spending affects serious adverse drug reaction reporting rates.

**Baseline specification**:
- Formula: `poisson veryserious q1totalexp q2q4totalexp generic $char $dapp $age $yrm drug* if condition==k, exp(permonths) robust`
- Outcome: `veryserious` (count of serious ADRs)
- Treatment: Promotion expenditure variables (q1totalexp, q2q4totalexp; or disaggregated: professional promo, DTCA, cost-of-contact)
- Controls: generic indicator, patient demographic shares ($char = 8 vars), drug approval age ($dapp = 4 vars), age-gender shares ($age = 4 vars)
- FE: Drug dummies + year-month dummies (absorbed in Poisson)
- Exposure: Person-months (permonths)
- Robust SE
- Estimated separately by condition (1=cholesterol, 2=allergies, 3=arthritis, 4=depression)

**Additional baseline-like rows**: Condition-specific estimates for all 4 conditions, plus decomposed promotion channels (Tables 3-4).

### G2: FDA Labeling Changes (Table 2)

**Claim object**: Serious ADR reports predict subsequent FDA labeling changes (evidence that ADR reporting is informative).

**Baseline specification**:
- Formula: `logit any_fda c1 c2 c3 v1 v2 v3 v4 generic $char $dapp $age $yrm`
- Outcome: `any_fda` (binary: any FDA labeling change)
- Treatment: ADR count x condition interactions (v1-v4)
- Variants: Contemporaneous, 3-month cumulative (v_3*), 12-month cumulative (v_12*)

## RC Axes Included

### Controls
- **LOO**: Drop generic indicator
- **Block removal**: Drop patient characteristics ($char), drop age-gender ($age), drop drug approval age ($dapp)
- **Standard sets**: Minimal (generic only), full (all available)

### Sample restrictions
- Exclude specific drugs (Rofecoxib/Vioxx, Valdecoxib/Bextra -- withdrawn drugs)
- Outlier trimming on ADR counts

### Design variants
- Negative binomial (alternative to Poisson for overdispersion)
- OLS on log rate (linear alternative)
- Pooled across conditions (vs. condition-specific)
- LPM and probit (alternatives to logit for G2)

### Functional form
- Log rate transformation
- Lagged expenditure (temporal ordering)
- Cumulative ADR measures (3-month, 12-month)

### Fixed effects
- Drug + year (vs. drug + year-month)
- Condition + drug + year-month (pooled model)

## What Is Excluded and Why

- **Detailed promotion channel decomposition**: The paper tests multiple promotion breakdowns (professional, DTCA, cost-of-contact). These are captured as baseline-like rows rather than RC axes since they represent different treatment measurements.
- **Duration models (countduration)**: Reported in Table 2 but represent a different estimand (time to labeling change vs. occurrence).

## Budgets and Sampling

- **G1 max core specs**: 55
- **G2 max core specs**: 25
- **Seed**: 114747
- **Major data constraint**: Most variables referenced in the do-file are not in the provided .dta. Executable specifications are limited.

## Inference Plan

- **Canonical**: Robust (sandwich) SE for Poisson/logit
- **Variants**: Cluster at drug level, cluster at condition level
