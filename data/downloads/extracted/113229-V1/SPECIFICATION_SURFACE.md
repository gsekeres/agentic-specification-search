# Specification Surface: 113229-V1

## Paper Overview
- **Title**: Betrayal Aversion: Evidence from Brazil, China, Oman, Switzerland, Turkey, and the United States (Bohnet, Greig, Herrmann & Zeckhauser, AER 2008)
- **Design**: Randomized experiment (lab experiment)
- **Key claim**: Subjects require a higher minimum acceptable probability (MAP) of being paid in a Trust Game (where a human partner decides) than in a Risky Dictator Game (where nature decides), indicating "betrayal aversion" -- an aversion to being let down by another person beyond what risk aversion alone would predict.

## Baseline Groups

### G1: Betrayal Aversion (Trust Game vs Risky Dictator Game)

**Claim object**: The treatment effect of game type (Trust Game vs Risky Dictator Game) on minimum acceptable probability (MAP), measuring betrayal aversion.

**Baseline specifications**:
- **Table2-Col1**: `map ~ tg + dp if mover==1, cl(session)` -- No controls. Focal coefficient on `tg` measures betrayal aversion (MAP premium in TG over RDG).
- **Table2-Col2**: Adds 9 demographic/country controls.
- **Table2-Col3**: Adds gender-treatment and Oman-treatment interactions (14 total controls).

## RC Axes Included

### Controls
- **Single additions**: Each of 9 non-interaction controls added individually (female, age, income, econmajor, brazil, china, oman, swiss, turk)
- **Standard sets**: Demographics only (female/age/income/econmajor), country dummies only, demographics + country
- **Progression**: Build-up from no controls to full set
- **Random subsets**: 15 random draws from the control pool (seed=113229)

### Sample restrictions
- **Drop one country at a time**: 6 leave-one-country-out specifications
- **Gender subsamples**: Women only, men only
- **Outlier trimming**: MAP at [1,99] and [5,95] percentiles

### Functional form
- **Outcome transform**: Logit(MAP) -- MAP is bounded in [0,1]
- **Treatment isolation**: TG-only vs RDG (dropping DP observations); DP-only vs RDG (dropping TG observations)

### Preprocessing
- **Complete cases**: Drop observations with missing demographics

## What Is Excluded and Why

- **Table 1 rank-sum tests**: Non-parametric tests are not regression-based specifications. They are conceptual alternatives but not in scope for the regression specification surface.
- **Table A.4 country-specific regressions**: These are subsample explorations (each country separately), not the pooled main claim. Included as `rc/sample/restriction/drop_country_*` instead.
- **Bootstrapping**: The Bootstrapping.do file implements permutation-style inference for country-level comparisons. This is an inference method, not a specification variant.
- **Interaction effects (Table2-Col3)**: The female*treatment and Oman*treatment interactions are recorded as baseline__table2_col3 but heterogeneity analysis is not separately explored.

## Budgets and Sampling

- **Max core specs**: 60
- **Max control subsets**: 15
- **Seed**: 113229
- **Controls pool**: 9 non-interaction controls in 2 blocks + 5 interaction terms

## Inference Plan

- **Canonical**: Clustered SEs at session level (matches paper)
- **Variants**: HC1 (robust, no clustering), HC3 (jackknife)
- Inference variants recorded in `inference_results.csv`
