# Specification Surface: 112451-V1

## Paper Overview
- **Title**: Biological Resource Centers and Scientific Research (Furman & Stern, 2011)
- **Design**: Difference-in-differences with conditional FE negative binomial
- **Data**: Article-year panel. BRC-linked articles vs. matched controls. Annual citations as outcome.
- **Key tables**: Table 4 (main results with age-profile controls), Table 5 (substitution test using most-related other articles)

## Baseline Groups

### G1: Effect of BRC Deposit on Citations (Table 4, Column 1)

**Claim object**: BRC deposit increases subsequent citations to linked articles. The treatment is `post_brc` (indicator for post-deposit period). Identification comes from comparing the citation trajectory of BRC-linked articles to matched controls, differencing out article FE and flexible age/year trends.

**Baseline specification**:
- Formula: `xtnbreg cites window post_brc age_brc1 $age1 $year1, irr fe i(rart_num) vce(bootstrap)`
- Outcome: `cites` (annual citation count)
- Treatment: `post_brc` (post-deposit indicator)
- Controls: `window` (time-window indicator), `age_brc1` (BRC-article age interaction), 30 individual age dummies, ~22 year dummies
- Sample: `sample3==1` (BRC-linked articles + matched controls)
- Inference: Bootstrap SE
- Reported as IRR (incidence rate ratio); focal coefficient is the IRR for post_brc

**Additional baseline**: Table 4, Column 2 adds `post_brc_yrs` (time since deposit).

## RC Axes Included

### Controls
- **Add post_brc_yrs**: Time-since-deposit trend (Table 4, Col 2)
- **LOO**: Drop `window`, drop `age_brc1` (the 30 age and 22 year dummies are mandatory)
- **Grouped age/year**: Replace individual dummies with 5-year grouped dummies (Table 4, Cols 3-4 use this)

### Sample restrictions
- **BRC-only sample**: `sample0==1` restricts to only BRC-linked articles (no controls), identifying from within-BRC timing variation (Table 4, Cols 3-4)
- **Drop short panels**: Exclude articles with few observation years
- **Trim/winsorize extreme citations**: Top 1% of citation counts

### Functional form
- **Poisson FE**: Replace negative binomial with Poisson fixed-effects regression
- **OLS on log(cites+1)**: Linear model on transformed outcome
- **OLS on asinh(cites)**: Inverse hyperbolic sine transform
- **Grouped age dummies**: 5-year age groups instead of individual year dummies

### Fixed effects
- **Pair FE**: Add matched-pair fixed effects (in addition to article FE)

### Data construction
- **Matching variations**: Different caliper widths for the matched control design
- **Drop unmatched**: Restrict to articles with valid matched pairs only

### Joint variations
- Sample x functional form combinations
- FE x functional form combinations

## What Is Excluded and Why

- **Table 5 (substitution test)**: This tests whether citations switch away from related articles by the same author. It uses a different outcome (citations to most-related articles) and is a robustness/falsification check, not the main claim. Excluded from core baseline groups.
- **Event study (Figure 2)**: Year-by-year pre/post coefficients are included as a diagnostic, not a core spec.
- **Tables 6-8 (additional robustness from other do files)**: These test additional mechanisms and are appropriately treated as exploration rather than core RC.

## Budgets and Sampling

- **Max core specs**: 60
- **Max control subsets**: 10 (limited variation since age/year dummies are mandatory)
- **Seed**: 112451
- Most variation comes from functional form, sample, and FE axes rather than controls

## Inference Plan

- **Canonical**: Bootstrap SE (matches paper's `vce(bootstrap)`)
- **Variants**: Cluster-robust at article level, HC1 (for OLS variants)

## Key Linkage Constraints

- Age dummies ($age1) and year dummies ($year1) must always be included together (the paper never drops them)
- The window variable identifies the pre/post-deposit time window for the matched sample; dropping it changes the sample interpretation
