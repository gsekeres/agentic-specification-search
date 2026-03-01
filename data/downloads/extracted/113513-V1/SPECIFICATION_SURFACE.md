# Specification Surface: 113513-V1

## Paper Overview
- **Title**: Trends in Economics Major Shares by Demographics (descriptive study using IPEDS data)
- **Design**: Cross-sectional OLS (descriptive correlations, no causal identification)
- **Key claim**: Economics undergraduate major shares correlate with shares of other disciplines, controlling for year trend. Purely descriptive analysis of changing composition of undergraduate majors.

## Baseline Groups

### G1: Economics Share vs Other Discipline Shares (All Students)

**Claim object**: Descriptive partial correlation between economics major share and other disciplines' shares, conditional on linear year trend.

**Baseline specification**:
- Formula: `econshare ~ level_d{j} + year, robust`
- Outcome: `econshare` (share of economics BAs among all BAs)
- Treatment: Share of another discipline (11 disciplines: business/mgmt, poli sci, psych, other social science, math/eng/CS, physical science, life science, arts/architecture, education, humanities, other)
- Controls: `year` only
- N ~ 16 (year-level collapsed data, 2000-2015)
- Inference: Heteroskedasticity-robust SEs

**Note**: This paper has 11 parallel bivariate regressions (one per comparison discipline). We treat the business/management regression as the primary baseline and the other 10 as additional baseline spec IDs, since they all represent the same claim structure.

## RC Axes Included

### Sample restrictions
- **Demographic subsamples**: Females only, nonwhites only (the paper reports these separately)
- **Time splits**: Pre-2008, post-2008, drop first year, drop last year

### Functional form
- **Second major share**: Use `econshare2d` (second/double major share) instead of first major share
- **Year trend**: Add year-squared for quadratic trend; drop year control entirely
- **No year control**: Bivariate regression without year

### No controls multiverse
Each regression uses exactly one discipline share plus year. The paper never combines multiple discipline shares.

## What Is Excluded and Why

- **Tables and graphs**: Summary statistics tables and time series plots are not regressions.
- **Multiple discipline shares in one regression**: The paper never does this; it would change the estimand fundamentally.
- **Data construction variants (IPEDS cleaning)**: The data is provided pre-cleaned at the discipline-year level. No raw data construction choices are available.
- **Causal inference methods**: The paper makes no causal claim; adding IV/matching/etc. would be inappropriate.

## Budgets and Sampling

- **Max core specs**: 55
- **Full enumeration**: The space is small enough for complete enumeration
- **Seed**: 113513 (unused since no random sampling needed)

## Inference Plan

- **Canonical**: HC1 robust SEs (matches paper)
- **Variants**: Newey-West HAC (addresses serial correlation in ~16 time-series obs), Classical OLS
