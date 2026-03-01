# Specification Search: 148301-V1

## Paper
Laffitte & Toubal (2022), "Multinational's Sales and Tax Havens", AEJ: Economic Policy.

## Surface Summary
- **Baseline groups**: 2
  - G1: Export platform share (ep ~ lfma), GLM fractional logit with marginal effects
  - G2: Profit shifting (lprofit ~ ep_haven), OLS with sector-year FE
- **Budgets**: G1 max 80, G2 max 30
- **Sampling seed**: 148301

## Data Construction
Dataset was constructed from raw BEA Excel files, PWT, CEPII Gravity, and tax agreement data.
- BEA data: years 1999-2013, 11 sectors (excl. finance, utilities, totals)
- FMA: Constructed using market potential measure from gravity data
- Tax rates: KPMG extended rates
- Haven indicator: Dharmapala & Hines (2009) list + NLD
- Agreements: Constructed from bilateral treaty .dta files

## Execution Summary

### Specification Results
- **Total planned**: 68
- **Successful**: 67
- **Failed**: 1

### Inference Results
- **Total**: 2
- **Successful**: 2
- **Failed**: 0

## G1 Specifications (51 total)
- Baseline: Table 2 Col 4 (GLM fractional logit, full controls with haven)
- Baseline variants: Table 2 Cols 3, 5, 6, 7, 8
- Design: OLS within estimator
- Controls LOO: 6 specs (drop each control)
- Control sets: minimal, lrgdp only, lrgdp+taxr, full with haven
- Control progression: Cols 1-4 steps
- Control additions: lemp, leqpmt, lemp+leqpmt
- Sample subgroups: manufacturing, services, non-haven, haven, big5, Caribbean
- Sample restrictions: drop imputed EP, drop imputed profits
- Outlier trimming: EP 1/99, EP 5/95
- FE variants: country-year, sector only, year only, add country
- Estimator: OLS reghdfe, OLS robust
- Outcome: EP no US sales
- Treatment: Haven split (big5 + otherh)
- Random control subsets: 10 draws

## G2 Specifications (17 total)
- Baseline: Table 4 Col 1 (OLS log profit)
- Baseline variants: GPML Poisson, cube root profit
- Controls LOO: 7 specs
- Sample: positive profit only, drop imputed profits
- Outlier trimming: profit 1/99
- Outcome transforms: cube root, GPML Poisson, level profit
- FE: add country

## Software
- Python 3.12.7
- pyfixest, statsmodels, pandas, numpy
- Surface hash: sha256:518c9a89ef103898daf5f21adb00810512b1d97466167ec94044509c26cda06c

## Deviations
- FMA variable was constructed using a simplified market potential measure (GDP/distance)
  rather than the full gravity-based FMA from the paper. This may cause small differences
  in coefficient magnitudes but preserves the qualitative direction of results.
- GLM fractional logit marginal effects are computed using average marginal effects (AME)
  with delta method standard errors, matching the paper's `margins, dydx(...)` approach.
- The two-way clustering inference variant for G1 uses pyfixest's built-in two-way
  clustering with iso3 + sector.
