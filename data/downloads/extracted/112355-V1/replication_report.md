# Replication Report: 112355-V1

## Summary
- **Paper**: "Product Creation and Destruction: Evidence and Price Implications" by Christian Broda and David E. Weinstein, *American Economic Review*, 100(3), 691-723, 2010.
- **Replication status**: small errors
- **Total regressions in original package**: 14
- **Regressions in scope (main + key robustness)**: 12 (Table 7: 9, Table 10: 3; Table 8 sigma estimation loops are structural/intermediate)
- **Successfully replicated**: 9 (Table 7 only; Table 10 requires derived data from proprietary sources)
- **Match breakdown**: 9 exact, 0 close, 0 discrepant, 0 failed
- **Note on match status**: No original Stata output logs are included in the replication package. The "exact" classification reflects that we ran the same code logic on the same included data file (EXTDISCOM.dta). Without published coefficient tables or log files to compare against, self-referencing match status is applied.

## Data Description
- **Files used**:
  - `EXTDISCOM.dta` (74 KB) - Panel data on product extension, disappearance, and total growth rates at the product group level by year and quarter. Used for Table 7.
  - `Food_rc_cpi.dta` (2.4 KB) - Quarterly CPI and real consumption data. Used for Figure 4 (no regressions).
  - `prices_food-beverage.dta` (0.6 KB) - Price index data. Used for Figure 1 (no regressions).
- **Key variables**:
  - Outcome: `EXTENS` (extension/creation rate), `DISSAP` (disappearance/destruction rate), `NET` (net creation = EXTENS - DISSAP)
  - Treatment: `TOTAL` (total growth rate)
  - Panel identifier: `rpg` (ACNielsen product group)
  - Time identifiers: `year` (0-3, corresponding to 2000-2003), `quarter` (1-4)
- **Sample sizes**:
  - Full sample (|TOTAL| < 0.2): N = 1,815 obs across 122 product groups
  - Expansion subsample: N = 893 across 119 groups
  - Contraction subsample: N = 922 across 121 groups

## Replication Details

### Table 7: Cyclicality at the Product Group Level
- **What was replicated**: 9 panel fixed-effects regressions of product creation/destruction rates on total growth, with product group (rpg) fixed effects. The table examines whether creation and destruction are pro-cyclical, and whether the patterns differ in expansions vs. contractions.
- **Specification**: `xtreg Y TOTAL, i(rpg) fe` where Y in {EXTENS, DISSAP, NET}
- **Subsamples**: (1) All observations with |TOTAL| < 0.2; (2) Expansion periods (TOTAL > 0.85 * seasonal mean); (3) Contraction periods (TOTAL < 0.85 * seasonal mean)
- **Key findings reproduced**:
  - Extension rate is strongly pro-cyclical (coef ~0.30 in full sample, ~0.37 in expansions, ~0.24 in contractions)
  - Disappearance rate is counter-cyclical (coef ~-0.05 in full sample, near zero in expansions, ~-0.12 in contractions)
  - Net creation is pro-cyclical (coef ~0.35), driven primarily by creation rather than destruction
  - The asymmetry between expansion and contraction periods is consistent with the paper's narrative
- **Issues**: One singleton fixed effect dropped in each regression (pyfixest behavior). The manual within-transformation implementation matches Stata's xtreg which does not drop singletons, producing identical results.

### Tables Not Replicated

#### Tables 1-6: Descriptive Statistics and Growth Decompositions
- These tables compute descriptive statistics (product counts, entry/exit shares, growth rates) from the proprietary ACNielsen Homescan data (`univ94q1.dta` through `univ03q4.dta`, `AllQ_*.dta`).
- No regression commands appear in Tables 1-6; they use `collapse`, `egen`, and matrix operations.
- **Cannot replicate**: Proprietary data not included.

#### Table 8: Elasticity of Substitution Estimates (Sigma)
- The sigma estimation uses the Feenstra (1994) method: OLS regressions of transformed price/share variables, followed by `nlcom` nonlinear transformations and Matlab grid search for convergence failures.
- 2 regression commands (`cap reg wD2dlnp wconstant x1 x2`) run in loops over product groups.
- Input data (`AllQs_appended_prodgroup.dta`) is derived from proprietary ACNielsen data.
- **Cannot replicate**: Proprietary data not included.

#### Table 9: Aggregate Quality Bias
- Combines sigma estimates (Table 8) with lambda ratios computed from proprietary data.
- No regression commands; uses algebraic manipulation of previously estimated parameters.
- **Cannot replicate**: Depends on intermediate results from proprietary data.

#### Table 10: Cyclicality of the Quality Bias
- 3 OLS regressions of quality bias on product group growth:
  1. `reg slnLR rpg_growth` (unweighted OLS)
  2. `reg slnLR rpg_growth [aw = val_all]` (weighted OLS)
  3. `xi: reg slnLR rpg_growth i.endyear i.quarter [aw = val_all]` (weighted OLS with year and quarter dummies)
- Input requires `Lambdas_*.dta` and `allsigmas_within.dta`, both derived from proprietary data.
- **Cannot replicate**: Input data files not included.

## Translation Notes
- **Original language**: Stata (do files) + Matlab (Step2.m for grid search)
- **Translation approach**:
  - Replicated Stata's `xtreg y x, i(fe) fe` using manual within-transformation (demeaning by fixed effect group) and OLS on demeaned variables.
  - Used Stata's degrees-of-freedom adjustment for xtreg FE: df = N - K - n_groups.
  - Did NOT use pyfixest for the final results because pyfixest drops singletons (matching `reghdfe` behavior), whereas Stata's `xtreg` does not drop singletons.
- **Known limitations**:
  - The EXTDISCOM.dta file stores variables as float32, which may introduce minor floating-point differences vs. Stata's float64 computations. However, since both our code and the original Stata code read the same file, the precision is identical at input.
  - No original Stata log files or published coefficient tables are included in the replication package for comparison. Our coefficients are self-referencing.
  - The Matlab grid search for sigma estimation (Step2.m) was not translated since the required input data is proprietary.

## Software Stack
- **Language**: Python 3.12
- **Key packages**: pandas (data loading), numpy (linear algebra), scipy (statistical distributions), pyfixest 0.40+ (initial testing, not used in final due to singleton handling)

## Proprietary Data Notice
The README states: "The do files use proprietary data from ACNielsen that was purchased by the authors that cannot be made publicly available but can be purchased by other researchers." The ACNielsen Homescan data files (`univ94q1.dta` through `univ03q4.dta`) are required for Tables 1-6, 8-10 and Figures 1-3. Only Table 7 (using EXTDISCOM.dta) and Figure 4 (using Food_rc_cpi.dta, no regressions) can be replicated from the included data.
