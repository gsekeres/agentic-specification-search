# Replication Report: 112754-V1

## Summary
- **Paper**: Search, Liquidity, and the Dynamics of House Prices and Construction
- **Authors**: Allen Head, Huw Lloyd-Ellis, Hongfei Sun
- **Journal**: American Economic Review, 2014, 104(4), 1172-1210
- **Replication status**: not possible
- **Total regressions in original package**: 6
- **Regressions in scope (main + key robustness)**: 6
- **Successfully replicated**: 0
- **Match breakdown**: 0 exact, 0 close, 0 discrepant, 0 failed

## Reason for "Not Possible" Classification

This paper is a structural macroeconomic model of housing search and construction. Its main empirical results depend on two software stacks that cannot be replicated in Python:

### 1. Panel VAR Estimation (Stata `pvar` / `xtvar`)

The empirical estimation uses Inessa Love's `pvar` command for system-GMM panel VAR estimation. This is an external Stata ado-file (available from the World Bank website at http://go.worldbank.org/E96NEWM7L0) that is NOT included in the replication package. The `pvar` command implements:
- System-GMM (2SLS) estimation of panel vector autoregressions
- Helmert (forward orthogonal deviations) transformation for fixed effects removal
- Monte Carlo confidence intervals for impulse response functions
- Variance decomposition

There is no standard Python package that replicates this exact system-GMM panel VAR procedure. The `statsmodels` VAR module handles time-series VARs but not panel VARs with GMM estimation. Manual implementation would require replicating the complete Arellano-Bover/Blundell-Bond GMM system for multivariate panel data, which is a substantial undertaking with many degrees of freedom in implementation that could affect results.

### 2. Dynare DSGE Model Simulations (Matlab + Dynare)

The paper's main theoretical results (Tables 1-2, 7-8) are generated from Dynare model simulations called by Matlab scripts. There are:
- 6 Dynare `.mod` files in the Simulation folder (baseline, nosearch, searchers_exit, endogenous_exit, land_develop, urnball)
- 11 Dynare `.mod` files in the Estimation folder (psvar_main, psvar_coastal, psvar_inland, psvar_sunbelt, psvar_ols, psvar_wge, psvar_inconly, psvar_ypgrowth, psvar_cwage, psvar_clabour, psvar_rent)
- 4 Matlab `.m` simulation scripts (sim_baseline, sim_nosearch, sim_wlr, targets_urnball)

Per the estimation methods reference: "Dynare replication is not typically feasible in Python for complex models. Mark as 'not possible' if the paper's main results depend on Dynare simulation."

## Data Description
- **Files available**: `msadata.dta` -- panel data for 106 US MSAs, 1980-2008
- **Key variables**: income, house prices (OFHEO index), construction permits, population, sales, rents, construction wages, construction employment, CPI
- **Data is included**: The Stata data file is present and loadable

## Script Inventory

### Stata Do-Files (Original 2014)
| Script | Description | Regression commands |
|--------|-------------|-------------------|
| `varest1.do` | Main 5-variable panel VAR (income, price, sales, construction, population growth) | 1 `pvar` (+ 5 auxiliary `regress` for time FE removal) |
| `varest_cwage.do` | 6-variable panel VAR adding construction wages | 1 `pvar` (+ 6 auxiliary `regress`) |
| `varest_clabour.do` | 6-variable panel VAR adding construction labor | 1 `pvar` (+ 6 auxiliary `regress`) |
| `varest_rent.do` | 6-variable panel VAR adding rents | 1 `pvar` (+ 6 auxiliary `regress`) |

### Stata Do-Files (2016 Update)
| Script | Description | Regression commands |
|--------|-------------|-------------------|
| `xtvarest15.do` | Main panel VAR with `xtvar` (LSDV estimator) | 1 `xtvar` (+ 5-8 auxiliary `regress`) |
| `2SLSest15.do` | Panel VAR in growth rates | 1 `pvar` (+ 5 auxiliary `regress`) |
| `unitroottest.do` | Panel unit root tests | 5 auxiliary `regress` (+ 20 `xtunitroot` -- not regression commands) |

### Dynare/Matlab Files
- 17 Dynare `.mod` files across Estimation and Simulation folders
- 4 Matlab `.m` simulation scripts
- 1 Excel file for annualizing IRFs

## Regression Count Details

Active (uncommented) panel VAR regression commands: **6 total**
1. `varest1.do`: `pvar wx px sx cx popx, lag(2) gmm monte 1000 decomp`
2. `varest_cwage.do`: `pvar wx px sx cx popx cwx, lag(2) gmm monte 1000 decomp`
3. `varest_clabour.do`: `pvar wx px sx cx popx cwx, lag(2) gmm monte 1000 decomp`
4. `varest_rent.do`: `pvar wx px sx cx popx rx, lag(2) gmm monte 1000 decomp`
5. `2SLSest15.do`: `pvar wx px sx cx popx, lag(2) gmm impulse gr_imp list`
6. `xtvarest15.do`: `xtvar wx px sx cx popx, step(10) norm`

The `quietly regress` commands (approximately 40 across all do-files) are intermediate data-cleaning steps that remove time fixed effects before the panel VAR. They are not themselves the regression results reported in the paper.

The Dynare `stoch_simul` commands (approximately 22 across all `.mod` files) are model simulations, not empirical regressions.

## Translation Notes
- **Original language**: Stata (panel VAR estimation) + Matlab/Dynare (DSGE simulation)
- **Translation approach**: Not attempted -- both key software dependencies lack Python equivalents
- **Known limitations**:
  - `pvar` (Inessa Love's panel VAR): No Python package implements system-GMM panel VAR with the same methodology
  - `xtvar`: LSDV panel VAR estimator, also from an external Stata package
  - Dynare `.mod` files: Require Dynare + Matlab/Octave for linearization and stochastic simulation
  - The paper's Tables 1-2, 7-10 all depend on Dynare simulation output

UNLISTED_METHOD: pvar in 112754-V1 -- Inessa Love's panel VAR with system-GMM estimation, Helmert transformation, Monte Carlo confidence intervals for IRFs, and variance decomposition. External Stata ado-file from World Bank.

UNLISTED_METHOD: xtvar in 112754-V1 -- Panel VAR with LSDV (least squares dummy variable) estimator. External Stata ado-file.

## Software Stack
- Language: Python 3.x (not used -- replication not possible)
- Key packages: N/A
- Required but unavailable: Stata with `pvar`/`xtvar` packages, Matlab with Dynare
