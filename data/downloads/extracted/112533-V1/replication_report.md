# Replication Report: 112533-V1

## Summary
- **Paper**: "The Finnish Great Depression: From Russia with Love" by Gorodnichenko, Mendoza, and Tesar (AER, 2012)
- **Replication status**: not possible
- **Total regressions in original package**: 162 (Stata `reg` invocations, mostly detrending in loops; 0 regression commands in Matlab code)
- **Regressions in scope (main + key robustness)**: 1 (Figure 1, Panel C OLS)
- **Successfully replicated**: 1
- **Match breakdown**: 0 exact, 1 close, 0 discrepant, 0 failed

## Paper Nature

This is a calibrated DSGE/IRBC (International Real Business Cycle) model paper. The paper builds a multi-sector open economy model to explain Finland's severe recession in the early 1990s following the collapse of trade with the Soviet Union. The main results (Tables 1-5, Figures 2-5) are produced by solving and simulating a structural model in Matlab using the Anderson-Moore (AIM) algorithm for perturbation solutions.

**The paper's main tables contain NO regression results.** They report:
- Table 1: Calibration parameters
- Table 2: Steady-state ratios (model vs. data)
- Table 3: Contributions of spending components to output decline
- Table 4: Baseline model simulation vs. actual data
- Table 5: Robustness/sensitivity analysis (alternative model specifications)

All of these are outputs of calibrated model simulations, not statistical estimation.

## Data Description
- **Files used**: `raw_production_data.dta`, `export_shares.dta`
- **Key variables**: Employment by industry (31 manufacturing sectors), share of exports to USSR by industry (1988)
- **Sample sizes**: 31 Finnish manufacturing industries (cross-sectional)
- **Matlab data**: `Finland_1973.mat` contains calibration data for the DSGE model; `.txt` files contain time series for Soviet-bloc trade

## Replication Details

### Figure 1, Panel C: Soviet Exposure and Cumulative Fall in Output
- **What was replicated**: Cross-sectional OLS regression of employment deviations from trend (1993) on share of exports to USSR (1988). This is the scatter plot in Figure 1, Panel C.
- **Original**: slope = -14.54, SE = (6.04), annotated directly on the figure
- **Replicated**: slope = -14.5458, SE = 6.4414 (HC1 robust)
- **Match status**: `close` -- coefficient matches to 4 significant figures (relative error 0.04%). SE differs by ~6.6%, possibly due to data version differences or rounding in the figure annotation.
- **Data pipeline**: Followed the Stata do-file exactly: reclassified industries, detrended employment series using OLS on 1980-1989 (or 1986-1990 for specific industries), extracted 1993 residuals, merged with export shares, ran bivariate OLS.

### Tables 1-5 (DSGE Model Results)
- **Not replicated**: These results come from solving a calibrated multi-sector IRBC model in Matlab. The code uses the AIM algorithm (`aim2.m`) for perturbation-based solution of rational expectations models, followed by impulse response simulation. This requires a Matlab environment and is a structural model, not a regression.
- Translation to Python is not feasible for this type of complex DSGE model (see estimation_methods.md: "Dynare replication is not typically feasible in Python for complex models").

## Regression Count Details

The 162 Stata `reg` commands break down as:
- **124**: Detrending regressions for `prod`, `va`, `export`, `inv` (4 variables x 31 industries) -- `reg log(x) year if year>=1980 & year<=1989`
- **31**: Detrending regressions for `empl` (31 industries) -- same trend specification
- **4**: Re-detrending for `empl` industries 3, 12, 21, 30 (constant-only on 1986-1990)
- **1**: Re-detrending for `empl` industry 3 (linear trend on 1986-1990)
- **1**: Main scatter plot OLS with robust SE: `reg r0_empl_1993 row_export_1988, robust`
- **1**: Same regression without robust SE (for fitted line in graph)

The Matlab code contains 0 regression commands. It uses matrix backslash (`\`) for detrending (same as OLS) in `Import_data.m` and `Table3_components_contributions_data.m`, but these are data processing, not estimation results. The core Matlab code solves a system of nonlinear equations for the model steady state and uses perturbation methods for dynamics.

The `.wf1` file is an EViews workfile for computing standard errors (proprietary format, not readable).

## Translation Notes
- **Original language**: Matlab (model), Stata (figures/data processing)
- **Translation approach**: Translated the Stata do-file for Figure 1 Panel C exactly, following the industry reclassification, detrending, reshaping, and regression steps. Used statsmodels OLS with HC1 robust standard errors.
- **Known limitations**: The paper's main results depend on Matlab DSGE model solution code that cannot be translated to Python. The only replicable regression is a secondary/illustrative result in a figure panel.

UNLISTED_METHOD: AIM algorithm (aim2.m) in 112533-V1 -- Anderson-Moore algorithm for solving linear rational expectations models via perturbation methods. Used for DSGE/IRBC model solution. Not a regression estimator.

## Software Stack
- Language: Python 3.x
- Key packages: pandas, numpy, statsmodels 0.14.x
