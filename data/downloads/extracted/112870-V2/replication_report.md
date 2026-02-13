# Replication Report: 112870-V2

## Summary
- **Paper**: Optimal Life Cycle Unemployment Insurance (Michelacci & Ruffo, AER 2015)
- **Replication status**: not possible
- **Total regressions in original package**: 38
- **Regressions in scope (main + key robustness)**: 38
- **Successfully replicated**: 0
- **Match breakdown**: 0 exact, 0 close, 0 discrepant, 0 failed

## Reason for "Not Possible" Classification

The PSID data required by `PSID_consumption_regressions.do` (which produces Figure 2 and Figure 3 results) is **not included** in the replication package. The README states: "PSID data has been moved to https://doi.org/10.3886/E228166V1." This is a separate restricted-access deposit that must be obtained independently.

Per replication protocol, when any regressions in the package require unavailable data, the entire paper is classified as "not possible" even if other regressions could be run with included data.

## Data Description
- **Files included**:
  - `SIPP_durations.dta` - SIPP unemployment duration data (used by elasticity_SIPP.do)
  - `sevpay_pooled_sel.dta` - Mathematica severance pay data (used by sevpay_M.do)
  - `statecps.dta` - CPS state-level panel data (used by unemploymentCPS.do)
  - `wbaofemployedxstatexyearxage.dta` - Complementary CPS data for benefit imputation
  - `wbaunempxyearxstatexage.dta` - Unemployment benefit data by year/state/age
  - `dataselect.dta` - SIPP wage loss data (used by WagelossesSIPP.do)
  - `stats.dta` - Summary statistics for Table A1
- **Files missing**:
  - `data.dta` - PSID consumption panel data (required by PSID_consumption_regressions.do for Figure 2 and Figure 3)
- **Quantitative model files**: Multiple Matlab `.m` and `.mat` files in `quantitative/` directory (structural model, not regression-based)

## Package Structure

The replication package has two major components:

### 1. Empirical Analysis (Stata do-files)
| Script | Tables/Figures | Data | Regressions | Status |
|--------|---------------|------|-------------|--------|
| `elasticity_SIPP.do` | Table 1, Table 2, Figure 1a, Figure A1 | `SIPP_durations.dta` + `wbaunempxyearxstatexage.dta` | 28 (stcox) | Data available |
| `sevpay_M.do` | Table 3 | `sevpay_pooled_sel.dta` | 3 (stcox) | Data available |
| `unemploymentCPS.do` | Figure 1b, Figure A2 | `statecps.dta` | 3 (reg, ivreg) | Data available |
| `PSID_consumption_regressions.do` | Figure 2, Figure 3 | `data.dta` (PSID) | 4 (xtreg) | **Data MISSING** |
| `WagelossesSIPP.do` | Wage loss tables | `dataselect.dta` | 0 (table commands only) | Data available |

### 2. Quantitative/Structural Model (Matlab .m files)
The `quantitative/` directory contains Matlab code for the structural life-cycle model:
- `Main.m` - Solves and simulates the worker's problem
- `Outcomes.m` - Produces baseline economy tables and figures
- `Elasticity.m`, `Elasticityin1step.m`, `Elasticity_Welf.m` - Computes elasticities
- `Optimization.m` - Finds optimal benefit/tax profiles
- `FirstBest.m` - First-best economy solution
- `Results.m` - Generates comparison figures/tables

These are structural model simulations, not regression-type estimations, and are not counted in the regression total.

## Regression Count Detail

### elasticity_SIPP.do (28 regressions)
- **Table 1, Panel (a)** - 3 Cox PH models: all workers, young (20-40), old (41-60) using individual benefit measure (`l_wba`)
- **Table 1, Panel (b)** - 3 Cox PH models: same subgroups using age-specific average benefits (`lgwbau_c`)
- **Figure 1, Panel (a)** - 8 Cox PH models: one per 5-year age group (15-25 through 50-60) using individual benefits
- **Figure 1, Average benefits** - 8 Cox PH models: same age groups using average benefits
- **Table 2, Individual benefits** - 3 stratified Cox PH models with asset quartile interactions (all, young, old)
- **Table 2, Aggregate benefits** - 3 stratified Cox PH models with asset quartile interactions (all, young, old)

### sevpay_M.do (3 regressions)
- **Table 3** - 3 Cox PH models with time-varying coefficients: all workers, young (20-40), old (41-60)

### unemploymentCPS.do (3 regressions)
- **Figure 1b** - 1 OLS regression of log unemployment on age-benefit interactions
- **Figure A2** - 2 IV regressions using lagged benefits as instruments

### PSID_consumption_regressions.do (4 regressions)
- **Figure 2a** - 1 random effects regression of food consumption
- **Figure 2b** - 1 random effects regression of nondurables consumption
- **Figure 3a** - 1 fixed effects regression of food consumption losses
- **Figure 3b** - 1 fixed effects regression of nondurables consumption losses

## Translation Notes
- Original language: Stata (empirical), Matlab (quantitative model)
- The primary estimation method is Cox Proportional Hazards (`stcox`), which is a duration/survival model
- Cox PH models with `tvc()` option use time-varying coefficients, requiring specialized handling
- Stratified Cox models (`strata()` option) are used for Table 2
- The quantitative Matlab model is a structural calibration exercise, not translatable to regression format

UNLISTED_METHOD: stcox with tvc() in 112870-V2 -- Cox proportional hazards model with time-varying coefficients (Stata's tvc option creates interactions of specified variables with analysis time _t)

UNLISTED_METHOD: stcox with strata() in 112870-V2 -- Stratified Cox proportional hazards model where baseline hazard varies by strata groups

## Software Stack
- Language: Python 3.x (not used - replication not possible)
- Key packages: N/A
