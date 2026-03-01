# Specification Search: 112840-V1

## Paper
Kumhof, Ranciere & Winant (2015), "Inequality, Leverage, and Crises", AER 105(3).

## Surface Summary
- **Paper ID**: 112840-V1
- **Baseline groups**: 1 (G1: Financial liberalization and public debt growth)
- **Design**: Panel fixed effects (within estimator), country FE, HC1 robust SE
- **Data**: 22 OECD countries, 1975-2005 annual panel
- **Surface hash**: sha256:d25cc34b9003a15117c20bffa1cbd24d90d1b2ce0c2ed29b6ac5efd458745b63
- **Sampling seed**: 112840

### Baseline Specifications
- Table O1, Col 1 (minimal): changerealdebt ~ lag_d_ave_finindex1 + lag_debtgdp + lagchangerealgdp | cn
- Table O1, Col 5 (extended): adds lag_emu_dum, size_product1, d_dep_ratio_old, changetop1incomeshare
- Table O1, Col 6 (Gini variant): replaces changetop1incomeshare with lag_changeave_gini_gross, drops Korea

### Budgets
- Max core specs: 70
- Max control subsets: 25
- Seed: 112840

## Execution Summary

### Counts
- **Total specification rows**: 59
  - Baseline: 3
  - Design variants: 1
  - RC variants: 55
- **Inference variant rows**: 5
- **Total (specs + inference)**: 64
- **Successful**: 56 / 59 specs, 3 / 5 inference
- **Failed**: 3 specs, 2 inference

### RC Axes Executed
1. **Controls progression** (Col 2-4): 3 specs
2. **Controls LOO** (drop each optional from Col5): 4 specs
3. **Controls swap** (inequality measure): 1 spec
4. **Controls random subsets**: 23 specs (seed=112840)
5. **Sample outliers** (trim y [1,99], [5,95], Cook's D): 3 specs
6. **Sample country exclusions** (Korea, Greece, small countries): 3 specs
7. **Sample period** (1980-2005, 1975-2000): 2 specs
8. **Functional form / treatment** (Index 2, level debt change): 3 specs
9. **Fixed effects** (add year FE, region FE): 2 specs
10. **Preprocessing** (unweighted index, alternative debt): 2 specs
11. **Joint variations** (index x controls, sample x controls): 9 specs

### Inference Variants
1. **Cluster by country** (CRV1, cn): on Col5 and Col1 baselines
2. **HC3**: on Col5 and Col1 baselines
3. **Driscoll-Kraay** (HAC approximation, 3 lags): on Col5 baseline

### Deviations / Notes
- Driscoll-Kraay SE approximated via HAC (Newey-West, 3 lags) on entity-demeaned data, since pyfixest does not natively support DK SE.
- The paper's structural Fortran model (calibration) is excluded per the surface.
- Table O3 (interest rate elasticity) is excluded per the surface.
- The size interaction variable (size_product1) uses the GDP-weighted index; for Index 2, the equivalent sizefin2_product1 is used.

## Software Stack
- Python 3.12.7
- pyfixest (panel FE estimation)
- pandas, numpy (data manipulation)
- statsmodels (Cook's D diagnostics, HAC SE)
- scipy (statistical tests)

## Data Construction
All variables reconstructed from source data files following the Stata do file exactly:
- gini_gross.csv + subset.dta + DepRatioOld.dta + incomeinequality.dta
- GDP-weighted average financial liberalization indexes
- First differences of logs for debt and GDP
- EMU dummy, size-finlib interaction, dependency ratio change
- Sample restricted to 22 OECD countries, 1975-2005

## Key Results
- **baseline**: coef=0.6883, se=0.2555, p=0.0072***, N=677
- **baseline__tableO1_col5**: coef=1.5340, se=0.3758, p=0.0001***, N=435
- **baseline__tableO1_col6**: coef=1.2928, se=0.2863, p=0.0000***, N=648

## Robustness Assessment
- Specifications with p < 0.05: 52/56 (93%)
- Specifications with p < 0.10: 54/56 (96%)
- Specifications with same sign as baseline: 56/56 (100%)
- Median coefficient: 1.1047
- Mean coefficient: 1.0398
- Coefficient range: [0.1126, 1.7346]
