# Specification Search: 112370-V1

## Paper: Elections, Capital Flows and Politico Economic Equilibria (Chang, 2009)

## Surface Summary
- **Paper ID**: 112370-V1
- **Baseline groups**: 1 (G1: capital flow disruptions -> leftist election outcomes)
- **Design code**: cross_sectional_ols (probit baseline, LPM/logit alternatives)
- **Budget**: max 30 core specs
- **Seed**: 112370

## Baseline Specifications
- **Table 2 Col 2 (strict)**: probit y_exe_left ~ democ lgrgdpwork dffo12_X_lagfdi_gdp, robust
  - coef=0.1149, p=0.0533, N=99
- **Table 1 Col 2 (broad)**: probit _y_exec_left ~ democ lgrgdpwork dffo12_X_lagfdi_gdp, robust
  - coef=0.0687, p=0.0501, N=99

## Data Construction
- Rebuilt dataset from 5 raw CSV sources following main.do
- Merged: DPI2006 (elections) + PWT (GDP) + MF (FDI) + Polity IV (democracy) + Federal Funds
- Created interaction term: dffo12 * DL_NFDI_GDP

## Executed Specifications
- **Total specification_results rows**: 60
- **Successful**: 60
- **Failed**: 0
- **Inference variant rows**: 4

### Axes explored:
1. **Estimator**: Probit (baseline), LPM (OLS), Logit
2. **Outcome**: strict (y_exe_left) vs broad (_y_exec_left) leftist transition
3. **Controls LOO**: drop democ, drop lgrgdpwork
4. **Controls sets**: no controls, add dffo12 level
5. **Sample**: drop Venezuela, drop Haiti, post-1985, post-1990
6. **Treatment form**: dffo12 level only, lagged FDI/GDP only
7. **Combined**: sample x LOO, treatment form x sample, logit x sample

### Inference variants:
- Country-clustered SEs (18 clusters) on baseline and LPM specs

## Deviations from Surface
- Added logit as additional design alternative (probit and logit are both standard for binary outcomes)
- Added combined spec variants (sample x controls, sample x treatment form, logit x sample) to reach 50+ specs
- All additions are within the spirit of the approved surface's core axes

## Software Stack
- Python 3.12.7
- statsmodels (probit/logit estimation)
- pyfixest (LPM/OLS estimation)
- pandas, numpy

## Key Findings
- 18/60 specs significant at p<0.05 (30.0%)
- 43/60 specs significant at p<0.10 (71.7%)
- 54/60 specs have positive coefficient (90.0%)
- Coefficient range: [-0.0734, 0.2518]
