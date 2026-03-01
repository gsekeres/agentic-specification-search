# Specification Search Report: 184041-V1

## Surface Summary

- **Paper ID**: 184041-V1
- **Design**: Randomized experiment (lab experiment)
- **Baseline groups**: 3
  - G1: Experiment I bid factors (BF/BEBF/NEBF ~ CV), OLS + median regression
  - G2: Experiment II price factors (BF ~ CV), OLS + median regression
  - G3: Decision weights (lnbid ~ lnfix + lnsignal), median regression + OLS
- **Budgets**: G1=50, G2=30, G3=30
- **Seed**: 184041
- **Surface hash**: sha256:91e5bb87f4226b26d131199b633ec6f4c0ef3707b5b3bb184dbbdb02ed6bdcd0

## Execution Summary

| Group | Planned | Executed | Success | Failed |
|-------|---------|----------|---------|--------|
| G1    | 30  | 30   | 30 | 0 |
| G2    | 9  | 9   | 9 | 0 |
| G3    | 12  | 12   | 12 | 0 |
| **Total** | **51** | **51** | **51** | **0** |

### Inference variants: 5 rows written to inference_results.csv

## Specification Details

### G1: Experiment I Bid Factors
- **Baseline**: OLS BF ~ CV, Exp I reduced sample (exp==1 & rrange==1 & domnBid<=8), cluster(subject)
- **Additional baselines**: BF median, BEBF OLS, BEBF median, NEBF OLS, NEBF median
- **Design variants**: Diff-in-means for BF, BEBF, NEBF
- **RC/sample**: Full sample (no rrange/domnBid exclusion), Winners only (dWin==1)
  - Each with OLS and median for BF, BEBF, NEBF
- **Inference**: HC1 (robust, no clustering), Session-level clustering

### G2: Experiment II Price Factors
- **Baseline**: OLS BF ~ CV, stage 22 (compound lottery with signal), cluster(subject)
- **Additional baselines**: Median regression stage 22
- **Design variants**: Diff-in-means
- **RC/sample**: Stage 21 (OLS + median), Stage 1 (OLS + median)
- **Inference**: HC1, Session-level clustering

### G3: Decision Weights
- **Baseline**: Median regression lnbid ~ lnfix + lnsignal, CV subsample, Exp I reduced
- **Additional baselines**: CP subsample, Interaction model (pooled with CV interactions)
- **RC/sample**: Full sample without rrange/domnBid exclusions (CV, CP, interaction)
- **RC/form**: OLS instead of median regression (CV, CP, interaction)
- **Inference**: HC1 for CV baseline

## Deviations and Notes

1. **Median regression SEs**: Python's `QuantReg` uses kernel-based standard errors, not clustered SEs as in Stata's `qreg2`. The coefficient estimates match but SEs/p-values may differ.
2. **Diff-in-means**: Implemented as Welch two-sample t-test rather than OLS regression, which gives equivalent point estimates but Welch-corrected SEs.
3. **No control-subset variations**: The main bid factor regressions include no controls, as noted in the surface. All variation comes from outcome measure, estimator, and sample selection.

## Software Stack

- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
- statsmodels: 0.14.6
- scipy: 1.15.1
