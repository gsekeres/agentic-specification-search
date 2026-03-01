# Specification Search Report: 130141-V1

**Paper:** Gortz, Tsoukalas & Zanetti (2018), "News Shocks under Financial Frictions", AEJ: Macroeconomics 10(4)

## Baseline Specification

- **Design:** Structural Bayesian VAR with Minnesota prior (BIC-selected hyperparameters)
- **Identification:** Medium-run MFEVD (maximize forecast error variance of TFP at h=40)
- **Outcome:** GDP impulse response to TFP news shock
- **Focal horizon:** h=8 quarters
- **VAR lags:** 5
- **Variables:** TFP, GDP, Consumption, Hours, gzspr, SP500, Inflation
- **Sample:** Quarterly, 1984Q1-2017Q1 (133 observations)
- **Posterior draws:** 500

| Statistic | Value |
|-----------|-------|
| IRF (h=8) | 0.005831 |
| SE (posterior) | 0.004227 |
| 68% CI | [0.001930, 0.010384] |
| N (obs) | 133 |

## Specification Counts

- Total specifications: 52
- Successful: 52
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Coef Range |
|----------|-------|------------|
| Baseline | 1 | [0.005831, 0.005831] |
| Lag Length | 5 | [0.005681, 0.006446] |
| Financial Variable | 6 | [0.005744, 0.007614] |
| Add Investment | 1 | [0.005171, 0.005171] |
| Sample Period | 3 | [0.004312, 0.005801] |
| Identification | 2 | [-0.004704, 0.007880] |
| IRF Horizon | 5 | [0.000935, 0.007898] |
| Outcome Variable | 7 | [-0.000861, 0.042693] |
| Variable Ordering | 2 | [-0.002202, 0.005103] |
| Posterior Draws | 1 | [0.006396, 0.006396] |
| Grid (lags x fin) | 6 | [0.005214, 0.007135] |
| Outcome x Horizon | 13 | [-0.004956, 0.017069] |

## Inference Variants

| Spec ID | SE | p-value | CI |
|---------|-----|---------|--------|
| infer/bayesian/posterior_5_95 | 0.005484 | 0.2877 | [-0.001512, 0.016529] |
| infer/frequentist/bootstrap | 0.002156 | 0.0296 | [0.000585, 0.004699] |

## Overall Assessment

- **GDP IRF specs (h=8, TFP news shock):** 28 specifications
- **Sign consistency:** Mixed signs across specifications
- **Direction:** Median IRF is positive (0.005793)
- **Range:** [-0.002202, 0.007614]
- **Robustness assessment:** MODERATE

Surface hash: `sha256:66bad59eae391bc5a14d118582e1c8d4eefc6c0f181f2e5702681bf648489756`

## Notes

- This paper uses a Bayesian VAR with Minnesota prior. Hyperparameters are selected via BIC grid search.
- TFP news shock identified via medium-run MFEVD: the rotation that maximizes the forecast error variance
  of TFP at the VAR horizon, subject to zero contemporaneous effect on TFP (Barsky-Sims type).
- The specification search varies: lag length (2-8), financial variable in VAR (GZ spread, EBP, default risk,
  bank equity, BAA spread, or none), sample period (full vs. pre-Great Recession vs. post-1990),
  identification scheme (MFEVD news vs. max-FEV financial), IRF horizon, outcome variable, and variable ordering.
- Posterior draws from Normal-Inverse-Wishart distribution, with credible intervals at 16th/84th percentiles.
