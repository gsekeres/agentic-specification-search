# Specification Search Report: 114295-V1

**Paper:** Charnavoki & Dolado (2014), "The effects of global shocks on small commodity-exporting economies: Lessons from Canada", AEJ: Macroeconomics 6(2)

## Baseline Specification

- **Design:** Factor-Augmented Structural VAR (FAVAR)
- **Identification:** Recursive (Cholesky) with ordering [activity, commodity, inflation]
- **Outcome:** Canadian real GDP impulse response to commodity price shock
- **Focal horizon:** h=4 quarters (1 year)
- **VAR lags:** 2
- **Canadian factors:** 8
- **Sample:** Quarterly, 1975Q2-2010Q4

| Statistic | Value |
|-----------|-------|
| IRF (h=4) | -0.000060 |
| N (obs) | 143 |

## Specification Counts

- Total specifications: 56
- Successful: 56
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Coef Range |
|----------|-------|------------|
| Baseline | 1 | [-0.000060, -0.000060] |
| Lag Length | 3 | [-0.000563, 0.000498] |
| Factor Count | 4 | [-0.000318, 0.000247] |
| Cholesky Ordering | 5 | [0.000856, 0.002754] |
| Sample Period | 4 | [-0.002431, -0.000031] |
| IRF Horizon | 6 | [-0.001772, 0.000142] |
| Alternative Outcomes | 8 | [-0.005354, 229.039097] |
| Alternative Shocks | 2 | [-0.001760, 0.006009] |
| Lags x Factors Grid | 11 | [-0.000501, 0.000780] |
| Unrestricted VAR | 1 | [0.000038, 0.000038] |
| Outcome x Horizon | 11 | [-0.005354, 0.003289] |

## Inference Variants

| Spec ID | SE | p-value | 68% CI |
|---------|-----|---------|--------|
| infer/frequentist/bootstrap | 0.001301 | 0.9633 | [-0.001522, 0.001133] |
| infer/bayesian/posterior_5_95 | 0.001358 | 0.9648 | [-0.001387, 0.001239] |

## Overall Assessment

- **GDP IRF specs (h=4, commodity shock):** 29 specifications
- **Sign consistency:** Mixed signs across specifications
- **Direction:** Median IRF is negative (-0.000031)
- **Range:** [-0.002431, 0.002754]
- **Robustness assessment:** FRAGILE

Surface hash: `sha256:88a7f7bfb09922412a5ed2eef49fe0e0c35169caa73b8f390d93a6ae6c40ae18`

## Notes

- This paper uses a Bayesian FAVAR model. The specification search uses frequentist OLS-estimated VARs
  with Cholesky identification, which produces point estimates for the IRFs.
- The original paper reports Bayesian posterior credible intervals from Gibbs sampling.
- Bootstrap confidence intervals are computed as inference variants on the baseline specification.
- The specification surface varies: lag length (1-4), number of Canadian factors (4-12),
  Cholesky ordering of global factors (6 permutations), sample period, IRF horizon, and outcome variable.
