# Specification Search Report: 113046-V1

**Paper:** Romer & Romer (2017), "New Evidence on the Aftermath of Financial Crises in Advanced Countries", AER 107(10)

## Baseline Specification

- **Design:** Jorda local projection impulse response
- **Outcome:** log(GDP) x 100 at horizon h=0
- **Treatment:** CRISIS (new semi-annual measure of financial distress, 0-15 scale)
- **Controls:** 4 crisis lags + 4 GDP lags
- **Fixed effects:** Country + Time (semiannual)
- **Standard errors:** Conventional OLS
- **Panel:** 24 OECD countries, semiannual 1967-2012

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.297016 |
| Std. Error | 0.047885 |
| p-value | 0.000000 |
| 95% CI | [-0.390926, -0.203106] |
| N | 2087 |
| R-squared | 0.9999 |

## Specification Counts

- Total specifications: 50
- Successful: 50
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [-0.2970, -0.2970] |
| Alt. Outcomes | 2 | 2/2 | [-0.3135, 0.0661] |
| Horizons | 7 | 7/7 | [-0.8520, -0.4132] |
| Sample (Pre-GFC/No-country) | 27 | 27/27 | [-0.3933, -0.1952] |
| Fixed Effects | 3 | 3/3 | [-0.4492, -0.2809] |
| Lag Structure | 5 | 5/5 | [-0.2992, -0.2917] |
| WLS | 1 | 1/1 | [-0.1771, -0.1771] |
| Alt. Crisis Measures | 4 | 1/4 | [-0.7048, -0.6108] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/robust | 0.073452 | 0.000055 | [-0.441068, -0.152964] |
| infer/se/cluster/country | 0.102707 | 0.008225 | [-0.509481, -0.084551] |
| infer/se/cluster/time | 0.075616 | 0.000171 | [-0.447311, -0.146722] |

## Overall Assessment

- **Sign consistency (GDP/IP specs):** All specifications have the same sign
- **Significance stability:** 45/45 (100.0%) GDP/IP specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.3024)
- **Interpretation:** Financial distress reduces log GDP
- **Robustness assessment:** STRONG

- **Total specifications (all treatment vars):** 50
- **Significant at 5% (all):** 47/50

Surface hash: `sha256:969efaea7e9c95de1892772b0aaa376dc73687dbce5e7bfde65eed85c0ff41dc`
