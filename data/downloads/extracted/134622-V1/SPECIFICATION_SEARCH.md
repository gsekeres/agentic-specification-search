# Specification Search Report: 134622-V1

**Paper:** Azoulay, Jones, Kim & Miranda (2022), "Immigration and Entrepreneurship in the United States", AER: Insights 4(1)

## Baseline Specifications

### G1: Administrative Data (Firm Size Power Law)

- **Design:** Cross-sectional OLS (aggregated log-log regression)
- **Outcome:** log_freq_norm (log10 of population-normalized firm count)
- **Treatment:** immigrant (indicator)
- **Controls:** log_firm_size, immi_x_size (interaction)
- **Fixed effects:** None
- **SEs:** HC1

| Statistic | Value |
|-----------|-------|
| Coefficient (immigrant) | 0.533116 |
| Std. Error | 0.191684 |
| p-value | 0.019410 |
| 95% CI | [0.106018, 0.960214] |
| N | 14 |
| R-squared | 0.9657 |

### G2: Fortune 500 Firm-Level

- **Design:** Cross-sectional OLS
- **Outcome:** ln_nb_employees (log of employee count)
- **Treatment:** immi (at least 1 foreign-born founder)
- **Controls:** None (bivariate baseline)
- **Fixed effects:** None
- **SEs:** HC1

| Statistic | Value |
|-----------|-------|
| Coefficient (immi) | -0.128595 |
| Std. Error | 0.131559 |
| p-value | 0.328862 |
| 95% CI | [-0.387146, 0.129955] |
| N | 449 |
| R-squared | 0.0019 |

## Specification Counts

- Total specifications: 52
- Successful: 52
- Failed: 0
- Inference variants: 4

### G1: Admin Data

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [0.5331, 0.5331] |
| Treatment Alternatives | 15 | 8/15 | [-1.2414, 1.2179] |
| Source Alternatives | 7 | 2/7 | [-1.2414, 1.2179] |
| Sample Variants | 4 | 1/4 | [0.5176, 0.5859] |
| Functional Form | 5 | 3/5 | [-0.9667, 0.4876] |
| Estimation | 2 | 2/2 | [-0.6619, -0.6037] |

### G2: Fortune 500

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 0/1 | [-0.1286, -0.1286] |
| Treatment Alternatives | 1 | 0/1 | [-0.1871, -0.1871] |
| Sample Variants | 9 | 0/9 | [-0.3097, -0.0309] |
| Functional Form | 8 | 0/8 | [-17366.8027, 9.5866] |
| Controls | 5 | 0/5 | [-0.1577, -0.0314] |
| FE Variants | 1 | 0/1 | [-0.0436, -0.0436] |

## Inference Variants

| Spec ID | Group | SE | p-value | 95% CI |
|---------|-------|-----|---------|--------|
| infer/se/nonrobust | G1 | 0.223769 | 0.038452 | [0.034527, 1.031705] |
| infer/se/nonrobust | G2 | 0.139994 | 0.358812 | [-0.403724, 0.146533] |
| infer/se/hc/hc1_decade_fe | G2 | 0.143021 | 0.760385 | [-0.324762, 0.237470] |
| infer/se/nonrobust_decade_fe | G2 | 0.146055 | 0.765214 | [-0.330725, 0.243433] |

## Overall Assessment

### G1
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 15/27 (55.6%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.155522)
- **Robustness assessment:** WEAK

### G2
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 0/25 (0.0%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.071829)
- **Robustness assessment:** FRAGILE

Surface hash: `sha256:312480033bdd99adf7a836cd1a47c7ba6156bf642aa4c75f57c88ff642be1b47`
