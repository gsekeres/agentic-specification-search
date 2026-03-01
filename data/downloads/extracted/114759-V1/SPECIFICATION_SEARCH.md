# Specification Search Report: 114759-V1

**Paper:** Camacho & Conover (2011), "Manipulation of Social Program Eligibility", AEJ: Economic Policy 3(2)

## Baseline Specification

- **Design:** Sharp Regression Discontinuity
- **Running variable:** SISBEN poverty score (puntaje)
- **Cutoff:** 47
- **Outcome:** Share of surveys at each score (density)
- **Treatment:** jump (score <= 47 indicator)
- **Estimator:** Local linear regression with triangular kernel
- **Bandwidth:** Data-driven (Imbens-Kalyanaraman)
- **Inference:** HC1 robust SEs

| Statistic | Value |
|-----------|-------|
| Coefficient (jump) | 1.546106 |
| Std. Error | 0.131791 |
| p-value | 0.000001 |
| 95% CI | [1.247973, 1.844239] |
| N (score bins in BW) | 13 |
| R-squared | 0.9899 |

## Specification Counts

- Total specifications: 50
- Successful: 50
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline (year-specific) | 5 | 4/5 | [0.0257, 1.6685] |
| Design: Bandwidth | 8 | 8/8 | [1.3233, 3.2763] |
| Design: Polynomial | 3 | 3/3 | [1.0335, 1.5461] |
| Design: Kernel | 3 | 3/3 | [1.5461, 1.6842] |
| Design: Procedure | 2 | 2/2 | [1.2742, 1.5461] |
| RC: Donut Hole | 3 | 3/3 | [2.2268, 3.9369] |
| RC: Bandwidth Multiples | 6 | 6/6 | [1.3233, 3.2763] |
| RC: SES Restrictions | 3 | 3/3 | [0.8779, 2.1560] |
| RC: Time Period | 13 | 8/13 | [0.0062, 1.6685] |
| RC: Score Construction | 2 | 2/2 | [1.5461, 1.5461] |
| RC: Placebo Cutoffs | 8 | 2/8 | [-0.8754, 0.0131] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.131791 | 0.000001 | [1.247973, 1.844239] |
| infer/se/cluster/score | 0.131791 | 0.000000 | [1.258957, 1.833255] |
| infer/se/iid | 0.193506 | 0.000022 | [1.108365, 1.983846] |

## Overall Assessment

- **Sign consistency:** All non-placebo specifications have the same sign
- **Significance stability:** 36/42 (85.7%) non-placebo specifications significant at 5%
- **Direction:** Median coefficient is positive (1.546106), indicating upward bunching below cutoff
- **Placebo cutoffs:** 2/8 significant at 5% (lower is better)
- **Robustness assessment:** STRONG

Surface hash: `sha256:fb86cbc21dd335e350f827b8e37a76226164db989aaaea9c5c597c638348375b`
