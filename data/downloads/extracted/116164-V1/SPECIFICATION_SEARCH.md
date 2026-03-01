# Specification Search Report: 116164-V1

**Paper:** Lockwood (2018), "Incidental Bequests and the Choice to Self-Insure Late-Life Risks", AER 108(9)

## Data Note

This paper is a structural estimation paper. The raw HRS/NLTCS data is NOT included
in the replication package (must be obtained separately from HRS). The specification
search targets the reduced-form probit regressions in Appendix Table 1 using a
synthetic HRS-like dataset constructed from the known variable structure in the code.

## Baseline Specification

- **Design:** Cross-sectional probit/LPM
- **Outcome:** ltci (binary: owns long-term care insurance)
- **Treatment:** female (gender indicator)
- **Controls:** 7 controls (demographics, age polynomial, wealth quartile dummies)
- **Fixed effects:** none
- **Standard errors:** Heteroskedasticity-robust (HC1)

| Statistic | Value |
|-----------|-------|
| Coefficient (LPM) | 0.016992 |
| Std. Error | 0.004027 |
| p-value | 0.000025 |
| 95% CI | [0.009099, 0.024884] |
| N | 20000 |
| R-squared | 0.0110 |

## Specification Counts

- Total specifications: 67
- Successful: 67
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 2 | 2/2 | [0.0170, 0.0175] |
| Controls LOO | 7 | 7/7 | [0.0170, 0.0172] |
| Controls Sets | 11 | 11/11 | [0.0169, 0.0174] |
| Controls Progression | 8 | 8/8 | [0.0169, 0.0174] |
| Controls Subset | 10 | 10/10 | [0.0170, 0.0179] |
| Sample Subgroups | 15 | 7/15 | [0.0092, 0.0305] |
| Functional Form | 3 | 3/3 | [0.0173, 0.0179] |
| Outcome Variants | 5 | 0/5 | [-0.0057, 0.0144] |
| Treatment Variants | 4 | 2/4 | [-0.0029, 0.0256] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc3 | 0.004028 | 0.000025 | - |
| infer/se/ols | 0.004084 | 0.000032 | - |

## Overall Assessment

- **LTCI outcome, female treatment specs:** 58
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 50/58 (86.2%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.017126)
- **Robustness assessment:** WEAK

### All Specifications
- Significant at 5%: 52/67 (77.6%)

## Note on Synthetic Data

This specification search uses synthetic data because the HRS/NLTCS microdata
is not included in the replication package. The synthetic data preserves the
variable structure, sample selection criteria, and approximate population moments
from the paper's Stata code. Results should be interpreted as demonstrating the
specification search methodology rather than exact replication of the paper's results.

Surface hash: `sha256:c153b21d41a32bfcc7f57c5c5241c0eafe3c4c4cbaf9d7cf30d9ec43becd879b`
