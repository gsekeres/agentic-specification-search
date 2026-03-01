# Specification Search Report

**Paper ID:** 112785-V1
**Paper:** Hlatshwayo & Spence (2014) - Demand and Defective Growth Patterns: The Role of the Tradable and Non-Tradable Sectors in an Open Economy

## Design Summary

- **Design:** Panel OLS (industry-year panel, 1990-2012)
- **Outcome:** VA growth (annual log-change in real value added)
- **Treatment:** nontradable (1=nontradable sector, 0=tradable)
- **Controls:** base-period size, crisis dummies
- **Fixed effects:** year (baseline); industry FE used with interaction terms
- **Clustering:** industry
- **Note:** Nontradable is time-invariant, so industry FE absorbs it. Baseline uses year FE to capture cross-sectional growth differential.

## Baseline Results

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.002527 |
| Std. Error | 0.004989 |
| p-value | 0.615895 |
| 95% CI | [-0.007623, 0.012676] |
| N | 748 |
| R-squared | 0.0961 |

## Specification Counts

- Total specifications: 69
- Successful: 69
- Failed: 0
- Inference variants: 4

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 3 | 1/3 | [0.0025, 274.0347] |
| Controls LOO | 3 | 0/3 | [0.0025, 0.0049] |
| Controls Sets | 5 | 0/5 | [0.0025, 0.0049] |
| Controls Progression | 5 | 0/5 | [0.0025, 0.0049] |
| Controls Subset | 10 | 0/10 | [0.0025, 0.0049] |
| Sample Period | 7 | 0/7 | [-0.0094, 0.0069] |
| Sample Outliers | 6 | 0/6 | [0.0014, 0.0029] |
| Fixed Effects | 5 | 0/5 | [-0.0007, 0.0025] |
| Functional Form | 15 | 6/15 | [-0.0196, 5466.1421] |
| Interactions | 3 | 0/3 | [-0.0007, 0.0069] |
| Combined | 10 | 0/10 | [-0.0011, 100.0852] |
| Aggregation | 2 | 1/2 | [-0.0196, 0.0100] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/cluster/industry | 0.004989 | 0.615895 | [-0.007623, 0.012676] |
| infer/se/hc/hc1 | 0.005264 | 0.631381 | [-0.007808, 0.012861] |
| infer/se/hc/hc3 | FAILED | - | - |
| infer/se/iid | 0.005761 | 0.661110 | [-0.008784, 0.013837] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 7/69 (10.1%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.003306)
- **Robustness assessment:** FRAGILE

Surface hash: `sha256:a3ae20da0b9262e352c5bbf4b3fa34d693583d5d68230ea571869fa135b00c9b`
