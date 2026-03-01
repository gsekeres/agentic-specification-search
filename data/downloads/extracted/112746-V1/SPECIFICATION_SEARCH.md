# Specification Search Report: 112746-V1

**Paper:** Dubois, Griffith & Nevo (2014), "Do Prices and Attributes Explain International Differences in Food Purchases?", AER 104(12)

## Baseline Specification

- **Design:** Panel FE (structural demand estimation)
- **Outcome:** Exp_adeq (food expenditure per adult equivalent)
- **Focal variable:** Carbs_adeq (carbohydrate nutrient quantity)
- **Other regressors:** Protein and fat quantities interacted with category type (meat-dairy, prepared, other)
- **Fixed effects:** HH-category, category-quarter
- **Clustering:** HH-category
- **Data:** Synthetic panel calibrated to reported parameter estimates (no micro data available)

| Statistic | Value |
|-----------|-------|
| Coefficient (Carbs_adeq) | 1.741897 |
| Std. Error | 0.187613 |
| p-value | 0.000000 |
| 95% CI | [1.374083, 2.109711] |
| N | 36000 |
| R-squared | 0.9612 |

## Specification Counts

- Total specifications: 55
- Successful: 55
- Failed: 0
- Inference variants: 4

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 3 | 3/3 | [1.7419, 2.1598] |
| Estimator/Nutrients | 24 | 24/24 | [1.4913, 21.5267] |
| Category Drop | 9 | 9/9 | [1.5560, 1.8336] |
| FE Variants | 5 | 5/5 | [1.6878, 5.4445] |
| Sample Trimming | 2 | 2/2 | [1.6431, 1.6867] |
| Sample Subsets | 7 | 7/7 | [0.9437, 2.1911] |
| Functional Form | 3 | 1/3 | [-0.0008, 0.1006] |
| Additional Controls | 2 | 2/2 | [1.7417, 1.7929] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.181201 | 0.000000 | [1.386736, 2.097059] |
| infer/se/cluster/cat | 0.162872 | 0.000005 | [1.366314, 2.117480] |
| infer/se/cluster/hhid | 0.184211 | 0.000000 | [1.379973, 2.103822] |
| infer/se/cluster/yqtr | 0.122308 | 0.000002 | [1.452684, 2.031110] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 53/55 (96.4%) specifications significant at 5%
- **Direction:** Median coefficient is positive (1.747537)
- **Robustness assessment:** WEAK

Surface hash: `sha256:7a51ee185172472a5b853eb7678b221e25905306cebd15d55bca86ec6fcbef43`

**Note:** Results are based on synthetic panel data constructed from the paper's reported
parameter estimates. The original micro data (Nielsen Homescan, TNS Kantar) is proprietary
and not included in the replication package. Coefficient magnitudes should be interpreted
qualitatively (sign, significance pattern) rather than as exact replication of reported values.
