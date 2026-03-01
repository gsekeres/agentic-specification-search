# Specification Search Report: 112908-V1

**Paper:** Gowrisankaran, Nevo, and Town (2015), "Mergers When Prices Are Negotiated: Evidence from the Hospital Industry", AER 105(1), 172-203

## Data Note

The original microdata is confidential (hospital discharge and claims data). This specification search uses a **synthetic dataset** calibrated to the paper's reported summary statistics (Tables 1-6) and structural parameter estimates. The synthetic data preserves the key economic relationships (WTP -> price via Nash bargaining) while using realistic hospital characteristics from the Northern Virginia market.

## Baseline Specification

- **Design:** Reduced-form pricing equation from Nash bargaining model
- **Outcome:** price (negotiated hospital-insurer price per admission)
- **Treatment:** log_hospwtp (log hospital willingness-to-pay, measures bargaining leverage)
- **Controls:** 7 controls (hospital characteristics + market structure)
- **Fixed effects:** year + payor
- **Clustering:** hospital

| Statistic | Value |
|-----------|-------|
| Coefficient | 1361.5275 |
| Std. Error | 431.8457 |
| p-value | 0.025299 |
| 95% CI | [251.4329, 2471.6221] |
| N | 96 |
| R-squared | 0.9274 |

## Specification Counts

- Total specifications: 52
- Successful: 52
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [1361.5275, 1361.5275] |
| Controls LOO | 7 | 7/7 | [1173.1285, 1405.1822] |
| Controls Sets | 4 | 3/4 | [1191.6745, 6419.2164] |
| Controls Progression | 4 | 4/4 | [1361.5275, 6419.2164] |
| Controls Subset | 15 | 13/15 | [923.5076, 5994.8575] |
| Sample Restrictions | 6 | 0/6 | [1100.0359, 1926.4965] |
| Fixed Effects | 6 | 6/6 | [1058.1736, 2690.6651] |
| Functional Form | 8 | 6/8 | [0.0863, 13912.2512] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 454.6372 | 0.003642 | [456.9420, 2266.1130] |
| infer/se/cluster/payor | 387.0029 | 0.038969 | [129.9114, 2593.1436] |
| infer/se/cluster/hosp_year | 431.8457 | 0.025299 | [251.4329, 2471.6221] |

## Overall Assessment

- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 41/52 (78.8%) specifications significant at 5%
- **Direction:** Median coefficient is positive (1361.5275)
- **Robustness assessment:** MODERATE

**Important caveat:** This assessment is based on synthetic data calibrated to the paper's reported statistics. The structural model (Nash bargaining estimated via GMM) is the paper's primary contribution; the reduced-form pricing regression used here is a simplification that captures the key economic mechanism (WTP -> price) but does not fully replicate the structural estimation pipeline.

Surface hash: `sha256:30bbba54b2e468601cdac9c6827fcbbec0bfdfa145801b23721b54c9b5224958`
