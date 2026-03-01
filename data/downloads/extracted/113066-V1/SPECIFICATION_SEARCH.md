# Specification Search Report: 113066-V1

**Paper:** Steinwender (2018), "Real Effects of Information Frictions: When the States and the Kingdom became United", AER 108(3)

## Design

- **Design:** Before-after comparison (pre vs post transatlantic telegraph, 1866)
- **Data:** Daily cotton price data, Liverpool and New York, ~1865-1867
- **Treatment:** tele (= 1 after telegraph introduction)
- **SE:** Newey-West HAC with 2 lags (baseline)

## Baseline Group G1: Price Level

- **Outcome:** difffrctotal (cotton price difference net of freight cost)
- **Controls:** l1nyrec (lagged NY receipts, in thousand bales)
- **Sample:** Excludes no-trade days

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.907616 |
| Std. Error | 0.147934 |
| p-value | 0.000000 |
| 95% CI | [-1.197561, -0.617671] |
| N | 575 |
| R-squared | 0.2299 |

## Baseline Group G2: Price Variance

- **Outcome:** dev2_difffrctotal (sample-corrected squared deviation from pre/post mean)
- **Controls:** l1nyrec (lagged NY receipts)

| Statistic | Value |
|-----------|-------|
| Coefficient | -1.974156 |
| Std. Error | 0.425125 |
| p-value | 0.000003 |
| 95% CI | [-2.807386, -1.140926] |
| N | 575 |
| R-squared | 0.0895 |

## Specification Counts

- Total specifications: 52
- Successful: 52
- Failed: 0
- Inference variants: 5

## Category Breakdown

### G1: Price Level

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 3 | 3/3 | [-0.9078, -0.8697] |
| Controls | 6 | 5/6 | [-1.0286, -0.2195] |
| Sample | 7 | 6/7 | [-1.2583, -0.2631] |
| Outcome Form | 6 | 5/6 | [-1.1122, 0.3370] |
| Joint | 13 | 11/13 | [-1.2583, 0.3509] |

### G2: Price Variance

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 3 | 3/3 | [-3.8016, -1.9742] |
| Controls | 2 | 2/2 | [-2.0488, -1.9868] |
| Sample | 3 | 3/3 | [-2.0405, -1.4721] |
| Outcome Form | 5 | 5/5 | [-3.8016, -0.7301] |
| Joint | 4 | 4/4 | [-3.1166, -0.6475] |

## Inference Variants

| Spec ID | Group | SE | p-value | 95% CI |
|---------|-------|-----|---------|--------|
| infer/se/hac/newey_west_4 | G1 | 0.184157 | 0.000001 | [-1.268557, -0.546675] |
| infer/se/hac/newey_west_8 | G1 | 0.231604 | 0.000089 | [-1.361552, -0.453680] |
| infer/se/hc/hc1 | G1 | 0.090978 | 0.000000 | [-1.085930, -0.729302] |
| infer/se/hac/newey_west_4_var | G2 | 0.522900 | 0.000160 | [-2.999021, -0.949291] |
| infer/se/hc/hc1_var | G2 | 0.263711 | 0.000000 | [-2.491021, -1.457291] |

## Overall Assessment

### G1: Price Level

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 30/35 (85.7%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.846739)
- **Robustness assessment:** WEAK

### G2: Price Variance

- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 17/17 (100.0%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-2.040498)
- **Robustness assessment:** STRONG

Surface hash: `sha256:6d3c0911b0415a85da06d8cfa5ec8f61635e36fb7ed5502d5e12665aea7f122b`
