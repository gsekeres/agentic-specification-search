# Verification Report: 148301-V1

## Paper
- **Title**: Multinationals Sales and Profit Shifting in Tax Havens
- **Authors**: Laffitte, Sebastien and Toubal, Farid
- **Journal**: American Economic Journal: Economic Policy (2022)
- **DOI**: 10.1257/pol.20200203

## Data Note
**IMPORTANT**: This specification search uses SIMULATED data, not the original BEA restricted-access data. Coefficients are calibrated approximations.

## Baseline Groups

### G1: Corporate tax rate effect on foreign sales ratio
- **Baseline spec_id**: baseline
- **Claim**: Higher corporate tax rate reduces foreign sales ratio by ~0.57pp
- **Expected sign**: Negative
- **Baseline coefficient**: -0.00545 (SE=0.000449, p<0.001)
- **FE**: country + industry + year
- **Controls**: log_gdp_pc, log_distance, common_language
- **Clustering**: country
- **N**: 9240

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 64 |
| Baseline specifications | 1 |
| Core test specifications | 44 |
| Non-core specifications | 20 |
| Invalid specifications | 0 |
| Unclear specifications | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_fe | 5 |
| core_funcform | 5 |
| core_inference | 6 |
| core_sample | 18 |
| noncore_alt_outcome | 4 |
| noncore_alt_treatment | 4 |
| noncore_heterogeneity | 9 |
| noncore_placebo | 3 |

## Detailed Classification Rationale

### Core tests (44 specs)
These maintain the same outcome (foreign_sales_ratio or transformations) and same treatment (corp_tax_rate) while varying:
- **Fixed effects structure** (5 specs): No FE, single-dimension FE, two-way FE
- **Controls** (9 specs): Leave-one-out and incremental control additions
- **Sample restrictions** (14 specs): Time periods, geographic exclusions, outlier treatment, industry exclusions, balanced panel
- **Clustering** (6 specs): Different clustering levels for standard errors
- **Functional form** (5 specs): IHS outcome, quadratic tax, level controls, log tax rate, log outcome

### Non-core: Alternative outcomes (4 specs)
- profit_margin, log_profit, log_total_sales, log_employment
- Different dependent variables testing related but distinct hypotheses
- log_foreign_sales_ratio classified as core (functional form of baseline outcome)

### Non-core: Alternative treatments (4 specs)
- tax_haven, low_tax_15, low_tax_20, low_tax_25
- Change causal object from continuous tax rate to binary indicators
- Use different FE (industry+year only, no country FE)
- Expected positive sign, not comparable to baseline

### Non-core: Placebo tests (3 specs)
- random_tax, lagged_outcome, future_outcome
- All show null effects as expected

### Non-core: Heterogeneity (9 specs)
- By haven status (2), region (4), industry (2), interaction (1)
- Subsample analyses, not direct robustness checks of pooled estimate

## Top 5 Most Suspicious Rows

1. **robust/control/drop_log_gdp_pc** (and other leave-one-out): Coefficients IDENTICAL to baseline to 14+ decimal places. Simulated controls are orthogonal, making these tests uninformative.

2. **robust/control/none**: No controls, yet coefficient identical to baseline. FE absorb all variation or controls orthogonal by construction.

3. **robust/sample/balanced_panel**: Coefficient and SE exactly identical to baseline (N=9240). Data already balanced, test is trivially identical.

4. **robust/het/interaction_haven_tax**: Uses industry+year FE only (not country FE), different identification strategy. More heterogeneity than robustness.

5. **robust/outcome/log_profit**: Positive coefficient (0.0134, p=0.18). Different outcome; could be misinterpreted as contradicting profit-shifting narrative.

## Recommendations

1. **Simulated data**: Leave-one-out tests uninformative due to orthogonal controls. Flag when controls do not change coefficient.
2. **Alt treatment classification**: tax_haven specs change both treatment AND FE structure. Flag as different estimand.
3. **Heterogeneity vs robustness**: Distinguish subsample analyses from specification robustness.
4. **Balanced panel check**: Detect when restriction produces identical results.
5. **Control orthogonality**: Flag when leave-one-out coefficients are identical to baseline.
