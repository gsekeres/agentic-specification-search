# Specification Search: 112338-V1

## Surface Summary
- **Paper**: Duggan & Scott Morton (2010), "The Effect of Medicare Part D on Pharmaceutical Prices and Utilization", AER
- **Baseline groups**: 2 (G1: price effects, G2: quantity effects)
- **Design**: cross_sectional_ols
- **Budgets**: G1 max 80 core specs, G2 max 60 core specs
- **Seeds**: G1=112338, G2=112339
- **Surface hash**: sha256:85ff7a803db6aae9e27eb29530033f826883dc71eafde62e816d62d36452a20f

## Critical Data Limitation

**The IMS Health pharmaceutical sales/dose data (ims0106data2.dta, ims0106all.dta) is proprietary
and NOT included in the ICPSR replication package.** This data provides the drug-level sales
revenue and dose quantities for 2001-2006 that are needed to construct ALL outcome variables:

- `lppd0603` (log change in price-per-day, 2003-2006)
- `ldoses0603` (log change in total doses, 2003-2006)
- `lsalesq0603` (log change in sales revenue, 2003-2006)
- `ppd0603` (level change in price-per-day)

Without this data, we cannot run any regressions beyond what is recorded in the Stata log file
(`regs-partd-final.log`). The log contains output for all regressions in the paper's Tables 2-5.

**Approach**: Extract coefficients from the Stata log for all specifications that match the
surface's approved universe. Record all other specifications as failures with the reason
"proprietary IMS data unavailable."

## Execution Summary

### G1: Price Effects (Table 2)

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 4 | 4 | 0 |
| RC specs | 28 | 6 | 22 |
| **Total** | **32** | **10** | **22** |

Specifications extracted from log:
- `baseline` (Table 2 Col 4) -- coef=-0.1364, p=0.015, N=517
- `baseline__table2_col1` (Table 2 Col 1) -- coef=-0.1278, p=0.026, N=548
- `baseline__table2_col2` (Table 2 Col 2) -- coef=-0.1272, p=0.026, N=548
- `baseline__table2_col3` (Table 2 Col 3) -- coef=-0.1376, p=0.014, N=538
- `rc/sample/subset/include_cancer` (=Table 2 Col 3) -- coef=-0.1376, p=0.014, N=538
- `rc/sample/subset/top292` (Table 2 Col 6) -- coef=-0.1427, p=0.009, N=200
- `rc/form/treatment/spending_share` (Table 2 Col 5) -- coef=-0.1329, p=0.018, N=517
- `rc/form/treatment/self_pay_decomposed` (Table 4 Col 2) -- coef=-0.2253, p=0.001, N=517
- `rc/form/treatment/dual_decomposed` (Table 4 Col 3) -- coef=-0.2429, p=0.002, N=517
- `rc/joint/no_trim_include_cancer` (=Table 2 Col 2) -- coef=-0.1272, p=0.026, N=548
- `rc/joint/spending_share_no_cancer` (=Table 2 Col 5) -- coef=-0.1329, p=0.018, N=517

### G2: Quantity Effects (Table 3)

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 4 | 4 | 0 |
| RC specs | 20 | 7 | 13 |
| **Total** | **24** | **11** | **13** |

Specifications extracted from log:
- `baseline` (Table 3 Col 4) -- coef=0.3892, p=0.218, N=517
- `baseline__table3_col1` (Table 3 Col 1) -- coef=0.5160, p=0.108, N=548
- `baseline__table3_col2` (Table 3 Col 2) -- coef=0.4338, p=0.175, N=548
- `baseline__table3_col3` (Table 3 Col 3) -- coef=0.3805, p=0.224, N=538
- `rc/sample/subset/include_cancer` (=Table 3 Col 3) -- coef=0.3805, p=0.224, N=538
- `rc/sample/subset/top293` (Table 3 Col 6) -- coef=0.5168, p=0.273, N=200
- `rc/sample/subset/drop_generic_facing` (Table 3 Col 7) -- coef=0.2651, p=0.054, N=318
- `rc/form/treatment/spending_share` (Table 3 Col 5) -- coef=0.3212, p=0.306, N=517
- `rc/form/treatment/self_pay_decomposed` (Table 4 Col 5) -- coef=0.4608, p=0.277, N=517
- `rc/joint/no_trim_include_cancer` (=Table 3 Col 2) -- coef=0.4338, p=0.175, N=548
- `rc/joint/drop_generic_no_cancer` (=Table 3 Col 7) -- coef=0.2651, p=0.054, N=318

### Exploration (non-core)
- `explore/outcome/log_sales_change` (G1, Table 4 Col 7) -- coef=0.2728, p=0.385, N=517 [SUCCESS]
- `explore/outcome/log_sales_change` (G2) -- FAILED (data unavailable)
- `explore/outcome/log_doses_change` (G1) -- FAILED (data unavailable)

### Inference Variants
- All 5 inference variants FAILED: proprietary data prevents re-estimation with alternative SEs

## Overall Counts

| | Planned | Succeeded | Failed |
|---|---|---|---|
| Core specs (G1+G2) | 56 | 22 | 34 |
| Inference variants | 5 | 0 | 5 |
| Exploration | 3 | 1 | 2 |

## Failure Reason

All failures share the same root cause: **The IMS Health pharmaceutical sales and dose data
(`ims0106data2.dta`, `ims0106all.dta`) is proprietary (purchased from IMS Health / IQVIA) and
was not included in the ICPSR replication package.** The readme explicitly states: "Because of
the proprietary nature of our data we cannot share these files."

Without this data, ALL outcome variables (price changes, dose changes, sales changes) cannot
be computed, making it impossible to run any regression not already recorded in the Stata log.

## Deviations from Surface

1. The surface requests ~80 G1 specs and ~60 G2 specs. Only 22 core specs could be
   extracted from the log (matching paper Tables 2-5).
2. Table 5 regressions use `cluster(ther1)` rather than HC1 (the canonical inference).
   These are recorded as failures for the canonical inference version, since the log only
   contains clustered SEs for these specifications.
3. Some surface RC specs exactly correspond to paper table columns (e.g., `rc/sample/subset/top292`
   = Table 2 Col 6), while others require novel regressions not in any table.

## Software Stack

- Source: Stata log file extraction (original analysis used Stata)
- Coefficients, SEs, p-values, CIs, N, R-squared all extracted verbatim from `regs-partd-final.log`
- Python packages used for output generation: pandas 2.2.3, numpy 2.1.3
