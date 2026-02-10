# Verification Report: 116336-V1

## Paper
**Title**: Six Randomized Evaluations of Microcredit: Introduction and Further Steps
**Authors**: Abhijit Banerjee, Dean Karlan, and Jonathan Zinman
**Journal**: American Economic Journal: Applied Economics, 2015
**Method**: Descriptive statistics (summary statistic computation) and cross-sectional OLS

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Mean APR for top MFI lenders across 6 study countries is ~60% (Table 1) | + | 313 |
| G2 | Cross-country APR differences are large and significant; Mexico has highest premium vs. India | + | 1597, 1598, 1599, 1600, 1601, 1602 |
| G3 | Gross loan portfolio sizes characterize MFI market scale | + | 1699 |

### G1: Mean APR (primary finding)
- **Baseline estimate**: 0.603 (60.3%), matching paper's Table 1 value of ~60%
- **Baseline SE**: 0.139 (cross-MFI standard deviation)
- **Baseline p-value**: N/A (descriptive statistic, not a regression coefficient)
- **Baseline N**: 54 MFI-country observations (top 10 lenders x 6 countries, with some missing)
- **Parameters**: non_missing_3vars sample, top 10 lenders, all 6 countries, country-specific compounding, loan loss rate, Roodman formula

### G2: OLS Country Coefficients
- **Mexico premium (spec 1599)**: estimate = 1.700, p = 0.006 (170 pp higher APR than India)
- **India intercept (spec 1602)**: estimate = 0.177, p < 0.001
- **Bosnia (spec 1597)**: estimate = 0.136, p < 0.001
- **Ethiopia (spec 1598)**: estimate = 0.125, p = 0.051
- **Mongolia (spec 1600)**: estimate = 0.233, p < 0.001
- **Morocco (spec 1601)**: estimate = 0.250, p < 0.001
- **Parameters**: top 10 lenders, baseline compounding, loan loss rate, HC1 standard errors

### G3: Gross Loan Portfolio
- **Baseline estimate**: $77.9M mean gross loan portfolio for top 10 lenders
- **Baseline p-value**: N/A (descriptive statistic)

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **1,736** |
| Baselines | 8 |
| Core tests (non-baseline) | 1,696 |
| Non-core tests | 32 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 8 | Exact replication of paper Table 1 statistics and OLS coefficients |
| core_multidim | 762 | Multi-dimensional variations of APR mean (joint changes to 2+ dimensions) |
| core_summarystat | 798 | Median instead of mean (APR median: 780; portfolio median: 18) |
| core_ols_robustness | 102 | OLS country coefficients with varied compounding, top-N, or loss variable |
| core_portfolio_robustness | 17 | Mean portfolio with varied top-N or country subset |
| core_compounding | 5 | Single-dimension compounding method variation |
| core_country | 5 | Single-dimension country subset variation |
| core_topn | 4 | Single-dimension top-N lender cutoff variation |
| core_lossvariable | 1 | Single-dimension loss variable variation |
| core_formula | 1 | Single-dimension APR formula variation |
| core_sample | 1 | Single-dimension sample restriction variation |
| noncore_auxiliary | 32 | Study lender statistics (18) and correlations (14) |

## Classification Decisions

### Core Test Classifications

**APR Mean specifications (780 specs, 1 baseline + 779 core)**: These are the heart of the specification search. Each specification computes the mean annual percentage rate for a subset of MFIs using a specific combination of analytical choices. The 7 dimensions varied are:

1. **Sample restriction** (2 values): non_missing_3vars (require all 3 key variables) vs. non_missing_yield_portfolio (allow missing loss data). This tests sensitivity to missing data handling.

2. **Top-N lenders** (5 values): 3, 5, 10 (baseline), 15, or all MFIs per country. This tests sensitivity to how the "market rate" is defined -- whether it represents the largest or the full set of lenders.

3. **Country subset** (6 values): All 6 countries, weekly-payment only (4 countries), monthly-payment only (2 countries), excluding Mexico, high-income countries, low-income countries. This tests whether the finding depends on any single country.

4. **Compounding method** (6 values): Country-specific baseline, all weekly, all monthly, simple annual, continuous, biweekly. This is the most influential dimension, changing APR by 50-100%.

5. **Loss variable** (2 values): Loan loss rate (baseline) vs. write-off ratio. Tests sensitivity to which loss metric enters the APR formula.

6. **APR formula** (2 values): Roodman formula (baseline, adjusts for losses) vs. yield-only (ignores losses). Tests whether the loss adjustment matters.

All 780 APR mean specs are classified as core because they all test the same fundamental claim (the level of the mean APR) under alternative reasonable analytical choices. The cross-product of these 7 dimensions produces 2 x 5 x 6 x 6 x 2 x 2 = 1,440 potential combinations, but the yield-only formula makes the loss variable irrelevant (reducing some combinations), and the actual count of 780 specs (including 30 yield-only specs) reflects the realized search grid.

**APR Median specifications (780 specs, all core)**: These replace the summary statistic with the median, testing sensitivity to outliers. Since Mexico has extremely high APR (mean ~188%), the median is much lower than the mean (37.2% vs. 55.2% across specs). All 780 are classified as core because the median is a standard robustness check for the same quantity (central tendency of APR). They are assigned to G1 because they test the same claim about APR levels.

**OLS Country Coefficients (108 specs, 6 baseline + 102 core)**: These regress APR on country dummies (India as reference category) and vary 3 dimensions: top-N (5 or 10), compounding (baseline or simple), and loss variable (loan loss rate or write-off ratio). All are classified as core because they test the same cross-country comparison claim under alternative analytical choices. The 6 baseline specs correspond to the 5 country coefficients plus the India intercept at baseline settings.

**Portfolio Size (36 specs, 1 baseline + 35 core)**: These compute mean or median gross loan portfolio across varying top-N and country subsets. All are classified as core because they test the same market characterization claim.

### Non-Core Classifications

**Study Lender Statistics (18 specs)**: These compute mean, median, and standard deviation of loan loss rate and write-off ratio for the 6 specific study lenders (not the top-N market lenders) across different year filters (all years, baseline only, 2012 only). These are classified as non-core because they address a different analytical question -- characterizing the study lenders specifically, rather than the broader MFI market. They are supplementary to the main Table 1 findings.

**Correlation Specifications (14 specs)**: These compute Pearson/Spearman correlations between portfolio size and APR (8 specs) or between loan loss rate and yield (6 specs). These are classified as non-core because correlations test a fundamentally different claim (the relationship between variables) rather than the level of any individual statistic. The paper does not emphasize these correlations as primary findings.

## Robustness Assessment

### G1: Mean APR Robustness

The mean APR is remarkably robust in qualitative terms:
- **100% of 780 APR mean specs show mean APR > 20%**, confirming that microcredit rates are far above zero regardless of analytical choices.
- **Range**: [0.207, 1.526] -- the minimum mean APR (20.7%) occurs under simple annual compounding with monthly-payment countries only, while the maximum (152.6%) occurs with continuous compounding for weekly-payment countries only.
- **Mean across specs**: 0.552 (55.2%)
- **Standard deviation across specs**: 0.320

Key sensitivity findings:
- **Compounding method** is the single largest source of variation. Simple annual compounding reduces APR from 60% to 39%; all-weekly compounding increases it to 60.4%.
- **Excluding Mexico** drops the mean APR from 60% to 31%, reflecting Mexico's extreme outlier status.
- **Top-N cutoff** has moderate effects: top 3 gives 80% APR, top 15 gives 59%, and all lenders gives 57%.
- **Loss variable** has minimal impact (< 5% change).

### G2: OLS Country Coefficient Robustness

- **Mexico premium positive and significant (p < 0.05) in all 18 Mexico-specific specs** -- the most robust cross-country finding.
- Mexico coefficient ranges from 0.609 (top 10, simple compounding, write-off ratio) to 2.533 (top 5, baseline compounding, write-off ratio).
- Morocco and Mongolia coefficients are consistently positive and significant.
- Ethiopia coefficient is borderline: p = 0.051 at baseline, and insignificant (p > 0.05) in 3 of 18 specs.
- Bosnia coefficient is consistently positive and significant.

### G3: Portfolio Size Robustness

- Mean portfolio ranges from $46M (top 5, low-income countries) to $201M (top 5, high-income countries).
- Median portfolio is consistently lower than mean, indicating right-skewed distribution.

## Notable Issues

### 1. This is a descriptive statistics paper
Unlike most specification searches, there are no causal claims, no treatment effects, and no hypothesis testing in the primary analysis. The APR mean and portfolio size specs have no p-values. The robustness question is whether the descriptive characterization of microfinance markets is sensitive to analytical choices, not whether a treatment effect is significant.

### 2. Compounding assumption drives most variation
The choice between simple annual, monthly, weekly, and continuous compounding mechanically changes the APR through the exponential compounding formula. This is not a specification choice that could be "p-hacked" -- it is a transparent methodological decision that should be disclosed. The paper's use of country-specific compounding (matching actual payment frequency) is a reasonable default.

### 3. Mexico as an extreme outlier
Mexico dominates the cross-country variation. Compartamos Banco's high rates (mean APR ~188%) are well-known in the microfinance literature. Excluding Mexico halves the mean APR. The paper discusses this prominently and the specification search confirms it.

### 4. Large spec count from combinatorial explosion
The 1,736 specifications arise primarily from the 7-dimensional grid of analytical choices, most of which are standard "researcher degrees of freedom" for descriptive statistics. The large count does not reflect cherry-picking risk; rather, it shows that the paper's findings are robust to a comprehensive grid of alternatives.

### 5. No convergence or estimation failures
All 1,736 specifications produced valid estimates. This is expected given the simplicity of the computations (summary statistics and simple OLS).

## Recommendations

1. **Focus on the APR mean specs (G1) for specification curve analysis**: These 780 specs cover the paper's primary finding with a well-structured grid of alternatives.

2. **The Mexico premium (G2) is the most testable claim**: With 18 OLS specs all showing significance, this is a clean robustness result.

3. **Non-core specs (32) are genuinely supplementary**: The study lender statistics and correlations address different questions and should not be included in a specification curve of the baseline claims.

4. **Compounding should be flagged as a methodological choice**: Since it mechanically drives most variation, it may be useful to present the specification curve conditional on compounding method.
