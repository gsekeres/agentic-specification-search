# Verification Report: 130141-V1

## Paper: News Shocks under Financial Frictions
**Authors**: Gortz, Tsoukalas, and Zanetti  
**Journal**: American Economic Journal: Macroeconomics  
**Method**: Structural VAR (Cholesky identification as proxy for medium-run identification)

---

## Baseline Groups

### G1: TFP News Shocks and GDP Variance
- **Claim**: TFP news shocks explain a significant share (15-25%) of GDP forecast error variance at business cycle frequencies (20 quarters) in a 7-variable SVAR with financial frictions.
- **Baseline spec_id**: baseline
- **Baseline coefficient (FEVD share)**: 0.223 (22.3%)
- **Expected sign**: Positive
- **Key parameters**: 7-variable VAR (TFP, GDP, Consumption, Hours, GZ spread, SP500, Inflation), 5 lags, Cholesky identification with TFP ordered first, full sample 1984Q2-2017Q1

**Important caveat**: The paper actually uses medium-run identification (maximizing FEVD at specific horizons), not Cholesky. The replication uses Cholesky as a computationally tractable proxy. This means all specifications are approximate rather than exact replications of the paper's methodology.

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **58** |
| Baseline | 1 |
| Core tests | 49 |
| Non-core tests | 5 |
| Invalid | 4 |
| Unclear | 0 |

### Core Test Breakdown

| Core Category | Count |
|---------------|-------|
| core_method (lag length, ordering, info criteria) | 17 |
| core_controls (VAR variable composition) | 20 |
| core_sample (time period, rolling windows) | 12 |
| core_funcform | 0 |
| core_fe | 0 |
| core_inference | 0 |

### Non-Core Breakdown

| Non-Core Category | Count |
|-------------------|-------|
| noncore_placebo | 2 |
| noncore_heterogeneity | 2 |
| noncore_alt_outcome | 1 |
| noncore_alt_treatment | 0 |
| noncore_diagnostic | 0 |

### Invalid Specs

| Invalid Category | Count |
|------------------|-------|
| invalid | 4 |

---

## Top 5 Most Suspicious Rows

### 1. robust/outcome/hours (INVALID)
- **Issue**: Labeled as "Hours as main outcome" but the reported FEVD (0.22297) is numerically identical to the baseline. Inspection of the coefficient_vector_json shows the same fevd_gdp_tfp_h20 value. The only change is that Hours is moved to the second position in the variable list, which in a Cholesky decomposition does not change the FEVD of GDP with respect to the TFP shock (since TFP remains first). This is NOT a genuine alternative outcome specification -- it is just a variable reordering that produces an identical result.

### 2. robust/outcome/consumption (INVALID)
- **Issue**: Same problem as robust/outcome/hours. Labeled as "Consumption as main outcome" but FEVD is identical to baseline (0.22297). Consumption is moved to second position but TFP stays first, so FEVD of GDP w.r.t. TFP shock is unchanged.

### 3. robust/outcome/sp500 (INVALID)
- **Issue**: Same problem. SP500 moved to second position, FEVD identical to baseline (0.22297). Not a genuine alternative outcome.

### 4. robust/loo/drop_gdp (INVALID)
- **Issue**: GDP is dropped from the VAR system, but the outcome variable is still reported as "GDP (FEVD share)". If GDP is not in the VAR, the FEVD of GDP cannot be computed. The reported FEVD (0.125) likely refers to the FEVD of TFP (the first remaining variable in the Cholesky order) or some other variable, but this is mislabeled.

### 5. robust/placebo/shuffled_tfp (NON-CORE, concerning result)
- **Issue**: This placebo test shuffles the TFP series randomly, destroying all temporal structure. Despite this, the FEVD is still 12.1%, which is more than half the baseline value. This suggests substantial spurious variance attribution in the Cholesky framework, potentially undermining the baseline finding. While correctly classified as a placebo (non-core), the high FEVD is a concerning signal about the methodology's reliability.

---

## Duplicate Specifications

Several specifications are exact or near-exact duplicates:

1. **svar/lags/aic_optimal**, **svar/lags/bic_optimal**, **svar/lags/hq_optimal** are all identical to **svar/lags/p1** (all information criteria select p=1). These produce identical coefficients (0.07479).

2. **robust/loo/drop_gzspr** is identical to **svar/vars/no_financial** (both drop the GZ spread, producing the same 6-variable VAR). Coefficient: 0.2458.

3. **robust/loo/drop_sp500** is identical to **svar/vars/no_sp500** (both drop SP500). Coefficient: 0.2209.

4. **svar/control/add_sp500** (6-var control progression) is identical to **svar/vars/no_financial** and **robust/loo/drop_gzspr** (all are the same 6-variable VAR without GZ spread). Coefficient: 0.2458.

5. **svar/control/add_consumption** is identical to **svar/vars/small_var4** (both are 4-variable VARs with TFP, GDP, Hours, Consumption, though variable ordering may differ). Coefficient: 0.2238.

After accounting for duplicates, there are approximately 45-46 truly distinct specifications rather than 58.

---

## Recommendations for Fixing the Specification Search Script

1. **Fix alternative outcome specifications**: The robust/outcome/hours, robust/outcome/consumption, and robust/outcome/sp500 specs do not actually change the FEVD target variable. To properly test alternative outcomes, the script should compute FEVD for the alternative variable (e.g., FEVD of Hours w.r.t. TFP shock) rather than just reordering the VAR and still computing GDP FEVD. This requires changing the FEVD target index, not just the variable ordering.

2. **Fix drop_gdp specification**: When GDP is dropped from the VAR, the script should either (a) skip this specification as invalid, or (b) clearly label which variable's FEVD is being reported.

3. **Remove or flag duplicate specifications**: The three IC-optimal lag specs (aic_optimal, bic_optimal, hq_optimal) are all identical (p=1). Either consolidate them into one spec or flag them as duplicates. Similarly, cross-category duplicates (drop_gzspr = no_financial = add_sp500) should be noted.

4. **Consider placebo calibration**: The shuffled TFP placebo showing 12% FEVD suggests the Cholesky FEVD metric has a high noise floor. The script could add a proper Monte Carlo placebo distribution to calibrate what FEVD values are spurious.

5. **Acknowledge identification gap**: The SPECIFICATION_SEARCH.md correctly notes that Cholesky identification differs from the paper's medium-run identification. However, this should be more prominently flagged, as it means all specifications are testing a different identification assumption than the paper's actual claim.

---

## Overall Assessment

The specification search is well-structured and covers the key dimensions of variation for a VAR analysis: lag length, variable composition, sample period, and variable ordering. Of 58 specifications, 49 are classified as valid core tests of the baseline claim, 5 are non-core (placebos, heterogeneity, alternative outcome), and 4 are invalid (mis-implemented outcome changes and an undefined drop-GDP spec).

The main finding -- that TFP news shocks explain a substantial share of GDP variance -- shows considerable sensitivity to sample period (2.4% excluding crisis vs. 59% post-2000), lag length (7.5% at p=1 vs. 42.6% at p=8), and Cholesky ordering (5.9% reversed vs. 22.3% baseline). The rolling window analysis shows high instability (0.5% to 62.5%). These sensitivities, combined with the 12% FEVD from the shuffled-TFP placebo, suggest the result is moderately robust but strongly influenced by the 2008 financial crisis period and specification choices.
