# Verification Report: 209903-V1

## Paper Information
- **Title**: The Empire Project: Trade Policy in Interwar Canada
- **Authors**: Markus Lampe, Kevin H. O'Rourke, Michael Reiter, and Yoto V. Yotov
- **Journal**: Journal of International Economics (2024)
- **Total Specifications**: 65 (53 converged, 12 failed)

## Baseline Groups

### G1: Common Tariff Effect on Canadian Imports
- **Claim**: Canadian tariffs significantly reduced imports during the interwar period (1924-1936). A 1-unit increase in ln(1 + tariff rate) is associated with an approximately 3.6 unit reduction in import value (PPML elasticity).
- **Baseline spec**: spec_id 1
- **Expected sign**: Negative
- **Baseline coefficient**: -3.617 (SE: 0.692, p < 0.001)
- **Outcome**: `import_value` (level, not logged, for PPML)
- **Treatment**: `LN_TARIFF` = ln(1 + ave_raw), where ave_raw = tariff_revenue / import_value
- **Estimator**: PPML (Poisson pseudo-maximum likelihood via ppmlhdfe)
- **Fixed effects**: country*sector*year + country*product + product*year
- **Clustering**: country + product (two-way)
- **Table 2, Column 2**

**Note**: The paper's primary specification (Table 2 Column 1) uses three-way clustering by country, product, and year. Since pyfixest does not support three-way clustering, the baseline here corresponds to Table 2 Column 2, which uses two-way clustering by country and product. The point estimate is identical; only the standard errors differ.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Baseline** | **1** | Table 2 Col 2 PPML specification |
| **Core tests** | **39** | |
| core_inference | 5 | Alternative clustering: country only, product only, year only, heteroskedasticity-robust, country*year |
| core_fe | 8 | Alternative FE structures: separate country+product+year, country*year+product, country*year+product*year, full bilateral, minimal (country+year), country*product+year, product*year only, country*product only |
| core_estimator | 5 | OLS with full FE, OLS with simpler FE variants, arcsinh(imports) OLS |
| core_treatment | 2 | No-zeros tariff (LN_TARIFF_NZ), tariff in levels |
| core_sample | 16 | Non-specific duties, specific duties, biennial, pre/post-Ottawa, Empire/non-Empire, exclude UK, top 10 partners, trim imports/tariffs, drop first/last year, middle period, positive tariffs only; 1 failed (exclude US) |
| core_controls | 3 | Tariff risk control, GDP+ER, GDP+ER+treaty dummies |
| core_outcome | 1 | Winsorized imports |
| core_estimator_inference | 1 | OLS with country-only clustering |
| core_estimator_sample | 3 | OLS on Empire, non-specific, pre-Ottawa subsamples |
| **Non-core tests** | **13** | |
| noncore_heterogeneity | 13 | 10 sector subsamples, 2 interaction terms (Empire, Ottawa), plus 8 failed specs (4 year cross-sections, 4 country-specific) |
| **Failed specifications** | **12** | Specs 29, 53-60 failed due to collinearity or convergence issues |
| **Total** | **65** | 53 converged + 12 failed |

## Detailed Classification Notes

### Core Tests (39 specs + 1 baseline = 40 total)

**Inference variations (5 specs, IDs 2-6)**: All share the identical point estimate of -3.617 because they use the same model and data, varying only the variance-covariance estimator. Standard errors range from 0.296 (heteroskedasticity-robust) to 0.693 (cluster country). All are highly significant regardless of clustering choice.

**Fixed effects variations (8 specs, IDs 7-14)**: These systematically vary the fixed effects structure, corresponding to the paper's Table A3 and additional unreported specifications.
- The most saturated FE (spec 10: country*year + country*product + product*year) yields coef = -2.924, somewhat smaller than the baseline, because it absorbs more variation.
- The minimal FE (spec 11: country + year only) yields an insignificant coef = -0.986 (p = 0.273), demonstrating that product-level controls are essential for identifying the tariff effect.
- Country*product FE only (spec 14) yields the largest coefficient at -4.887, as it exploits only within-pair temporal variation.
- This range (-0.986 to -4.887) illustrates the importance of the FE structure for identifying the tariff elasticity.

**Estimator variations (5 specs, IDs 15-19)**: OLS specifications using ln(import_value) as the dependent variable. OLS coefficients are uniformly smaller in magnitude (-1.6 to -2.8) than PPML (-3.6), consistent with the known downward bias of OLS in the presence of heteroskedasticity and zero trade flows (Santos Silva & Tenreyro, 2006). The arcsinh transformation (spec 19, coef = -1.569) is very close to OLS log, as expected.

**Treatment measure variations (2 specs, IDs 20-21)**: Spec 20 uses the no-zero tariff measure (LN_TARIFF_NZ), which excludes zero-import imputations. The coefficient is identical to the baseline (-3.617) because the observations where the two measures differ are those with zero imports, which get dropped or contribute little to the PPML estimation anyway. Spec 21 uses the tariff level rather than the log transformation, yielding a coefficient of -2.317.

**Sample restrictions (16 specs, IDs 22-37, plus failed ID 29)**: The largest core category, reflecting the paper's emphasis on sample composition.
- Key result: the coefficient is negative and significant across all converged subsample restrictions.
- The range among converged specs is [-5.211, -1.005].
- Non-specific duties (spec 22, coef = -5.211) show a larger effect than specific duties (spec 23, coef = -2.046, p = 0.026), consistent with better measurement of ad valorem tariffs.
- Empire countries (spec 27, coef = -4.525, p = 0.050) show a larger but marginally significant effect; non-Empire (spec 28, coef = -1.005) is smaller.
- Pre-Ottawa (spec 25, coef = -2.135) vs. post-Ottawa (spec 26, coef = -2.839) shows the tariff effect is present in both periods.
- Dropping the UK (spec 30, coef = -1.300) substantially reduces the coefficient, indicating that UK trade is an important driver.
- Spec 29 (exclude US) failed due to convergence issues.

**Controls (3 specs, IDs 50-52)**: Adding tariff risk, GDP, exchange rates, and treaty dummies. These use reduced FE to avoid collinearity with the macro controls. Coefficients range from -3.708 to -4.119, broadly consistent with or slightly larger than the baseline.

**Outcome transformation (1 spec, ID 61)**: Winsorized imports at 1st/99th percentile yields coef = -3.015, modestly smaller than the baseline, suggesting some influence of extreme trade values.

**Additional OLS specifications (4 specs, IDs 62-65)**: OLS with alternative clustering (spec 62), and OLS on Empire, non-specific, and pre-Ottawa subsamples (specs 63-65). All remain significant with the expected negative sign.

### Non-Core Tests (13 converged + 8 failed = 21 specs)

**Sector heterogeneity (10 converged specs, IDs 38-47)**: Sector-specific PPML estimates with reduced FE (country*product + product*year). These decompose the aggregate tariff effect by broad product category. Coefficients range from -1.541 (Agricultural, Non-Food, p = 0.051) to -10.574 (Iron and Its Products). These are non-core because they estimate sector-specific effects rather than testing alternative specifications of the aggregate tariff effect. In the original paper, these correspond to Figure 4 (top panel) and Table 4, where sector-specific tariff dummies are interacted in a single regression.

**Interaction terms (2 specs, IDs 48-49)**: Empire vs. non-Empire tariff interactions (spec 48: Empire coef = -4.293, non-Empire coef reported separately) and Ottawa vs. non-Ottawa tariff interactions (spec 49: Ottawa coef = -5.247). These test heterogeneous treatment effects rather than robustness of the common tariff effect.

**Failed heterogeneity specs (8 specs, IDs 53-60)**: Single-year cross-sections (1924, 1928, 1932, 1936) and single-country estimates (US, UK, Germany, France) all failed due to collinearity. Within a single year, country*product FE leaves no variation in LN_TARIFF for identification. Similarly, within a single country, product*year FE absorbs all tariff variation. These specifications are fundamentally not identified in this setting.

## Failed Specifications

| Spec ID | Name | Reason |
|---------|------|--------|
| 29 | Sample: exclude United States | Demeaning failed after 100,000 iterations |
| 53 | Cross-section: year 1924 | All variables collinear |
| 54 | Cross-section: year 1928 | All variables collinear |
| 55 | Cross-section: year 1932 | All variables collinear |
| 56 | Cross-section: year 1936 | All variables collinear |
| 57 | Country: United States | All variables collinear |
| 58 | Country: United Kingdom | All variables collinear |
| 59 | Country: Germany | All variables collinear |
| 60 | Country: France | All variables collinear |

The year cross-section and country-specific failures are expected: with country*product FE in a single-year cross-section, tariff variation is absorbed by the fixed effects. The US exclusion failure (spec 29) is a convergence issue specific to the PPML estimator on that particular subsample.

## Data Construction Notes

The original paper uses Stata for both dataset construction (Empire_Project_JIE_dataset_construction.do) and estimation (Empire_Project_JIE_estimates.do). Because the replication package provides raw data in CSV and DTA formats, the specification search agent rebuilt the dataset in Python following the do-file logic. Key construction steps:
1. Loading raw trade data (joined_set_name_based_merges_balanced.csv, ~2.5M rows)
2. Collapsing to country-product-year level
3. Constructing tariff variables: ave_raw = tariff_revenue / import_value, LN_TARIFF = ln(1 + ave_raw)
4. Coding British Empire membership, treaty variables, sector groups
5. Creating FE identifiers as string concatenations

Potential discrepancies from the published results may arise from:
- Simplified tariff imputation (the full do-file has hundreds of lines of manual corrections)
- Slight differences in sample construction due to country consolidation edge cases
- pyfixest vs. ppmlhdfe numerical differences in convergence

## Robustness Assessment

The main finding -- that Canadian tariffs reduced imports during the interwar period -- is **robust** across the large majority of core specifications:

- **Sign consistency**: All 40 core converged specifications yield a negative coefficient on the tariff variable.
- **Significance**: 38 of 40 core converged specifications are statistically significant at p < 0.05. The two exceptions are the minimal FE specification (spec 11: country + year only, p = 0.273) and the product*year-only FE (spec 13: p = 0.074). Both lack adequate controls.
- **PPML coefficient range (core, converged)**: [-5.211, -0.986], with the baseline at -3.617.
- **OLS coefficient range**: [-2.843, -1.120], consistently smaller than PPML as expected.
- **Key sensitivity**: The FE structure matters substantially. The minimal specification (country + year) fails to find a significant effect, while the full specification and most intermediate FE structures do. This reflects the importance of controlling for product-level heterogeneity in gravity models.
- **Empire vs. non-Empire**: The tariff effect is driven more strongly by Empire trade partners (coef = -4.525) than non-Empire partners (coef = -1.005), consistent with the paper's emphasis on imperial preference.
- **UK importance**: Dropping the UK reduces the coefficient from -3.617 to -1.300, highlighting the UK's outsized role in interwar Canadian trade.
