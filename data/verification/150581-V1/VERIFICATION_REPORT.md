# Verification Report: 150581-V1

## Paper Overview

**Title**: Wage Cyclicality and Labor Market Sorting
**Paper ID**: 150581-V1
**Core Claim**: Wages are negatively related to the unemployment rate (negative wage-unemployment semi-elasticity), estimated via panel fixed effects regression of log hourly wages on the unemployment rate using NLSY79 data.
**Important Caveat**: The specification search used **simulated data** due to NLSY data access constraints, as noted in SPECIFICATION_SEARCH.md.

---

## Baseline Groups

### G1: Negative wage-unemployment semi-elasticity

- **Baseline spec_ids**: `baseline/table2_col1`, `baseline/table2_col2`, `baseline/table2_col3`, `baseline/table2_col4`, `baseline/table2_col5`
- **Outcome**: `lhrp2` (log hourly pay)
- **Treatment**: `unempl` (aggregate unemployment rate)
- **Expected sign**: Negative
- **Fixed effects**: Individual, industry-year, occupation-year
- **Clustering**: Individual
- **Notes**: Columns 1-5 of Table 2 progressively add controls (transition indicators, mismatch, full interactions). Coefficients range from -0.0044 to -0.0065. All are highly significant. These represent a single canonical claim with varying control sets.

---

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 61 |
| Baseline specifications | 5 |
| Core test specifications (is_core_test=1) | 51 |
| Non-core specifications (is_core_test=0) | 10 |
| Invalid specifications | 0 |
| Unclear specifications | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 16 |
| core_sample | 17 |
| core_inference | 5 |
| core_fe | 6 |
| core_funcform | 7 |
| noncore_alt_treatment | 2 |
| noncore_heterogeneity | 5 |
| noncore_placebo | 3 |

---

## Classification Rationale

### Core tests (51 specs)

**Controls variations (16)**: Leave-one-out tests (dropping age, education, tenure, mismatch, time trend, month FE) and control progression tests (minimal to full). All preserve the same outcome, treatment, FE, and sample - only varying control sets. This is the largest category and a strong test of the claim's robustness to confounders.

**Sample restrictions (17)**: Time period exclusions (drop recession, pre/post-2000, post-1985, drop 1990s/2000s/2010s), demographic subsamples (young/older workers, college/noncollege, white, tenure under 10y, high/low mismatch), and winsorization (1%, 5%). All preserve the core estimand on subsets of the data. Some demographic splits (college vs noncollege, young vs older) border on heterogeneity analysis, but since they estimate the main effect within a subsample (not an interaction), they qualify as core sample restrictions.

**Inference variations (5)**: Clustering by heteroskedasticity-robust, year-month, year, region, or IID. All share the same point estimate (-0.00446) and only differ in standard errors. Clearly core inference robustness.

**FE structure variations (6)**: No FE (OLS), individual FE only, year FE only, individual+year FE, individual+industry FE, individual+occupation FE. These are important because the FE structure is critical for identification: specs with only individual FE (or individual+industry or individual+occupation) yield positive coefficients (+0.0056), while the full baseline FE structure yields negative coefficients. This demonstrates that industry-year and occupation-year FE are essential for the paper's identification strategy.

**Functional form (7)**: IHS wages, standardized wages, level wages, standardized unemployment, cubic tenure, cubic age, log mismatch. All preserve the core claim (wage-unemployment relationship) while varying how variables enter the regression.

### Non-core tests (10 specs)

**Alternative treatments (2)**: `robust/treatment/regional_unemp` uses regional (not aggregate) unemployment, which changes the causal object. `robust/treatment/high_unemp_dummy` uses a binary indicator for high unemployment, fundamentally changing the estimand from a semi-elasticity to a discrete comparison. Both are informative but test different questions.

**Heterogeneity (5)**: Interaction terms (unemployment x education, race, region, mismatch level, triple interaction with education). These add interaction terms to the model, meaning the reported coefficient on `unempl` represents the effect for the omitted category, not the average effect. They test a different question (does wage cyclicality vary across groups?) rather than robustness of the main effect.

**Placebo tests (3)**: Lagged unemployment, lead unemployment, and randomized transition indicators. These are diagnostic tests of the identification strategy, not alternative estimates of the main claim.

---

## Top 5 Most Suspicious Rows

1. **`robust/estimation/id_fe_only` (positive coefficient +0.0056)**: With only individual FE, the unemployment coefficient flips sign to positive. This is a well-known identification issue in wage cyclicality research (composition bias). The coefficient is valid but shows the FE structure is essential, not a problem with extraction.

2. **`robust/estimation/no_fe` (positive coefficient +0.0061)**: OLS without any fixed effects also yields a positive coefficient. Same identification concern as above. Extraction is correct.

3. **`robust/estimation/id_industry_fe` and `robust/estimation/id_occ_fe` (positive coefficients +0.0056 each)**: Adding only industry or occupation (not interacted with year) FE to individual FE still yields positive coefficients. The coefficients and standard errors are nearly identical to the id_fe_only spec, suggesting these non-interacted FE are not absorbing relevant variation. This may indicate that in the simulated data, the industry and occupation FE are not well-constructed (possible simulation artifact).

4. **`robust/placebo/random_transitions` (significant negative coefficient -0.0048, p<0.001)**: The placebo test with randomized transition indicators still shows a significant negative effect. This is expected because the main effect of unemployment on wages does not operate solely through transition indicators, but it is worth noting that the "placebo" does not null the result.

5. **`baseline/table2_col2` and `baseline/table2_col3` (identical coefficients -0.00647)**: These two baseline specs have exactly identical coefficients, standard errors, p-values, n_obs, and R-squared despite col3 adding the mismatch variable. This suggests that either: (a) the mismatch variable has zero variance or is perfectly collinear in the simulated data, or (b) there is a data generation issue. This is likely a simulation artifact.

---

## Recommendations

1. **Simulated data is the main concern**: The use of simulated rather than actual NLSY79 data undermines the validity of all results. The identical coefficients for baseline cols 2 and 3, and the nearly identical coefficients across the individual+industry/occupation FE specs, suggest the simulated data may not capture key features of the real data. If possible, the analysis should be re-run on the actual data.

2. **Consider whether heterogeneity specs should be included as core**: The paper's richer claim is about heterogeneity in wage cyclicality by transition type and mismatch. If the verifier decides this is a separate baseline claim, the heterogeneity interaction specs could be reclassified as core tests of that claim. I classified them as non-core because the reported coefficient is the main effect (not the interaction), so the interaction specs test a different question than the baseline.

3. **Regional unemployment treatment**: The `robust/treatment/regional_unemp` spec uses a different level of variation (regional vs aggregate). The paper's identification uses aggregate unemployment, so this is arguably an alternative identification strategy rather than a robustness check. Its insignificance (p=0.66) is informative but tests a different estimand.

4. **FE structure diagnostics**: The positive coefficients under simpler FE structures are important for understanding identification but make the FE variation specs somewhat unusual as "robustness checks" - they demonstrate the mechanism rather than robustness. They are classified as core because they vary the FE structure while preserving the estimand concept.

5. **No extraction errors detected**: All specifications appear correctly extracted with sensible values for coefficients, standard errors, sample sizes, and R-squared. No rows appear invalid or misattributed.
