# Verification Report: 217741-V1

## Paper: AI and Women's Employment in Europe
- **Journal**: AER Papers and Proceedings 2025
- **Method**: Cross-sectional WLS with sector and country fixed effects
- **Data**: European Labour Force Survey, 16 countries, pooled 2011-2019

---

## Baseline Groups

### G1: Webb AI Exposure Measure
- **Baseline spec_id**: baseline_Webb
- **Claim**: Higher AI exposure (Webb patent-based percentile measure) is associated with positive changes in female employment shares
- **Treatment**: PCT_aiW
- **Outcome**: DHSshEmployee (DHS percent change in female employment share)
- **Baseline coefficient**: 0.076, SE: 0.042, p=0.068 (marginally insignificant)

### G2: Felten AI Exposure Measure
- **Baseline spec_id**: baseline_Felten
- **Claim**: Higher AI exposure (Felten ability-application percentile measure) is associated with positive changes in female employment shares
- **Treatment**: PCT_aiF
- **Outcome**: DHSshEmployee
- **Baseline coefficient**: 0.185, SE: 0.053, p=0.0005 (strongly significant)

---

## Classification Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **136** |
| **Core tests** | **114** |
| **Non-core tests** | **22** |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 4 | 2 baselines + 2 unweighted OLS |
| core_controls | 2 | No controls/FE specifications |
| core_fe | 4 | Sector-only and country-only FE |
| core_sample | 96 | LOO sectors, LOO countries, country-group subsamples, occupation exclusions, year variations, winsorization/trimming |
| core_inference | 4 | Clustering by country only or sector only |
| core_funcform | 4 | Quadratic and log treatment |
| noncore_heterogeneity | 18 | 12 sector-specific, 4 exposure-quartile, 2 country-interaction |
| noncore_alt_outcome | 2 | Female share level (shWomen) instead of DHS change |
| noncore_alt_treatment | 2 | Both AI measures included jointly (horse race) |

---

## Classification Rationale

### Core tests (114 specs)
The majority of specifications are core tests because the specification search focused on reasonable robustness checks:
- **Leave-one-out country** (32 specs): Dropping each of the 16 countries one at a time, for both Webb and Felten measures. These preserve the full-sample estimand.
- **Leave-one-out sector** (10 specs): Dropping each of 5 sectors, both measures. Preserves the cross-sector estimand.
- **Exclude-occupation** (18 specs): Dropping each 1-digit occupation, both measures.
- **Country-group subsamples** (16 specs): Splitting by LFP participation, education, upskilling. These are substantive sample restrictions but test the same directional claim.
- **Year variations** (16 specs): Using different base years (2012-2019) instead of 2011. Same estimand, different time window.
- **Winsorization/trimming** (4 specs): Alternative outlier treatment.
- **FE variations** (4 specs): Sector-only, country-only FE.
- **No-controls** (2 specs): No FE at all.
- **Clustering variations** (4 specs): Country-only or sector-only clustering.
- **Unweighted** (2 specs): OLS instead of WLS.
- **Functional form** (4 specs): Quadratic and log transformations of treatment.

### Non-core: Heterogeneity (18 specs)
- **Sector-specific** (12 specs: sec1-sec6 for each measure): Running the regression within a single sector (e.g., Agriculture only, Financial Services only) changes the estimand fundamentally from the pooled cross-sector effect to a within-sector effect. The paper discusses these as heterogeneity results.
- **Exposure-quartile** (4 specs): Restricting to top or bottom quartile of AI exposure. These are nonlinear heterogeneity tests, not robustness of the main linear effect.
- **Country-interaction** (2 specs): Adding country x AI-exposure interactions changes the treatment variable to a country-specific interaction term, not the main effect.

### Non-core: Alternative outcome (2 specs)
- robust/outcome/shWomen_Webb and robust/outcome/shWomen_Felten: These use the level of female employment share rather than the DHS change. This is a fundamentally different estimand.

### Non-core: Alternative treatment (2 specs)
- robust/treatment/both_measures_Webb and robust/treatment/both_measures_Felten: Including both AI measures simultaneously in a horse-race regression changes interpretation from the marginal effect of one measure to a conditional-on-the-other effect.

---

## Top 5 Most Suspicious Rows

1. **robust/outcome/shWomen_Webb**: p-value reported as ~8.88e-280 which is numerically zero. This extreme p-value suggests either a data issue or that the coefficient (-0.004) is picking up a near-mechanical relationship. The outcome variable shWomen (level) differs from baseline DHSshEmployee (change). Correctly classified as non-core alternative outcome.

2. **robust/form/log_treatment_Felten**: Coefficient of 5.11 is an order of magnitude larger than baseline (0.185). This is expected since the treatment is now log(percentile) rather than percentile, but the magnitude shift could confuse downstream analysis. Correctly classified as core_funcform.

3. **robust/form/log_treatment_Webb**: Coefficient of 1.81, also much larger than baseline (0.076) due to log transformation. Not significant (p=0.17). Correctly classified as core_funcform.

4. **robust/het/country_interaction_Webb and robust/het/country_interaction_Felten**: Treatment variable is PCT_aiW_cty1 / PCT_aiF_cty1, which is a country-specific interaction coefficient, not the main effect. The reported coefficient (0.089 / 0.239) appears to be for country 1 (Austria) specifically. Correctly classified as noncore_heterogeneity.

5. **robust/sample/sec1_Webb**: Coefficient is -0.30 (negative) for Agriculture sector only. This is a single-sector subsample with fundamentally different dynamics. The sign reversal is consistent with the paper discussion of sector heterogeneity. Correctly classified as noncore_heterogeneity.

---

## Recommendations for Spec-Search Script

1. **Sector-specific regressions should be pre-labeled as heterogeneity**: The spec_tree_path uses robustness/sample_restrictions.md for sector-specific subsamples (sec1-sec6), but these are better understood as heterogeneity analyses since they change the estimand. Future runs should use a heterogeneity/ prefix.

2. **The shWomen outcome p-value needs investigation**: The p-value of ~8.88e-280 for robust/outcome/shWomen_Webb is implausibly small and may indicate a numerical issue in the estimation code (e.g., the SE is near-zero because the coefficient is mechanically determined).

3. **Both-measures horse race could be core**: A reasonable argument exists that including both measures simultaneously is a robustness check (testing whether the preferred measure survives conditional on the other). However, it is classified as non-core because the interpretation of the coefficient changes when conditioning on a correlated measure.

4. **Country-group subsamples (clas1-clas5) are borderline**: These restrict to roughly half the countries based on country-level characteristics. While classified as core_sample, they could be considered heterogeneity analyses if the paper claim is specifically about the pooled effect. The confidence for these is set at 0.75.

5. **Exposure-quartile heterogeneity specs report the linear effect within a nonlinear subsample**: The specs robust/het/high_exposure_* and robust/het/low_exposure_* restrict to the top/bottom quartile of AI exposure and re-estimate the linear effect within that restricted range. This is a heterogeneity/nonlinearity test, not a robustness check of the main linear effect.
