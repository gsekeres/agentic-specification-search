# Verification Report: 193942-V1

**Paper**: Effective Health Aid: Evidence from Gavi Vaccine Program
**Journal**: AEJ: Policy
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Vaccine Coverage (Primary Claim)
- **Claim**: Gavi vaccine program introduction increases vaccine coverage rates in recipient countries.
- **Baseline spec_id**: baseline
- **Outcome**: coverage (vaccine coverage percentage)
- **Treatment**: post (binary indicator for post-Gavi-introduction)
- **Expected sign**: Positive (+)
- **Baseline estimate**: 3.60 pp (SE=0.64, p<1e-7)
- **Fixed effects**: country-disease + cohort-disease + country-cohort
- **Clustering**: country

### G2: Child Mortality (Secondary Claim)
- **Claim**: Gavi vaccine program introduction reduces child mortality rates.
- **Baseline spec_id**: did/outcome/mortality_baseline
- **Outcome**: rate (postneonatal mortality rate)
- **Treatment**: post
- **Expected sign**: Negative (-)
- **Baseline estimate**: +654.71 (SE=266.13, p=0.015) -- WRONG SIGN (positive, not negative)
- **Fixed effects**: country-disease + cohort-disease + country-cohort
- **Clustering**: country

**Note on G2**: The positive mortality coefficient is counter to the paper claim that Gavi reduces mortality. This likely reflects data reconstruction issues, as the SPECIFICATION_SEARCH.md notes that derived data had to be reconstructed from raw sources as Stata-generated intermediate files were not available.

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **59** |
| Baselines | 2 |
| Core tests (incl. baselines) | 40 |
| Non-core tests | 19 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_method | 2 (baselines) |
| core_fe | 4 |
| core_sample | 29 |
| core_inference | 2 |
| core_funcform | 3 |
| noncore_heterogeneity | 17 |
| noncore_placebo | 1 |
| noncore_alt_treatment | 1 |
| noncore_alt_outcome | 0 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

### By Baseline Group

| Group | Baselines | Core (non-baseline) | Non-core | Total |
|-------|-----------|---------------------|----------|-------|
| G1 (coverage) | 1 | 36 | 15 | 52 |
| G2 (mortality) | 1 | 1 | 4 | 6 |
| (alt treatment, G1) | -- | -- | 1 | 1 |

Note: did/treatment/intensity is assigned to G1 for baseline_group_id but marked non-core because the treatment definition change (binary to continuous years_since_intro) changes the estimand. The non-core heterogeneity count of 17 includes 9 disease-specific vaccine splits, 3 regional splits, 2 baseline-coverage splits, and 3 mortality cause-of-death splits.

---

## Core Tests Summary (G1 only, excluding baseline)

The 36 core tests for the coverage claim break down as:
- **FE variations** (4): unit-only, time-only, two-way, none
- **Sample restrictions** (29): vaccine-type subsets (2), Gavi-recipients-only (1), time-period restrictions (3), drop-one-year (5), time-windows (4), exclude-one-vaccine (4), exclude-one-country (7), ever-treated-only (1), winsorize/trim (2)
- **Inference variations** (2): robust SE, cluster at disease level
- **Functional form** (2): log coverage, IHS coverage

Among the 36 G1 core tests:
- 35 of 36 show positive coefficients (time-only FE is negative due to omitted variable bias)
- 30 of 36 are significant at 5%
- Coefficients range from 1.08 (1990-2010 window) to 44.40 (new vaccines only)

---

## Top 5 Most Suspicious Rows

### 1. did/outcome/mortality_baseline (WRONG SIGN)
- **Issue**: Mortality baseline coefficient is +654.71, meaning Gavi introduction is associated with HIGHER postneonatal mortality. The paper claims Gavi REDUCES mortality.
- **Likely cause**: Data reconstruction from raw sources. The intermediate Stata data files were not available, and the mortality variable may have been constructed incorrectly.
- **Recommendation**: Verify the mortality data construction against the original Stata code. Check whether rate is per-100,000 or another unit, and whether the country-disease-cohort merge is correct.

### 2. robust/placebo/bcg_unaffected (PLACEBO FAILS)
- **Issue**: BCG placebo test shows a significant POSITIVE effect (coef=5.14, p<0.001). BCG is not Gavi-funded, so its coverage should NOT increase post-Gavi-introduction.
- **Likely cause**: (a) General trends in coverage improvements coinciding with Gavi timing, (b) spillover effects from Gavi infrastructure, or (c) treatment variable post_any captures any Gavi introduction in the country.
- **Recommendation**: Investigate whether post_any correctly identifies Gavi introduction timing. Compare BCG trends for Gavi vs non-Gavi countries.

### 3. robust/heterogeneity/mortality_diarrhea (CONTRADICTS BASELINE SIGN)
- **Issue**: Diarrhea mortality coefficient is -872.55 (p=0.053), which IS the expected negative sign. This contradicts the all-cause mortality baseline showing a positive sign.
- **Likely cause**: The all-cause mortality may be poorly measured, while diarrhea-specific mortality (directly vaccine-preventable) shows the expected effect.
- **Recommendation**: Re-examine the mortality outcome construction. Cause-specific results may be more reliable than the aggregate.

### 4. robust/sample/new_vaccines (VERY LARGE COEFFICIENT)
- **Issue**: Coefficient of 44.40 is 12x larger than the baseline (3.60).
- **Likely cause**: Mechanical effect -- PCV and ROTA had zero coverage pre-introduction, so any post-introduction coverage contributes entirely to the treatment effect.
- **Recommendation**: Valid but interpretation differs from baseline. Consider flagging the magnitude difference.

### 5. did/fe/time_only (WRONG SIGN)
- **Issue**: With only time FE (cohort-disease), coefficient is -5.60 (p<0.001) -- opposite sign of baseline.
- **Likely cause**: Omitted variable bias without unit FE. Countries that adopted Gavi-funded vaccines may have lower average coverage.
- **Recommendation**: Expected behavior demonstrating importance of unit FE. Not a data error.

---

## Recommendations for Fixing the Spec-Search Script

1. **Mortality data pipeline**: The mortality baseline (G2) has a wrong-sign coefficient. Before including G2 in aggregate analysis, verify the mortality data pipeline against original Stata replication code. Consider dropping G2 specs if data cannot be validated.

2. **Placebo interpretation**: The BCG placebo test failing is substantively important. The search correctly ran this test but the result raises questions about identification.

3. **Heterogeneity classification**: The search correctly distinguishes between disease-specific heterogeneity (robust/heterogeneity/disease_*) and leave-one-out sample checks (robust/sample/exclude_*). This is appropriate.

4. **Treatment variation**: The did/treatment/intensity spec changes the estimand from level shift to dose-response and should not be pooled with binary-treatment core specs.

5. **Missing specifications**: No controls-variation specs are needed since the paper identification relies entirely on fixed effects without additional regression controls.
