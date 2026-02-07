# Verification Report: 130784-V1

## Paper
**Title**: Child Marriage Bans and Female Schooling and Labor Market Outcomes  
**Journal**: AER: Papers and Proceedings  
**Author**: Wilson (2020)

## Data Caveat
All results were generated using **simulated data** that matches the structure of the original DHS/MACHEquity datasets. Actual results require DHS data access. Coefficients are approximations.

---

## Baseline Groups

| Group ID | Claim | Expected Sign | Baseline spec_id(s) | Outcome | Treatment |
|----------|-------|---------------|---------------------|---------|-----------|
| G1 | Child marriage bans reduce probability of marriage before age 18 | - | baseline | childmarriage | bancohort_pcdist |
| G2 | Child marriage bans increase years of female education | + | did/sample_full/outcome_educ | educ | bancohort_pcdist |
| G3 | Child marriage bans increase female employment probability | + | did/sample_full/outcome_employed | employed | bancohort_pcdist |
| G4 | Child marriage bans increase the age at first marriage | + | did/sample_full/outcome_marriage_age | marriage_age | bancohort_pcdist |

G1 is the primary claim of the paper. G2-G4 are secondary/ancillary outcomes reported in the same framework.

---

## Summary Counts

| Metric | Count |
|--------|-------|
| **Total specifications** | 77 |
| **Baselines** | 4 |
| **Core tests** | 67 |
| **Non-core tests** | 10 |
| **Invalid** | 0 |
| **Unclear** | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_sample | 38 | Sample restrictions (age, urban/rural, country subsets, leave-one-out) |
| core_funcform | 13 | Alternative outcome thresholds (childmarriage17/16/15), log/IHS transforms |
| core_method | 10 | Treatment definition variations, baselines, binary treatment |
| core_fe | 3 | Fixed effects structure variations |
| core_inference | 3 | Clustering and SE variations |
| noncore_heterogeneity | 7 | Urban/rural, age group, and intensity subgroup splits |
| noncore_placebo | 3 | Placebo tests (childmarriage14, pre-ban cohort trends) |

---

## Top 5 Most Suspicious / Noteworthy Rows

1. **did/fe/twoway** (spec_id): This specification produces an **identical** coefficient and SE to the baseline. It appears to be a complete duplicate, suggesting the spec search created redundant entries. Not problematic per se, but inflates the apparent spec count.

2. **robust/cluster/countryregionurban**: Also produces an **identical** coefficient and SE to the baseline, because countryregionurban clustering is already the baseline clustering choice. This is a pure duplicate.

3. **robust/outcome/childmarriage14 / did/outcome/married_before_14**: Marriage before age 14 is well below typical ban thresholds (age 16-18). The near-zero, insignificant coefficient (p=0.67) suggests this functions more as a placebo test than a robustness check. Classified as noncore_placebo.

4. **robust/heterogeneity/by_urban_rural vs robust/sample/rural_only**: The heterogeneity rural split (coef=-0.0752, countryage FE only) is nearly identical to the rural-only sample restriction (coef=-0.0751, countryage+countryregionurban FE) but uses different fixed effects. The heterogeneity version drops countryregionurban FE, making it a slightly different specification rather than a pure duplicate.

5. **did/sample_urban/outcome_childmarriage vs robust/sample/urban_only**: These produce **identical** coefficients and SEs (-0.0179, SE=0.00151). They are complete duplicates with different spec_ids, one from the "did" branch and one from the "robustness" branch.

---

## Duplicate Specifications

The spec search produced many exact duplicates across the "robust/" and "did/" branches:

| Spec A | Spec B | Identical? |
|--------|--------|------------|
| baseline | did/fe/twoway | Yes (coef, SE, n_obs) |
| baseline | robust/cluster/countryregionurban | Yes (coef, SE, n_obs) |
| robust/outcome/educ | did/sample_full/outcome_educ | Yes |
| robust/outcome/employed | did/sample_full/outcome_employed | Yes |
| robust/outcome/marriage_age | did/sample_full/outcome_marriage_age | Yes |
| robust/outcome/childmarriage17 | did/outcome/married_before_17 | Yes |
| robust/outcome/childmarriage16 | did/outcome/married_before_16 | Yes |
| robust/outcome/childmarriage15 | did/outcome/married_before_15 | Yes |
| robust/outcome/childmarriage14 | did/outcome/married_before_14 | Yes |
| robust/sample/urban_only | did/sample_urban/outcome_childmarriage | Yes |
| robust/sample/rural_only | did/sample_rural/outcome_childmarriage | Yes |
| robust/treatment/bancohort_pc | did/treatment/binary | Yes |

This means roughly 12 of the 77 specs are exact duplicates. The effective unique specification count is closer to 65.

---

## Recommendations for the Spec-Search Script

1. **Deduplicate**: The script creates many specifications under both the "did/" and "robust/" branches that produce identical results. A deduplication pass after running all specs would reduce the count from 77 to approximately 65 unique specifications.

2. **Separate baselines by outcome**: The script implicitly treats G1 (childmarriage) as the single baseline but also runs G2-G4 as outcomes. The spec search would benefit from explicitly tagging the secondary outcomes (educ, employed, marriage_age) as their own baseline groups from the start.

3. **Distinguish heterogeneity from sample restrictions**: The "robust/heterogeneity" specs and some "robust/sample" specs overlap conceptually (e.g., rural-only appears in both). The heterogeneity branch should be reserved for interaction-term specifications, while simple subsample splits belong under sample restrictions.

4. **Placebo classification**: The childmarriage14 outcome should be explicitly tagged as a placebo in the spec search (it tests whether bans affect marriage below the ban threshold), rather than being grouped with the outcome measurement variations.

5. **Baseline identification**: The spec_id "baseline" is correctly tagged, but the identical specs (did/fe/twoway, robust/cluster/countryregionurban) should reference the baseline rather than appearing as independent robustness checks.
