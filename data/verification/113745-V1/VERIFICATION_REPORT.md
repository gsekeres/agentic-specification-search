# Verification Report: 113745-V1

## Paper
- **Title**: Disclosure by Politicians
- **Authors**: Djankov, La Porta, Lopez-de-Silanes, Shleifer (2010)
- **Journal**: American Economic Journal: Applied Economics
- **Method**: Cross-sectional OLS (cross-country regression with 175 countries)

## Baseline Groups

### G1: Public disclosure and government effectiveness
- **Baseline spec_id**: baseline
- **Claim**: Public disclosure of politicians financial asset values is associated with better government effectiveness, conditional on income, democracy, and press freedom.
- **Outcome**: goveff_96_07 (World Bank Government Effectiveness)
- **Treatment**: v_cit_mp_prac_all (Values publicly available, continuous 0-1)
- **Coefficient**: -0.202 (p=0.137) -- INSIGNIFICANT
- **Controls**: lngni06, democ_5006, negpres06
- **Expected sign**: Negative (conditional on development controls)
- **Notes**: The raw bivariate correlation is strongly positive (coef=1.23, p<0.001) but once income is controlled, the relationship turns negative and insignificant. This is the paper's reported main specification.

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 69 |
| Baseline | 1 |
| Core tests | 38 |
| Non-core tests | 31 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 14 |
| core_sample | 9 |
| core_inference | 6 |
| core_funcform | 4 |
| core_method | 5 (incl. baseline) |
| noncore_alt_outcome | 9 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 13 |
| noncore_placebo | 6 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

## Classification Decisions

### Core tests (38 total, including baseline)

1. **Control variations (14)**: All specs varying control sets while keeping goveff_96_07 and v_cit_mp_prac_all. Includes bivariate, progressive control additions, leave-one-out, legal origin dummies, continent dummies, and full kitchen-sink model.

2. **Sample restrictions (9)**: Continent exclusions (5), legal-origin exclusions (2), trimming (1), winsorizing (1). All keep baseline outcome/treatment/controls but restrict the sample geographically or by outlier handling.

3. **Inference variations (6)**: HC0/HC1/HC2/HC3 robust SEs (4) and clustering by continent (1) and legal origin (1). Coefficient unchanged; only p-values differ.

4. **Functional form (4)**: Includes quadratic treatment (1), IHS-transformed treatment (1), and two alternative public disclosure measures (s_cit_mp_prac_all = sources publicly available, ft_pubprac_all = binary publicly available). The latter two are classified as core_funcform because they are alternative operationalizations of the same concept (public disclosure) rather than genuinely different causal objects.

5. **Estimation method (4)**: Quantile regressions at 25th/50th/75th percentiles (3) and robust M-estimation (1). All keep same outcome/treatment/controls.

### Non-core tests (31 total)

1. **Alternative outcomes (9)**: Six specs use different governance/corruption indices (Kaufmann, TI, ICRG, log cost, Heritage, GCR). Three funcform specs (log_ti, log_her, log_gcr) also change the outcome variable to a log-transformed version of a non-baseline outcome. These change the estimand.

2. **Alternative treatments (3)**: Congress-only disclosure (v_mc_all, s_mc_all) and any-disclosure-required (disc_req) are different causal objects from public disclosure. The paper explicitly contrasts public vs congress disclosure.

3. **Heterogeneity (13)**: Income quartile subgroups (4), democracy/press-freedom splits (4), and interaction terms (5). These test whether the effect varies by context, not the main effect itself.

4. **Placebo tests (6)**: Congress disclosure as placebo treatment (2) and unrelated outcomes (fuel exports, religion shares) as placebo outcomes (4).

## Top 5 Most Suspicious Rows

1. **robust/placebo/congress_v_mc_all vs robust/treatment/v_mc_all**: These have IDENTICAL coefficients (-0.4955, p=0.00027). The placebo spec is an exact duplicate of the alternative treatment spec. The same regression appears twice under different labels. This is a data quality issue.

2. **robust/placebo/congress_s_mc_all vs robust/treatment/s_mc_all**: Same duplication issue. Coefficient -0.1109 (p=0.459) appears identically in both rows.

3. **robust/funcform/log_ti_03_07, log_her_03_08, log_gcr2003_2008**: These are labeled as functional form variations but they change the outcome variable to a log of a completely different governance index. They are misclassified in the original spec search -- they should be labeled as alternative outcomes with a log transform, not functional form variations of the baseline.

4. **robust/sample/gni_ilow (p=0.007)**: Low-income subsample shows a strong significant positive effect (coef=0.559) while all other income subgroups are insignificant. This could indicate that the disclosure-governance relationship is driven entirely by low-income countries, raising questions about external validity.

5. **robust/placebo/outcome_muslim80 (p=6.59)**: The p-value of 6.59 is greater than 1, which is impossible for a valid p-value. This indicates a computational error in the estimation script for this specification.

## Recommendations

1. **Fix duplicate specs**: The congress disclosure placebo specs are exact duplicates of the alternative treatment specs. Remove the duplicates or differentiate them (e.g., the placebo version could include public disclosure as a control to test the incremental effect of congress disclosure).

2. **Reclassify funcform specs**: The log_ti, log_her, log_gcr specs should be moved to alternative outcomes, since they change the dependent variable entirely.

3. **Fix p-value computation**: The muslim80 placebo spec has p_value=6.59, which is invalid. Check the estimation code for this specification.

4. **Reconsider income/democracy subgroup classification**: The spec search labeled these as sample restrictions, but they are heterogeneity analyses. Consider labeling them under heterogeneity in the spec_tree_path.

5. **Consider adding continent/legal-origin FE**: The spec search adds legal origin and continent as control dummies but does not test them as fixed effects. Given the cross-country nature of the data, this could be a useful addition.
