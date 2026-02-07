# Verification Report: 146041-V1

## Paper
"Human Capital in the Presence of Skilled-Biased Technical Change" (AER)

## Baseline Groups

### G1: AQ-GDP Relationship (Domestic Data)
- **Baseline spec_id**: `baseline`
- **Claim**: Relative skill efficiency (AQ = A_H/A_L) is positively associated with log GDP per worker across countries
- **Outcome**: `l_irAQ53_dum_skti_hrs_secall` (log relative skill efficiency, sigma=1.5)
- **Treatment**: `l_y` (log GDP per worker)
- **Sample**: 12 countries with micro data (IPUMS International)
- **Baseline coefficient**: 1.408 (SE=0.497, p=0.005)

### G2: Q-GDP Relationship (Immigrant Data)
- **Baseline spec_id**: `ols/Q/baseline_us`
- **Claim**: Relative human capital quality (Q) of immigrants is positively associated with origin-country GDP per worker
- **Outcome**: `l_irQ53_dum` (log relative human capital quality from US immigrant wages)
- **Treatment**: `l_y` (log GDP per worker)
- **Sample**: ~101 origin countries of US immigrants
- **Baseline coefficient**: 0.105 (SE=0.017, p<0.001)

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 67 |
| Baseline specifications | 2 |
| Core test specifications | 55 |
| Non-core specifications | 12 |
| Invalid specifications | 0 |
| Unclear specifications | 0 |

### Core tests by baseline group
- G1 (AQ-GDP): 31 core tests (including baseline)
- G2 (Q-GDP): 24 core tests (including baseline)

## Category Counts

| Category | Count | Description |
|----------|-------|-------------|
| core_sample | 27 | Leave-one-out, subsample restrictions, alternative sample definitions |
| core_funcform | 13 | Alternative sigma values, measurement approaches, levels vs logs |
| core_controls | 7 | Experience/gender controls, bilateral gravity controls |
| core_method | 6 | Baselines plus quantile regression and pooled estimation |
| core_inference | 2 | Classical and bootstrap standard errors |
| noncore_heterogeneity | 8 | Sector-specific AQ decompositions and sector wage ratios |
| noncore_alt_outcome | 2 | Wage ratio and labor supply ratio (decomposition components) |
| noncore_diagnostic | 2 | AQ measures run on wrong sample (Q sample instead of AQ sample) |

## Classification Rationale

### Why two baseline groups?
The paper makes two conceptually distinct claims using different data sources:
1. AQ analysis uses domestic wage and labor supply data from 12 countries to measure skill efficiency
2. Q analysis uses US immigrant wage data from 101 origin countries to measure human capital quality

These have different outcome variables, different sample sizes, and coefficient magnitudes that differ by an order of magnitude (~1.4 vs ~0.10). They cannot be pooled into a single specification curve.

### Why sectoral specs are non-core
The 4 sector-specific AQ specs (Agriculture, Manufacturing, LowSkillServices, HighSkillServices) and their 4 wage ratio counterparts partition the baseline outcome by industry. This is heterogeneity analysis: they test whether the AQ-GDP relationship holds within each sector, not whether the overall relationship is robust to specification choices. The sector-specific AQ is a different estimand than the economy-wide AQ.

### Why wage ratio and labor supply are non-core
The wage ratio (`l_wrat53_dum_skti_secall`) and labor supply ratio (`l_H5L3_dum_skti_hrs_secall`) are mathematical components of AQ. Testing their individual relationships with GDP is a decomposition exercise, not a robustness check on the AQ claim.

### Why comparison specs are non-core diagnostic
Specs `ols/compare/baseline_aq` and `ols/compare/barro-lee_aq` run the AQ outcome variable on the US immigrant sample (Q dataset), which is a different sample than the baseline AQ analysis. This is a diagnostic cross-check, not a standard robustness test of either baseline claim.

## Top 5 Most Suspicious Rows

1. **ols/compare/baseline_aq** (coef=1.408, n=12): This is identical to the baseline but run on the Q dataset. It produces the exact same coefficient, suggesting it picked up the same 12 micro-data observations within the Q dataset. This is a duplicate of the baseline, not an independent test.

2. **ols/compare/barro-lee_aq** (coef=1.107, n=92): Runs AQ (Barro-Lee version) on the Q dataset sample. The n=92 vs n=139 for the Barro-Lee broad spec suggests subsample differences. This is a cross-dataset diagnostic.

3. **ols/form/quadratic_gdp** (coef=-9.158, p=0.208): The linear GDP coefficient becomes large and negative when a quadratic term is added. With only 12 observations, this is severely overfitted and should be interpreted with extreme caution.

4. **robust/sample/low_income** (coef=-0.036, p=0.175): The only Q-GDP spec with a negative coefficient. The Q-GDP relationship may not hold for low-income countries, which is a meaningful finding but also reflects sample splitting on the treatment variable.

5. **ols/Q/selection_adjusted** (coef=0.039, p=0.091): The selection-adjusted Q coefficient is only 37% of the unadjusted baseline (0.039 vs 0.105). If immigrant selection drives much of the measured Q variation, the core claim is substantially weakened.

## Recommendations

1. **No script fixes needed for baseline definitions**: The baseline spec_id correctly identifies the main AQ analysis, and the Q baseline is correctly labeled as `ols/Q/baseline_us`.

2. **Consider dropping duplicate**: `ols/compare/baseline_aq` appears to be a duplicate of the baseline (same outcome, same 12 observations from the micro sample). If the intent was to compare AQ across a broader sample, the sample selection may be incorrect.

3. **Sector specs should be explicitly tagged as heterogeneity**: The spec_tree_path already marks them as `robustness/heterogeneity.md`, which is appropriate.

4. **The sigma parameter variations are correctly treated as robustness**: Since AQ is mechanically constructed using sigma, varying sigma is analogous to a functional form change rather than an outcome change.

5. **The small sample size (n=12) for G1 specs is a fundamental limitation**: All G1 leave-one-out specs have n=11, and the quadratic spec has only 9 effective degrees of freedom. These small-sample issues cannot be resolved by specification search.
