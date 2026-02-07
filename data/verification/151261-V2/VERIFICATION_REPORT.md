# Verification Report: 151261-V2

## Paper Overview
- **Paper ID**: 151261-V2
- **Title**: The Effects of Working while in School: Evidence from Employment Lotteries
- **Journal**: AEJ: Applied
- **Method**: Instrumental Variables (2SLS) with lottery as instrument

## Baseline Groups Found

| Group ID | Claim | Outcome | Baseline spec_id | Expected Sign |
|----------|-------|---------|-------------------|---------------|
| G1 | YET program increases employment | s_emp | baseline | + |
| G2 | YET program increases enrollment | s_enrolled_edu | baseline_enrollment | + |
| G3 | YET program increases total income | s_tot_income | baseline_income | + |
| G4 | YET program increases work-study status | s_wstudy | baseline_wstudy | + |
| G5 | YET program decreases NEET status | s_nwnstudy | baseline_nwnstudy | - |

The primary claim (G1: employment effect) is the focus of most robustness specifications. G2 (enrollment) and G3 (income) have moderate robustness coverage. G4 and G5 have no robustness specs beyond their baselines.

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 103 |
| Baseline specifications | 5 |
| Core test specifications | 62 |
| Non-core specifications | 41 |
| Invalid specifications | 0 |
| Unclear specifications | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 30 | Control set variations (LOO, stepwise addition, strata-only, no controls, full controls) |
| core_sample | 19 | Sample restriction variations (gender, age, SES, vulnerability, track, trimming) |
| core_method | 7 | Estimation method variations (OLS, ITT/reduced form, pension contrib proxy) |
| core_funcform | 3 | Functional form variations (log income, IHS income, winsorized income) |
| core_inference | 3 | Inference variations (robust SE -- identical to baseline) |
| noncore_alt_outcome | 21 | Alternative outcomes not part of baseline claims (soft skills, education details, industry composition, employment types) |
| noncore_heterogeneity | 7 | Heterogeneity subgroup analyses (TUS participation, first-time, school schedule, ability) |
| noncore_placebo | 7 | Balance/placebo tests (lottery predicting pre-treatment covariates) |
| noncore_diagnostic | 6 | First-stage and reduced-form diagnostic regressions |

## Top 5 Most Suspicious Rows

### 1. Leave-One-Out (LOO) Specifications (11 rows: robust/loo/drop_*)
**Issue**: All 11 LOO specifications produce *identical* coefficients and standard errors to the baseline (coef = 0.4718602641417221, SE = 0.03298080014972392). This is extremely suspicious and suggests the variables being "dropped" (female, age, number_kids, etc.) were never in the baseline control set to begin with. The baseline uses 70 controls that appear to be location strata + quota variables + application/school variables, but NOT demographic variables like female, age, number_kids. These LOO specs are likely attempting to drop variables from a hypothetical extended control set but mistakenly producing the same regression as the baseline. While classified as core_controls, these should be treated with low confidence (0.6).

### 2. Clustering Variations (3 rows: robust/cluster/robust_*)
**Issue**: The baseline already uses robust standard errors (cluster_var = "robust"), and these clustering variation specs also use robust SE with identical coefficients and SEs. These specs provide no additional information beyond the baseline. Classified as core_inference with confidence 0.7 but they are effectively duplicates.

### 3. s_pension_contrib classified as core_method (robust/outcome/s_pension_contrib)
**Issue**: This uses pension contributions as the outcome variable, which is a different variable than s_emp. However, pension contributions can be viewed as a proxy measure of formal employment, which is conceptually closely related to the employment claim (G1). Classified as core_method with confidence 0.7, but reasonable people could classify this as noncore_alt_outcome.

### 4. First-Stage Reduced Forms vs ITT Method Specs
**Issue**: The reduced form specs (iv/first_stage/reduced_form_*) and the ITT specs (iv/method/itt_*) produce identical results because they are the same regression (OLS of outcome on offered). The first-stage reduced forms are classified as noncore_diagnostic while the ITT specs are classified as core_method. This is because the ITT specs are explicitly framed as alternative estimation methods (intent-to-treat analysis), while the first-stage reduced forms serve as IV diagnostics. This distinction is conceptual rather than statistical.

### 5. Heterogeneity vs Sample Restrictions
**Issue**: The heterogeneity specs (robust/heterogeneity/*) and sample restriction specs (robust/sample/*) both involve running the baseline regression on subsamples. The difference is that sample restrictions are classified as core_sample (testing whether the baseline result holds in subpopulations), while heterogeneity specs are classified as noncore_heterogeneity. The distinction is based on the spec_tree_path labeling and the fact that heterogeneity analyses typically test whether effects *differ* across groups rather than whether the baseline holds. However, in practice both groups run the same type of regression. The heterogeneity subgroups (TUS participant, morning/afternoon school, ability) are more specialized and less standard, supporting their non-core classification.

## Recommendations

1. **Fix LOO specifications**: The leave-one-out script should verify that the dropped variable is actually present in the baseline control set before running the regression. Currently, the LOO specs drop variables (female, age, etc.) that are not in the baseline 70-control set, producing duplicate results. The baseline controls are location strata + quota + application/school variables, while the LOO vars are demographic.

2. **Remove duplicate clustering specs**: The robust/cluster/* specs are exact duplicates of the baselines since the baseline already uses robust SE. Either add alternative clustering structures (e.g., cluster by location strata) or remove these specs.

3. **Clarify ITT vs reduced-form distinction**: The ITT method specs and first-stage reduced-form specs are statistically identical. Consider keeping only one set and labeling it clearly.

4. **Consider adding more G4/G5 robustness**: The wstudy (G4) and nwnstudy (G5) baselines have zero robustness specifications beyond their baseline. If these are important claims, the search should include control variations and sample restrictions for these outcomes.

5. **Consider re-evaluating heterogeneity classification**: Some heterogeneity specs (e.g., morning vs afternoon school) could potentially be reclassified as core_sample if the paper treats these as pre-registered subgroup analyses rather than exploratory heterogeneity.
