# Verification Report: 158081-V7

## Paper Overview
- **Paper**: Work from Home before and after the COVID-19 Outbreak (RPS Data)
- **Authors**: Bick, Blandin, Mertens
- **Journal**: AEJ-Macroeconomics
- **Paper ID**: 158081-V7

## Baseline Groups

### G1: College education and WFH share
- **Claim**: College-educated workers have substantially higher WFH share than non-college workers (approx. 18 percentage point gap)
- **Baseline spec_id**: baseline
- **Outcome**: wfh_share (share of work days from home)
- **Treatment**: college (Bachelor's degree or higher)
- **Baseline coefficient**: 0.181 (SE = 0.003, p < 0.001, N = 69,373)
- **Expected sign**: Positive
- **Controls**: Age, age squared, female

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 70 |
| Core tests (is_core_test=1) | 39 |
| Non-core tests | 30 |
| Unclear | 1 |
| Invalid | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls (incl. baseline) | 18 |
| core_sample | 12 |
| core_inference | 3 |
| core_fe | 3 |
| core_funcform | 2 |
| core_method | 1 |
| noncore_alt_outcome | 5 |
| noncore_alt_treatment | 7 |
| noncore_heterogeneity | 15 |
| noncore_placebo | 3 |
| unclear | 1 |

## Classification Notes

### Core tests (39 specs)
These maintain the same outcome (wfh_share) and treatment (college) while varying:
- **Control sets** (18): Progressive addition of controls (none through full + industry) and leave-one-out robustness
- **Sample restrictions** (12): Year subsets (2020-2024), COVID peak, post-pandemic, exclude first/last year, interior WFH, full-time/part-time
- **Inference** (3): Clustering by wave, clustering by year, classical SEs
- **Fixed effects** (3): Year FE, industry FE, year + industry FE
- **Functional form** (2): IHS transformation of outcome, age categories instead of quadratic
- **Estimation method** (1): WLS with survey weights

### Non-core tests (30 specs)
- **Alternative outcomes** (5): wfh_days (count), any_wfh (binary), full_wfh (binary), days_commuting_num, days_working_num. These change the outcome concept/scale.
- **Alternative treatments** (7): graduate degree, bachelors only, industry dummies (professional, information, finance, education), high income. These test different causal objects.
- **Heterogeneity** (15): 11 subgroup regressions (by gender, age group, income, marital status, children) and 4 interaction models. These do not test the full-sample claim.
- **Placebo tests** (3): days_working as placebo outcome, construction industry and retail industry as placebo treatments.

### Unclear (1 spec)
- **robust/funcform/educ_categories**: Uses bachelors as treatment with education category dummies. This is a different parameterization that changes how the education gradient is estimated.

## Top 5 Most Suspicious Rows

1. **robust/outcome/days_working and robust/placebo/days_working_outcome**: These two rows have identical coefficients (0.0615), standard errors (0.0102), p-values, and sample sizes (N=69,373). The same regression appears under both alternative outcome and placebo test paths. This is a duplicate specification.

2. **robust/control/add_white and robust/control/add_black**: Both white and black variables have coefficient = 0.0, SE = 0.0, and p-value = NaN. The college coefficient is identical to the preceding step (add_has_children = 0.1827). These race variables appear to be perfectly collinear with the intercept or other variables. The control progression does not actually change the model at these steps.

3. **robust/funcform/educ_categories**: Reports the coefficient on bachelors (0.206) rather than college, with some_college, associates, and graduate as additional regressors. This changes the reference category and parameterization, making it not directly comparable to the baseline binary college indicator.

4. **robust/treatment/education_industry**: Treatment is ind_education (education industry indicator) with coefficient 0.018. This tests whether working in the education industry predicts WFH, which is a completely different question from whether college education predicts WFH.

5. **robust/het/interaction_income**: Reports college coefficient of 0.143, but because income_high and college:income_high are in the model, this is the college effect for low-income workers only (omitted income_high = 0). This is not comparable to the unconditional baseline estimate.

## Recommendations

1. **Remove duplicate specification**: robust/placebo/days_working_outcome and robust/outcome/days_working are identical regressions. One should be removed or they should be reconciled with different labels.

2. **Fix race variable collinearity**: The progressive control addition of white and black produces identical estimates to the prior step. The underlying script should either drop these steps or fix variable construction (possible dummy variable trap or all-zero columns).

3. **Tag heterogeneity separately in search script**: The specification search should distinguish subgroup/interaction analyses from robustness checks more clearly in the spec_tree_path.

4. **Clarify interaction coefficient interpretation**: When interaction terms are included, the specification search should either report the average marginal effect or note that the main effect coefficient only applies to the reference group.

5. **Consider whether alternative outcomes should be core**: WFH days count, any WFH binary, and full WFH binary could arguably be considered core tests of the same claim (college increases WFH). However, because they change the scale and interpretation of the coefficient, they are classified as non-core here. The underlying search script should be explicit about whether these are intended as robustness or as distinct claims.
