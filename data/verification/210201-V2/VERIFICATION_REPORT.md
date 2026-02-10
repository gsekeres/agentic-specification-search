# Verification Report: 210201-V2

## Paper: Life Insurance and Household Consumption (AER 2012)
**Authors**: Jay H. Hong, Jose-Victor Rios-Rull

## Context

This is a **structural calibration** paper, not a reduced-form estimation paper. The replication package contains PSID microdata (1994-1999) used to construct marital transition matrices as inputs to a life-cycle dynamic programming model. The specification search constructs reduced-form regressions examining gender-specific marital transition probabilities, treating "female" as the primary treatment variable. Because the paper's core contribution is a calibrated structural model rather than a causal reduced-form claim, these specifications should be interpreted as descriptive analyses of the underlying data patterns.

## Baseline Groups

Four baseline groups were identified, each testing the effect of gender (female) on a different marital transition outcome:

| Group | Spec ID(s) | Outcome | Coefficient | SE | p-value | N | Sample |
|-------|-----------|---------|-------------|-----|---------|---|--------|
| G1 | 1, 5 | divorce / stay_married | -0.000274 / +0.000274 | 0.00739 | 0.971 | 5,994 | Married 1994 |
| G2 | 2 | new_marriage | -0.02656 | 0.01405 | 0.059 | 4,080 | Single 1994 |
| G3 | 3 | became_widow | +0.02367 | 0.00484 | <0.001 | 10,074 | Full sample |
| G4 | 4 | married99 | -0.02026 | 0.00688 | 0.003 | 10,074 | Full sample |

**G1 note**: Specs 1 and 5 are algebraic complements. Since `stay_married = 1 - divorce`, the coefficient on `female` is exactly negated between them. Both are classified as baselines in G1.

All baselines use OLS with controls: age group dummies + child94 + dep94 (and married94 for full-sample outcomes).

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 85 |
| Baselines | 5 |
| Core tests (non-baseline) | 45 |
| Non-core specifications | 35 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 21 (incl. 5 baselines) | Control variations for G1/G2 outcomes |
| core_sample | 11 | Age, children, dep, widowed/single-other subsamples |
| core_inference | 6 | HC0/HC2/HC3, clustered, homoskedastic, bootstrap |
| core_funcform | 9 | Logit, probit, polynomial age, log controls |
| core_method | 3 | WLS variations (by q, dep, child) |
| noncore_alt_treatment | 11 | Treatment switched to child94, dep94, age, q, married94, has_child94 |
| noncore_alt_outcome | 8 | delta_dep, delta_child, gained_child, lost_child, has_child99, single_widow99, dep99, child99 |
| noncore_heterogeneity | 11 | Interaction models reporting main effects (incl. female_x_child in functional form) |
| noncore_placebo | 4 | Randomized gender, unrelated outcomes (col10, col11) |
| noncore_diagnostic | 1 | dep94 on changed_spouse |

## Classification Rationale

### Core Tests (45 specs, excluding 5 baselines)

**Control variations (16 non-baseline specs)**: Specs 6-16 systematically vary controls for the G1 (divorce) outcome -- no controls, age-only, children-only, dep-only, partial combos, and additional controls (spouse age, age gap, q rate). Specs 17-20 do the same for G2 (marriage formation). These are genuine robustness checks testing whether the gender coefficient is sensitive to control set composition. All maintain the same outcome, treatment, and sample as their respective baseline.

**Sample restrictions (11 specs)**: Specs 23-27 restrict the married sample by age group or children status for G1. Specs 30-31 restrict the single sample by marital history type for G2. Specs 32-34 trim age extremes or split by dependents. Spec 41 restricts the widowhood outcome to the married-only subsample. Spec 42 is an algebraic complement of G2 (stay_single = 1 - new_marriage). These test the same claim on meaningful subpopulations.

**Inference variations (6 specs)**: Specs 51-54, 56-57 apply the same G1 model with different standard error computations (HC0, HC2, HC3, age-clustered, homoskedastic, bootstrap). Point estimates are identical; only SEs and p-values change. Note that spec 57 (bootstrap) has an anomalous p-value of 1.95, which appears to be a coding error.

**Functional form (9 specs)**: Specs 58-62 use logit/probit instead of OLS for G1, G2, G3, and G4. Specs 63-65 replace age dummies with polynomial age (linear, quadratic, cubic). Spec 67 log-transforms the control variables. These test whether the G1/G2/G3/G4 results are sensitive to modeling choices.

**Method variations (3 specs)**: Specs 68-70 use weighted least squares with different weighting variables. These test whether particular observations drive the results.

### Non-core Tests (35 specs)

**Alternative treatments (11 specs)**: Specs 21-22, 28-29, 35, 46-50, 55, and 75 (as diagnostic) change the treatment variable from `female` to `child94`, `dep94`, `age_94`, `q`, `married94`, or `has_child94`. These answer fundamentally different questions than the baselines. For example, "does the number of dependents predict divorce?" is a different causal object than "does gender predict divorce?" These cannot be compared to the baseline coefficients.

**Alternative outcomes (8 specs)**: Specs 36-40, 43-45 use outcomes that are not marital transitions: change in dependents, change in children, gained/lost child, has child in 1999, single-widowed in 1999, dependents level in 1999, children level in 1999. These measure different phenomena than the baseline marital transition probabilities.

**Heterogeneity interactions (11 specs)**: Specs 66, 76-85 add interaction terms (e.g., female x young, female x dep, female x q) but report the main effect of female rather than the interaction term itself. In an interaction model, the main effect represents the gender effect for the reference category only (e.g., non-young, zero dependents), not the average treatment effect. This makes them non-comparable to the baseline ATE. Spec 84 additionally shows signs of severe collinearity (age dummy coefficients ~10^11).

**Placebo tests (4 specs)**: Specs 71-72 predict unrelated variables (col10, col11) with gender. Specs 73-74 use randomized gender assignments. These are validation checks, not tests of the baseline claim.

**Diagnostic (1 spec)**: Spec 75 examines dep94 predicting changed_spouse, which uses both a different treatment and a different outcome from all baselines.

## Top 5 Most Suspicious Rows

1. **Spec 57 (inference/bootstrap_200)**: The p-value is reported as 1.95, which is impossible for a valid p-value (range should be [0,1]). This indicates a bug in the bootstrap SE or p-value calculation. The point estimate and SE appear reasonable, but the p-value is unreliable. Classified as core but flagged with lower confidence.

2. **Spec 84 (heterogeneity/young_x_child_divorce)**: The age dummy coefficients are approximately 9.9 x 10^11 and the constant is -9.9 x 10^11, indicating severe multicollinearity. The `young` indicator is likely perfectly collinear with a subset of age dummies, causing numerical instability. The coefficient on `female` (0.002) is still in a reasonable range, but this specification is unreliable.

3. **Specs 21-22, 28-29, 35, 55 (sample splits that change treatment)**: These are labeled as "sample" restrictions but actually change the treatment variable from `female` to `child94` or `married94`. This is not a sample restriction -- it is a different regression. The labeling in `spec_tree_path` (e.g., `sample/female_only_divorce`) is misleading because the gender-specific subsample eliminates variation in the original treatment variable, forcing a different covariate to become the treatment.

4. **Spec 71 (placebo/gender_col10)**: This "placebo" test predicting col10 (which appears to be a numeric variable like dependents count in 1999) with female shows a highly significant result (p=0.0003). If col10 is actually `dep99` or a similar variable, this is not a valid placebo -- gender plausibly affects household composition. The variable labeling as "col10" obscures what is being tested.

5. **Spec 20 (controls/marriage_full_plus_col10)**: Adds `col10` as a control variable, but `col10` is an unidentified numeric column. If this is a post-treatment variable (e.g., 1999 outcomes), including it as a control would bias the gender coefficient. Its inclusion without clear documentation is concerning.

## Notes on Spec-Search Design

1. **Structural vs. reduced-form mismatch**: This paper is a structural calibration study. The specification search imposes a reduced-form framework that does not correspond to any hypothesis test in the original paper. The gender variable is not a "treatment" in any causal sense -- it is a demographic category. The specification search results document the stability of descriptive correlations, not the robustness of a causal claim.

2. **Algebraic complements counted separately**: Specs 1/5 (divorce/stay_married) and 2/42 (new_marriage/stay_single) are exact algebraic complements. Each pair contains no independent information. The effective number of distinct baseline specifications is 4, not 5.

3. **Treatment switching is over-represented**: 11 of 85 specs (13%) change the treatment variable, which changes the estimand entirely. These should not be aggregated with specs that vary controls or samples for the purpose of assessing robustness of the baseline claims.

4. **Heterogeneity specs report wrong coefficient**: All 10 heterogeneity specifications report the main effect of `female` from an interaction model, not the interaction coefficient itself. This means they report the gender effect for the omitted reference category, which is not the same estimand as the baseline average effect. For these to be informative about heterogeneity, the interaction coefficients should be extracted instead.
