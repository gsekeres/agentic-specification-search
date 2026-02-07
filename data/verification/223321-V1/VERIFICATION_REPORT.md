# Verification Report: 223321-V1

## Paper Overview
- **Paper**: Effect of Parent's Age at Arrival on Child Not Identifying as Hispanic
- **Journal**: AER Papers and Proceedings 2025
- **Paper ID**: 223321-V1

## Baseline Groups

### G1: Age-at-arrival effect on child Hispanic identification
- **Claim**: Children of Mexican immigrant parents who arrived in the US at ages 0-8 are more likely to not identify as Hispanic than children of parents who arrived at ages 9-17.
- **Expected sign**: Positive (arriving young increases probability of not identifying as Hispanic)
- **Outcome**: not_hisp (binary: child does not identify as Hispanic)
- **Treatment**: par_Arrived0_8 (binary: parent arrived ages 0-8 vs 9-17)
- **Baseline spec_ids**: baseline (full controls, p=0.10), baseline_no_controls (p<0.001), baseline_basic_controls (p<0.001)
- **Note**: The paper's preferred specification (baseline with full controls including intermarriage) shows an insignificant effect (coef=0.0014, p=0.10). The effect is highly significant without intermarriage controls. This is a single baseline group because all three baselines test the same claim with different control sets.

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 62 |
| Core test specifications | 53 |
| Non-core specifications | 9 |
| Invalid specifications | 0 |
| Unclear specifications | 0 |
| Baseline specifications | 3 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 19 | Leave-one-out control drops and control-progression builds |
| core_sample | 20 | Time period, age group, gender, state, and survey subsamples |
| core_funcform | 6 | Alternative treatment cutoffs (ages 5-10) and continuous treatment |
| core_inference | 4 | HC2, HC3, state-clustered, year-clustered SEs |
| core_method | 4 | Logit, probit, unweighted OLS, LPM duplicate |
| noncore_heterogeneity | 8 | Interaction terms and heterogeneity subgroup splits |
| noncore_placebo | 1 | Parent own Hispanic identification as outcome |
| noncore_alt_outcome | 0 | -- |
| noncore_alt_treatment | 0 | -- |
| noncore_diagnostic | 0 | -- |
| invalid | 0 | -- |
| unclear | 0 | -- |

## Classification Rationale

### Core tests (53 specs)
- **Control variations (19)**: The leave-one-out and control-progression specs all use the same outcome (not_hisp) and treatment (par_Arrived0_8) on the full sample, varying only which control groups are included. These are classic robustness checks of the baseline claim.
- **Sample restrictions (20)**: These restrict the sample by time period, child age group, child gender, parent gender, family structure, state, or survey source. All keep the same outcome, treatment, and full controls (minus any group-specific exclusions). These are valid subsample tests of the baseline claim.
- **Functional form / treatment definition (6)**: Alternative treatment cutoffs (ages 5, 6, 7, 9, 10) and continuous age-at-arrival test the same conceptual hypothesis about whether younger arrival age increases non-Hispanic identification. The treatment variable name changes but the estimand concept is preserved.
- **Inference (4)**: HC2, HC3, state clustering, and year clustering change only the standard errors, not the point estimate. These test whether statistical significance is robust to inference assumptions.
- **Method (4)**: Logit and probit test the same hypothesis with nonlinear models (different scale, so coefficients are not directly comparable to LPM). Unweighted OLS tests sensitivity to survey weights. The LPM-weighted spec is an exact duplicate of the baseline.

### Non-core tests (9 specs)
- **Heterogeneity interactions (4)**: The interaction specs (treat_x_female, treat_x_par_female, treat_x_both_parents, treat_x_english_fluent) report the coefficient on the interaction term, not the main treatment effect. These test differential effects, not the level of the baseline claim.
- **Heterogeneity subgroup splits (4)**: The by_low_education, by_high_education, by_intermarried, and by_not_intermarried specs split the sample along dimensions that are themselves endogenous or that test heterogeneity rather than the baseline claim. These are informative but not robustness checks of the main effect.
- **Placebo (1)**: The parent_hisp spec changes the outcome to parent's own Hispanic identification (par_not_hisp), which is a different estimand used as a falsification test.

## Top 5 Most Suspicious Rows

1. **ols/method/lpm_weighted** (confidence: 0.95): This is an exact duplicate of the baseline specification (identical coefficient=0.001383, p=0.1023). It appears redundant and should be flagged -- it inflates the spec count without adding information.

2. **robust/build/bivariate** (confidence: 0.95): This is an exact duplicate of baseline_no_controls (identical coefficient=0.01137, p<0.001). Same concern about redundancy.

3. **robust/build/par_demographics** (confidence: 0.95): This appears to be an exact duplicate of baseline_basic_controls (identical coefficient=0.01153, p<0.001). The controls description differs slightly but the estimates are identical.

4. **robust/placebo/parent_hisp** (confidence: 0.95): The placebo test is actually significant (coef=0.0018, p=0.001), which is concerning for the identification strategy. Parents who arrived younger are themselves more likely to not identify as Hispanic, suggesting the child-level effect may partly reflect parent-level identity changes rather than intergenerational cultural transmission.

5. **robust/estimation/logit and robust/estimation/probit** (confidence: 0.80): These show highly significant effects (p<1e-59) but use a "subset of controls (for convergence)" rather than the full control set. The different control set and different scale make these less directly comparable to the baseline. The extremely high significance likely reflects the absence of intermarriage controls rather than the logit/probit functional form.

## Recommendations for Spec-Search Script

1. **Remove exact duplicates**: ols/method/lpm_weighted duplicates baseline; robust/build/bivariate duplicates baseline_no_controls; robust/build/par_demographics duplicates baseline_basic_controls. These should either be removed or merged.

2. **Logit/probit control set**: The logit and probit specs use a reduced control set for convergence. If possible, the script should try to include intermarriage controls or at minimum flag the different control set more explicitly, since the key sensitivity finding is about intermarriage controls.

3. **Heterogeneity classification**: Consider separating by_education and by_intermarriage splits from interaction-term specs in the spec_tree_path, since they use different approaches (subsample splits vs interaction terms) even though both test heterogeneity.

4. **Baseline identification**: The spec_id="baseline" is the full-controls version, which is the paper's preferred spec. However, since this spec is borderline insignificant (p=0.10), the researcher's actual claim may be more nuanced. The SPECIFICATION_SEARCH.md correctly identifies this sensitivity. No change needed, but downstream analysis should note the baseline is not significant at conventional levels.
