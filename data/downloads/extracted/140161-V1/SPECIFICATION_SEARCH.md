# Specification Search Report: 140161-V1

## Surface Summary

- **Paper**: Henry, Zhuravskaya & Guriev — Fake news & fact-check sharing experiment (France, 2019 EU elections)
- **Design**: Randomized experiment (individual-level, online survey)
- **Baseline groups**: 2
  - G1: ATE of fact-checking on desire to share alt-facts (surveys 1,2,3; N=2537)
  - G2: ATE of voluntary vs imposed fact-check on fact-check sharing (surveys 2,3; N=1692)
- **Budget G1**: max 80 core specs, 10 random subsets
- **Budget G2**: max 40 core specs, 5 random subsets
- **Seed G1**: 140161 | **Seed G2**: 140162
- **Surface hash**: sha256:d01ac5fab4502dc8f055a081ecffe9647708cb603128ed01a5cd451ff493e207

## Execution Summary

| Category | Rows | Successful | Failed |
|----------|------|------------|--------|
| Total    | 222 | 222 | 0 |
| G1       | 196 | 196 | 0 |
| G2       | 26 | 26 | 0 |
| Inference (HC3) | 5 | 5 | 0 |

## Spec Types Executed

| spec_id prefix | Count |
|----------------|-------|
| baseline__factcheck_action | 1 |
| baseline__intent_share_fb | 2 |
| baseline__share_action_g1 | 2 |
| design/randomized_experiment/estimator/diff_in_means | 5 |
| design/randomized_experiment/estimator/strata_fe | 4 |
| design/randomized_experiment/estimator/with_covariates | 5 |
| rc/controls/loo/drop_age | 4 |
| rc/controls/loo/drop_altruism | 4 |
| rc/controls/loo/drop_catholic | 4 |
| rc/controls/loo/drop_children | 4 |
| rc/controls/loo/drop_image | 4 |
| rc/controls/loo/drop_income | 4 |
| rc/controls/loo/drop_log_nb_friends_fb | 5 |
| rc/controls/loo/drop_low_educ | 5 |
| rc/controls/loo/drop_male | 5 |
| rc/controls/loo/drop_married | 4 |
| rc/controls/loo/drop_mid_educ | 5 |
| rc/controls/loo/drop_muslim | 4 |
| rc/controls/loo/drop_negative_image_UE | 5 |
| rc/controls/loo/drop_no_religion | 4 |
| rc/controls/loo/drop_often_share_fb | 4 |
| rc/controls/loo/drop_reciprocity | 4 |
| rc/controls/loo/drop_religious | 5 |
| rc/controls/loo/drop_second_mlp | 5 |
| rc/controls/loo/drop_single | 4 |
| rc/controls/loo/drop_town | 4 |
| rc/controls/loo/drop_use_FB | 5 |
| rc/controls/loo/drop_village | 4 |
| rc/controls/progression/strata | 5 |
| rc/controls/progression/strata_socio | 5 |
| rc/controls/progression/strata_socio_vote | 5 |
| rc/controls/progression/strata_socio_vote_fb | 5 |
| rc/controls/progression/strata_socio_vote_fb_behavioral | 4 |
| rc/controls/progression/strata_socio_vote_fb_behavioral_reported | 4 |
| rc/controls/sets/full | 5 |
| rc/controls/sets/none | 5 |
| rc/controls/sets/strata_only | 5 |
| rc/controls/sets/strata_socio_vote_fb | 5 |
| rc/controls/subset/random_001 | 5 |
| rc/controls/subset/random_002 | 5 |
| rc/controls/subset/random_003 | 5 |
| rc/controls/subset/random_004 | 5 |
| rc/controls/subset/random_005 | 5 |
| rc/controls/subset/random_006 | 4 |
| rc/controls/subset/random_007 | 4 |
| rc/controls/subset/random_008 | 4 |
| rc/controls/subset/random_009 | 4 |
| rc/controls/subset/random_010 | 4 |
| rc/form/outcome/share_click2 | 2 |
| rc/form/outcome/share_click3 | 2 |
| rc/form/outcome/share_fact_click3 | 1 |
| rc/form/outcome/share_facts_click2 | 1 |
| rc/treatment/binary/any_factcheck_vs_control | 2 |
| rc/treatment/pairwise/imposed_vs_control | 2 |
| rc/treatment/pairwise/imposed_vs_voluntary | 2 |
| rc/treatment/pairwise/voluntary_vs_control | 2 |

## Data Construction Notes

- Dataset reconstructed from raw Qualtrics CSV files (Survey 1, 2, 3) + GA_hours.dta
- Duration filter (>=250s) and gc==1 applied during raw CSV import (matching Stata 1.infile_data.do)
- Google Analytics merge by day/hour/survey_id for share_click2/3 and share_facts_click2/share_fact_click3
- Variables constructed to match Stata do-file definitions exactly
- Education dummies (i.educ) expanded as educ_2 through educ_9 (educ_1 as reference)
- All outcome variables zero-filled when missing (matching do-file: replace X=0 if X==.)

## Deviations from Surface

- Duration filter variants (rc/sample/duration_filter/*) excluded per surface removal
- The `educ` variable in column 3+ regressions is expanded as 8 dummies (i.educ in Stata)

## Software Stack

- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
- statsmodels 0.14.6 (for HC3 inference)
