# Verification Report: 113673-V1

## Paper Information
- **Paper**: Vocational Training for Disadvantaged Youth in Colombia: A Long Term Follow Up
- **Authors**: Attanasio, Guarin, Medina, Meghir (2017)
- **Journal**: AEJ-Applied
- **Paper ID**: 113673-V1

## Baseline Groups

Three baseline groups were identified, each with two baselines (one per sample):

| Group | Claim | Outcome | Baseline Spec IDs |
|-------|-------|---------|-------------------|
| G1 | Program increases formal sector income | contrib_inc_max | baseline/ec/formal_income, baseline/es/formal_income |
| G2 | Program increases formal sector employment | pareado_max | baseline/ec/formal_employment, baseline/es/formal_employment |
| G3 | Program increases large firm employment | N200 | baseline/ec/large_firm, baseline/es/large_firm |

The paper uses two independent samples: the Entire Cohort (EC, N=372,648, admin data only) and the Evaluation Sample (ES, N=306,696, survey + admin data). The treatment variable differs by sample (select_h1 for EC, TK for ES) but both reflect lottery selection into the program.

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **50** |
| Baselines (is_baseline=1) | 6 |
| Core tests (is_core_test=1, not baseline) | 15 |
| Non-core specs | 29 |
| Invalid | 0 |
| Unclear | 0 |

### Core test breakdown (excluding baselines):

| Category | Count |
|----------|-------|
| core_controls | 5 |
| core_fe | 1 |
| core_sample | 2 |
| core_inference | 2 |
| core_funcform | 3 |
| core_method | 2 |

### Non-core breakdown:

| Category | Count |
|----------|-------|
| noncore_heterogeneity | 18 |
| noncore_diagnostic | 11 |

## Duplicate Specifications Issue

A major issue with this specification search is that **12 of the 15 non-baseline core specs are exact duplicates of baseline results**, re-labeled under different robustness categories. Specifically:

1. **sample/ec_only/formal_employment** = baseline/ec/formal_employment (identical coeff=0.0382614, SE=0.005441, N=372648)
2. **sample/es_only/formal_employment** = baseline/es/formal_employment (identical coeff=0.0423553, SE=0.0121037, N=306696)
3. **inference/cluster_individual/ec** = baseline/ec/formal_employment (identical; baseline already clusters at individual)
4. **inference/cluster_individual/es** = baseline/es/formal_employment (identical; baseline already clusters at individual)
5. **outcome/formal_income/ec** = baseline/ec/formal_income (identical coeff=26824.92, SE=4267.12)
6. **outcome/formal_binary/ec** = baseline/ec/formal_employment (identical)
7. **outcome/large_firm/ec** = baseline/ec/large_firm (identical)
8. **weights/weighted/es/formal_employment** = baseline/es/formal_employment (identical; ES baseline is already weighted)
9. **weights/unweighted/ec/formal_employment** = baseline/ec/formal_employment (identical; EC baseline is already unweighted)
10. **controls/full_gender_interactions/ec** = baseline/ec/formal_employment (identical; baseline already uses full gender interactions)

These duplicates inflate the apparent robustness of results without providing new information.

### Genuinely distinct core specs (3 only):

1. **estimation/no_fe/formal_employment**: Drops course x gender FE. Coeff=0.0443798 vs baseline 0.0382614 (slightly larger without FE). N=365,292 vs 372,648 (slightly different sample).
2. **displacement/no_fe/p3**: Adds treatment intensity control p3_t4, drops FE. N=365,292.
3. **displacement/fe/p3**: Adds treatment intensity control p3_t4, keeps FE.
4. **displacement/no_fe/p5**: Adds treatment intensity control p5_t4, drops FE. N=372,396.
5. **displacement/fe/p5**: Adds treatment intensity control p5_t4, keeps FE.

Note: The displacement main-effect specs (4 total) add a treatment intensity variable as a control. The coefficient on the treatment indicator is still testing the same claim (program effect on employment). These are legitimate controls variations.

## Top 5 Most Suspicious Rows

1. **controls/no_gender_interactions/women** and **controls/no_gender_interactions/men**: These are labeled as controls variations but are actually women-only and men-only subsamples. Their coefficients are identical to the heterogeneity/ec/women/formal_employment and heterogeneity/ec/men/formal_employment specs respectively. They should be classified as heterogeneity, not controls variations.

2. **estimation/no_fe/formal_employment** has identical coefficient/SE to **displacement/no_fe/p3** (coeff=0.0443798, SE=0.0053174, N=365,292). This is likely the same regression extracted twice. The no-FE displacement regression with p3_t4 as a control produces the exact same treatment coefficient as the no-FE regression without it, which is suspicious.

3. **All 10 duplicate specs** listed above: These provide zero additional robustness information and suggest the specification search script re-extracted the same regressions under multiple labels when it could not vary the specifications (due to data unavailability).

4. **displacement/interaction/p3_no_fe** and similar interaction specs: The treatment variable is listed as "select_h1#c.p3_t4" which is the Stata interaction term. These test a different estimand (whether displacement moderates the treatment effect) and should not be compared to the main effect baseline.

5. **balance/es/educ_lb**: Shows a significant imbalance (coeff=0.255, p<0.001) in education between treatment and control. This raises questions about the validity of the randomization, though the paper addresses this by controlling for baseline characteristics.

## Recommendations for Spec-Search Script

1. **Do not re-label baseline regressions as robustness specs.** If the data is unavailable to run actual variations, the spec search should report fewer specs rather than duplicating baselines under different category labels.

2. **Distinguish displacement main effects from interaction terms.** The treatment main effect in a regression that also includes treatment x intensity interactions should be clearly labeled as a conditional main effect, not conflated with the unconditional baseline.

3. **Gender subgroup specs should be categorized as heterogeneity, not controls variations.** The controls/no_gender_interactions/women and controls/no_gender_interactions/men specs are mislabeled.

4. **Flag when data is unavailable for genuine specification variation.** The search report notes that the data is not included in the replication package, but the resulting CSV contains 50 rows that give the false impression of extensive robustness testing. In reality, only approximately 5-6 genuinely distinct specifications exist for any given baseline.

5. **Consider cross-outcome robustness more carefully.** The three outcome measures (income, employment, large firm) are distinct claims. Cross-listing them as outcome variations of each other conflates different estimands.
