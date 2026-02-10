# Verification Report: 113768-V1

## Paper Information
- **Title**: The Importance of Being Wanted
- **Authors**: Do, Quy-Toan and Tung D. Phung (2010)
- **Journal**: American Economic Journal: Applied Economics
- **Method**: OLS/WLS with quadratic birth-year trends
- **Total Specifications**: 93

## Baseline Groups

### G1: Cohort Fertility Effect -- Auspicious Years Have Larger Birth Cohorts
- **Claim**: Birth cohorts are approximately 11% larger in zodiac-auspicious years, indicating superstition-driven fertility timing.
- **Expected sign**: Positive
- **Baseline spec**: `baseline` (Table 3 Column 1)
- **Baseline outcome**: `log_cohort_size_all` (log provincial birth cohort size, 1977-1998)
- **Baseline treatment**: `BorG` (born in auspicious year for boys or girls)
- **Baseline coefficient**: 0.111 (SE: 0.035, t=3.14, p=0.002)
- **N**: 22 year-level observations
- **Specs assigned**: 45 (including 22 jackknife)

### G2: Education Effect -- Auspicious Year Birth Increases Schooling
- **Claim**: Children born in auspicious years have about 0.3 more years of schooling.
- **Expected sign**: Positive
- **Baseline spec**: `educ_baseline_col1` (Table 4 Column 1)
- **Baseline outcome**: `School` (years of schooling)
- **Baseline treatment**: `BorG`
- **Baseline coefficient**: 0.305 (SE: 0.125, t=2.43, p=0.015)
- **N**: 6,571,225 individual-level observations
- **Clustering**: birth year (only 21 clusters)
- **Specs assigned**: 34

### G3: Health Effect -- Auspicious Year Birth Affects Height
- **Claim**: Children born in auspicious years have different height outcomes (proxy for health).
- **Expected sign**: Negative (note: sign is counterintuitive; the paper uses height-for-age z-scores which could not be replicated)
- **Baseline spec**: `health_baseline` (Table 4 Column 5 analog)
- **Baseline outcome**: `Height` (raw height, not z-score)
- **Baseline treatment**: `BorG`
- **Baseline coefficient**: -1.778 (SE: 0.610, t=-2.92, p=0.004)
- **N**: 57,592 individual-level observations
- **Estimation**: WLS with survey weights, birthyr-clustered SE
- **Specs assigned**: 14

## Classification Summary

| Category | Count | Core? | Description |
|----------|-------|-------|-------------|
| core_sample | 27 | Yes | Sample restrictions: 22 jackknife, drop years, boys/girls cohort decomposition, pre-1995 |
| core_controls | 13 | Yes | Baseline specs + control additions/removals (birth order, siblings, household, cohort size, month FE) |
| core_funcform | 7 | Yes | Functional form: linear trend, cubic trend, no trend, levels outcome |
| core_inference | 4 | Yes | SE/clustering alternatives: robust SE, province clustering, district clustering |
| core_fe | 2 | Yes | Fixed effects: province FE, district FE |
| core_method | 1 | Yes | Estimation method: unweighted OLS instead of WLS |
| **Total Core** | **54** | | |
| noncore_alt_treatment | 20 | No | Treatment decompositions (Bonly, Gonly, BandG, Good, B, G, BorGSex) |
| noncore_heterogeneity | 16 | No | Subgroup analyses (boys/girls, Kinh, urban/rural, Buddhist, firstborn, ethnic, religious subpop cohorts) |
| noncore_alt_outcome | 3 | No | Different outcomes (sex ratio, binary enrollment, log(School+1)) |
| **Total Non-Core** | **39** | | |
| **Grand Total** | **93** | | |

## Classification Decisions and Rationale

### Core Test Decisions

1. **Jackknife leave-one-year-out (22 specs, G1)**: All 22 jackknife specifications drop one birth year at a time from the cohort regression. These are standard leave-one-out sensitivity checks and are classified as core_sample. They all retain the same outcome, treatment, and model specification.

2. **Sample restrictions -- cohort (4 specs, G1)**: Dropping 1991, dropping 1991-1994, pre-1995 only, and the boys/girls cohort decompositions are core tests. The boys and girls log cohort sizes are natural subcomponents of the total cohort (core_sample).

3. **Control additions -- education (6 specs, G2)**: Progressive addition of birth order + siblings (Col 2), full household controls (Col 3), and commune/district/province cohort size controls are standard robustness checks that keep the same outcome and treatment.

4. **Control additions -- health (3 specs, G3)**: Adding birth order + siblings (Col 6), full controls (Col 7), and removing month FE are core control variations.

5. **Functional form (7 specs, G1/G2/G3)**: Linear trend, cubic trend, no trend, and levels specifications are critical core functional form tests. These are especially important because the SPECIFICATION_SEARCH.md identifies trend specification as the key vulnerability.

6. **Inference variations (4 specs, G2/G3)**: Robust SE (education and health), province clustering, and district clustering test the same coefficients under different variance estimators. These are core inference tests.

7. **Fixed effects (2 specs, G2)**: Province FE and district FE are standard core FE variations for the education outcome.

8. **Estimation method (1 spec, G3)**: Unweighted OLS vs WLS for the health outcome is a core estimation method variation.

9. **Education interaction main effect (1 spec, G2)**: `educ_interaction_BorG` reports the BorG coefficient from a model that also includes BorG*Sex. Since the treatment variable is still BorG and the additional interaction is a control, this is classified as core_controls.

### Non-Core Decisions

1. **Treatment decompositions (20 specs)**: Specifications using Bonly, Gonly, BandG, Good, B, G, or BorGSex as treatment variables fundamentally change the estimand. These are not testing the same claim (BorG effect) but rather decomposing or redefining the treatment. The gender-matched "Good" variable tests a different mechanism entirely.

2. **Heterogeneity subgroups (16 specs)**: Boys-only, girls-only, Kinh-only, urban-only, rural-only, Buddhist, non-Buddhist, firstborn, laterborn, ethnic minority (education and health), plus rural, Christian/Muslim, and ethnic subpopulation cohorts. These test heterogeneous effects rather than the average treatment effect.

3. **Alternative outcomes (3 specs)**: The sex ratio outcome tests a fundamentally different hypothesis (sex selection vs. fertility timing). Binary enrollment and log(School+1) are alternative outcome parameterizations that test whether the education result is robust to outcome measurement. These are classified as noncore_alt_outcome because they change the estimand.

### Borderline Decisions

- **Boys/girls cohort sizes (cohort_lBoys, cohort_lGirls)**: Classified as **core_sample** because the total cohort is the sum of boys and girls cohorts. These are natural subcomponents.

- **Binary enrollment and log(School+1)**: Classified as **noncore_alt_outcome** rather than core because they fundamentally change the outcome variable. The SPECIFICATION_SEARCH.md notes these show *no* significant effect (t=-1.11 and t=-1.23), suggesting the education result may be sensitive to outcome parameterization. This is substantively important but represents a different claim.

- **Health boys-only and girls-only**: Classified as **noncore_heterogeneity** because they split the sample by sex, which changes the estimand. Notably, boys-only loses significance (t=-1.50) while girls-only is very strong (t=-3.76).

- **Duplicate specification**: `cohort_jackknife_drop1991` has identical results to `cohort_drop1991` (both drop 1991, yielding coefficient 0.124, t=3.69). This is counted as a jackknife spec but is effectively a duplicate.

## Data Quality Notes

1. **Duplicate specification**: `cohort_jackknife_drop1991` and `cohort_drop1991` produce identical results (same coefficient, SE, t-stat, p-value, N=21). They appear under different tree paths (sample_restriction/jackknife vs sample_restriction/drop_year).

2. **Height outcome caveat**: The health baseline uses raw Height rather than the paper's height-for-age z-scores (computed via Stata's zanthro module). The negative sign on the BorG coefficient for raw Height is counterintuitive and likely reflects age confounds that the z-score transformation would address.

3. **Few-cluster inference**: The education and health specifications cluster at the birth-year level with only 21 and 23 clusters respectively. This makes clustered t-statistics unreliable. The robust SE specifications show dramatically different t-statistics (education: 2.43 clustered vs. 175.8 robust; health: -2.92 clustered vs. -25.8 robust), confirming that within-cluster correlation is important but the few-cluster inference is fragile.

4. **Trend sensitivity**: All three baseline groups are sensitive to trend specification. With linear trend only: cohort t=1.28, education t=-0.008 (zero), health t=0.05 (zero). This is the single most important robustness concern.

## Verification Summary

- **93 total specifications** verified
- **3 baseline specifications** identified (1 per group)
- **54 core tests** (58% of total) -- same claim tested under alternative samples, controls, trends, inference, and estimation methods
- **39 non-core tests** (42% of total) -- treatment decompositions, heterogeneity subgroups, alternative outcomes
- **3 baseline groups**: G1 (cohort fertility, 45 specs), G2 (education, 34 specs), G3 (health, 14 specs)
- **1 duplicate specification** noted (cohort_jackknife_drop1991 = cohort_drop1991)
