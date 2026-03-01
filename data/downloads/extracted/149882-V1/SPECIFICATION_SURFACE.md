# Specification Surface: 149882-V1

## Paper
"Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India"
Dhar, Jain, and Jayachandran, American Economic Review, 2022

## Overview

This is a school-level RCT evaluating the "Breakthrough" gender-equality curriculum delivered in government schools in Haryana, India. The intervention lasted 2.5 years (grades 6-9) and outcomes were measured at two endlines. The paper's headline claims are the ITT effects on (1) gender attitudes and (2) self-reported gender-related behaviors.

## Baseline Groups

### G1: Effect on gender attitudes (endline 1)

**Claim object:**
- **Outcome concept**: Gender attitude index (standardized, inverse-covariance-weighted index of survey items on gender norms)
- **Treatment concept**: School-level Breakthrough gender-equality curriculum
- **Estimand concept**: ITT effect on combined (boys + girls) gender attitudes at endline 1
- **Target population**: Students in government schools in Haryana, India (grades 6-7 at baseline, both genders)

**Baseline spec**: Table 1.2, Combined column, Gender Attitudes row
- `E_Sgender_index2 ~ B_treat + B_Sgender_index2 + district_gender_* + gender_grade_* + [endline component missing flags], cluster(Sschool_id)`
- This is the paper's primary specification: OLS with baseline outcome, strata FE, and endline missing flags. SE clustered at school (randomization unit).

**Why a separate baseline group**: Gender attitudes are the paper's primary outcome family. The paper reports this as its headline result (Abstract, Introduction p.3, Section 4.2 p.14).

### G2: Effect on self-reported behavior (endline 1)

**Claim object:**
- **Outcome concept**: Self-reported behavior index (standardized, inverse-covariance-weighted index of gender-related behaviors like opposite-sex interactions, household chores, decisions)
- **Treatment concept**: School-level Breakthrough gender-equality curriculum
- **Estimand concept**: ITT effect on combined (boys + girls) gender-related behavior at endline 1
- **Target population**: Students in government schools in Haryana, India (grades 6-7 at baseline, both genders)

**Baseline spec**: Table 1.2, Combined column, Behavior row
- `E_Sbehavior_index2 ~ B_treat + B_Sbehavior_index2 + district_gender_* + gender_grade_* + [endline component missing flags], cluster(Sschool_id)`

**Why a separate baseline group**: The paper treats behavior as a separate primary outcome (Table 1.2 presents attitudes, aspirations, and behavior as the three primary outcomes). The behavior index is conceptually distinct from attitudes, measuring whether students translate attitudes into actions.

### Why not additional baseline groups?

- **Aspirations index (girls only)**: This is a third primary outcome in Table 1.2 but is estimated only for girls (the aspirations battery was only asked of girls). This constitutes a target population change that interacts with the treatment concept, making it less suitable as a separate standalone baseline group for our specification surface.
- **Endline 2 (medium-run)**: Tables 1.8 and 1.10 report 2-year follow-up results. These use a different sample (higher attrition) and for some outcomes (scholarship, petition) different estimands. We treat these as a separate follow-up confirmation, not additional baseline groups.
- **Social desirability robustness (Table 1.3/1.9)**: These are heterogeneity analyses by social desirability score, not separate claims.

## Revealed Search Space

The paper reveals the following specification dimensions:

1. **Control sets**: The paper uses a complex control selection procedure:
   - Basic controls: strata FE (district x gender, gender x grade) + baseline index value
   - Extended controls: LASSO-selected from a pool of ~50 baseline variables (demographics, school characteristics, census data), selected via double LASSO (Appendix Table 1.10)
   - Missing flags: variables tracking imputed/missing baseline values
   - The paper runs multiple LASSO versions (cntrls_all_8 is the version used in the paper)

2. **Sample restrictions**:
   - Girls only vs boys only vs combined (Table 1.4 panels)
   - Endline 1 vs endline 2 samples (different attrition)
   - Grade 6 vs grade 7 at baseline

3. **Outcome variants**:
   - Gender attitude sub-indices: education, employment, subjugation/autonomy, fertility
   - Behavior sub-indices: opposite-sex interactions, household chores, relatives' discouragement, decision-making (girls only), mobility (girls only)

4. **Social desirability**: Interaction with social desirability score (Table 1.3) -- this is heterogeneity, not an RC.

## Core Universe

### G1: Gender Attitudes

#### Baseline specs (1)
- `baseline`: Table 1.2 combined gender attitudes (baseline index + strata FE + missing flags, clustered at school)

#### Design specs (2)
- `design/randomized_experiment/estimator/diff_in_means`: Simple difference in means (no controls)
- `design/randomized_experiment/estimator/strata_fe`: Strata FE only (no baseline index, no LASSO)

#### RC: Controls (10)
- **Sets**: no controls, strata only, strata + baseline index, strata + LASSO-selected, no missing flags
- **LOO**: Drop baseline gender index, drop district-gender FE, drop gender-grade FE, drop missing flags
- **Progression**: Bivariate, strata FE only, + baseline index, + baseline index + demographics, + LASSO extended

#### RC: Sample (5)
- Girls only, boys only (from Table 1.4 panels)
- Grade 6 only, grade 7 only
- Co-ed schools only (since single-sex schools are a special subpopulation)
- Trim outcome outliers (1/99)

#### RC: Functional form / data construction (4)
- Gender sub-index: education component
- Gender sub-index: employment component
- Gender sub-index: subjugation/autonomy component
- Equal-weight index (instead of inverse-covariance-weighted)

### G2: Behavior Index

Same structure as G1 with outcome-appropriate substitutions:
- Baseline behavior index replaces baseline gender index
- Behavior sub-indices replace gender sub-indices (opposite-sex, household chores, relatives)

## Total budget: ~50 core specs per baseline group

## Inference Plan

**Canonical**: Cluster-robust SE at the school level (Sschool_id), matching the paper. The randomization is at the school level, so clustering at school is the natural choice.

**Variants** (recorded in inference_results.csv):
- HC1 (robust, individual-level; ignores clustering)
- HC3 (small-sample leverage correction)
- Cluster at district level (coarser; only 4 districts, so this is a very conservative test)

## What Is Excluded (and why)

1. **Aspirations index (girls only)**: Changes target population (girls-only sample for an outcome only measured for girls). Treated as exploration.
2. **Endline 2 outcomes**: Different time horizon and sample. The petition and scholarship outcomes are revealed preferences (different outcome concept).
3. **Table 1.3/1.9 social desirability**: Heterogeneity analysis (interaction with social desirability score). Changes the estimand.
4. **Table 1.5 heterogeneity by gender**: Already captured as sample subvariants (girls only / boys only).
5. **Appendix Table 1.11 heterogeneity by parent attitudes**: Changes the estimand.
6. **School-level outcomes (Appendix Table 1.26)**: Different unit of observation and outcome concept.
7. **IAT outcomes (Appendix Table 1.9)**: Different measurement technology; implicit attitudes are a conceptually different outcome.
8. **Individual attitude item regressions (Appendix Table 1.9)**: Disaggregation of the index; not a separate claim.

## Linkage Constraints

Not applicable. The baseline is a simple OLS regression (no bundled estimator).

## Control Blocks (for progression)

1. **Treatment**: B_treat (always included as focal regressor)
2. **Strata FE**: district_gender_1-8, gender_grade_1-4 (randomization strata)
3. **Baseline index**: B_Sgender_index2 (or B_Sbehavior_index2 for G2)
4. **Endline missing flags**: ${el_gender_flag} (or ${el_behavior_common_flag})
5. **Demographics**: B_Sage, B_Sgrade6, B_rural, B_Scaste_sc, B_Scaste_st, B_Smuslim, B_no_female_sib, B_no_male_sib, B_Sparent_stay, B_m_secondary, etc.
6. **School characteristics**: B_fulltime_teacher, B_pct_female_teacher, B_q13_counselor, etc.
7. **Census characteristics**: Cfem_lit_rate, Cmale_lit_rate, Cfem_lab_part
8. **LASSO-selected extended**: Subset of 5-7 automatically selected via double LASSO from the above pool

## Data Notes

- Analysis dataset: `bt_analysis_final.dta` (constructed by 04a_merge_indices.do)
- ~14,809 students at endline 1 (combined boys and girls)
- ~314 schools (half treatment, half control)
- 4 districts in Haryana: Panipat, Sonipat, Rohtak, Jhajjar
- Outcomes are standardized indices (mean 0, SD 1 in control group)
- School 2711 is excluded from baseline analyses (missing at baseline but present at endline)
- Child 3205037 is excluded (blind at baseline)
