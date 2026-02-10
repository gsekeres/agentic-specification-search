# Verification Report: 216882-V1

## Paper Information
- **Title**: The Franchise, Policing, and Race: Evidence from Arrests Data and the Voting Rights Act
- **Authors**: Giovanni Facchini, Brian Knight, Cecilia Testa
- **Journal**: American Economic Journal: Applied Economics
- **Method**: Cross-sectional OLS (long differences)
- **Total Specifications**: 78

## Baseline Groups

### G1: Main Claim -- VRA Reduced Discriminatory Policing (Black Arrest Rates)
- **Claim**: Counties with higher pre-VRA Black population shares that were covered by VRA Section 5 experienced larger relative declines in Black arrest rates.
- **Expected sign**: Negative
- **Baseline spec**: `baseline` (basic controls: pop, black_share; state FE; judicial division clustering)
- **Baseline outcome**: `ln_diff_tot_arrest_black` (long-difference in log total Black arrest rates, sheriff + police)
- **Baseline treatment**: `treatment` (black_share60_proxy x VRA coverage)
- **Baseline coefficient**: -0.0049 (SE: 0.0041, p = 0.230)
- **Specs assigned**: 65

### G2: Placebo -- White Arrest Rates
- **Claim**: VRA treatment had no significant effect on white arrest rates (falsification test).
- **Expected sign**: Zero (null)
- **Baseline spec**: `baseline_white`
- **Baseline outcome**: `ln_diff_total_white`
- **Baseline treatment**: `treatment`
- **Baseline coefficient**: -0.0072 (SE: 0.0144, p = 0.619)
- **Specs assigned**: 13

## Classification Summary

| Category | Count | Core? | Description |
|----------|-------|-------|-------------|
| core_controls | 16 | Yes | Control set variations (build-up, drop-one-at-a-time, interaction terms) |
| core_sample | 11 | Yes | Sample restrictions (drop state, trim outliers, agency decomposition) |
| core_inference | 5 | Yes | Standard error / clustering alternatives (HC1, HC3, state clusters, bootstrap, classical) |
| core_funcform | 3 | Yes | Functional form (raw counts, quadratic, quadratic in treatment) |
| core_method | 2 | Yes | Estimation method (quantile regression, WLS) |
| core_fe | 1 | Yes | Fixed effects variation (no state FE) |
| **Total Core** | **38** | | |
| noncore_alt_outcome | 20 | No | Different outcomes (white arrests, political representation, granular agency x crime-type decompositions) |
| noncore_heterogeneity | 13 | No | Subgroup analyses (Deep South, Upper South, high/low black share, protest counties, NAACP, KKK, black police) |
| noncore_placebo | 4 | No | Placebo tests (white arrest rates with different controls/agencies) |
| noncore_alt_treatment | 3 | No | Different treatment definitions (coverage only, black share only, log treatment) |
| **Total Non-Core** | **40** | | |
| **Grand Total** | **78** | | |

## Classification Decisions and Rationale

### Core Test Decisions

1. **Control progression (16 specs)**: The build-up from bivariate to kitchen-sink controls, plus drop-one-at-a-time sensitivity tests, are classic core robustness checks for the same claim. Also includes the `both_terms` specification which adds main effects of the interaction components.

2. **Sample restrictions (11 specs)**: Dropping individual states (5 specs), trimming outliers (2 specs), and agency-level decomposition into sheriff vs. police black arrests (4 specs including 2 duplicates) are core tests of the same claim. The sheriff and police decompositions are classified as core because they are natural subcomponents of the combined outcome.

3. **Inference variations (5 specs)**: HC1, HC3, classical, state-clustered, and bootstrap SEs are standard core inference robustness checks. They have the same point estimates (or very close for bootstrap) with different standard errors.

4. **Functional form (3 specs)**: Raw count log differences (instead of per-capita), quadratic in black share, and quadratic in treatment are meaningful functional form variations for the same claim.

5. **Estimation method (2 specs)**: Quantile (median) regression and WLS weighted by population are alternative estimators for the same claim.

6. **Fixed effects (1 spec)**: Removing state FE entirely is a core FE variation.

### Non-Core Decisions

1. **Alternative outcomes (20 specs)**: White arrest outcomes (5 specs assigned to G2), political representation outcomes (3 specs: county governing body, municipal, judges), and granular agency x crime-type/age decompositions (12 specs). The political representation outcomes test an entirely different channel. The granular decompositions (e.g., sheriff felony black, police juvenile white) are too far removed from the combined total to constitute core tests of the same claim.

2. **Heterogeneity (13 specs)**: All subgroup analyses (Deep South, Upper South, high/low unskilled, protest counties, NAACP presence, KKK presence, black police presence, high/low black share, covered-only, non-covered-only) are classified as non-core because they test heterogeneous effects rather than the average treatment effect claim.

3. **Placebo tests (4 specs)**: White arrest rate specifications explicitly framed as placebos, plus the white arrest baseline.

4. **Alternative treatments (3 specs)**: Coverage-only indicator, black share only, and log treatment fundamentally change the treatment variable and thus test different causal channels.

### Borderline Decisions

- **Sheriff/Police Black arrest decompositions** (`ln_diff_she_black`, `ln_diff_pol_black`): Classified as **core_sample** because the combined (sheriff + police) outcome is a sum of these two components. These test whether the effect operates through elected sheriffs vs. appointed police, which is central to the paper's mechanism but uses the same treatment on a strict subcomponent of the baseline outcome.

- **Black felony/non-felony arrests** (`ln_diff_total_felony_black`, `ln_diff_total_nofelony_black`): Classified as **core_controls** (medium confidence) because these are natural decompositions of total Black arrests and test the same claim at a slightly different level of aggregation. However, the granular agency x crime-type cross-decompositions (sheriff felony, police non-felony, etc.) are classified as non-core because they are too far from the baseline.

- **High/low black share subsamples**: Classified as **noncore_heterogeneity** because splitting the sample by the treatment variable's intensity fundamentally changes the estimand.

- **Covered-only and non-covered-only subsamples**: Classified as **noncore_heterogeneity** because they also change the treatment variable (using `black_share60_proxy` alone instead of the interaction), making them joint treatment-sample changes.

## Data Quality Notes

1. **Duplicate specifications**: 4 specs appear to be exact duplicates across different tree paths:
   - `ols/outcome/she_black` = `ols/outcome/ln_diff_she_black` (identical coefficients)
   - `ols/outcome/pol_black` = `ols/outcome/ln_diff_pol_black` (identical coefficients)
   - `placebo/she_white` = `ols/outcome/ln_diff_she_white` (identical coefficients)
   - `placebo/pol_white` = `ols/outcome/ln_diff_pol_white` (identical coefficients)

2. **Proxy treatment variable**: The treatment variable uses child population nonwhite share as a proxy for the true black_share60, which was not publicly available. This affects all specifications.

3. **Missing controls**: Several paper controls (cotton suitability, farm size, Green Book establishments, rural farm share) require private ICPSR data and are not included.

## Verification Summary

- **78 total specifications** verified
- **2 baseline specifications** identified (1 per group)
- **38 core tests** (49% of total) -- meaningful alternative implementations of the main claim
- **40 non-core tests** (51% of total) -- placebos, different outcomes, heterogeneity, alternative treatments
- **2 baseline groups**: G1 (main Black arrest claim, 65 specs) and G2 (white arrest placebo, 13 specs)
- **4 duplicate specifications** noted across different tree paths
