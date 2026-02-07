# Verification Report: 136741-V1

## Paper
- **Title**: Historical Lynchings and Black Voter Registration
- **Author**: Williams
- **Journal**: AEJ-Applied
- **Paper ID**: 136741-V1

## Baseline Groups

### G1: Historical lynchings and black voter registration
- **Claim**: Higher historical black lynching rates (per 10,000 black population using 1900 denominator) lead to lower contemporary black voter registration rates in Southern US counties.
- **Expected sign**: Negative
- **Baseline spec_ids**: baseline
- **Baseline coefficient**: -0.469 (SE = 0.156, p = 0.003)
- **Outcome**: Blackrate_regvoters (black voter registration rate, %)
- **Treatment**: lynchcapitamob (black lynching rate per 10,000 black population, 1900 denominator)
- **Controls**: Historical controls (illiteracy, county formation year, newspapers, farm value, small farms, land inequality, free blacks)
- **Fixed effects**: State FE
- **Inference**: Robust (heteroskedasticity-consistent) standard errors
- **Sample**: 267 Southern US counties in 6 states

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **70** |
| **Core tests** | **56** |
| **Non-core tests** | **14** |
| **Invalid** | **0** |
| **Unclear** | **0** |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 21 | Control variations (leave-one-out, build-up, add controls) |
| core_sample | 11 | Sample restrictions (drop states, trim, winsorize, cap) |
| core_funcform | 12 | Functional form (log/IHS Y and X, alt treatment denominators, quadratic, binary, terciles, standardized) |
| core_inference | 6 | Inference variations (clustering, HC1-HC3, classical) |
| core_fe | 1 | Fixed effects variation (pooled OLS without FE) |
| core_method | 5 | Estimation method (WLS population/black pop, quantile regressions) |
| noncore_placebo | 3 | Placebo tests (white registration, white lynching, double placebo) |
| noncore_alt_outcome | 2 | Alternative outcomes (registration count, count with extra controls) |
| noncore_heterogeneity | 9 | Heterogeneity (interaction terms and split-sample analyses) |

## Classification Rationale

### Core tests (56 specs)

**Control variations (21)**: All leave-one-out, sequential control build-up, and additional control specifications are core tests. They vary the control set while keeping the same outcome, treatment, sample, and method. The kitchen sink and Table 5 col 5 specifications add potential mediators (incarceration, polling places) but still estimate the same causal object.

**Sample restrictions (11)**: Dropping individual states, trimming/winsorizing the outcome, and handling registration rates above 100% are all legitimate robustness checks on the same estimand.

**Functional form (13)**: This is the largest judgment-call category.
- Alternative treatment denominators (1910, 1920, 1930 population): These change the treatment variable name but measure the same conceptual object (lynching intensity). Classified as core because the paper explicitly discusses these as sensitivity checks on the preferred measure.
- EJI/Stevenson data: Alternative data source for the same treatment concept. Core.
- Log/IHS outcome and treatment: Standard functional form variations. Core.
- Quadratic: Reports the linear term from a quadratic specification. The near-zero coefficient (-0.019, p=0.95) reflects the linear term being absorbed by the quadratic, not a refutation. Core but low confidence (0.80).
- Binary treatment: Any lynching vs. none. Changes treatment concept (intensive to extensive margin) and yields a positive coefficient (+1.6, p=0.41). Core as functional form test, but worth noting.
- Tercile treatment: High lynching vs. omitted. Also changes treatment concept. Core but lowest confidence (0.75).
- Standardized: Same model with standardized variables. Core.

**Inference (6)**: State clustering, county clustering, HC1/HC2/HC3, classical SEs. All core inference tests. HC1 appears identical to baseline.

**FE (1)**: Pooled OLS without state FE. Core test of whether results hold without geographic controls.

**Method (5)**: WLS (population/black pop weights) and quantile regressions (25th/50th/75th percentiles). All core.

### Non-core tests (14 specs)

**Placebo (3)**: White registration outcome, white lynching treatment, double placebo. Diagnostic tests of causal mechanism, not alternative estimates of the main claim.

**Alt outcome (2)**: register_black (count of registered black voters, not rate) and the same count with additional controls. Count has fundamentally different scale from rate.

**Heterogeneity (9)**: Five interaction specs (education, earnings, church, incarceration, slavery) and four split-sample analyses (by education, by black share). These test heterogeneity, not the average effect.

## Top 5 Most Suspicious Rows

1. **robust/form/x_binary**: Positive coefficient (+1.61, p=0.41). Only positive spec. Extensive margin conflates high-registration and high-lynching counties.

2. **robust/form/quadratic**: Linear term near zero (-0.019, p=0.95). Expected when squared term is included -- absorbed by quadratic.

3. **robust/form/x_terciles**: High tercile coefficient small and insignificant (-0.53, p=0.82). Discretization loses much of the variation.

4. **robust/se/hc1**: Identical coefficient and SE to baseline, suggesting baseline already uses HC1. Redundant but not wrong.

5. **robust/control/with_shareblack**: Uses register_black (count) not Blackrate_regvoters (rate). Outcome mismatch with baseline.

## Recommendations for Spec-Search Script

1. **No major issues detected.** The specification search is well-structured with clear categories and a sensible baseline.

2. **Minor: HC1 redundancy.** The robust/se/hc1 specification appears identical to the baseline. The script could check whether the baseline already uses a specific HC variant and skip redundant inference tests.

3. **Minor: Interaction heterogeneity coefficients.** For the interaction specifications, the reported coefficient is the main effect (lynchcapitamob), not the interaction term. This is the correct choice for assessing the base relationship, but the main effect in an interaction model has a different interpretation (effect when moderator = 0) than the unconditional baseline.

4. **Minor: register_black outcome.** The robust/control/with_shareblack specification uses register_black (count) rather than Blackrate_regvoters (rate). If the intent was to test additional controls on the baseline claim, the outcome should remain Blackrate_regvoters.
