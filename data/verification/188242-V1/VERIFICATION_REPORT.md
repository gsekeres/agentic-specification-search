# Verification Report: 188242-V1

## Paper Information
- **Title**: Entrepot: Hubs, Scale, and Trade Costs
- **Authors**: Sharat Ganapati, Woan Foong Wong, Oren Ziv
- **Journal**: AEJ: Macroeconomics (2024)
- **Paper ID**: 188242-V1
- **Verified**: 2026-02-09
- **Verifier**: verification_agent

## Summary

The specification search produced **74 specifications** for gravity equation regressions estimating the distance elasticity of trade. All specifications relate to a single baseline group (G1): the effect of log sea distance on log bilateral trade.

## Baseline Group

### G1: Distance elasticity of trade
- **Claim**: Sea distance has a negative effect on bilateral trade flows, with a distance elasticity around -1.1 in the standard gravity specification.
- **Expected sign**: Negative
- **Baseline specs**: 5
  - `baseline_seadist_only`: Bivariate OLS (coef = -0.93)
  - `baseline_gravity`: Standard gravity with GDP + bilateral controls (coef = -1.12)
  - `baseline_origin_fe`: Origin fixed effects (coef = -1.22)
  - `baseline_twoway_fe`: Two-way FE, no bilateral controls (coef = -1.46)
  - `baseline_twoway_fe_controls`: Two-way FE with bilateral controls (coef = -1.29)
- **Outcome**: ltrade (log bilateral trade value)
- **Treatment**: lseadist (log sea distance)

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **51** | |
| core_controls | 17 | Control variable build-up and leave-one-out variations |
| core_sample | 14 | Sample restriction robustness checks |
| core_fe | 9 | Fixed effects variations (none, origin, dest, two-way, continent-pair) |
| core_inference | 6 | Clustering and standard error variations |
| core_funcform | 3 | Functional form changes (quadratic, GDP product, levels) |
| core_method | 2 | Weighted regressions (GDP-weighted, population-weighted) |
| **Non-core tests** | **23** | |
| noncore_heterogeneity | 10 | Interaction terms and continent subsamples |
| noncore_alt_treatment | 5 | Alternative distance/trade-cost measures (weighted dist, contiguity, colony) |
| noncore_placebo | 4 | Placebo tests (GDP/pop as outcome, scrambled distance) |
| noncore_alt_outcome | 2 | Alternative trade measures (log quantity, trade intensity) |
| invalid | 2 | Trade intensity specs with estimation failure (r_squared = -Inf) |

## Classification Rationale

### Core tests (51 specs)

**Control variations (17)**: These are the most numerous core tests. They include:
- A control build-up sequence from bivariate to kitchen-sink (7 specs)
- Leave-one-out exercises dropping each baseline control (5 specs)
- Alternative size-variable parameterizations (GDP product, population, GDP per capita + population) (3 specs)
- The bivariate and base_controls specs duplicate baseline specs but are retained as part of the build-up sequence (2 specs)

All use the same outcome (ltrade), same treatment (lseadist), same sample, and same inference approach. They test whether the distance elasticity is sensitive to the control set, which is a meaningful alternative implementation of the same claim.

**Sample restrictions (14)**: These restrict the estimation sample while keeping the baseline specification otherwise unchanged. They include trimming outliers, excluding specific countries (USA, China), splitting by origin country size, geography (intra/inter-continent, short/long distance), and EU membership. These test whether the distance elasticity holds across subpopulations, which is a core robustness concern.

**Fixed effects variations (9)**: These vary the fixed effect structure (none, origin, destination, two-way, continent-pair), with and without bilateral controls. Fixed effects absorb unobserved country heterogeneity that could bias the distance coefficient; varying them is a core test of the same claim. Some overlap with baseline specs (ols/fe/none = baseline_gravity, ols/fe/twoway = baseline_twoway_fe_controls, ols/fe/twoway_no_controls = baseline_twoway_fe).

**Inference variations (6)**: These keep the same point estimate but vary the clustering structure (origin, destination, two-way, robust, continent) for both the no-FE and two-way FE baseline specs. They test whether statistical significance is robust to different assumptions about error structure.

**Functional form (3)**: The quadratic distance spec, GDP product gravity form, and levels (non-log) specification test whether the distance-trade relationship is robust to different functional form assumptions while maintaining the same core claim.

**Method (2)**: GDP-weighted and population-weighted regressions test whether the distance elasticity is driven by small-economy observations.

### Non-core tests (23 specs)

**Heterogeneity (10)**: Six interaction specifications (distance x large origin, distance x contiguity, distance x language, distance x colony, distance x EU, distance x FTA) and four continent-specific subsamples (Asia, Europe, Americas, Africa). While these still report a main distance coefficient, the purpose is to test whether the effect varies across subgroups, not to test the baseline claim itself. The interaction terms change the interpretation of the main coefficient. The continent subsamples are arguably borderline (they could be viewed as sample restrictions), but I classify them as heterogeneity because they were explicitly grouped under the heterogeneity category by the specification search and the purpose is to document continental variation in the distance effect.

**Alternative treatments (5)**: Two specs use CEPII weighted distance (ldistw) instead of sea distance (OLS and two-way FE), one uses contiguity, one uses colony, and one is the quadratic weighted distance spec. These test a different claim -- the effect of a different trade cost measure on trade -- rather than the same claim with different implementation choices.

**Placebos (4)**: Two specs use non-trade outcomes (GDP, population), and two use scrambled (randomly permuted) distance. These are falsification tests rather than tests of the same claim.

**Alternative outcomes (2)**: Trade quantity (ltrade_q) and its FE variant. These measure a different dimension of bilateral trade flows (quantity vs. value), changing the economic claim being tested.

**Invalid (2)**: Both trade intensity (ltrade_intensity) specifications produce r_squared = -Infinity, indicating the outcome variable construction or estimation failed. The coefficients are near machine precision (10^-13 to 10^-25), confirming these are not meaningful estimates.

## Key Observations

1. **High robustness of the core claim**: Among the 51 core specs, 49 are highly statistically significant (p < 0.001). The only exception is `robust/sample/long_distance` (coef = -0.12, p = 0.55), which is expected given the restricted distance variation in above-median distance pairs.

2. **Stable coefficient range**: Core estimates of the distance elasticity range from about -0.85 (continent-pair FE, GDP-weighted) to -1.75 (quadratic specification linear term) with median around -1.12. The interquartile range is approximately [-1.23, -0.93].

3. **Some duplicate specs**: Several non-baseline specs are numerically identical to baselines (e.g., robust/build/bivariate = baseline_seadist_only, robust/build/base_controls = baseline_gravity, ols/fe/none = baseline_gravity, ols/fe/twoway = baseline_twoway_fe_controls, ols/fe/twoway_no_controls = baseline_twoway_fe). These appear in different specification tree paths but produce identical results.

4. **Conservative classification**: I classified the continent-specific subsamples (Asia, Europe, Americas, Africa) as noncore_heterogeneity rather than core_sample. While they could be viewed as sample restrictions, they were designed to test heterogeneity in the distance effect across continents, and the specification search grouped them under heterogeneity. Similarly, the interaction specs report a main distance coefficient but the interpretation changes when interactions are present.

5. **Invalid specs flagged**: Two trade intensity specifications are clearly invalid due to estimation failure. These should be excluded from any analysis.

## Files Produced
- `verification_baselines.json` - Baseline group definitions
- `verification_spec_map.csv` - Per-specification classification (74 rows)
- `VERIFICATION_REPORT.md` - This report
