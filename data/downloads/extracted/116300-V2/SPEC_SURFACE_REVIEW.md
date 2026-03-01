# Specification Surface Review: 116300-V2

## Summary of Baseline Groups

One baseline group (G1) for the elasticity of risky asset share to wealth changes. This is correct -- the paper's core claim is a single regression coefficient (dfw on dpstk) testing whether portfolio allocation responds to wealth fluctuations.

The IV specifications are correctly excluded from the core universe -- they use different instruments and address endogeneity, making them a different estimand-preserving approach but computationally distinct. They could be added as design variants if desired.

The participation probits (Table 2) and inertia tests are correctly excluded as they address different questions.

## Design Selection

- **Design code**: `panel_fixed_effects` -- correct. The paper uses first-differenced OLS on PSID panel data.
- **Design audit**: Includes panel unit/time, FE structure, differencing approach, and clustering. The `first_difference` differencing label is accurate -- all variables are in changes (d-prefixed).
- The design variant `design/panel_fixed_effects/estimator/first_difference` is appropriate.

## RC Axes

- LOO control drops cover the key control variables (inheritance, family composition, employment, wealth components).
- Control set progressions (minimal, demographics+income, full, full+composition) mirror the paper's own revealed structure.
- Random control subsets (10 draws) provide additional coverage of the large control space.
- Sample period variation (1984-1999 vs 1999-2003) is the paper's own primary sample split.
- Functional form variations on both the outcome and treatment are directly from the paper's appendix tables (dlogpstk, dlogitpstk, dpstklev, dpbhstk/dtwnocar).
- Weight variation (famwt vs unweighted) captures the paper's appendix robustness.

## Controls Multiverse

- controls_count_min = 20, controls_count_max = 32 reflects the paper's practice of always including the core demographic + education + income controls.
- The year*region interactions are correctly noted as mandatory.
- 10 random control subsets with stratified_size sampling is appropriate for the 28-32 control pool.

## Inference Plan

- Canonical cluster at famid is correct.
- The bootstrap quantile regression variant captures the paper's appendix analysis (bsqreg).
- HC1 robust is a standard alternative.

## Budget and Sampling

- 80 total core specs is reasonable given the rich control multiverse.
- 10 random control subsets is appropriate.
- Seed 116300 ensures reproducibility.

## What's Missing

- The IV/2SLS specifications (instrumenting dfw with dincd1, dincd2, loginher) could be added as a design variant. However, since the instruments are testing endogeneity rather than providing a standard RC, excluding them from core is defensible.
- The spline regression (dfwsp1-dfwsp4) could be added as a functional form variant.
- The paper's Table 7 (lagged wealth changes, dlfw instead of dfw) could be added as an additional treatment definition variant.
- Could add the Hausman endogeneity test from the ivreg2 output as a diagnostic.

## Final Assessment

**Approved to run.** The surface correctly captures the first-differenced panel design with a rich control multiverse, appropriate functional form variations, and the key sample period split. The budget is sufficient for meaningful coverage.
