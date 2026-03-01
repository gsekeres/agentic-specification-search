# Specification Surface Review: 112431-V1

## Summary of Baseline Groups
- **G1**: Effect of reelection incentives on corruption (pcorrupt ~ first | uf)
  - Well-defined claim object: ATE of term-limit eligibility on corruption share
  - Baseline spec matches paper's Table 4 Col 6 exactly
  - Additional baselines (ncorrupt, ncorrupt_os) are alternative measures of the same concept

## Changes Made
1. No changes to baseline group definition -- well-specified.
2. Verified control-count envelope is appropriate: [0, 37] spanning bivariate to full.
3. Confirmed LOO candidates exclude party dummies and sorteio dummies (design-critical).
4. Confirmed no linkage constraints needed (single-equation OLS, not bundled).

## Key Constraints and Linkage Rules
- No bundled estimator: single-equation OLS with absorbed state FE
- HC1 robust SEs match the original Stata 'robust' option
- Party dummies and lottery dummies are kept as "mandatory" in most specifications (design-critical: party controls address selection, lottery dummies address audit randomization)

## Budget/Sampling Assessment
- 53 planned specs is within the 80-spec budget
- 20 random control subset draws with seed=112431 is reproducible
- LOO covers 13 droppable controls -- sufficient for sensitivity analysis
- Control progression provides 6 build-up steps -- informative for Oster-style reasoning

## What's Missing (minor)
- No exploration of Table 7 experience subsamples (correctly excluded as population change)
- No matching/IPW estimator (correctly excluded as unavailable)
- Could add winsorization of pcorrupt as rc/preprocess variant, but trim already covers outlier sensitivity

## Final Assessment
**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's revealed search space, and the budget is feasible.
