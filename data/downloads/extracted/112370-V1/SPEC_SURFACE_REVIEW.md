# Specification Surface Review: 112370-V1

## Summary of Baseline Groups
- **G1**: Effect of capital flow disruptions on leftist election outcomes
  - Claim object is well-defined: probit of leftist election wins on US rate x FDI interaction
  - Two outcome definitions (strict vs broad leftist transition) correctly treated as baselines within same group
  - N=101 is very small; this fundamentally constrains the search space

## Changes Made
1. No changes to baseline group definition -- well-specified given the paper's scope.
2. Verified that the control-count envelope [0, 2] is appropriate given the paper uses exactly {democ, lgrgdpwork} in all main specs.
3. Added LPM as a design alternative (OLS instead of probit) since the paper exclusively uses probit but LPM is a natural robustness check for binary outcomes.
4. Noted that country-level clustering has only 18 clusters, which makes it unreliable -- included as a variant but not canonical.

## Key Constraints and Linkage Rules
- No bundled estimator: single-equation probit with robust SEs
- Only 2 controls available in the analysis dataset -- no scope for control subset sampling
- N=101 means sample restriction axes are inherently noisy
- Probit is the native estimator; LPM preserves the claim object but changes the estimator

## Budget/Sampling Assessment
- ~15-20 planned specs is within the 30-spec budget
- No random control subset draws needed (only 2 controls)
- The spec search will be thin but still informative for this paper's structure

## What's Missing (minor)
- Could add region-of-origin birth country dummies if data were richer, but the provided CSV files do not support this
- Interaction decomposition (separate effects of dffo12 and lagfdi) partially captured by rc/form variants
- Panel structure exists in the data but the paper does not exploit it (no country FE in probit due to small N)

## Final Assessment
**APPROVED TO RUN.** The surface is conceptually coherent given the severe constraints of this paper (N=101, 2 controls, probit). The search space is necessarily thin but faithfully reflects the manuscript's revealed surface. The main value will come from LPM vs probit comparison, outcome definition sensitivity, and sample trimming.
