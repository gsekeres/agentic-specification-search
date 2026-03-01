# Specification Surface Review: 113229-V1

## Summary of Baseline Groups

**G1 (Betrayal Aversion)**: Correctly identified as the paper's main claim. The three Table 2 columns represent progressively richer control sets for the same claim object (MAP ~ game type). The focal coefficient is on `tg` (Trust Game dummy) with RDG as the omitted category.

- Baseline group is well-defined: single outcome concept (MAP), single treatment concept (game type), single estimand (ITT), single population (pooled lab participants).
- No missing baseline groups: the paper's headline claim is that TG MAP > RDG MAP, and this is fully captured.

## Design Selection

- `randomized_experiment` is correct. Subjects were randomly assigned to game types within sessions.
- `design_audit` is adequate: records randomization unit, estimand, clustering, and the multi-arm nature of the treatment.
- Design variants are minimal and appropriate: `diff_in_means` (simple mean comparison) and `with_covariates` (adding pre-treatment covariates).

## RC Axes Assessment

- **Controls**: Appropriate. The pool of 9 non-interaction controls is small enough for near-exhaustive enumeration. Block structure (demographics vs country dummies) is well-motivated.
- **Sample restrictions**: Leave-one-country-out is a high-value axis given the cross-country nature of the study. Gender subsamples are appropriate given Table 2 Col 3 interactions.
- **Functional form**: Logit(MAP) is a reasonable transform since MAP is bounded. Treatment isolation specs are appropriate for a multi-arm design.
- **Missing axes**: No weighting axis is included, which is fine since the experiment has no survey weights. No data construction axis is needed (data is clean from lab).

## Controls Multiverse Policy

- `controls_count_min=0, controls_count_max=14` correctly derived from Table 2 columns.
- The 5 interaction terms (female_tg, female_dp, oman_tg, oman_dp, plus the implicit fifth from the code) are appropriately treated as part of the expanded control set rather than as separate treatment concepts.
- `linked_adjustment=false` is correct since there is no bundled estimator.

## Inference Plan

- Canonical `infer/se/cluster/session` matches the paper exactly.
- HC1 and HC3 variants are reasonable stress tests for the small number of clusters (sessions).
- No wild cluster bootstrap variant is listed; this could be valuable given that the number of clusters may be small, but is not blocking.

## Budget Assessment

- 60 max core specs is adequate for this paper's relatively simple specification space.
- 15 control subsets is reasonable given the small pool.

## What's Missing

- Wild cluster bootstrap inference could be added as a variant (small number of clusters concern).
- No Bonferroni/MHT correction for multiple comparisons across the two focal coefficients (tg and dp), but this would be a `post/*` object, not core.

## Verdict

**Approved to run.** The surface is conceptually coherent, faithful to the manuscript, and appropriately scoped.
