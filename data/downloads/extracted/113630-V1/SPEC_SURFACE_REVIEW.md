# Specification Surface Review: 113630-V1

## Summary of Baseline Groups

**G1_mortality (Child Mortality)**: Correctly identified as the paper's flagship result. The three mortality measures (crude, infant, exposure-corrected) represent different operationalizations of the same underlying concept. The two experiments (I&P and P) are included as baselines and sample restrictions.

**G2_weight (Weight-for-Age Z-Score)**: Correctly identified as a secondary headline outcome. The age group split (infants vs older children) is appropriate given the paper's separate presentation.

- No missing baseline groups for the primary outcomes. Immunization (Table 8) could be a third group but is correctly treated as secondary.
- The cross-experiment comparison (Panel C) is correctly excluded from core (it changes the estimand to a diff-in-diff between experiments).

## Design Selection

- `randomized_experiment` is correct for both groups. This is a cluster-RCT.
- `design_audit` blocks are adequate: record randomization unit (health facility), strata (district), estimand (ITT), and clustering.
- The distinction between G1 (cluster-level data, robust SEs) and G2 (individual-level data, cluster SEs) is correctly handled.

## RC Axes Assessment

### G1_mortality
- **Controls**: Baseline facility controls from Panel C are the right addition. The pool is very small (3 controls + 3 squares).
- **Sample**: Experiment-specific, age-group, and time restrictions are appropriate.
- **Outcome transforms**: Multiple mortality measures are well-captured (crude, infant, neonatal, exposure-corrected).
- **Design variants**: Difference-in-means and Poisson with exposure offset are the right alternatives.
- **Weights**: Exposure weighting is important for mortality data.

### G2_weight
- **Sample**: Age-group split and outlier trimming are the main axes.
- **Height-for-age**: Good alternative outcome that preserves the nutritional status concept.
- **Missing axes**: No age-continuous analysis (the paper's Figure 5 shows treatment effect by month of exposure). This is `explore/*`.

## Controls Multiverse Policy

- `controls_count_min=0, controls_count_max=6` is correct for both groups.
- The baseline regressions use no controls beyond strata FE, which is standard for well-balanced RCTs.
- The Panel C baseline controls are only used for the cross-experiment comparison, not the main results. Including them as optional controls for the main specifications is a mild extension.

## Inference Plan

- G1 canonical (HC1 robust at cluster level) is correct since data is already collapsed.
- G2 canonical (cluster at hfcode) is correct for individual-level data.
- District-level clustering is an important stress test given the small number of districts.

## Budget Assessment

- 60 + 40 = 100 total core specs is adequate.
- The specification space is modest given the simple experimental design.

## What's Missing

- Wild cluster bootstrap for G2 (individual-level data with potentially few clusters per district).
- Randomization inference (permutation tests) would be a natural inference variant for an RCT.
- Lee bounds for differential attrition are mentioned in the diagnostics plan but not in the inference variants.

## Verdict

**Approved to run.** The surface is faithful to the two-experiment RCT structure and correctly distinguishes between cluster-level (mortality) and individual-level (weight) analyses. The two baseline groups capture the paper's main headline results well.
