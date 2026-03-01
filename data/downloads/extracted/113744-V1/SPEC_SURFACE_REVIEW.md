# Specification Surface Review: 113744-V1

## Summary

The surface was reviewed against the paper's SPSS data and code descriptions. This is a field experiment at Subway restaurants studying how calorie information and menu ordering affect food choices.

## Baseline Groups

### G1: Total Meal Calories
- **Status**: Correctly defined. The main claim is about the effect of information/convenience nudges on total caloric intake.
- **Design code**: `randomized_experiment` is correct for a field experiment with random assignment.
- **Design audit**: Present. Correctly identifies individual-level randomization, treatment indicators, and the OLS estimator.

## Checklist Results

### A) Baseline groups
- Single baseline group for total calorie consumption -- correct as the paper's headline result.
- SandwichCal and NonSandwichCal are included as additional baseline-like rows, which is appropriate since the paper reports these as decompositions of the main outcome.
- No missing baseline groups: secondary outcomes (ChoseLowCalSandwich, CalEstimate accuracy) are correctly excluded as different claim objects.

### B) Design selection
- `randomized_experiment` is appropriate.
- Design variants correctly include study-level subsamples and pooling strategies.

### C) RC axes
- **Controls**: Good coverage. The experiment has a small but meaningful set of demographic and behavioral controls. LOO, additions, standard sets, and random subsets are all included.
- **Sample**: Appropriate outlier trimming and meaningful subgroup restrictions (non-dieters, seal-openers).
- **Functional form**: Log and asinh transforms of calorie outcome are reasonable.
- No high-leverage axes appear missing for a simple experimental design.

### D) Controls multiverse policy
- `controls_count_min=0` and `controls_count_max=10` -- appropriate. Experiments do not require controls, but adding them can improve precision and serve as balance checks.
- No linked adjustment needed.

### E) Inference plan
- HC1 is appropriate for individual-level experiments.
- Classical OLS and HC3 are useful variants.

### F) Budgets + sampling
- Budget of 65 specs is reasonable for a simple experiment with ~7 optional controls.
- Seed specified (113744).

### G) Diagnostics plan
- Balance check is standard and appropriate for RCTs.

## Key Constraints and Linkage Rules
- No bundled estimators or linked adjustment.
- Control pool is modest (7 optional variables), making the combinatorial space manageable.

## What's Missing
- Nothing material for the core calorie consumption claim.

## Final Assessment
**Approved to run.** The surface correctly identifies the main experimental claim, defines appropriate RC axes for an RCT, and sets reasonable budgets. Data is available in SPSS format.
