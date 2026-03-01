# Specification Surface Review: 112415-V1

## Summary of Baseline Groups
- **G1**: Effect of choice set size on attention efficiency in lab experiment
  - Within-subject design: each of ~39 subjects sees all 3 set-size conditions
  - Claim object is clear: set size affects fixation quality/efficiency
  - Multiple outcome measures (efficiency, stopping, duration, choice quality) treated as additional baselines within same group

## Changes Made
1. No changes to baseline group definition -- well-specified for a lab experiment.
2. Added explicit pairwise condition comparisons (4v9, 9v16, 4v16) as RC sample variants.
3. Confirmed no control variables are needed (within-subject design eliminates observables).
4. Added data construction variants for alternative efficiency measures available in the data.

## Key Constraints and Linkage Rules
- Within-subject design: no observational controls needed or appropriate
- Paired t-test is the native inference method; OLS with subject FE is equivalent
- Small number of subjects (~39) limits power
- Fixation-level data requires careful aggregation to subject-condition level before testing

## Budget/Sampling Assessment
- ~30-40 planned specs is within the 50-spec budget
- No random control subset draws needed (no controls)
- Multiple outcome measures and data processing choices provide the main variation

## What's Missing (minor)
- Could include non-parametric tests (Wilcoxon signed-rank) as inference variants
- Could explore trial-order effects (learning across trials) but this would be explore/*
- No structural model estimation (correctly excluded -- different claim object)

## Final Assessment
**APPROVED TO RUN.** The surface correctly identifies a within-subject experimental design with no controls. The main specification dimensions are outcome definition, sample restrictions (RT thresholds, refixation filtering), and pairwise condition comparisons. The search space is appropriately constrained by the experimental design.
