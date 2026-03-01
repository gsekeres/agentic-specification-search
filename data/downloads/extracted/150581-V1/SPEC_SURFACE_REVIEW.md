# Specification Surface Review: 150581-V1

## Summary of Baseline Groups

- **G1**: Wage cyclicality, job transitions, and skill mismatch (panel FE)
  - Well-defined claim object: wage semi-elasticity to unemployment rate, and how it varies with job transitions (EE/UE) and skill mismatch
  - Baseline spec matches Table 2 Col 4 exactly: `reghdfe lhrp2 unempl c.unempl#i.dummy1 c.unempl#i.dummy2 c.mismatch1w#c.unempl c.unempl#i.dummy1#c.mismatch1w c.unempl#i.dummy2#c.mismatch1w ... age agesq i.dummy_educ c.time##c.time time_trend i.month if HOURSM>=75 & age>=20, a(ID i.industry#i.year i.occupation_agg#i.year) cluster(ID)`
  - Additional baselines (Table 2 Cols 1-3, 5) are nested or extended versions of the same claim -- correctly included
  - Design code `panel_fixed_effects` is correct: within-individual FE regression with industry-year and occupation-year FE

## Changes Made

1. **Added regional unemployment note**: When using `rc/data/unemployment/regional_unempl`, the code also interacts `time_trend` with `regionfe` dummies (`c.time_trend#i.regionfe`). This is a joint data/controls change that should be documented for the runner. Added a constraint note.

2. No changes to baseline group definition, RC axes, design_audit, or budget.

## Key Constraints and Linkage Rules

- No bundled estimator: single-equation panel FE with absorbed high-dimensional FE (ID, industry x year, occupation x year)
- Cluster at individual level (ID) matches paper's `cluster(ID)`
- The triple-interaction structure (unempl x transition type x mismatch) is the core model specification -- robustness variations should preserve this structure
- Month dummies (i.month) are part of the standard control set
- `unempl` is divided by 100 before regressions -- runner must replicate this scaling
- The `rc/data/*` variants change the definitions of key variables (transition dummies, mismatch measure, unemployment rate) rather than just adding/removing controls -- the runner should be aware these require regenerating the interaction terms with the new variable definitions

## Budget/Sampling Assessment

- ~46 planned core specs within the 70-spec budget -- feasible
- 10 random control subset draws with seed=150581 is reproducible
- 5 LOO specs cover the main non-interaction controls (age/agesq, education, time polynomial, time trend, month dummies)
- 4 control progression specs provide informative build-up from bivariate to full
- 4 `rc/data/*` variants faithfully reflect Figure 3 robustness checks in the paper

## What's Missing (minor)

- The paper's Figure 3 includes 7 robustness variants total. The surface captures 4 of the most substantive ones (3-month transitions, recalls, skill-specific weights, regional unemployment). The remaining 3 (occupational skill requirements, occupational tenure, cumulative mismatch) are included as extended control sets rather than data construction variants -- this is a reasonable mapping.
- No design estimator variants (e.g., first-difference) are included; this is appropriate given the panel structure with three-way FE
- The regional unemployment variant involves a joint change (treatment variable + time trend interaction), which is correctly noted but could be formalized as `rc/joint/*` rather than `rc/data/*`. This is a minor classification issue that does not affect execution.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's Figure 3 robustness exercise and Table 2 column progression, and the budget is feasible.
