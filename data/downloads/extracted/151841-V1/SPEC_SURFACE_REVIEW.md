# Specification Surface Review: 151841-V1

## Summary of Baseline Groups

**G1**: Single baseline group capturing the paper's main claim -- that peer rankings predict heterogeneous returns to capital grants. The focal coefficient is on Winner * Quint_Rank_NS (the interaction of lottery winning with average peer quintile rank excluding self-rank) in a household panel FE regression with IPW weights and group-level clustering.

This is appropriate: the paper's headline finding is Table 2, which shows this interaction is positive and significant for both income and profits outcomes.

No additional baseline groups needed. The information treatment analysis (Tables 5-6) is a separate experimental arm with a different estimand and is not the paper's headline claim.

## Checklist Assessment

### A) Baseline groups
- Single claim object is well-defined: heterogeneous ITT by peer rank.
- No missing baseline groups. Table 3 (business inputs) is a secondary outcome analysis, not a separate main claim.
- No exploration items incorrectly included as baselines.

### B) Design selection
- `randomized_experiment` is correct. This is a field experiment with a weighted lottery for grant allocation.
- `design_audit` includes all key design-defining parameters: panel structure, FE, clustering, IPW weights, and the weighted lottery mechanism.
- ANCOVA design variant is appropriate -- the paper reports this in Table A6.

### C) RC axes
- **Controls**: Well-structured using the paper's four named control panels. LOO, subset, and add (psychometric) strategies cover the control space well.
- **Data construction (rank)**: Includes the three main rank variants the paper itself examines (self-rank, relative rank, median rank) plus SD interaction. These are high-value because the rank construction is the key regressor.
- **Functional form**: Log outcomes and tercile rank parameterization both present in the paper's appendix tables.
- **Sample**: All 5 waves, groups of 5, and outlier handling all correspond to paper appendix tables.
- **Weights**: Unweighted check is important given the IPW weighting.
- **Fixed effects**: Three FE variations including the high-leverage strata-FE-instead-of-HH-FE variant.
- **Joint**: Extensive joint specifications that combine axes. These are warranted because many appendix tables combine multiple changes.

No missing high-leverage axes. The surface covers the paper's revealed search space comprehensively.

### D) Controls multiverse policy
- Controls count envelope [0, 43] correctly derived: paper shows 0-control and 26-control specs; psychometric adds 17.
- Mean-imputation + miss_control interactions is the paper's approach and is correctly documented.
- Mandatory controls: Winner must always be included as a control (it is the main effect).

### E) Inference plan
- Canonical: cluster at GroupNumber matches the paper exactly.
- Variants: HC1 (robust without clustering) and cluster at Id (household) are reasonable alternatives.
- No issues.

### F) Budgets + sampling
- Total budget of ~100 specs is adequate for the structured search space.
- Control subsets are deterministic (named panel combinations), not random -- no seed needed for sampling.
- Feasible and informative.

### G) Diagnostics plan
- Empty diagnostics plan is acceptable. Standard RCT balance checks are in the paper (Table A1) but are not estimates of the focal estimand.

## Changes Made to Surface

No changes made. The surface is well-constructed and faithful to the paper's revealed search space.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, statistically principled, faithful to the manuscript's revealed search space, and auditable. The budget is feasible and the inference plan is correct.
