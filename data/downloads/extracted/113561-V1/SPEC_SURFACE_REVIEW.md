# Specification Surface Review: 113561-V1

## Summary

The surface defines four baseline groups (G1-G4) for the four headline outcome measures among white respondents. This is appropriate: the paper treats all four outcomes as independent main results in Tables 4-5, and each represents a conceptually distinct measure of generosity.

## Checklist

### A) Baseline groups
- **Correct**: Each group maps to one outcome x one treatment x one population.
- **No missing groups**: The paper's main claims are about white respondents. Full-sample and Black-respondent results are secondary/mechanism analysis.
- **No spurious groups**: Table 3 perception outcomes and Table 6 interaction effects are correctly excluded as exploration/mechanism.

### B) Design selection
- **Correct**: `randomized_experiment` is the right design. This is a pure survey experiment with random assignment.

### C) RC axes
- **Controls**: Appropriate. Block-level LOO is more informative than individual-variable LOO given 29+ controls. The progression from bivariate to full is well-structured.
- **Sample**: All four sample restrictions from Table 5 are included (main variant, Slidell, Biloxi, race-shown).
- **Weights**: Weighted vs unweighted comparison matches the paper's own robustness.
- **Functional form**: Topcode variant for G1 (giving) is a reasonable addition given the censoring structure. G2 already uses topcoded outcome. G3/G4 are ordinal scales where functional form RC is less relevant.
- **Missing high-value axis**: No preprocessing/coding RC was included. However, the data construction is straightforward for this experiment (no complex variable coding choices), so this omission is acceptable.

### D) Controls multiverse policy
- **Mandatory controls**: picraceb and picobscur are correctly marked mandatory -- they capture the other randomly assigned photo conditions and must always be included.
- **Count envelope**: min=10, max=34 is reasonable given the baseline has 29-31 controls and the extended set adds 3 more.

### E) Inference plan
- **Canonical**: HC1 matches the paper's `robust` option. Correct.
- **Variant**: HC3 for G1 only is a reasonable stress test. The sample sizes (~900 for white respondents) are large enough that HC1 vs HC3 differences should be minimal.

### F) Budgets + sampling
- Total ~87 specs across 4 groups. This comfortably exceeds the 50-spec target.
- No combinatorial subset sampling needed -- block-level LOO and progression provide adequate coverage.

### G) Diagnostics
- No diagnostics planned. This is acceptable for a clean RCT where randomization is the identification strategy.

## Changes Made

1. No changes to the surface JSON were needed. The surface is well-structured and faithful to the revealed search space.

## Final Assessment

**Approved to run.** The surface is conceptually coherent, statistically principled, faithful to the manuscript, and auditable.
