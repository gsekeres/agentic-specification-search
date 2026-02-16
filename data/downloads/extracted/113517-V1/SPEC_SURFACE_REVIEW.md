# Specification Surface Review: 113517-V1

## Summary

The surface defines 4 baseline groups (G1-G4), one per dependent variable (log nominal earnings, log real earnings, log nominal hourly wage, log real hourly wage). Each group's baseline is the "all flows" specification (column 6 of the corresponding Table 1 panel).

## Review Checklist

### A) Baseline Groups
- **4 groups are appropriate**: The paper explicitly presents all 4 depvar panels as separate results, and the patterns differ materially (xee is large positive for nominal earnings, small positive for nominal hourly wages, and negative for real hourly wages). Each depvar is a distinct outcome concept.
- **No missing groups**: The paper has only Table 1 as its main results table. No other main claims.
- **No spurious groups**: All 4 are headline results.

### B) Design Selection
- **cross_sectional_ols is correct**: The second-stage regressions are OLS with absorbed market FE. The two-stage procedure is a data-construction step (generating predicted values), not a distinct design family.
- **No design variants needed**: The only implementation is OLS with absorbed FE.

### C) RC Axes
- **Control progressions**: Correctly capture the paper's revealed surface (Table 1 columns 1-7 vary which flow controls are included).
- **Leave-one-out**: Appropriate for the 5 flow controls + time trend in the baseline.
- **Job stayers subsample**: Correctly identified as a sample RC (spec 8), not a population change, since the paper frames it as a robustness check within the same Table 1.
- **EE interaction**: Correctly identified as a functional-form RC (spec 9 adds xee*eetrans_i).
- **Unweighted**: High-value RC since the paper uses survey weights throughout.
- **Drop market FE**: Useful to test whether within-market vs pooled results differ.

### D) Controls Multiverse Policy
- **Linked adjustment is correct**: First-stage controls are invariant; only second-stage flow controls vary.
- **Control count envelope (1-7)**: Correct -- ranges from EE only (1 RHS var + time trend) to all flows (6 flow vars + time trend).
- **No control-subset sampling needed**: The flow control space is small enough for full enumeration (only ~15 combinations of 6 flow variables).

### E) Inference Plan
- **Canonical = classical SE**: The paper's second-stage areg commands do not specify robust or cluster. This is correct for replication. However, since variation is at market*time level but observations are individual-level, classical SE likely understate uncertainty.
- **HC1 and cluster(mkt) variants are well-chosen**: These address the likely standard-error understatement.

### F) Budgets + Sampling
- **~17 specs per group x 4 = ~68 total**: Feasible and exceeds the 50-spec target.
- **Full enumeration is feasible**: No combinatorial explosion.

### G) Missing High-Leverage Axes
- **First-stage control variation**: Not included, correctly -- the paper does not vary first-stage controls and doing so would be very expensive with ~6M observations. This is documented in the constraints.
- **Time period splits**: Could add early/late half splits as RC, but the paper does not reveal these. Not critical.

## Changes Made
- No changes to the surface JSON. The surface is well-specified.

## Assessment
**Approved to run.** The surface faithfully captures the paper's revealed specification search space. The 4 baseline groups correctly correspond to the 4 panels of Table 1. The core universe covers all 9 columns of Table 1 plus additional RC axes (LOO, unweighted, drop FE). The inference plan appropriately flags that classical SE are the canonical choice but adds robust alternatives.
