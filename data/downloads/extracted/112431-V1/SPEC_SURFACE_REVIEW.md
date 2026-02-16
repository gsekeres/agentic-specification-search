# Specification Surface Review: 112431-V1

## Summary

The surface defines one baseline group (G1) for the main claim: effect of reelection incentives (first-term status) on corruption share (pcorrupt). This is appropriate -- the paper's headline result is the pcorrupt ~ first relationship in Table 4, Column 6.

## Checklist Results

### A) Baseline Groups: PASS
- G1 corresponds to a single well-defined claim object (pcorrupt ~ first | observables + state FE).
- The decision to exclude Table 5 outcomes (ncorrupt, ncorrupt_os), Table 9 (convenios panel), and Table 8 (pmismanagement placebo) as separate claim objects is correct -- the paper frames pcorrupt as the headline.
- No missing baseline groups for the core pcorrupt claim.

### B) Design Selection: PASS
- `cross_sectional_ols` is correct. The paper uses cross-sectional OLS (areg) with controls and state FE. While Table 6 adds RDD-style polynomial controls, the estimator is still OLS, not a formal RD design.
- `design/cross_sectional_ols/estimator/ols` is the appropriate (and only necessary) design variant.

### C) RC Axes: PASS with minor adjustments
- **Controls progression**: Well-designed, follows the paper's own Table 4 build-up. The 8-step progression plus LOO and random subsets is comprehensive.
- **LOO approach**: Treating party dummies as a block (drop all 17) and sorteio dummies as a block (drop all 10) is appropriate -- dropping individual party/sorteio dummies would be uninformative.
- **FE variants**: Dropping state FE and replacing with region FE are both reasonable.
- **Sample trimming**: Appropriate for pcorrupt which has a mass at 0 and right tail.
- **Functional form**: log(1+pcorrupt) and asinh(pcorrupt) are reasonable since pcorrupt is a share bounded [0,1] with many zeros. These preserve the qualitative claim object (direction of effect on corruption) even if the coefficient interpretation changes from level to semi-elasticity. Mark as RC with interpretation note.

**Issue found**: The `rc/controls/progression/bivariate` and `rc/controls/sets/none` are redundant (both are treatment-only with state FE). Removed `rc/controls/sets/none` to avoid duplication. Similarly, `rc/controls/sets/baseline` is identical to the baseline spec itself. Removed.

**Issue found**: `rc/controls/sets/minimal` and `rc/controls/sets/extended` need explicit definitions. Set minimal = audit controls (sorteio) only, extended = baseline + lfunc_ativ + lrec_fisc.

### D) Controls Multiverse Policy: PASS
- controls_count_min=0 (bivariate) and controls_count_max=43 (full extended) correctly span the paper's revealed surface.
- No bundled estimator, so linked_adjustment=false is correct.
- Party dummies treated as a block is appropriate (they should not be individually varied).

### E) Inference Plan: PASS
- Canonical HC1 matches Stata robust SE exactly.
- State-level clustering is a sensible stress test (26 clusters is borderline but informative).
- HC3 is a good small-sample correction for N=476.

### F) Budgets + Sampling: PASS
- 80-spec budget is generous; ~50 planned specs is feasible and informative.
- Seed 112431 (paper ID) for reproducibility.
- Stratified-size sampling for random subsets is appropriate.

### G) Diagnostics: N/A
- No diagnostics plan included. For cross-sectional OLS, balance tests and placebo outcomes would be natural but are not required for the core surface.

## Changes Made to Surface

1. Removed duplicate `rc/controls/sets/none` (redundant with bivariate progression).
2. Removed `rc/controls/sets/baseline` (identical to the baseline spec itself).
3. Clarified `rc/controls/sets/minimal` = sorteio dummies only (audit design controls).
4. Clarified `rc/controls/sets/extended` = baseline + lfunc_ativ + lrec_fisc.
5. Added interpretation note for functional-form transforms.

## Final Assessment

**APPROVED TO RUN.** The surface is well-defined, auditable, and faithful to the paper's revealed specification search. The ~50 planned specifications should provide a comprehensive robustness check of the headline pcorrupt ~ first claim.
