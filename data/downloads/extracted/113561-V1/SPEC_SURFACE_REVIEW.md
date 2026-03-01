# Specification Surface Review: 113561-V1

## Summary

The surface defines one baseline group (G1) for the effect of racial picture priming (picshowblack) on dictator-game giving, estimated via WLS with robust SEs. This is appropriate: the paper's headline result is the ITT effect on giving for all respondents.

## Checklist Assessment

### A) Baseline Groups
- **G1 is well-defined**: one outcome (giving), one treatment (picshowblack), one estimand (ITT), one population (all respondents). Correct.
- **No missing baseline groups**: The paper presents alternative outcomes (hypothetical giving, charity/govt support) and subsamples (white, black) in Tables 4-6, but these are extensions/heterogeneity, not separate headline claims. The paper's abstract and introduction focus on giving as the main outcome.
- **No exploration leaking into core**: Table 6 interactions with racial attitudes are correctly excluded as they change the estimand.

### B) Design Selection
- **design_code = randomized_experiment**: Correct. This is a lab-in-the-field experiment with random assignment of picture treatments.
- **design_audit**: Present and adequate. Includes randomization unit (individual), estimand (ITT), treatment arms, and weights.
- **Design variants**: Only `diff_in_means` included, which is appropriate for checking whether controls matter. No over-expansion.

### C) RC Axes
- **Controls**: LOO (20 specs), control sets (4), progression (5). Comprehensive and appropriate.
- **Sample**: Main survey only, city subsample, race-shown only. These are revealed by Table 5. Added outlier trimming (not in paper) as a standard stress test. Appropriate.
- **Functional form**: log(1+y), asinh, binary. The paper uses censored regression (cnreg) and ordered probit (oprob), which are hard to replicate exactly. Using log/asinh/binary as approximations is reasonable for the specification search purpose.
- **Weights**: Unweighted and mweight. Appropriate since the paper uses tweight throughout.

**Issue found**: The `rc/form/treatment/nraudworthy_composite` spec changes the coding of manipulation controls (aggregating worthiness dummies into a single index). This is a control-coding change, not a treatment transformation. **Changed**: Moved to `rc/controls/sets/nraudworthy` -- this is effectively a control-set variant where worthiness manipulation dummies are replaced by a composite variable.

**Issue found**: `rc/sample/outliers/topcode_giving_90` -- giving is already bounded [0,100] by the dictator game design. Topcoding at 90 makes limited sense. **Changed**: Removed this spec and replaced with `rc/sample/outliers/drop_extreme` (drop giving==0 and giving==100, i.e., only interior choices).

### D) Controls Multiverse Policy
- **controls_count_min = 13, max = 37**: Verified. The paper's "no demographics" spec (Table 5 row 5) has ~13 controls (pic vars + manipulation vars + black + other). The extended spec (Table 5 row 6) adds 3 extra controls to the ~34 baseline controls = 37.
- **Mandatory controls**: The treatment variables (picshowblack, picraceb, picobscur) must always be included. The manipulation audio dummies should be included in all specs except the "none" and "bivariate" specs, as they are part of the experimental design.
- **No bundled estimator**: Correct. Simple OLS/WLS.

### E) Inference Plan
- **Canonical = HC1**: Correct. The paper uses `robust` throughout, which is HC1 in Stata.
- **No clustering**: Correct. Randomization is at the individual level, not clustered.
- **Variants**: Classical, HC2, HC3. Appropriate small set of inference-only alternatives.

### F) Budgets + Sampling
- **Budget = 60 specs**: The enumerated core universe has ~55 specs. Full enumeration is feasible. No random sampling needed. Appropriate.

### G) Diagnostics Plan
- **Empty**: Acceptable for this paper. Standard RCT diagnostics (balance checks, attrition) could be added but are not part of the specification search pipeline.

## Changes Made to Surface

1. Removed `rc/form/treatment/nraudworthy_composite` from rc_spec_ids (this is a control-set variant, not a functional form change). Replaced with `rc/controls/sets/nraudworthy`.
2. Removed `rc/sample/outliers/topcode_giving_90` (giving is already bounded 0-100). Replaced with `rc/sample/outliers/drop_extreme_choices`.
3. No other blocking issues.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithfully reflects the paper's revealed search space, and the budget is feasible for full enumeration. The core universe of ~55 specifications provides good coverage across controls, sample, functional form, and weights dimensions.
