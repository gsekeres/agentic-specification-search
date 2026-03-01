# Specification Surface Review: 114759-V1

## Summary of Baseline Groups

One baseline group (G1) for the density discontinuity at the SISBEN cutoff. This is appropriate -- the paper's core claim is about manipulation of the poverty score, evidenced by the RD density jump.

Tables 6-7 (electoral tightness correlates) are correctly excluded from the core universe and classified as exploration. They address a different question (why manipulation occurs) using municipality-level panel FE regressions, not the RD itself.

## Design Selection

- **Design code**: `regression_discontinuity` -- correct for the sharp RD at score=47.
- **Design audit**: Includes running variable, cutoff, RD type, bandwidth rule, kernel, polynomial order, and bias correction. This is sufficient for interpretability.
- Design variants cover the standard RD robustness battery: bandwidth, polynomial order, kernel, and bias-corrected inference.

## RC Axes

- Bandwidth variations and donut exclusions are the highest-leverage RC axes for this RD, and both are included.
- Placebo cutoffs are included as RC (they test the sharpness of the effect at the specific cutoff).
- Sample restrictions (by SES stratum, pre/post 1998) are reasonable given the paper's discussion of temporal variation in manipulation.
- Data construction variants (floor vs round score) are appropriate given the discrete running variable.
- No covariate controls to vary -- the RD is on collapsed density data.

## Controls Multiverse

Not applicable. The RD specification has no separate covariate controls beyond the running variable polynomial. The `controls_count_min=0` and `controls_count_max=2` correctly reflect that the only "controls" are polynomial terms.

## Inference Plan

- Canonical HC1 is correct (matches Stata `robust`).
- Clustering at score level is a natural variant for the discrete running variable.

## Budget and Sampling

- 60 total core specs is reasonable for the design-variant-heavy RD surface.
- No controls-subset sampling needed.
- Full enumeration is feasible.

## What's Missing

- Could consider a "global polynomial" design variant (e.g., global 4th-order polynomial on full sample as in the paper's Table 3 manual bandwidth computation), but this is less standard and the local polynomial variants provide good coverage.
- The paper runs year-by-year estimates in Table 3; these could be treated as additional baseline specs or as exploration by time period. The current surface treats the pooled estimate as the single baseline, which is defensible.

## Final Assessment

**Approved to run.** The surface is well-defined, conceptually coherent, and faithful to the manuscript's revealed search space. The budget is sufficient for meaningful coverage of the design-variant space.
