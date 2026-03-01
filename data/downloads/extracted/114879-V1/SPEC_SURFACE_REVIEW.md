# Specification Surface Review: 114879-V1

## Summary of Baseline Groups

One baseline group (G1) for the refrigerator replacement effect on electricity consumption. This is correct -- the paper has a single primary claim. The AC replacement effect (rAC) is a secondary treatment that enters as a control, not a separate baseline.

## Design Selection

- **Design code**: `difference_in_differences` -- correct for the household-level panel DiD.
- **Design audit**: Includes panel unit, time, FE structure, clustering, treatment timing, and the double-demeaning estimator. The `double_demeaning: reg2hdfe` note is useful context for implementation.
- TWFE is the only design variant listed, which is appropriate -- the massive FE dimensions make alternative DiD estimators (Callaway-Sant'Anna, etc.) computationally challenging, and the staggered adoption is relatively clean.

## RC Axes

- **Control group choice** is the paper's most important robustness axis. The surface correctly includes random, location-matched, and usage-matched variants.
- **FE structure** variation (simple vs county-interacted month FE) captures the paper's Table 3 column variation.
- Outlier trimming is important given the `drop if usage > 200000` in the code.
- Functional form transformations (log, asinh) are appropriate for the positive consumption outcome.
- Sample restrictions (summer/winter, drop transition months) capture seasonal aspects of the electricity consumption setting.

## Controls Multiverse

- Controls_count_min = 1, controls_count_max = 2 is correct -- only rAC and rAC_summer are available.
- This is a very thin control multiverse, but that accurately reflects the paper's design (most variation is absorbed by the rich FE structure).

## Inference Plan

- Canonical cluster at county level matches the code.
- State-level clustering is a natural coarser alternative.

## Budget and Sampling

- 70 total core specs is reasonable.
- The main variation comes from control group x FE x sample restrictions, not from control subsets.

## What's Missing

- The pre-treatment trend test (pretreatmenttrend in the code) is correctly included in the diagnostics plan.
- Could consider adding the specifications that drop control households entirely (specs 4-5 in the code), which are included via `rc/sample/restrict/no_control_hh`.
- Modern DiD estimators (e.g., Sun-Abraham, Callaway-Sant'Anna) could be considered as design variants, but the computational burden with the massive household x month FE may be prohibitive.

## Final Assessment

**Approved to run.** The surface correctly identifies the household DiD design, focuses on the paper's primary robustness axes (control group choice, FE structure, outlier handling), and appropriately handles the thin control multiverse.
