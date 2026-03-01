# Specification Surface Review: 114875-V1

## Summary of Baseline Groups

One baseline group (G1) for the subsidy effect on investment at the score threshold. This is correct -- the paper has a single core claim object (effect of R&D subsidy on investment). The subgroup analyses (small vs large firms, high vs low coverage, young vs old firms) are properly treated as RC sample restrictions rather than separate baselines.

## Design Selection

- **Design code**: `regression_discontinuity` -- correct for the sharp RD at score=75.
- **Design audit**: Includes running variable, cutoff, bandwidth rules, polynomial orders, and the separate-polynomial estimator approach. Notes the parametric style (not modern local polynomial).
- Design variants appropriately cover polynomial order, bandwidth, kernel, and bias-corrected inference.

## RC Axes

- Polynomial order variation (0-3) is the paper's primary revealed search space and is well covered.
- Bandwidth variations span the paper's own choices (full sample, 50%, 35%) plus additional windows.
- Subgroup restrictions (firm size, coverage ratio, firm age) are from Tables 5-6 and represent the paper's own robustness battery.
- Donut exclusions and placebo cutoffs are standard RD checks.
- Functional form transformations (log, asinh) of INVSALES are appropriate given the ratio outcome.

## Controls Multiverse

Not applicable in the traditional sense. The "controls" are entirely polynomial terms in the running variable, and the paper does not include separate covariate controls. The envelope of 0-6 polynomial terms is correct.

## Inference Plan

- Canonical cluster at score level matches the code (`cluster(score)`). This is appropriate for the discrete running variable.
- HC1 robust is a natural alternative.

## Budget and Sampling

- 70 total core specs is reasonable.
- Full enumeration is feasible.

## What's Missing

- The paper does not report modern CCT-style bandwidth selection or bias-corrected inference. Including `design/regression_discontinuity/procedure/robust_bias_corrected` is a good addition that goes beyond the paper's own approach.
- Could consider an RD estimate using the `rdrobust` package's data-driven bandwidth as a design variant.

## Final Assessment

**Approved to run.** The surface captures the paper's parametric-style RD approach with appropriate polynomial, bandwidth, and subgroup variations. The design audit correctly documents the separate-polynomial estimator.
