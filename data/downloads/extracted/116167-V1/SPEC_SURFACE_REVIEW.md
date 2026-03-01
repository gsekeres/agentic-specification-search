# Specification Surface Review: 116167-V1

## Summary of Baseline Groups

One baseline group (G1) for the effect of Mediaset signal on Forza Italia vote share in 1994. This is appropriate -- while the paper also examines other parties (Table 5) and other election years, the core claim is about Berlusconi's party in 1994 (the first election after Mediaset's expansion). Other parties and years are correctly treated as exploration.

Individual-level analyses (ITANES, PIAAC) are correctly excluded -- they use different data and unit of analysis.

## Design Selection

- **Design code**: `cross_sectional_ols` -- correct. The identification relies on geographic signal variation conditional on controls/FE.
- **Design audit**: Includes model formula, selection story, FE structure, clustering, and weights. Sufficient for interpretability.
- No design variants beyond OLS are listed, which is appropriate given the continuous treatment variable.

## RC Axes

- Control progressions mirror the paper's own Table 3 column structure (none, signal only, land, land+FE, full). This is good.
- LOO drops of individual covariates are appropriate.
- Adding civic81 captures the paper's later analysis (Table A8/A10).
- Population restrictions from Table 4 (no capitals, pop caps) are high-value robustness checks the paper itself performs.
- The matched-neighbor analysis (Table 4 Cols 6-8) is included as a data/matching RC variant, which is appropriate.
- Weight variation (weighted vs unweighted) captures Table 3 Col 6.
- Treatment definition (capped signal) captures Table 3 Col 7.
- Election-year variation is included to test persistence.

## Controls Multiverse

- controls_count_min = 0, controls_count_max = 9 correctly reflects the range from bivariate to full controls.
- The note about signalfree being a near-mandatory control is appropriate -- dropping it is a useful but interpretively different specification.

## Inference Plan

- Canonical cluster at district is correct.
- The spatial SE variants (Conley at 10/30/50km) are particularly important and directly reflect the paper's own Table A3 robustness. This is a strong addition.
- Two-way clustering and SLL clustering round out the inference variants well.

## Budget and Sampling

- 80 total core specs is reasonable given the rich RC space.
- Full enumeration is feasible.

## What's Missing

- The paper uses the `acreg` command for spatial SEs with various distance cutoffs. The surface captures this with the Conley variants.
- Could add `rc/form/treatment/binary_signal` (converting continuous signal to binary above/below threshold) as an additional treatment definition variant.
- The paper's Table 1 balance tests could be included as diagnostics.

## Final Assessment

**Approved to run.** The surface correctly captures the cross-sectional OLS design with geographic signal variation, includes the paper's own revealed robustness battery (control progressions, population caps, matched neighbors, spatial SEs), and appropriately excludes mechanism/exploration analyses.
