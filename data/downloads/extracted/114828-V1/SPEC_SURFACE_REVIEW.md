# Specification Surface Review: 114828-V1

## Summary of Baseline Groups

One baseline group (G1) for the attitude discontinuities at the Pale of Settlement boundary. This is appropriate -- all four outcomes (prefer_market, prefer_democracy, selfemp, trust_d) are jointly treated as measuring the "antimarket culture" concept. They share the same RD design and sample.

The fuzzy RD (Table 5), population movements IV (Table 6), and pogroms analysis (Table 7) are correctly excluded. These address mechanisms or alternative estimands, not the core reduced-form RD claim.

## Design Selection

- **Design code**: `regression_discontinuity` -- correct for the geographic/spatial RD.
- **Design audit**: Includes running variable, cutoff, bandwidth, kernel, polynomial order, and notes the spatial nature. The `spatial_rd: true` flag is a useful addition.
- Design variants cover bandwidth, polynomial order, kernel, and bias-corrected inference.

## RC Axes

- LOO control drops for the parametric specification are appropriate -- the paper shows both nonparametric (no controls) and parametric (with controls) versions.
- Control set progressions (none, geographic, demographic, full) capture the paper's own revealed search space.
- Bandwidth variations (30, 90, 120 km) span a reasonable range around the baseline 60km.
- Country-specific subsamples are high-value given the cross-country nature of the geographic boundary.
- Donut exclusions (5km, 10km) are standard for geographic RD.

## Controls Multiverse

- For the nonparametric RD, controls_count_min = 0 is correct.
- For the parametric version, controls_count_max = 18 reflects the full covariate set.
- LOO drops are applied only to the parametric specification, which is appropriate.

## Inference Plan

- Canonical cluster at PSU is correct (matches `cl(psu1)` in code).
- HC1 robust and country clustering are reasonable variants. The country clustering caveat (very few clusters) is appropriately noted.

## Budget and Sampling

- 80 total core specs is reasonable for the combined design-variant and RC space.
- Full enumeration is feasible.

## What's Missing

- The paper shows bandwidth robustness at bw=120 (Table B2). This is already captured in the surface.
- Rural-specific nonparametric RD results are shown in Table 3. These are included via the urban/rural sample restriction RC specs.
- Could add the parametric RD as a separate design variant (`design/regression_discontinuity/procedure/parametric_control_function`), since the paper explicitly contrasts parametric vs nonparametric approaches.

## Final Assessment

**Approved to run.** The surface correctly identifies the core geographic RD claim, includes appropriate design and RC variants, and properly excludes mechanism/exploration analyses.
