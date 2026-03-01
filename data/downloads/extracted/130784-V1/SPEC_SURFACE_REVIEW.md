# Specification Surface Review: 130784-V1

## Summary

The surface defines one baseline group (G1) for the headline claim: child marriage bans reduce child marriage rates across 17 countries, using a generalized DiD with cohort x regional-intensity variation.

## Checklist

### A) Baseline groups
- **PASS**: Single baseline group correctly captures the paper's main claim object (effect of ban on child marriage).
- The paper's other outcomes (marriage_age, educ, employed) are appropriately included as additional baseline specs within G1 since the paper treats them as co-equal results in Table 4.
- age_firstbirth is excluded from additional baselines (less central to headline claim).

### B) Design selection
- **PASS**: `difference_in_differences` is correct. The paper exploits cohort x region-intensity variation from staggered adoption of child marriage bans.
- The `design_audit` correctly records the key design parameters: TWFE estimator, FE structure, clustering, treatment type.
- No design variants included (paper does not test alternative DiD estimators like Callaway-Sant'Anna); this is appropriate since the paper uses a continuous intensity measure with 2-way FE rather than a standard staggered binary treatment.

### C) RC axes
- **PASS**: Comprehensive set of RC axes covering:
  - Controls: linear time trend, quadratic time trend, CSL controls, demographics FE, region trends
  - Sample: urban/rural, country groups by baseline legal age, age restrictions, jackknife
  - Treatment construction: 7 alternative intensity measures, 3 alternative ban-cohort cutoffs
  - Outcome coding: 5 alternative child marriage age thresholds
  - Additional FE: interview year, ban year

### D) Controls multiverse policy
- **PASS**: Baseline has 0 controls; the paper's robustness adds controls one dimension at a time. No control-subset sampling needed. This is correctly reflected in `controls_count_min=0`, `controls_count_max=3`.

### E) Inference plan
- **PASS**: Canonical inference is CRV1 at countryregionurban, matching paper. Country-level clustering variant is a useful stress test (17 clusters). HC1 variant included.

### F) Budgets + sampling
- **PASS**: ~55-60 planned specs within budget of 100. Full enumeration, no sampling needed.

### G) Diagnostics
- No diagnostics planned. Pre-trends would require event-study framework which is complex for this continuous-intensity design. Acceptable for core surface.

## Changes Made
- None. Surface is well-designed and approved for execution.

## What is missing (minor)
- The paper also tests alternative distance measures constructed from different pre-ban age cohort pools (distance25, distance40, distance50). These are included as rc/data/treatment variants.
- Region-specific trends are listed but may be computationally expensive with 282 regions x age trends. Will handle in execution.

## Final Assessment
**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's revealed specification search space, and well-scoped within budget constraints.
