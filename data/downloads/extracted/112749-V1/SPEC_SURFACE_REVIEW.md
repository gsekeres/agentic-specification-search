# Specification Surface Review: 112749-V1

## Summary of Baseline Groups

### G1: Black Population Share (lnfrac_black)
- Claim object well-defined: log Black population share, flood intensity DiD, cotton-belt counties
- Baseline spec matches Table 2 Col 1 replication (coef = -0.156, se = 0.032, N = 2604)
- **Approved**: coherent and faithful to the paper's main claim

### G2: Farm Equipment Value (lnvalue_equipment)
- Claim object well-defined: log farm equipment value, flood intensity DiD, cotton-belt counties
- Baseline spec matches Table 4 Col 2 replication (coef = 0.440, se = 0.099, N = 2170)
- **Approved**: coherent and faithful to the paper's second main claim

No additional baseline groups needed. Tables 3 and 5 are correctly classified as exploration (alternative outcomes).

## Key Constraints and Linkage Rules

1. **County FE always included**: The DiD identification relies on within-county variation. Dropping county FE would fundamentally change the estimand. Surface correctly treats county FE as mandatory.

2. **State-year FE**: Implemented via dummy variables (d_sy_*) rather than absorbed FE. This is because the paper uses `areg` with `absorb(fips)` and includes state-year dummies as explicit regressors. The surface correctly includes "drop state-year FE" as an RC variant.

3. **Weights**: Paper uses analytic weights by county area (county_w). The surface includes unweighted and population-weighted alternatives, which is appropriate.

4. **Treatment block**: The f_int_{year} variables must always be included as a complete block for the relevant post-flood years. The surface correctly treats this as invariant.

5. **Control blocks are independent**: Geography controls, outcome lags, and New Deal controls are independent blocks that can be varied separately. No linkage constraint needed.

## Budget and Sampling Assessment

- Target of 50+ specifications is achievable with the planned RC axes
- Full enumeration is feasible (no combinatorial explosion)
- The surface defines approximately 15 RC variants per group, which with baseline gives ~30-35 specs per group and 60-70 total. This exceeds the 50-spec target.

## Changes Made to Surface

1. **No structural changes**: The surface is well-designed and faithful to the paper.
2. **Minor clarification**: Added note that the focal parameter for G1 is f_int_1930 (immediate post-flood effect) and for G2 is f_int_1940 (peak mechanization response), which are the most important coefficients from the paper's argument.

## What's Missing (Minor)

1. **Placebo/pre-trend test**: Not included as a core spec (correctly classified as diagnostic). Could be added to diagnostics_plan if needed.
2. **Alternative treatment definitions**: Could explore flood_intensity vs binary flood indicator vs pop_affected. The binary flood indicator is included; pop_affected is not (would change the treatment concept).
3. **Conley spatial SE**: Not available in Python environment. Noted as excluded.

## Final Assessment

**APPROVED TO RUN**

The surface is conceptually coherent, statistically principled, faithful to the revealed manuscript surface, and auditable. The two baseline groups correctly capture the paper's two main claims. The RC axes are well-chosen and cover the most important degrees of freedom (controls, FE, sample, functional form, weights).
