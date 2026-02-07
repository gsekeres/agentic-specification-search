# Verification Report: 113597-V2

## Paper: The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia

**Journal**: AEJ-Applied
**Method**: RCT with cross-sectional ANCOVA analysis
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

| Group | Claim | Outcome | Baseline spec_id | Expected Sign |
|-------|-------|---------|-------------------|---------------|
| G1 | Group microfinance loans increase enterprise ownership | enterprise | baseline | + |
| G2 | Group microfinance loans increase sole entrepreneurship | soleent | baseline/soleent | + |
| G3 | Group microfinance loans increase total consumption | ln_totalc | baseline/ln_totalc | + |
| G4 | Group microfinance loans increase food consumption | ln_foodc | baseline/ln_foodc | + |
| G5 | Group microfinance loans affect scaled profit | scaled_profit_r | baseline/scaled_profit_r | unknown |

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **66** |
| Baselines | 5 |
| Core tests | 42 |
| Non-core tests | 18 |
| Invalid | 1 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 22 (includes 5 baselines + 3 control progressions + 10 LOO + 4 outcome duplicates) |
| core_sample | 16 (aimag drops, winsorization, trimming, education/loan subsamples) |
| core_inference | 6 (clustering variations across outcomes) |
| core_fe | 2 (aimag FE variations) |
| core_funcform | 3 (level consumption, IHS profit, per capita consumption) |
| noncore_heterogeneity | 7 (interaction terms for enterprise, soleent, consumption) |
| noncore_alt_outcome | 4 (scaled assets, hours total, log non-durable, log durable) |
| noncore_placebo | 3 (buddhist, hahl, under16 as predetermined outcomes) |
| noncore_alt_treatment | 2 (individual loan vs control, different causal object) |
| invalid | 1 (soum FE failed to estimate) |

---

## Core Test Distribution by Baseline Group

| Baseline Group | Core Tests | Description |
|---------------|------------|-------------|
| G1 (enterprise) | 33 | Controls, LOO, sample drops, inference, FE, trim, subsamples |
| G2 (soleent) | 3 | Outcome duplicate, aimag clustering |
| G3 (ln_totalc) | 5 | Outcome duplicate, aimag clustering, level form, per capita form |
| G4 (ln_foodc) | 1 | Outcome duplicate only |
| G5 (scaled_profit_r) | 5 | Outcome duplicate, aimag clustering, winsorizations, IHS form |

---

## Top 5 Most Suspicious Rows

1. **robust/outcome/enterprise** (spec_id: robust/outcome/enterprise): This appears to be an exact duplicate of the baseline spec. Same outcome, treatment, controls, and identical coefficient (0.0775). Likely a coding artifact where the baseline was re-run under a different spec_id. Classified as core but flagged as redundant.

2. **robust/outcome/soleent, robust/outcome/scaled_profit_r, robust/outcome/ln_totalc, robust/outcome/ln_foodc**: Similarly, these four specs under robust/outcome/ appear to be exact duplicates of the respective baselines (baseline/soleent, baseline/scaled_profit_r, baseline/ln_totalc, baseline/ln_foodc). Their coefficients are identical to the baselines. These are redundant specifications that inflate the apparent spec count without adding information.

3. **robust/fe/aimag_fe** (spec_id: robust/fe/aimag_fe): The coefficient (0.07745) and p-value (0.0257) are virtually identical to the baseline. This is because the baseline already includes aimag dummies as controls, so adding "aimag fixed effects" changes nothing substantively. Not suspicious per se, but adds no new information.

4. **robust/sample/winsorize_***: These three specs use outcome "profit_r_wins" which is a winsorized version of scaled_profit_r. The coefficients are large negative numbers (-7517 to -9051) in the same scale as the baseline profit spec (-7.85). However, the winsorized variants have much larger coefficients in absolute terms, suggesting the winsorization variable may be in different units (not scaled). This warrants investigation.

5. **robust/fe/soum_fe**: This spec failed entirely (no coefficient, SE, or p-value extracted). The error message indicates the treatment variable "group" could not be found after including soum fixed effects, likely because treatment is assigned at the soum level and is collinear with soum FE.

---

## Recommendations

1. **Remove duplicate outcome specs**: The 5 specs under robust/outcome/ that exactly replicate baselines (enterprise, soleent, scaled_profit_r, ln_totalc, ln_foodc) should be removed or merged. They inflate spec counts without adding variation.

2. **Investigate winsorized profit scaling**: The winsorized profit specs (robust/sample/winsorize_*) show coefficients of -8000 to -9000 vs the baseline scaled_profit_r of -7.85. This suggests the winsorized variable may not be properly scaled (divided by baseline mean). The spec search script should ensure consistent scaling.

3. **Fix soum FE specification**: The robust/fe/soum_fe spec fails because soum FE absorbs the treatment variation (treatment assigned at soum level). This is an expected collinearity issue. The script could handle this more gracefully or skip this specification.

4. **Consider whether heterogeneity specs should be core**: The 7 heterogeneity interaction specs report the main effect of treatment conditional on the interaction, which changes interpretation. The current classification as noncore is conservative. If the paper explicitly reports these interaction models, they could be reclassified.

5. **Clarify baseline profit direction**: The baseline scaled_profit_r shows a negative treatment effect (-7.85), which is counterintuitive for a microfinance program. This may reflect measurement issues with the profit variable or genuine heterogeneity. The expected_sign is marked as "unknown" accordingly.
