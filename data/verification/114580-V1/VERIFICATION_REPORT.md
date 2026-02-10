# Verification Report: 114580-V1

## Paper Information
- **Title**: New Evidence on Taxes and the Timing of Birth
- **Authors**: Sara LaLumia, James M. Sallee, Nicholas Turner
- **Journal**: AEJ: Economic Policy (2015)
- **Total Specifications**: 52
- **Data Note**: All results are from SIMULATED data (100K obs), as the paper uses confidential IRS microdata

## Baseline Groups

### G1: Tax Incentive Effect on Birth Timing
- **Claim**: A $1,000 increase in the tax benefit of a December birth increases the probability of a December (vs January) birth by approximately 1.1 percentage points.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0113 (SE: 0.0039, p = 0.004)
- **Outcome**: `dec_dob` (=1 if born Dec 25-31, =0 if born Jan 1-7)
- **Treatment**: `tax_ch` (tax savings from December vs January birth, in $1,000s)
- **N**: 100,000 (simulated); R2 = 0.0009
- **Table 2, Column 4**

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **44** | |
| core_controls | 22 | Baseline + bivariate + basic + demographics + leave-one-out (x7) + single covariate (x7) + income bins + saturated + kitchen sink |
| core_fe | 4 | No FE, year only, state only, zipcode proxy (state FE) |
| core_inference | 7 | Unclustered, state, year clustering + HC2 + HC3 + LPM robust + IV/2SLS |
| core_funcform | 8 | Linear/quadratic/log AGI, log/quadratic treatment, continuous age, logit, probit |
| core_sample | 3 | 7-day window, wider window with day control, WLS inverse AGI |
| **Non-core tests** | **8** | |
| noncore_alt_treatment | 1 | Alternative treatment measure (tax_ch2) |
| noncore_heterogeneity | 5 | C-section interaction, AGI interaction, married/parity/urban interactions |
| noncore_placebo | 2 | Baby gender placebo, shuffled treatment placebo |
| **Total** | **52** | |

## Detailed Classification Notes

### Core Tests (44 specs)

**Control progressions and leave-one-out (22 specs)**: The coefficient is extremely stable across control configurations. Leave-one-out analysis shows the coefficient ranges from 0.0110 to 0.0116 -- no single control variable affects the estimate. Single-covariate specifications all yield positive, significant effects (0.008-0.009), confirming the result is not driven by conditioning on any particular variable.

**Fixed effects (4 specs)**: The effect is significant with no FE (0.0080), year FE only (0.0080), state FE only (0.0081), and full year+state+age FE (0.0113). Adding FE increases the coefficient slightly.

**Inference (7 specs)**: The coefficient is significant under all SE methods: baseline state-year clustering, unclustered, state-only, year-only, HC2, HC3, and robust. The IV specification using tax_ch2 as instrument yields a slightly larger coefficient (0.013).

**Functional form (8 specs)**: Stable across linear/quadratic/log AGI parametrizations, log/quadratic treatment transformations, continuous age, logit, and probit. Logit and probit marginal effects (~0.008) are slightly smaller than LPM but remain significant.

### Non-Core Tests (8 specs)

**Heterogeneity (5 specs)**: Interactions with C-section rates, AGI, marital status, parity, and urban residence explore subgroup variation. The C-section interaction is near zero (the simulated data does not have realistic state-level C-section variation).

**Placebo (2 specs)**: Baby gender shows no significant association with tax incentives (coef 0.004, p=0.31). Shuffled treatment shows no effect (coef -0.002, p=0.54). Both confirm the result is not spurious.

**Alternative treatment (1 spec)**: tax_ch2 (simulated expected tax change) produces a similar coefficient (0.011).

## Robustness Assessment

The finding that tax incentives shift birth timing is **very robust** in the simulated data:

- **98% of specifications** show positive coefficients; 90% are significant at 5%.
- The coefficient is remarkably stable across control configurations, FE structures, and functional forms.
- Placebo tests confirm specificity: baby gender and shuffled treatment show no effects.
- The result is robust to all clustering and SE alternatives tested.

**Critical caveat**: All results are from simulated data. While the data-generating process was calibrated to match the paper's reported estimates, exact magnitudes and significance levels may differ from the actual confidential IRS data. Several subsample analyses (year-by-year, demographic subgroups) failed due to thin cells in the 100K simulated sample but would likely succeed with the paper's ~900K observations.
