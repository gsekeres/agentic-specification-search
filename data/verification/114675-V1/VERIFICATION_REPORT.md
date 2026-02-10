# Verification Report: 114675-V1

## Paper Information
- **Title**: Mafia in the Ballot Box
- **Authors**: Giuseppe De Feo, Giacomo Davide De Luca
- **Journal**: AEJ: Economic Policy
- **Total Specifications**: 139

## Baseline Groups

### G1: DC Vote Share (Political Capture)
- **Claim**: Mafia presence increases Christian Democrat (DC) vote share in Sicilian municipalities via clientelistic political capture.
- **Baseline spec**: `baseline/iv/DC_VV/full`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.809 (SE: 0.269, p = 0.003)
- **Outcome**: `DC_VV` (DC vote share, Chamber of Deputies)
- **Treatment**: `Mafia1987_diff` (Mafia presence x time trend)
- **Instrument**: `Mafia1900_diff` (historical Mafia presence x time trend)
- **N**: 3,696; First-stage F = 61.9
- **Table 4, Column 6**

### G2: Construction Employment (Patronage Mechanism)
- **Claim**: Mafia presence increases construction employment share, consistent with patronage through public works allocation.
- **Baseline spec**: `baseline/iv/bui_labf/full`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.231 (SE: 0.111, p = 0.037)
- **Outcome**: `bui_labf` (construction workers as share of labor force)
- **Treatment**: `Mafia1987_diff`
- **N**: 3,696; First-stage F = 61.9
- **Table 5, Column 6**

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **95** | |
| core_controls | 36 | 2 baselines + progressive IV/OLS control additions (Tables 2-5) + drop-one-control variations + time-invariant interactions |
| core_sample | 38 | Neighbors-only, post-1960, leave-one-year-out (x24), pre-1987 |
| core_inference | 16 | Overidentification with sulfur instrument (x12) + robust HC SEs (x4) |
| core_funcform | 4 | Winsorized outcomes at 1/99 and 5/95 percentiles for both outcomes |
| **Non-core tests** | **44** | |
| noncore_alt_treatment | 12 | Mafia1994, Mafia x communism, post-1970 interaction -- all for both outcomes |
| noncore_alt_outcome | 19 | Senate DC vote, left/right vote shares, turnout, industry/trade/banking/public employment |
| noncore_heterogeneity | 8 | Decade-specific interactions (1960s/70s/80s x 2 outcomes), large/small pop subsamples |
| noncore_placebo | 4 | MSI-PRI falsification for both outcomes (OLS + IV) |
| noncore_diagnostic | 2 | Mafia=1 only subsamples (no cross-sectional variation in treatment) |
| **Total** | **139** | |

## Detailed Classification Notes

### Core Tests (95 specs)

**Control progressions (36 specs)**: The paper systematically builds up controls from year FE only to full (investment + socio-demographic + economic + geographic trends + church trends) for both IV and OLS, and for both outcomes. Drop-one-control-set variations (5 per outcome) test sensitivity to each control block. The coefficient is remarkably stable: IV DC_VV ranges from 0.45 (no controls) to 0.81 (full controls), monotonically increasing as confounders are absorbed.

**Sample restrictions (38 specs)**: The largest core category. Leave-one-year-out analysis (24 specs for DC_VV, both OLS and IV) shows no single election year drives the result. IV coefficients range from 0.68 to 1.00 across years. Post-1960 sample (6 specs) and pre-1987 sample (2 specs) confirm temporal stability. Neighbors-only (4 specs) restricts to municipalities adjacent to Mafia municipalities, providing a geographic discontinuity-like test.

**Inference (16 specs)**: Overidentification with sulfur production as a second instrument (12 specs) produces nearly identical coefficients to the just-identified baseline, supporting instrument validity. Robust HC SEs (4 specs) yield much smaller SEs than clustered SEs, with the IV coefficient remaining highly significant (p < 0.0001).

**Functional form (4 specs)**: Winsorization at 1/99 and 5/95 percentiles for both outcomes. Coefficients barely change (0.80 vs 0.77 for DC_VV), confirming outliers are not driving results.

### Non-Core Tests (44 specs)

**Alternative treatments (12 specs)**: Mafia1994 measure produces similar but larger IV coefficients (1.01 for DC_VV, 0.29 for bui_labf). Mafia x communism interactions test whether the effect strengthens where left-wing parties are stronger. Post-1970 interactions test temporal specificity.

**Alternative outcomes (19 specs)**: Senate DC vote share (DC_VVS), left/right vote shares, turnout, and sectoral employment (industry, trade, banking, public). Left vote share shows a negative Mafia effect (IV: -0.53, p=0.019), suggesting Mafia shifts votes from left to DC. Right-wing, turnout, trade, and banking show no significant effects. Public employment shows a marginally significant positive effect (IV: 0.12, p=0.033).

**Heterogeneity (8 specs)**: Decade-specific interactions show the effect strengthens over time (insignificant in 1960s, growing through 1970s-1980s). Population subsamples show the effect is stronger in large municipalities but the instrument is weak in small municipalities (F=8.0).

**Placebo (4 specs)**: MSI-PRI (right-wing) vote differentials show no significant Mafia effect in any specification, confirming that Mafia influence is specific to DC.

**Diagnostics (2 specs)**: Mafia=1 only subsamples have no cross-sectional variation in the Mafia indicator, so the treatment variation comes only from the time trend interaction within Mafia municipalities. The negative coefficients (DC_VV: -3.71, bui_labf: -4.66) with very large SEs reflect near-collinearity.

## Robustness Assessment

The paper's main finding is **very robust**:

- **G1 (DC vote share)**: The IV coefficient is positive and significant across virtually all core specifications. Progressive addition of controls increases the coefficient (from 0.45 to 0.81), suggesting OLS is attenuated. Leave-one-year-out analysis shows remarkable stability (0.68 to 1.00). The overidentification test with sulfur produces nearly identical results.

- **G2 (Construction employment)**: Also consistently positive and significant, though with somewhat wider confidence intervals. The coefficient is stable across control additions (0.23 to 0.30).

- **First stage is strong**: F-statistics consistently around 62, well above weak instrument thresholds. Only the small-municipality subsample has a weak first stage (F=8.0).

- **Key sensitivity**: The OLS baseline with investment controls shows anomalous behavior (negative R2 of -1.0, negative coefficient), suggesting a problematic demeaning or multicollinearity issue in that particular specification. All other OLS and all IV specifications are well-behaved.
