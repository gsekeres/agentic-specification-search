# Verification Report: 114728-V1

## Paper Information
- **Title**: Is the EITC as Good as an NIT? Conditional Cash Transfers and Tax Incidence
- **Author**: Jesse Rothstein
- **Journal**: AEJ: Economic Policy (2010)
- **Total Specifications**: 388
- **Method**: Structural calibration (not reduced-form regression)

## Baseline Groups

### G1: EITC Leakage (totxfer < 1)
- **Claim**: The EITC delivers substantially less than $1 per dollar of government spending due to wage depression effects.
- **Baseline spec**: `baseline`
- **Expected direction**: totxfer < 1 (leakage)
- **Baseline value**: 0.6438 (36% leakage)
- **Parameters**: sigma_e=0.75, sigma_i=0.0, rho=-0.3, LM1m
- **N**: 51,692 (March 1993 CPS)

### G2: NIT Amplification (totxfer > 1)
- **Claim**: The NIT delivers more than $1 per dollar due to wage increases from reduced labor supply.
- **Baseline spec**: `baseline/nit/All/totxfer`
- **Expected direction**: totxfer > 1 (amplification)
- **Baseline value**: 1.5484 (55% amplification)
- **Parameters**: Same as G1

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **222** | |
| core_controls | 2 | 2 baselines (EITC All totxfer, NIT All totxfer) |
| core_funcform | 204 | Parameter grid (64), rho sensitivity (20), sigma_e sensitivity (20), sigma_i sensitivity (40), LM models (48), cross-model x rho (24), parameter bounds (28), perfect competition (10) |
| core_sample | 16 | Workers only (4), families with kids (6), unmarried only (6) |
| **Non-core tests** | **166** | |
| noncore_alt_outcome | 74 | Baseline non-totxfer outcomes (totinc, totearn, wage, LS, partic x subgroups), hours margin (6), total income at different rho (16) |
| noncore_heterogeneity | 92 | Baseline subgroup decompositions (totxfer by singmom/marrmom/etc.), education heterogeneity (20), marital/kids x rho (16) |
| **Total** | **388** | |

## Detailed Classification Notes

### Core Tests (222 specs)

**Parameter sensitivity (204 specs)**: This is by far the largest category and constitutes the paper's primary robustness exercise. The 3x3x4 parameter grid systematically varies sigma_e (0.5, 0.75, 1.0), sigma_i (0.0, 0.25, 0.5), and rho (0, -0.3, -1.0, -inf). Additional sensitivity analyses examine finer rho/sigma_e/sigma_i grids. These are classified as core_funcform because in a structural model, parameter choice is analogous to functional form choice in reduced-form work.

**Labor market model variations (48 + 24 specs)**: The 6 alternative LM cell definitions (LM1-LM3, with/without marital status) test whether the skill-group classification affects results. Cross-model x rho tests these at extreme rho values.

**Sample restrictions (16 specs)**: Workers only, families with kids only, and unmarried women only.

### Non-Core Tests (166 specs)

**Alternative outcomes (74 specs)**: The baseline decomposition reports totinc, totearn, wage effects, labor supply, and participation changes for each subgroup. Hours margin and total income at different rho values provide complementary perspectives.

**Heterogeneity (92 specs)**: Subgroup decompositions (single mothers, married mothers, etc.) and education-specific results (Table 8) explore how the distributional effects vary across demographic groups.

## Robustness Assessment

The paper's central claims are **very robust**:

- **EITC totxfer < 1** in 82% of All-group specifications. The only exceptions occur at extreme parameter combinations (very high sigma_e and sigma_i with rho near 0).
- **NIT totxfer > 1** in 99% of All-group specifications.
- **NIT > EITC ranking** holds in 100% of comparable specification pairs -- this is the paper's most robust result.
- Results are stable across all 6 labor market cell definitions.
- The key sensitivity is to rho: as rho approaches 0 (very elastic demand), wage effects shrink and both programs approach $1 per dollar. As rho becomes more negative, the EITC-NIT gap widens.
- Replication matches Stata output to 5+ decimal places.

**Note**: This is a structural calibration paper with no statistical uncertainty (no SEs or p-values). Robustness is assessed through parameter sensitivity rather than statistical tests.
