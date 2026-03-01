# Specification Surface: 131981-V1

**Paper**: "Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey"
**Authors**: Altindag, Erten, and Keskin
**Journal**: American Economic Journal: Applied Economics

---

## Paper Summary

This paper uses a sharp regression discontinuity (RD) design to estimate the mental health effects of COVID-19 age-based curfews in Turkey. People born before December 1955 (aged 65+) were subject to strict stay-at-home orders during 2020, while those just below the threshold were not. The running variable is the birth month distance from the December 1955 cutoff. Outcomes are mental health indices constructed from the SRQ-20 questionnaire.

## Design

- **Design**: Sharp regression discontinuity
- **Running variable**: `dif` (birth month distance from December 1955 cutoff, in months)
- **Cutoff**: 0 (born before Dec 1955 => age >= 65 => subject to curfew)
- **Treatment**: `before1955` (1 = born before Dec 1955)
- **Polynomial**: Local linear (order 1), with quadratic as robustness
- **Kernel**: Uniform (all observations within bandwidth equally weighted)
- **Bandwidths**: 17, 24, 30, 36, 45, 48, 60, 72 months from cutoff

## Baseline Groups

### G1: Mental Health Effect of Curfew

- **Outcome**: `z_depression` (mental distress index, inverse-covariance-weighted average of 20 SRQ items, normalized to control-group SD)
- **Treatment**: `before1955` (born before Dec 1955 = subject to curfew)
- **Estimand**: Sharp RD treatment effect at age-65 curfew cutoff
- **Population**: Turkish adults near the age-65 curfew threshold

The paper's canonical specification (Table 4, Column 2) uses:
- Bandwidth = 30 months
- Local linear polynomial (separate slopes each side)
- Full controls: month FE, province FE, ethnicity FE, education FE, female, survey_taker_id FE
- SE clustered at survey month-year (modate)

### Additional Baseline Specs

The paper reports the same specification at bandwidths 17, 45, and 60 (Table 4, Columns 1, 3, 4). These are included as additional baseline rows.

## Core Universe

### Design Variants (`design/*`)

1. **Bandwidth variations**: 17, 24, 36, 45, 48, 60, 72 months -- the paper itself explores several of these across Tables 4 and A4
2. **Polynomial order**: Local quadratic (Table A8 uses quadratic at bw=45)

### Robustness Checks (`rc/*`)

1. **Controls leave-one-out (LOO by block)**: Drop each FE block one at a time:
   - Drop month FE
   - Drop province FE
   - Drop ethnicity FE
   - Drop education FE
   - Drop female
   - Drop survey_taker_id FE
2. **Control sets**:
   - No controls (RD polynomial only)
   - Minimal demographics (female + education + ethnicity only)
   - Full baseline controls
3. **Donut holes**: Exclude observations within 1, 2, or 3 months of cutoff
4. **Alternative outcomes** (preserving the mental health claim object):
   - `sum_srq` (sum of SRQ-20 yes answers)
   - `z_somatic` (somatic symptoms index)
   - `z_nonsomatic` (nonsomatic symptoms index)
5. **Joint specs**: Bandwidth x polynomial, bandwidth x controls, outcome x bandwidth

### Inference Plan

- **Canonical**: Clustering at `modate` (survey month-year), matching the paper
- **Variants**: HC1 (no clustering), clustering at province level

## Budget and Sampling

- **Target**: ~60-75 specifications total
- **Approach**: Full enumeration (no random sampling needed)
- Controls are block-structured (FE groups), so LOO at the block level yields a small, tractable set

## What Is Excluded (and Why)

- **Mobility outcomes** (Table 3: outside_week, under_curfew, never_out): These are mechanisms/first-stage outcomes, not the main claim
- **Channel outcomes** (Table 5: employment, income, social isolation): These are secondary/mechanism analysis
- **Political outcomes** (Table 6: curfew_support, gov_support): These are secondary outcomes
- **Religiosity** (Table A7): Secondary outcome
- **Individual SRQ items**: The paper's claim is about the index, not individual items
- **Sensitivity / Exploration**: Not part of the core surface

## Constraints

- Controls-count envelope: min=2 (polynomial terms only), max=~92 (all FE dummies)
- Controls move as blocks (FE groups), not individual dummies
- Linked adjustment: not applicable (single-equation estimator)
