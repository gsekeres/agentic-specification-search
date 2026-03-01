# Specification Surface: 150323-V1

## Paper Overview

"Political Turnover, Bureaucratic Turnover and the Quality of Public Services" by Akhtari, Moreira, and Trucco (AER). This paper uses a sharp regression discontinuity design exploiting close municipal elections in Brazil (2008 and 2012 cycles stacked) to estimate the effect of political party turnover on bureaucratic personnel replacement, school personnel changes, and student test scores. The running variable is the incumbent party vote margin (pX) with a cutoff at zero. Treatment is political turnover (pX_dummy = 1 when the incumbent party loses).

## Baseline Groups

### G1: Political Turnover and Test Scores (Main Claim)

**Claim object**: Political turnover lowers 4th grade test scores by 0.05-0.08 standard deviations in municipal schools one year after the election.

**Baseline spec**: Table 3, Panel A, Column 1 -- sharp RD of standardized combined test scores (both_score_indiv_4_stdComb) on political turnover dummy, with linear polynomial in the running variable (pX, pX_pD = pX * pX_dummy) and baseline test score control, within the data-driven optimal bandwidth (rdbwselect with uniform kernel), clustered at municipality.

**Why this is the main claim**: Page 4 of the paper states "Party turnover lowers test scores...by .05-.08 standard deviation units" -- this maps directly to Table 3 Panel A (4th grade) and Panel B (8th grade).

### G2: Political Turnover and Municipal Personnel Replacement

**Claim object**: Political turnover increases the share of municipal personnel that is new by ~7 percentage points (23% of the mean).

**Baseline spec**: Table 2, Column 1 -- RD of new personnel share (SHhired_Mun_lead) on political turnover, with year dummy, at optimal bandwidth, municipality-clustered SEs.

**Why this is a separate baseline group**: This is the first-stage mechanism documented in Figure 3 and Table 2. The paper explicitly frames this as a standalone finding (page 3).

### G3: Political Turnover and School Personnel (Headmaster/Teacher Replacement)

**Claim object**: Political turnover increases headmaster replacement by ~28 percentage points and teacher replacement by ~11 percentage points in municipal schools.

**Baseline spec**: Table 4, Column 1 (headmaster) -- RD of headmaster-in-school-less-than-2-years on political turnover, no controls, optimal bandwidth.

**Why this is a separate group**: Tables 4-5 report school-level personnel changes as a distinct claim from municipal-level RAIS outcomes.

## What Is Included and Why

### RD-specific design variations (all groups):

1. **Bandwidth**: Optimal (rdbwselect), fixed at 0.07, fixed at 0.11, half-optimal, double-optimal. The paper reports results at optimal, 0.07, and 0.11 for every table.
2. **Polynomial order**: Linear (baseline), quadratic, cubic.
3. **Kernel**: Uniform (baseline), triangular.

### Control variations (G1 primarily):

1. **No controls** vs. **baseline score** vs. **school controls** vs. **student controls** vs. **all controls**. The paper reports no-controls and with-controls columns for each bandwidth.
2. **School controls** ($controls_sch): urban_schl, waterpblcnetwork_schl, sewerpblcnetwork_schl, trashcollect_schl, eqpinternet_schl, eqpinternet_schl_miss
3. **Student controls** ($controls_stud): female_SPB, female_SPB_miss, white_SPB, white_SPB_miss, mom_reads_SPB, mom_reads_SPB_miss
4. **Year dummy**: Included in some specifications.
5. **LOO baseline score**: Drop the lagged test score control.

### Sample variations (all groups):

1. **Election cycle**: 2008 only, 2012 only, both stacked (baseline).
2. **School type**: Municipal (baseline), non-municipal (Table 6 serves as placebo/comparison).
3. **Urban/rural**: Urban schools only, rural schools only.
4. **Donut hole**: Exclude observations within 1% or 2% of the cutoff.

### Outcome variations (G1):

1. **Grade**: 4th grade (baseline), 8th grade.
2. **Subject**: Combined (baseline), math only, Portuguese only.
3. **Standardization**: Standardized (baseline), unstandardized.

### Outcome variations (G2):

1. **Direction**: New hires (baseline), personnel who left, net change.
2. **Timing**: Year of election +1 (lead, baseline), +2 (after).

### Outcome variations (G3):

1. **Personnel type**: Headmaster replacement (baseline), new teacher share, left teacher share.
2. **School type**: Municipal (baseline), non-municipal.

## What Is Excluded and Why

- **Fuzzy RD / IV**: The paper uses sharp RD throughout. Fuzzy RD (via ivreg2) appears only in appendix Tables A1-A2 and is not part of the main claims.
- **Bias-corrected inference (Calonico et al. robust)**: The paper uses conventional OLS within bandwidth. Bias-corrected inference could be added as an inference variant but is not part of the paper's revealed search space.
- **Controls for G2**: The municipality-level personnel regressions include only the RD polynomial and year dummy. Adding school-level controls would be incoherent for this outcome.
- **Sensitivity/exploration**: Not in core universe.
- **DML/matching**: Not applicable to RD design.

## Inference Plan

- **Canonical**: Cluster at municipality level (COD_MUNICIPIO), matching the paper.
- **Variants (G1 only)**: HC1 robust (no clustering) and coarser state-level clustering.

## Budgets and Sampling

- **G1 (test scores)**: ~60 specifications from bandwidth x polynomial x controls x grade x election cycle x school type combinations.
- **G2 (municipal personnel)**: ~30 specifications from bandwidth x outcome direction x timing x election cycle.
- **G3 (school personnel)**: ~25 specifications from bandwidth x controls x outcome type x election cycle x school type.
- **Combined target**: ~115 specifications across all groups (well above 50 minimum).
- No random control-subset sampling needed -- the control space is small and structured.

## Key Notes

- The RD regressions use the form: `reg Y pX_dummy pX pX_pD [controls] if abs(pX)<=bandwidth, cluster(COD_MUNICIPIO)`. Here pX_pD = pX * pX_dummy (slope interaction above cutoff), so the RD is a piecewise-linear regression.
- Bandwidth is determined by `rdbwselect` with uniform kernel (Calonico-Cattaneo-Farrell-Titiunik method).
- The paper stacks two election cycles (2008 and 2012) and includes a year dummy. Running the cycles separately is an important robustness check.
- Sample restrictions: supplementary elections are dropped, and large municipalities (with runoff) are excluded.
- The running variable axis is reversed in figures (negative margin = incumbent won), but the regression specification is standard.
