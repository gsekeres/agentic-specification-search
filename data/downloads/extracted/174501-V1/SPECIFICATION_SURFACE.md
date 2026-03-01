# Specification Surface: 174501-V1

**Paper**: "Interaction, Stereotypes and Performance: Evidence from South Africa" (Corno, La Ferrara, Burns)

**Design**: Randomized experiment (natural experiment exploiting random roommate assignment at University of Cape Town)

---

## Overview

This paper exploits a policy at the University of Cape Town that randomly assigns roommates within residence halls to study whether inter-racial interaction affects stereotypes, attitudes, and academic performance. The treatment variable `mixracebas` is a binary indicator for being assigned to a mixed-race room at baseline.

The paper has three main families of claims (three baseline groups):
1. **G1**: Effect on racial stereotypes (IAT scores)
2. **G2**: Effect on academic performance (GPA, exams passed, dropout)
3. **G3**: Effect on social outcomes (friendships, attitudes, communication, pro-social behavior)

Each claim is estimated separately for White students, Black students, and the full sample.

---

## Baseline Groups

### G1: Stereotypes (Race IAT)

**Claim object**: Inter-racial roommate exposure reduces White students' negative racial stereotypes (as measured by the Race IAT D-score) and has no significant effect on Black students' stereotypes.

**Baseline specs** (Table 3):
- White students: `DscoreraceIAT ~ mixracebas + L.DscoreraceIAT + controls_subsample + rocontrols + i.Res_base` (ANCOVA with lagged outcome)
- Black students: Same specification on Black subsample

**Why ANCOVA**: Table 3 includes the lagged IAT score (`L.DscoreraceIAT`) as a control, making this an ANCOVA-type estimator rather than a simple difference-in-means.

**Controls structure**:
- Own controls (`controls_subsample`, 11 vars): Female, Falseuct2012, missfalse, Foreign, foreign_missing, privateschool_nomiss, privateschool_miss, durpcabas_nomiss, durpcabas_miss, consbas_nomiss, consbas_miss
- Roommate controls (`rocontrols`, 10 vars): roFalseuct2012, missrofalse, roForeign_bas, roforeign_missingbas, roprivschool_nomiss, roprivschool_miss, rodurpcabas_nomiss, rodurpcabas_miss, roconsbas_nomiss, roconsbas_miss
- Fixed effects: Residence building (`i.Res_base`)

**Revealed robustness**: Table A7 shows the same IAT regressions without roommate controls. Table A6 shows placebo regressions (baseline IAT as outcome). Table A8 decomposes treatment by race of roommate.

### G2: Academic Performance

**Claim object**: Inter-racial roommate exposure improves Black students' GPA, number of exams passed, and continuation (lower dropout), with no significant effect on White students.

**Baseline specs** (Table 4):
- Black students: `GPA ~ mixracebas + controls_subsample + rocontrols + i.Res_base + i.regprogram` (no lagged outcome available for academic outcomes)
- Same for `examspassed`, `continue`, `PCAperf`
- White students and full sample as secondary

**Additional FE**: Academic program FE (`i.regprogram`) is included for academic outcomes but not for IAT/social outcomes.

**Revealed robustness**: Table A9 (no roommate controls), Table A10 (second-year outcomes: GPA2013, examspassed2013, continue2013, PCAperf2013), Table A11 (channels: interaction with samefaculty/samecourse).

### G3: Social and Attitudinal Outcomes

**Claim object**: Inter-racial roommate exposure increases cross-racial friendships and improves racial attitudes, especially for White students.

**Baseline specs** (Table 5):
- White students: `PCAfriend ~ mixracebas + controls_subsample + rocontrols + i.Res_base`
- Same for `PCAattitude`, `PCAcomm`, `PCAsocial`
- Black students and full sample as secondary

**Revealed robustness**: Table A15 (PCA indices constructed without missing values: PCAfriend_nomiss, PCAatt_nomiss, PCAcomm_nomiss, PCAsocial_nomiss).

---

## Design and Identification

**Randomization**: University policy randomly assigns roommates within residence buildings. The key identifying assumption is that conditional on residence FE (and race in the full sample), room composition is as-good-as-random.

**Strata**: Residence building (`Res_base`) is the blocking variable. In full-sample specifications, race x residence interactions (`blackRes`, `whiteRes`) are also included.

**Clustering**: Standard errors are clustered at the room level (`roomnum_base`), which is the randomization unit.

---

## Core Universe (what will be run)

### Design variants
- **Difference-in-means**: Treatment effect without covariate adjustment
- **ANCOVA**: With lagged outcome (G1 only, since lagged IAT is available)
- **With covariates**: Own controls only (no roommate controls)
- **Strata FE**: Residence FE only (the paper's minimum specification)

### RC axes

**Controls**:
- Drop all roommate controls (revealed in Tables A7, A9)
- Leave-one-out (LOO) over each own-control variable group (6 groups, counted as pairs with their missingness indicators)
- LOO over each roommate-control variable group (5 groups)

**Sample**:
- Full sample with race x residence FE (revealed in Tables 4-6 column 3)
- Restrict to White-Black roommate pairs only, excluding coloured/other (Table A8)
- Race subsample variants (White only, Black only)

**Functional form**:
- Academic IAT instead of Race IAT (Table 3 cols 3-4, for G1)
- PCA performance index vs individual academic outcomes (for G2)
- No-missing-values PCA indices (Table A15, for G3)
- Second-year academic outcomes (Table A10, for G2)

**Fixed effects**:
- Add/remove program FE (regprogram) for academic outcomes

---

## What is Excluded (and Why)

- **Table A3 (IAT correlates)**: These are observational correlations between IAT and behavior, not experimental estimates. They belong in `explore/*` or `diag/*`.
- **Table A5 (simulation)**: Simulation exercise about roommate assignment mechanism, not an experimental estimate.
- **Table A8 (race-specific roommate effects)**: This decomposes treatment by roommate race group (roblabas, rocolothbas). This changes the treatment concept and belongs in `explore/*`.
- **Tables A12-A14 (detailed sub-outcomes)**: Individual friendship/attitude/prosocial items and ordered logit models. These explore sub-components of PCA indices and use different estimators (ologit for ordinal outcomes). They are secondary explorations.
- **Table 6 (residential choices)**: Different outcome concept (whether student stays in residence, chooses mixed room in year 2). Could be considered a separate baseline group but is more of a mechanism/downstream outcome. Excluded from core.
- **FWER p-values**: Post-processing for multiple testing correction, not new estimates.

---

## Inference Plan

**Canonical**: Cluster-robust SEs at the room level (`roomnum_base`), matching the paper's approach throughout.

**Variants**:
- HC1 robust SEs (no clustering) -- to check sensitivity to clustering choice
- Cluster at residence level (`Res_base`) -- coarser clustering as stress test

---

## Budgets and Sampling

| Baseline Group | Max Core Specs | Max Control Subset | Sampling |
|---|---|---|---|
| G1 (Stereotypes) | 60 | 30 | Full enumeration |
| G2 (Academic) | 80 | 40 | Full enumeration |
| G3 (Social) | 60 | 30 | Full enumeration |

Full enumeration is feasible because:
- LOO has ~11 variants per race subsample
- Major revealed variants (drop roommate controls, full sample, alternative outcomes) are small in number
- The combinatorial explosion is modest given the paper's revealed search space

---

## Diagnostics Plan

- **Balance tests** (Table 1, Table 2): Check that baseline characteristics are balanced across mixed vs non-mixed rooms, within residence strata.
- **Attrition analysis** (Table A1): Check whether attrition is differential by treatment status and whether it correlates with baseline IAT.
- **Placebo regressions** (Table A6): Use baseline IAT as outcome (should show null effect).

---

## Key Linkage Constraints

1. **Missingness indicators are paired**: Variables like `privateschool_nomiss` and `privateschool_miss` must be dropped/added together. Same for all `_nomiss`/`_miss` pairs.
2. **Race subsample determines control set**: Full-sample specs use `controls` (with race dummies white, coloured, Else) plus race x residence FE. Subsample specs use `controls_subsample` (without race dummies).
3. **Lagged outcome is mandatory for IAT ANCOVA** (G1): `L.DscoreraceIAT` should not be dropped in ANCOVA specs.
4. **Program FE is mandatory for academic outcomes** (G2): `i.regprogram` is included in all Table 4 specifications.
