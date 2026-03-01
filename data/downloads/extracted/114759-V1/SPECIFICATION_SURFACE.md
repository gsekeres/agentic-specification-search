# Specification Surface: 114759-V1

## Paper: Camacho & Conover -- "Manipulation of Social Program Eligibility" (AEJ: Economic Policy)

This paper studies manipulation of the SISBEN poverty targeting system in Colombia. The core RD evidence documents a discontinuity in the density of poverty scores at the eligibility cutoff (score = 47), indicating strategic manipulation by local officials.

---

## Baseline Groups

### G1: Density discontinuity at the SISBEN cutoff

- **Design**: Regression discontinuity (sharp)
- **Running variable**: SISBEN poverty score (puntaje)
- **Cutoff**: 47
- **Outcome**: Share of surveys at each score value (sisses), measuring the density of the poverty index distribution
- **Treatment**: Eligibility for welfare programs (score <= 47)
- **Estimand**: Discontinuity in score density at cutoff -- evidence of manipulation/bunching
- **Target population**: Urban Colombian households in socioeconomic strata 1-3, surveyed 1994-2003

The paper's Table 3 is the core RD result. It estimates the density jump at the cutoff using local polynomial regression with data-driven (Imbens-Kalyanaraman) bandwidths, year by year and pooled. The estimation uses triangular kernel weighting and local linear regression on collapsed score-level data.

Note: Tables 6-7 use panel fixed-effects regressions of the "jump" variable on lagged electoral tightness at the municipality level. These are downstream analyses examining *correlates* of manipulation intensity, not the core RD evidence. They are better classified as `explore/*` (correlate analysis) rather than a separate baseline group, since they study a different question (why manipulation happens) rather than whether it happens.

---

## Baseline Specs

- **Table3-Nonparametric-Pooled**: Local linear RD with triangular kernel and data-driven bandwidth. Outcome is the collapsed score density. Controls are only the running variable polynomial terms (align, jump_align). Robust SEs.

---

## Core Universe

### Design variants
- Bandwidth: half baseline, double baseline, plus 50%, 75%, 125%, 150%, 200% of baseline
- Polynomial order: local linear (p=1), local quadratic (p=2), local cubic (p=3)
- Kernel: triangular, uniform, Epanechnikov
- Procedure: conventional, robust bias-corrected (Calonico et al.)

### RC axes
- **Sample/bandwidth**: Various bandwidth multiples
- **Sample/donut**: Exclude observations within 1 or 2 points of cutoff
- **Sample/restrict**: Subsets by SES stratum, pre/post 1998 (when manipulation intensified)
- **Data construction**: Floor vs round score aggregation
- **Placebo cutoffs**: Test at non-eligibility scores (40, 50, 52)

### Excluded from core
- Tables 6-7 (municipality-level panel FE regressions of jump on electoral tightness) are exploration/correlate analyses, not the core RD claim
- Tables 4-5 (score reconstruction, cheating detection) are descriptive/diagnostic

---

## Constraints

- No covariate controls beyond the running variable polynomial -- the RD is on collapsed score-level density data
- Control-count envelope: min=0, max=2 (polynomial terms only)
- Bandwidth and polynomial order are the primary design axes

---

## Inference Plan

- **Canonical**: HC1 robust SEs (matching baseline Stata code)
- **Variant**: Cluster at score level (common in RD with discrete running variable)

---

## Budget

- Total core specs: up to 60
- No controls-subset sampling needed (controls are only polynomial terms)
- Full enumeration of design variants is feasible
