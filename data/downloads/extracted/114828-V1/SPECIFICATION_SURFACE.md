# Specification Surface: 114828-V1

## Paper: Grosfeld, Rodnyansky & Zhuravskaya -- "Persistent Antimarket Culture: A Legacy of the Pale of Settlement" (AEJ: Economic Policy)

This paper studies the long-run cultural legacy of the Pale of Settlement using a geographic/spatial RD at the historical boundary. The core claim is that proximity to the Pale boundary produces discontinuities in pro-market attitudes, democratic preferences, entrepreneurship, and trust.

---

## Baseline Groups

### G1: Attitude discontinuities at the Pale boundary

- **Design**: Regression discontinuity (sharp, geographic/spatial)
- **Running variable**: Distance to the Pale of Settlement boundary (km, negative = inside Pale)
- **Cutoff**: 0 (the boundary itself)
- **Outcomes**: prefer_market, prefer_democracy, selfemp, trust_d (four main outcomes from Table 3)
- **Treatment**: Historical residence inside the Pale of Settlement
- **Estimand**: Discontinuity in attitudes/behavior at the boundary
- **Target population**: Urban respondents in Russia, Ukraine, and Latvia from the LITS 2006 survey, within 60km of the boundary

The paper presents both nonparametric RD (using Stata's `rd` command with bw=60) and parametric control-function approaches (Table 3). The nonparametric approach is treated as the primary baseline.

---

## Baseline Specs

Multiple outcomes share the same baseline group because the paper treats them as jointly characterizing the "antimarket culture" legacy:
- **prefer_market** (primary focal outcome for the baseline row)
- **prefer_democracy**, **selfemp**, **trust_d** (additional baseline spec IDs)

The parametric specification adds geographic controls, demographics, religion dummies, country dummies, elevation, and a linear control function in distance (separately inside/outside Pale, separately rural/urban).

---

## Core Universe

### Design variants
- Bandwidth: 30km, 90km, 120km (half, 1.5x, double of baseline 60km)
- Polynomial order: local linear, local quadratic
- Kernel: triangular, uniform
- Procedure: conventional, robust bias-corrected

### RC axes
- **Controls (parametric)**: LOO drops of individual covariates; control set progressions (none, geographic only, demographic only, full)
- **Sample/bandwidth**: Multiple bandwidth choices
- **Sample/restrict**: Country-specific subsamples (Russia, Ukraine, Latvia), urban vs rural
- **Sample/donut**: Exclude observations within 5km or 10km of boundary

### Excluded from core
- Table 5 (fuzzy RD instrumenting historical Jewish population share) -- this is an alternative estimand/mechanism, better as exploration
- Table 6 (IV for population movements) -- exploration of mechanisms
- Table 7 (pogroms distance) -- separate analysis, not the core RD claim
- Election data analysis (Table 2) -- separate dataset and analysis

---

## Constraints

- For nonparametric RD: no covariate controls (controls_count_min = 0)
- For parametric RD: up to 18 covariates (controls_count_max = 18)
- The geographic polynomial (control function) is part of the RD design, not a "control" in the usual sense

---

## Inference Plan

- **Canonical**: Cluster at PSU level (psu1), matching the baseline code
- **Variants**: HC1 robust (no clustering); country-level clustering (very few clusters, interpretive caveat)

---

## Budget

- Total core specs: up to 80
- Full enumeration is feasible given the moderate number of design + RC axes
