# Specification Surface: 128521-V1

**Paper**: "Recessions, Mortality, and Migration Bias: Evidence from the Lancashire Cotton Famine" (Beach, Hanlon)
**Design**: Difference-in-differences (TWFE with district and period FE)

---

## Overview

This paper estimates the mortality impact of the 1861-1865 Lancashire Cotton Famine using a difference-in-differences design. The treatment group is cotton-dependent registration districts; the control group is non-cotton districts. The key innovation is the use of individually linked death records (GRO death index linked back to census records), which allows deaths to be assigned to the district of enumeration rather than the district of death, addressing migration bias.

The data structure is a district-by-period panel with two periods: 1851-1855 (pre) and 1861-1865 (post). All regressions use `reghdfe` with district and period fixed effects, analytical (population) weights, and clustered standard errors.

---

## Baseline Groups

### G1: Total Mortality (Table 2)

**Claim object**: The Cotton Famine increased mortality in cotton-dependent districts relative to non-cotton districts. The main estimate is from standardized total mortality rates using linked death records assigned to district of enumeration.

**Baseline specs** (Table 2, Columns 1-4):
- Col 1: Minimal specification -- `census_mr_tot ~ cotton_dist_post | district + period`, pop weights, cluster(district)
- Col 2: Add log population density, linkable share, age shares, region x period FE
- Col 3: Add nearby district spillover rings (0-25km, 25-50km, 50-75km treatment intensities)
- Col 4: Use continuous cotton employment share (`cotton_eshr_post`) instead of binary treatment

**Key features**:
- Population weights (`aweight=pop_census_tot`) used throughout
- Linked death rate is scaled by lambda (ratio of linked to aggregate deaths) to make rates comparable
- 538 registration districts, 2 time periods

### G2: Age-Specific Mortality (Table 3)

**Claim object**: The mortality effect of the Cotton Famine varied by age group, with the largest effects concentrated among specific age categories.

**Baseline specs** (Table 3): Seven parallel DD regressions, one for each age group (under-15, 15-24, 25-34, 35-44, 45-54, 55-64, over-64), using the preferred specification from Table 2 Col 3 with age-group-specific population weights and linkable shares.

---

## What is Included (Core Universe)

### RC axes for G1:

**Controls (leave-one-out)**:
- Drop log population density
- Drop linkable share
- Drop age shares (under-15, elderly)
- Drop region x period FE
- Drop nearby district rings

**Sample**:
- Exclude Greater Manchester (largest cotton district)
- Exclude Manchester + Liverpool + Leeds (all large urban areas)

**Weights**:
- Unweighted regression (no population weights)

**Data construction**:
- No foreign-born links (AppendixD1, Table 8 Col 5)
- Six linking restriction variants (AppendixD1, Table 9): unique last name; unique first+last; distance < 200/100/50; exact match

### RC axes for G2:
- Exclude Manchester
- Unweighted

---

## What is Excluded (and Why)

- **Table 4 (Occupational exposure)**: Changes the unit of analysis to district x occupation, adds industry FE, and uses a different treatment variable (`cotdistXshock` or `cotton_ind_post`). This changes both the estimand and the population decomposition. Classified as `explore/*`.
- **Table 5 (Aggregate vs linked comparison)**: Uses three different death assignment methods (enumeration, death, registrar). The comparison across methods is a data construction diagnostic, not a treatment effect estimate. Classified as `explore/*`.
- **AppendixD3 (BMD-based results)**: Uses an entirely different death record linkage source (FreeBMD vs GRO). This is a data construction variant. The placebo test (Table 12, 1856-60 vs 1851-55) and harvesting test (Table 13, 1866-70 vs 1856-60) are diagnostics.
- **AppendixD4 (Fertility and infant mortality)**: Different outcome (births, infant mortality rates). Classified as `explore/*`.
- **Figures 1-4**: Descriptive visualizations (maps, time series, population trends).

---

## Inference Plan

**Canonical**: Cluster at district (`master_name`) level, matching the paper's baseline.

**Variants**:
- County-level clustering (Table 2 county panel): coarser, more conservative
- Robust SEs (no clustering): used as input for the permutation test
- Permutation test: reassigns cotton district treatment to each of 538 districts based on spatial distance structure, preserving the geographic concentration of treatment. P-values reported separately.

---

## Budgets and Sampling

| Group | Max Core Specs | Sampling | Rationale |
|---|---|---|---|
| G1 | 80 | Full enumeration | 4 baselines + 1 design + 5 LOO controls + 2 sample + 1 weights + 7 data construction = ~20 specs, well within budget |
| G2 | 30 | Full enumeration | 7 age groups x (1 baseline + 2 RC) = 21 specs |

Full enumeration is feasible because the control set is small (5 conceptual blocks, not individual variables) and the revealed robustness space is compact.

---

## Key Linkage Constraints

1. **Population weights and death rates are mechanically linked**: The mortality rate denominator uses the same population variable as the analytical weight. Changing weights changes the outcome construction.
2. **Lambda scaling**: The linked death rates are inflated by a lambda factor (linked/aggregate death count ratio). This scaling is age-group-specific. Data construction variants (linking restrictions) change the lambda and therefore the outcome level.
3. **Nearby rings are part of treatment specification**: The nearby_post_25/50/75 variables capture spillover effects and are conceptually part of the treatment model, not optional controls.
4. **Region x period FE**: These are generated via `xi i.region*post` and expand to multiple interaction dummies. They should be treated as a single conceptual block for LOO purposes.

---

## Diagnostics Plan

- **Population trends** (Figure 4): Visual check that cotton and non-cotton districts had parallel population trajectories before the famine.
- **Placebo test** (Table 12, AppendixD3): BMD-based DD using 1856-1860 vs 1851-1855 (both pre-famine periods). Should show null effects.
