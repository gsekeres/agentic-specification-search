# Specification Surface Review: 136741-V1

**Paper**: "Historical Lynchings and the Contemporary Voting Behavior of Blacks" (Williams)
**Design**: Cross-sectional OLS with state fixed effects
**Review date**: 2026-02-25

---

## Summary of Baseline Groups

### G1: Effect of Historical Black Lynching Rate on Contemporary Black Voter Registration
- **Claim**: Higher historical black lynching rates (1882-1930) are associated with lower contemporary black voter registration rates (2000-2012) in Southern US counties.
- **Baseline spec**: Table 2 Col 1 / Table 3 Col 1: `regress Blackrate_regvoters lynchcapitamob + 7 historical controls + i.State_FIPS`. Coefficient = -0.469 (SE = 0.144).
- **Design**: Cross-sectional OLS. N=267 counties in 6 Southern states (AL, FL, GA, LA, NC, SC). Homoskedastic SEs (no robust/cluster option in main tables). State FE absorbed.
- **Data**: Panel data (multiple election years per county) collapsed to county-level cross-section via Stata `collapse` before main analysis.

---

## Changes Made

### 1. Added falsification diagnostics plan

The original surface had an empty `diagnostics_plan`. Table 2 contains three falsification tests that are standard for this paper's identification: (a) white lynching rate on black registration (should be null), (b) black lynching rate on white registration (should be null), (c) white lynching rate on white registration (should be null). Added all three as `diag/falsification/*` entries.

### 2. Corrected controls_count_max from 15 to 14

The full control pool has exactly 14 variables:
- 7 historical: Black_share_illiterate, initial, newscapita, farmvalue, sfarmprop1860, landineq1860, fbprop1860
- 4 contemporary black: Black_beyondhs, Black_avgage, Black_Earnings, share_maritalblacks
- 3 additional: share_slaves, incarceration_2010, pollscapita

The previous count of 15 was off by one. Corrected to 14.

### 3. Added data construction note

The design_audit now includes a note that the panel data is collapsed to county-level cross-section in Createmain.do (line 200), and the sample is restricted to N=267 counties with complete data on all controls including contemporary variables.

### 4. Added sample restriction ordering note

Table B3 code (Maindo.do lines 225-229) first replaces Blackrate_regvoters > 100 with 100, then runs two regressions: one on capped data and one dropping values < 100 on the already-capped data. The runner must apply these modifications in order, not as independent operations. Added a `sample_restriction_note` to the constraints block.

### 5. Added functional form coefficient interpretation caveat

The functional_form_policy note now clarifies that asinh/log1p transforms change coefficient interpretation from level to semi-elasticity, though they preserve the direction/significance claim.

---

## Key Constraints and Linkage Rules

1. **No bundled estimators**: Single-equation OLS throughout. No linkage constraints.
2. **State FE**: All main regressions include `i.State_FIPS` (6 states). The `rc/fe/drop/State_FIPS` variant removes this. With only 6 states, state FE is a meaningful control.
3. **IID SEs**: The paper uses plain `regress` without robust or cluster options. This is confirmed by inspecting Maindo.do -- no `vce()` or `cluster()` options are specified in the main table regressions.
4. **Controls use `c.` prefix**: Stata's `c.` prefix in Maindo.do means "treat as continuous" -- this is Stata factor notation and does not change interpretation.
5. **Data sample**: The cross-sectional dataset is constructed by collapsing panel data and then restricting to counties with complete data on all controls (line 242-249 of Createmain.do). This means the sample is the same across all specifications.

---

## Budget and Sampling Assessment

### G1 Budget (80 max):
- 1 baseline
- 7 LOO (historical controls)
- 4 standard sets (no controls, historical+slaves, historical+contemporary, full)
- 9 progression specs (bivariate through full)
- 20 random control subsets (seed=136741)
- 4 sample variants (2 trimming, 1 drop >100, 1 cap at 100)
- 1 FE drop
- 4 data/treatment construction (1910, 1920, 1930 population denominators, Stevenson data source)
- 2 functional form (asinh, log1p)
= **52 core specs**
- Well within 80 budget. Full enumeration for deterministic specs plus 20 random draws.

---

## What's Missing

1. **Oster bounds (Table B5)**: The paper computes `psacalc` bounds for omitted variable bias. This is a sensitivity analysis that could be added as `sens/unobserved_confounding/oster_bounds`. The surface correctly excludes this from the core universe, but it could be listed as a planned sensitivity analysis.

2. **Table 4 (register_black level outcome)**: Uses count rather than rate. The surface correctly classifies this as `explore/variable_definitions` since it changes the outcome concept.

3. **Table 8 (heterogeneity interactions)**: Uses interaction terms between lynching rate and education/earnings/church membership. Correctly excluded as `explore/heterogeneity`.

4. **County-level clustering**: The Createmain.do file (line 198) runs one specification with `cluster(fips)` before collapsing data. This suggests the authors considered county-level clustering as relevant for the panel version. While the collapsed cross-section has N=267 counties (each appearing once), the inference variant `infer/se/hc/hc1` is more appropriate than state-level clustering (only 6 clusters). The surface already includes HC1 as a variant, which is good.

5. **Historical voter registration (Table B1)**: Uses 1867 voter registration as outcome. This is a different time period and could be a useful falsification/mechanism diagnostic but is correctly excluded.

6. **Missing LOO for contemporary controls**: When the full control set is used (e.g., in `rc/controls/sets/full_kitchen_sink`), individual LOO for contemporary controls (Black_beyondhs, Black_avgage, Black_Earnings, share_maritalblacks, incarceration_2010, pollscapita, share_slaves) would be informative. The progression specs partially cover this but not as ceteris-paribus LOO. This is a minor gap -- the 20 random subsets partially address it.

---

## Approved to Run

**Status**: APPROVED

The surface is well-constructed for this cross-sectional OLS paper. The core universe is comprehensive, covering LOO, progressive control building, random subset sampling, sample restrictions, treatment variable construction, and functional form. The budget is adequate, and the falsification diagnostics are now included. The main identification concern (unobserved confounding in a cross-sectional setting) is acknowledged in the Oster bounds exclusion as a planned sensitivity analysis.

**One note for runners**: The maindata.dta is created by Createmain.do with a sample restriction to counties having complete data on ALL controls including contemporary variables (line 242-248). This means the sample is fixed at N=267 regardless of which controls are included in a given specification. This is the correct behavior for the specification search.
