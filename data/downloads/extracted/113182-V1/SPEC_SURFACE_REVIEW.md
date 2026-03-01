# Specification Surface Review: 113182-V1

**Paper**: Condra, Long, Shaver, Wright (2018), "The Logic of Insurgent Electoral Violence," AER
**Design**: instrumental_variables (two distinct IV strategies)
**Date reviewed**: 2026-02-25

---

## Summary of Baseline Groups

| Group | Claim Object | Instrument | Data | Status |
|-------|-------------|-----------|------|--------|
| G1 | LATE of morning attacks on voter turnout (Table 2) | Wind speed at 4:30 AM | district_iv.dta (782 obs, 60 vars) | Verified |
| G2 | LATE of IED deployment on vote totals (Table 3) | Nighttime cloud cover | roads_iv.dta (15,056 obs, 35 vars) | Verified |

**Changes made**: Minor corrections to alternative outcome variable names in G2 and added a missing RC axis for additional controls (see below).

---

## A) Baseline Groups

- G1 and G2 are correctly separated. They use different instruments, different endogenous variables, different units of observation (district-election vs. polling-station-road-segment), and different datasets. This is exactly right.
- G1 baseline spec (Table 2, Col 4) verified against code:
  - `reghdfe total_v2_agcho10 [weather controls, quadratic] plus_wind_00Z_10_pre14D population_2010_adj (df_5to11= c.plus_wind_00Z_10) if no_voting_either==0&disrupt==1, absorb(first) vce(cluster DISTID)`
  - All 18 controls confirmed in the do-file. Weather variable names verified in data.
- G2 baseline spec (Table 3, Col 3) verified against code:
  - `reghdfe total_votes_wins ht_route_indicator rcv2 population_v2 shape_leng pre_event_indicator_6m_V2 march_rain march_rain2 (post_event_indicator= cloudz_perc_election) if closure_indicator==0&stations!=0, absorb(distid) vce(robust)`
  - All 7 controls confirmed. Variable names verified in roads_iv.dta.
- No missing baseline groups. Tables 6-7 (ANQAR survey regressions) are correctly excluded as secondary OLS analysis.

## B) Design Selection

- `design_code: instrumental_variables` is correct for both groups.
- Design audit blocks are comprehensive: endog_vars, instrument_vars, n_instruments, overid_df, fe_structure, cluster_vars, sample_condition, and bundle all recorded.
- Both groups are just-identified (1 instrument, 1 endogenous regressor, overid_df=0). This means overidentification tests are not applicable. Correct.
- Estimator variants (2SLS, LIML) are appropriate. LIML is more robust under weak instruments, which is relevant here given the weather-based instrument.
- `bundle.linked_adjustment: true` for both groups is correct -- controls appear in both first and second stages.

## C) RC Axes

### G1
- **Control progressions**: Linear weather (Col 2), quadratic weather (Col 3), quadratic + pre-14D wind (Col 4, baseline). Verified against TABLE_2.do. Correct.
- **LOO drops**: Drop population, drop pre-14D wind, drop quadratic terms. Reasonable tests of exclusion-restriction sensitivity.
- **Sample expansion**: Include all districts (drop disrupt==1). Verified in TABLES_SI-4-22.do (SI Table 5 runs on `no_voting_either==0` without `disrupt==1`). Correct.
- **Alternative outcomes**: `total_nc_TO`, `susp_turnout_v2`, `corruption` -- all verified as variables in district_iv.dta.
- **Alternative treatments**: `df_5to7`, `df_5to8`, `df_5to9`, `df_5to10`, `df_5to12` -- verified in data columns and SI Tables 11-12.
- **Missing axis (added)**: The SI tables also include additional controls (`pashto`, `terrain`, `DF_pretrend28d`, `intim`, `nighttime_observations`) as separate robustness checks (SI Tables 15-19). These are additional exogenous regressors that test whether the IV results are robust to controlling for potentially confounding district characteristics. Added `rc/controls/add/pashto`, `rc/controls/add/terrain`, `rc/controls/add/pretrend_28d`, `rc/controls/add/intimidation`, `rc/controls/add/nighttime_obs` to the core_universe.

### G2
- **LOO drops**: Drop rain, drop pre-event 6-month, drop shape_leng, drop population. Reasonable.
- **Sample**: Include non-disrupted areas. Correct.
- **Alternative outcomes**: The surface references `total_nc_wins`, `ghani_nc_wins`, `abdullah_nc_wins`, `corrupt_perc`. Verified in roads_iv.dta: `total_nc_wins`, `ghani_nc_wins`, `abdullah_nc_wins`, `corrupt_perc` all present. Correct.
- **Note on variable naming**: The Table 3 code uses `ashraf_wins` and `abdullah_wins` for the candidate-specific total vote outcomes (Cols 4-5). The surface correctly references these baseline spec IDs. The "Ghani" terminology refers to Ashraf Ghani, whose variables use the "ashraf" prefix in the data. This is consistent.

## D) Controls Multiverse Policy

### G1
- `controls_count_min: 10` (linear weather + population, matching Col 2). Verified.
- `controls_count_max: 18` (full quadratic + pre-14D + population, matching Col 4). Verified: 2 windspeed_06Z/12Z + 3 temp + 3 rain + 2 windspeed_sq + 3 temp_sq + 3 rain_sq + 1 pre14D + 1 population = 18. Correct.
- `linked_adjustment: true`. Correct -- controls are shared across stages.

### G2
- `controls_count_min: 5` (geographic controls without rain). Verified: ht_route_indicator + rcv2 + population_v2 + shape_leng + pre_event_indicator_6m_V2 = 5.
- `controls_count_max: 7` (geographic + march_rain + march_rain2). Verified.
- `linked_adjustment: true`. Correct.

## E) Inference Plan

### G1
- **Canonical**: Cluster at DISTID. Matches Table 2. Correct.
- **Variant**: Robust without clustering. Reasonable.

### G2
- **Canonical**: Robust SEs (HC1). The do-file confirms Table 3 reports robust SEs as primary (`vce(robust)`) with clustered SEs reported separately. Correct.
- **Variant**: Cluster at pc_clust_thiessen. Confirmed in do-file. Correct.

## F) Budgets and Sampling

- G1: 70 specs. Reasonable for 1 baseline outcome + control progressions + LOO + 5 alternative treatments + 3 alternative outcomes + sample expansion + LIML + additional controls.
- G2: 55 specs. Reasonable.
- Exhaustive enumeration for control subsets. Correct given the structured nature of the control progressions.

## G) Diagnostics Plan

### G1
- First-stage F (K-P): Confirmed in do-file (`rkf` statistic reported).
- Falsification RF: Verified in SI Table 6 (reduced form on non-disrupted districts).
- Balance of instrument: SI Table 4 (wind does not predict disruption indicator).

### G2
- First-stage F (K-P): Confirmed.
- Weak-IV robust p-value: Anderson-Rubin test via `weakiv` command. Confirmed in do-file.
- Falsification RF and exclusion restriction tests: SI Tables 24-25.

---

## Key Constraints and Linkage Rules

1. **Bundled estimator (IV)**: Controls/FE are linked across first and second stages in both groups. Any spec modification must apply to both stages simultaneously.
2. **Weather controls (G1)**: Critical for exclusion restriction. The progression (linear -> quadratic -> +pre14D) absorbs potential direct effects of weather on turnout. Dropping ALL weather controls would violate identification.
3. **Geographic controls (G2)**: Road-segment characteristics are essential spatial controls for the exclusion restriction.
4. **Election FE (G1)**: The `first` variable in district_iv.dta represents election fixed effects (absorbed via `absorb(first)`). This is mandatory for the panel structure (district-election panel).
5. **District FE (G2)**: Absorbed via `absorb(distid)`. Mandatory for the cross-sectional comparison within districts.

---

## Changes Made to JSON

1. Added 5 additional RC axes to G1 core_universe for SI robustness controls: `rc/controls/add/pashto`, `rc/controls/add/terrain`, `rc/controls/add/pretrend_28d`, `rc/controls/add/intimidation`, `rc/controls/add/nighttime_obs`.
2. Updated G1 controls_count_max note for clarity.
3. No structural changes to G2.

---

## Final Assessment

**APPROVED TO RUN.** Both IV strategies are well-documented with complete design audit blocks, appropriate bundled-estimator constraints, and comprehensive RC axes covering the paper's revealed specification search space. The diagnostics plan correctly includes first-stage strength, falsification, and balance tests. The addition of the extra controls from SI Tables 15-19 makes the G1 surface more comprehensive.
