# Specification Surface: 112749-V1

**Paper**: "When the Levee Breaks: Black Migration and Economic Development in the American South"
**Authors**: Hornbeck & Naidu (AER 2014)
**Design**: Continuous-treatment Difference-in-Differences (county panel)
**Created**: 2026-02-13

---

## 1. Paper Overview

This paper studies the long-run effects of the Great Mississippi Flood of 1927 on Black out-migration and agricultural development in Southern US counties. The identification strategy exploits cross-county variation in flood intensity (share of county area flooded) as a quasi-exogenous shock, using a county-level panel from 1900-1970 with county fixed effects and state-by-year fixed effects.

The key empirical design is a **continuous-treatment DiD**: the treatment variable `flood_intensity` (share of county flooded) is interacted with post-flood year dummies (`f_int_1930`, `f_int_1940`, ..., `f_int_1970`), estimated with county FE absorbed (`areg ... , absorb(fips)`) and state-by-year FE as dummies (`d_sy_*`), clustered at the county level.

**Important design note**: This is NOT staggered treatment adoption. All treatment occurs simultaneously (the 1927 flood), and treatment intensity is continuous. Modern staggered-DiD estimators (Sun-Abraham, Callaway-Santanna, Borusyak imputation) are not directly applicable.

---

## 2. Baseline Groups

### Why two baseline groups?

The paper makes two distinct headline claims that constitute independent baseline groups:

1. **G1 (Labor/Demographics)**: The flood caused persistent Black out-migration, reducing the Black share of population in more-flooded counties.
2. **G2 (Agricultural Capital)**: The labor outflow induced capital-labor substitution, increasing mechanization (farm equipment value) in more-flooded counties.

These are conceptually distinct claims (one about labor, one about capital) with different outcomes, and they are the paper's two main "stories." Other outcomes explored in the paper (tractors, mules/horses, farm size, farmland, land values, crop yields) are treated as **exploration** rather than additional baseline groups, because the paper's headline narrative focuses on the labor-outflow and capital-substitution channels.

### G1: Flood impact on Black population share

| Field | Value |
|---|---|
| **Outcome concept** | Log fraction Black population in county |
| **Treatment concept** | Flood intensity x post-flood year dummies |
| **Estimand concept** | Dynamic ATT path at each post-flood decade |
| **Target population** | Southern counties near Mississippi flood zone (balanced panel, >=10% Black, >=15% cotton in 1920) |
| **Direction expectation** | Negative (flood caused Black out-migration) |
| **Baseline specs** | Table 2, columns 1-2 |
| **Focal parameter** | `f_int_1950` (peak effect period) |

**Baseline spec details (Table 2 col 1)**:
- Outcome: `lnfrac_black`
- Treatment: `f_int_1930` through `f_int_1970`
- FE: county (absorbed), state-by-year (dummies)
- Controls: geography block (crop suitability x year, distance to MS x year, coordinates x year, ruggedness x year) + lagged dependent variable block (lag2, lag3, lag4 of `lnfrac_black` x year)
- Weights: `county_w` (county area in 1920)
- Cluster: `fips` (county)
- N = 978

**Baseline spec (Table 2 col 2)**: Same as col 1 plus New Deal spending controls (`lnpcpubwor_*`, `lnpcaaa_*`, `lnpcrelief_*`, `lnpcndloan_*`, `lnpcndins_*`).

### G2: Flood impact on agricultural capital intensity

| Field | Value |
|---|---|
| **Outcome concept** | Log value of farm equipment per county |
| **Treatment concept** | Flood intensity x post-flood year dummies |
| **Estimand concept** | Dynamic ATT path at each post-flood census year |
| **Target population** | Same as G1 |
| **Direction expectation** | Positive (labor outflow induced mechanization) |
| **Baseline specs** | Table 4, columns 3-4 |
| **Focal parameter** | `f_int_1950` (post-WWII mechanization period) |

**Baseline spec details (Table 4 col 3)**:
- Outcome: `lnvalue_equipment`
- Treatment: `f_int_1930` through `f_int_1970` (9 periods including 5-year intercensus)
- FE: county (absorbed), state-by-year (dummies)
- Controls: geography block + lagged DV block (lag1 through lag4 of `lnvalue_equipment` x year)
- Weights: `county_w`
- Cluster: `fips`

**Baseline spec (Table 4 col 4)**: Same as col 3 plus New Deal spending controls.

---

## 3. Revealed Search Space Analysis

### 3A. Controls axis (highest leverage, revealed extensively)

The paper's **RefTables 1-3** reveal a systematic 7-step control progression for each outcome:

| Step | Controls included | Source |
|---|---|---|
| 1 | Lagged DV only (no geography, no New Deal) | RefTable step 1 |
| 2 | + Geography (crop suitability, distance to MS, coordinates, ruggedness) | RefTable step 2 |
| 3 | + Geography + tenancy share + manufacturing (mfg establishments, mfg wages) | RefTable step 3 |
| 4 | + Geography + New Deal spending | RefTable step 4 |
| 5 | + Geography + New Deal + plantation dummy | RefTable step 5 |
| 6 | + Geography + New Deal + tenancy + manufacturing | RefTable step 6 |
| 7 | + Geography + New Deal + tenancy + manufacturing + propensity score | RefTable step 7 |

The control variables naturally form **blocks**:
- **Geography**: `cotton_s_*`, `corn_s_*`, `ld_*` (distance to MS), `dx_*`, `dy_*` (coordinates), `rug_*` (ruggedness) -- all interacted with year
- **Lagged DV**: `lag1_*` through `lag4_*` of the outcome variable -- interacted with year
- **New Deal spending**: `lnpcpubwor_*`, `lnpcaaa_*`, `lnpcrelief_*`, `lnpcndloan_*`, `lnpcndins_*` -- interacted with year
- **Tenancy + manufacturing**: `lag*_lnfarms_nonwhite_t_*`, `lag*_lnmfgestab_*`, `lag*_lnmfgavewages_*`
- **Plantation**: `plantation_*`
- **Propensity score**: `prop_plant_flstate_1*`

**Control-count envelope**: Approximately 36 (col 1, geography+lags) to 61+ (col 2 with New Deal; RefTable full specification is larger). The progression is block-based, so enumeration is feasible (no variable-level combinatorial search needed).

### 3B. Treatment definition axis (revealed)

The paper tests 3 alternative flood intensity measures:
1. **Baseline**: `flood_intensity` = share of county area flooded (from flood maps)
2. **Alternative 2**: `flood_intensity_2` = Red Cross flooded acres / county area
3. **Alternative 3**: `flood_intensity_3` = Red Cross affected population / total population

These are reported in RobustnessMeasures2 and RobustnessMeasures3 tables.

### 3C. Inference axis (revealed)

The paper explicitly varies inference methods:
1. **Baseline**: Cluster by county (`fips`)
2. **Conley spatial HAC**: 50mi, 100mi, and 200mi cutoffs (computed using `x_ols` custom ado file)

### 3D. Sample axis (partially revealed)

- **Main sample**: Balanced panel, counties with >=10% Black and >=15% cotton share in 1920 (~163 counties, ~978 obs for decadal outcomes)
- **Table 6**: Southern rivers sample (non-flooded counties along other rivers) -- different target population, treated as exploration/falsification
- **Table 7**: Non-flooded counties only (distance to flood as treatment) -- different target population, treated as exploration/falsification

### 3E. Fixed effects axis (NOT revealed)

All panel specifications use the same FE structure: county FE + state-by-year FE. The paper does not vary this. Alternative FE structures (e.g., year FE only, dropping state-by-year) would change identifying variation substantially.

### 3F. Weights axis (NOT revealed)

All regressions use `county_w` (county area in 1920) as analytic weights. The paper does not show unweighted results.

---

## 4. Core-Eligible Universe

### 4A. Design specifications

| spec_id | Description |
|---|---|
| `baseline` | Paper's Table 2 col 1 (G1) / Table 4 col 3 (G2) |
| `baseline__with_newdeal` | Paper's Table 2 col 2 (G1) / Table 4 col 4 (G2) |
| `design/difference_in_differences/estimator/twfe` | Baseline TWFE estimator (same as baseline; paper's method) |

**Note on alternative DiD estimators**: Modern heterogeneity-robust DiD estimators (Sun-Abraham, Callaway-Santanna, etc.) are designed for staggered binary treatment adoption. This paper has simultaneous continuous treatment from a single event. These estimators are not directly applicable.

### 4B. Controls robustness (RC)

**Block-based progression** (from revealed RefTable structure):

| spec_id | Description |
|---|---|
| `rc/controls/sets/none` | No time-varying controls (FE only) |
| `rc/controls/progression/lagged_dv_only` | Lagged DV only |
| `rc/controls/progression/geography` | Geography block only (no lags) |
| `rc/controls/progression/geography_and_lags` | Geography + lagged DV (= Table 2 col 1 baseline) |
| `rc/controls/progression/geography_lags_newdeal` | + New Deal spending (= Table 2 col 2 baseline) |
| `rc/controls/progression/geography_lags_tenancy_mfg` | Geography + lags + tenancy share + manufacturing |
| `rc/controls/progression/geography_lags_newdeal_tenancy_mfg` | Geography + lags + New Deal + tenancy + mfg |
| `rc/controls/progression/geography_lags_newdeal_plantation` | Geography + lags + New Deal + plantation |
| `rc/controls/progression/geography_lags_newdeal_tenancy_mfg_propscore` | Full kitchen sink (all revealed controls + propensity score) |

**Leave-one-block-out** (from baseline geography+lags+New Deal specification):

| spec_id | Description |
|---|---|
| `rc/controls/loo_block/drop_geography` | Drop all geography controls |
| `rc/controls/loo_block/drop_lagged_dv` | Drop all lagged DV controls |
| `rc/controls/loo_block/drop_new_deal` | Drop New Deal spending controls |
| `rc/controls/loo_block/drop_crop_suitability` | Drop cotton_s_* and corn_s_* only |
| `rc/controls/loo_block/drop_distance_ms` | Drop ld_* only |
| `rc/controls/loo_block/drop_coordinates` | Drop dx_* and dy_* only |
| `rc/controls/loo_block/drop_ruggedness` | Drop rug_* only |

### 4C. Treatment definition robustness (RC)

| spec_id | Description |
|---|---|
| `rc/form/treatment/alt_measure_redcross_acres` | Use `flood_intensity_2` (Red Cross flooded acres) |
| `rc/form/treatment/alt_measure_redcross_people` | Use `flood_intensity_3` (Red Cross affected population) |

### 4D. Sample robustness (RC)

| spec_id | Description |
|---|---|
| `rc/sample/time/drop_1970` | Drop last period (1970) |
| `rc/sample/time/drop_1930` | Drop first post-treatment period (1930) |
| `rc/sample/time/pre1960_only` | Restrict to 1930-1960 only (drop 1964, 1970) |
| `rc/sample/outliers/trim_treatment_p95` | Trim flood_intensity at 95th percentile |

### 4E. Weights robustness (RC)

| spec_id | Description |
|---|---|
| `rc/weights/main/unweighted` | Run without analytic weights |

### 4F. Inference variations

| spec_id | Description |
|---|---|
| `infer/se/cluster/unit` | Cluster by county (baseline) |
| `infer/se/hc/hc1` | Heteroskedasticity-robust only (no clustering) |
| `infer/se/spatial/conley_50km` | Conley spatial HAC, 50mi cutoff |
| `infer/se/spatial/conley_100km` | Conley spatial HAC, 100mi cutoff |
| `infer/se/spatial/conley_200km` | Conley spatial HAC, 200mi cutoff |
| `infer/se/cluster/state` | Cluster by state (conservative; few clusters ~9) |

---

## 5. Excluded from Core (and Why)

### 5A. Alternative outcomes (explore/*)

The paper examines many outcomes beyond the two baseline groups. These are **exploration** because they test different outcome concepts:

- `lnpopulation_black` (log Black population, not share)
- `lnpopulation` (total population)
- `lnfracfarms_nonwhite` (nonwhite farm share)
- `lntractors` (tractors)
- `lnmules_horses` (mules and horses)
- `lnavfarmsize` (average farm size)
- `lnfarmland_a` (farmland acreage)
- `lnlandbuildingvaluef`, `lnlandbuildingvalue` (land/building values)
- `lncotton_yield`, `lncorn_yield` (crop yields)
- `lncropval_p`, `lncropval_rp` (crop value per capita)

These would be `explore/outcome/*` specifications.

### 5B. Alternative samples (explore/*)

- Table 6 (southern rivers, non-flooded counties with `riverclose_elsewhere` as treatment): Different target population and treatment concept
- Table 7 (non-flooded counties, `distance` as treatment): Different target population and treatment concept

These are falsification exercises and belong in `explore/sample/*` or `diag/*`.

### 5C. Plantation interaction (explore/heterogeneity/*)

The paper interacts treatment with plantation status (`f_int_*_p`). This is heterogeneity analysis, not a main claim. Would be `explore/heterogeneity/plantation`.

### 5D. FE variations

The paper does not vary its FE structure (always county + state-by-year). Dropping state-by-year FE to use only year FE would materially change identification. Not included in core.

### 5E. Modern DiD estimators

Not applicable to this design (simultaneous continuous treatment, not staggered binary adoption).

---

## 6. Budgets and Sampling

### Budget per baseline group

For each of G1 and G2:
- **Control progressions**: 9 specs (including 2 baselines)
- **Leave-one-block-out**: 7 specs
- **Treatment alternatives**: 2 specs
- **Sample variations**: 4 specs
- **Weight variations**: 1 spec
- **Subtotal RC**: ~23 unique specification combinations

Cross with 6 inference variations = ~138 total if fully crossed. But inference variations are **inference-only** recomputations (same point estimate, different SE), so they do not multiply the computational burden of estimation.

**Estimated core total per baseline group**: ~25 distinct estimating equations x 6 inference variants = ~80 total rows (within budget).

### Sampling method

**Full enumeration** is feasible because:
1. Controls vary at the block level (not variable-by-variable), keeping the combinatorial space small
2. The paper itself reveals the progression structure in RefTables 1-3
3. The total number of distinct estimating equations is manageable (~25 per group)

No random subset sampling is needed.

---

## 7. Diagnostics Plan (Separate from Core)

| Diagnostic | Scope | Description |
|---|---|---|
| `diag/difference_in_differences/pretrends/event_study_plot` | baseline_group | Plot event-study coefficients including pre-flood periods (1900, 1910); visual pre-trends check. Paper does this in Figures 3-4. |
| `diag/difference_in_differences/pretrends/joint_test` | spec | Joint F-test that pre-flood period coefficients = 0 for each core specification |

---

## 8. Key Linkage Constraints and Implementation Notes

1. **Time-varying controls**: All controls in this paper are interacted with year dummies (e.g., `cotton_s_1930`, `cotton_s_1940`, etc.). When adding or dropping a control "block," all year-interacted versions of those variables must be added/dropped together.

2. **Lagged dependent variables**: The lagged DV controls are outcome-specific. For G1, the lags are of `lnfrac_black`; for G2, the lags are of `lnvalue_equipment`. When running the same control progression for a different outcome, the lagged DV block changes accordingly.

3. **Focal parameter selection**: Each regression produces a vector of treatment effects (one per post-flood year). The focal parameter for scalar summary should be `f_int_1950` (representing the cumulative medium-run effect). The full vector of period-specific coefficients must be stored in `coefficient_vector_json`.

4. **Weights**: The `county_w` variable is county area in 1920 (used as analytic weights via Stata's `[aweight=county_w]`). This corresponds to weighted least squares, not frequency/probability weights.

5. **Panel structure differences across outcomes**: Table 2 labor outcomes use decadal observations (1930-1970, 5 periods), while Table 4 capital outcomes use census years including 5-year intercensus observations (1930-1970, 9 periods). The treatment variable set differs accordingly.
