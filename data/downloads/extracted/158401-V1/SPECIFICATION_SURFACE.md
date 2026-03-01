# Specification Surface: 158401-V1

**Paper**: "Market Access and Quality Upgrading: Evidence from Four Field Experiments"
**Authors**: Tessa Bold, Selene Ghisolfi, Frances Nsonzi, and Jakob Svensson
**Design**: Cluster-randomized experiment (village-level randomization, ITT with ANCOVA + Fisher randomization inference)

---

## 1. Baseline Groups

This paper reports results from two cluster-randomized experiments in Uganda studying maize farmers:

1. **Market Access experiment** (sample frame 1): 12 of 20 villages randomly assigned to treatment (access to an agro-trading company buying high-quality maize). Treatment variable: `buy_treatment`.
2. **Extension Service experiment** (sample frame 2): 9 of 18 villages randomly assigned to an agricultural extension service. Treatment variable: `ext_treatment`.

The paper's title and primary narrative focus on the Market Access experiment. The Extension Service experiment serves as a comparison and falsification (showing that information alone, without market access, does not produce the same effects). We define **two baseline groups**, both for the Market Access experiment, corresponding to the paper's two main results tables:

- **G1**: Market Access -> Investment outcomes (Table 5, Panel A)
- **G2**: Market Access -> Productivity and Income outcomes (Table 6, Panel A)

### Why two baseline groups (not one)?

Tables 5 and 6 represent conceptually distinct claim objects:
- **G1 (Investment)**: Whether market access induces farmers to invest more in quality (seeds, fertilizer, post-harvest handling). This is the mechanism/behavioral channel.
- **G2 (Productivity/Income)**: Whether market access improves welfare (harvest, yield, profits). This is the welfare/reduced-form effect.

These share the same treatment and sample but target different outcome concepts. Keeping them separate preserves clean interpretation of robustness results within each claim family.

### What is excluded and why

- **Extension Service experiment (Panel B of Tables 5 and 6)**: Not a separate baseline group because the paper treats it as a secondary comparison/falsification. It could be added as exploration if needed.
- **Table 4 (maize quality)**: Uses a different dataset (`quality_for_analysis.dta`) with the quasi-control group, a different unit of observation (bags of maize), and different estimators (Horowitz-Manski and Lee bounds). Too different in structure to include in the same specification surface.
- **Table 7 (traders' prices/market shares)**: Uses seller-level data (`sellers_for_analysis.dta`) with a different unit of observation and sample. Different claim object entirely.
- **Appendix Table 12 (production function decomposition)**: Mediation analysis, not a direct ITT estimate.
- **Appendix Table 14 (selection-adjusted effects)**: Uses normalized outcomes and selection indicators; changes the estimand.

---

## 2. Baseline Specifications

### Canonical specification pattern (both groups)

The paper uses ANCOVA throughout:

```
reg outcome buy_treatment ancova_control i.survey_season if group==1 & season_post==1, cluster(ea_code)
```

Where:
- `ancova_control` = mean of the outcome variable at the last pre-treatment season (or pooled baseline if last-season value is missing)
- `i.survey_season` = season fixed effects
- `cluster(ea_code)` = village-clustered standard errors
- Fisher randomization inference: 10,000 permutations of village-level treatment assignment

### G1: Table 5 Panel A outcomes (investment)

| Column | Outcome | ANCOVA control | Notes |
|--------|---------|----------------|-------|
| 1 | `expenses_fert_seeds` | `expenses_fert_seeds_p3` | Expenses on seeds + fertilizer (USD) |
| 2 | `expenses_inputs` | `expenses_inputs_p3` | Expenses on all inputs (USD) |
| 3 | `tarpaulin_d` | `tarpaulin_d_p3` | Proper drying dummy |
| 4 | `sort_d` | (none) | Sorting dummy -- no baseline measure |
| 5 | `winnow_d` | (none) | Winnowing dummy -- no baseline measure |
| 6 | `expenses_labor_preharvest` | `expenses_labor_preharvest_p3` | Pre-harvest labor expenses (USD) |
| 7 | `expenses_postharvest` | `expenses_postharvest_p3` | Post-harvest expenses (USD) |
| 8 | `expenses_labor_postharvest` | `expenses_labor_postharvest_p3` | Post-harvest labor expenses (USD) |

### G2: Table 6 Panel A outcomes (productivity/income)

| Column | Outcome | ANCOVA control | Notes |
|--------|---------|----------------|-------|
| 1 | `price` | `price_p3` | Price per kg (USD) |
| 2 | `acreage` | `acreage_p3` | Maize acreage |
| 3 | `harvest_kg_tot` | `harvest_kg_tot_p3` | Harvest in kg |
| 4 | `yield` | `yield_p3` | Yield (kg/acre) |
| 5 | `harvest_value` | `harvest_value_p3` | Harvest value (USD) |
| 6 | `expenses` | `expenses_p3` | Total monetary expenses (USD) |
| 7 | `surplus` | `surplus_p3` | Profit = harvest value - expenses (USD) |
| 8 | `surplus_hrs` | `surplus_hrs_p3` | Profit including own labor hours (USD) |

---

## 3. Core Universe

### 3.1 Design variants (`design/randomized_experiment/*`)

| spec_id | Description | Rationale |
|---------|-------------|-----------|
| `design/randomized_experiment/estimator/diff_in_means` | Difference-in-means (no ANCOVA, no season FE) | Minimal specification; pure randomization-based estimate |
| `design/randomized_experiment/estimator/with_covariates` | ANCOVA + pre-treatment household covariates | Adds 5 household characteristics (mdm_female, mdm_primary, hhr_n, distance_kakumiro, main_road_min) |

### 3.2 Robustness checks (`rc/*`)

#### Controls axis

| spec_id | Description | Rationale |
|---------|-------------|-----------|
| `rc/controls/sets/none` | Treatment + season FE only (drop ANCOVA control) | Tests sensitivity to ANCOVA adjustment |
| `rc/controls/sets/extended_hh_chars` | ANCOVA + all 5 household characteristics | Tests precision gains and sensitivity to additional pre-treatment covariates |
| `rc/controls/loo/drop_ancova` | Drop the ANCOVA baseline outcome control | For outcomes with ANCOVA, tests whether the baseline outcome control drives the result |

#### Sample axis

| spec_id | Description | Rationale |
|---------|-------------|-----------|
| `rc/sample/outliers/trim_y_1_99` | Trim top 1% of outcome per season (and bottom 1% for profit variables) | Matches Appendix Table 11 exactly; tests outlier sensitivity |
| `rc/sample/time/drop_first_post_season` | Drop first post-treatment season | Tests whether effects are driven by initial novelty |
| `rc/sample/time/drop_last_post_season` | Drop last post-treatment season (spring 2020, collected during COVID) | Tests sensitivity to last-season data quality |
| `rc/sample/panel/balanced_only` | Restrict to households observed in both baseline and follow-up | Tests sensitivity to attrition |

#### Functional form axis

| spec_id | Description | Rationale |
|---------|-------------|-----------|
| `rc/form/outcome/asinh` | Inverse hyperbolic sine of continuous monetary outcomes | Standard transformation for right-skewed outcomes with zeros; preserves sign interpretation |
| `rc/form/outcome/log1p` | log(1+y) for continuous monetary outcomes | Alternative to asinh for semi-elasticity interpretation |

**Note on functional form**: asinh/log1p transformations are applied only to continuous monetary outcomes (expenses, harvest value, profit, price, harvest_kg). They are NOT applied to binary outcomes (tarpaulin_d, sort_d, winnow_d) or acreage/yield where the level interpretation is natural. This preserves the claim object (direction and approximate magnitude of the effect).

### 3.3 Focal outcomes for RC/design specs

To keep the specification count manageable while maintaining coverage across outcome families, we run RC and design variants on **3 focal outcomes per baseline group**:

**G1 focal outcomes**:
- `expenses_fert_seeds` (continuous monetary, primary investment channel)
- `tarpaulin_d` (binary, post-harvest quality practice)
- `expenses_postharvest` (continuous monetary, post-harvest investment)

**G2 focal outcomes**:
- `surplus` (continuous monetary, primary welfare outcome)
- `harvest_value` (continuous monetary, gross outcome)
- `yield` (continuous, productivity measure)

---

## 4. Inference Plan

### Canonical inference

**Clustered SE at the village level** (`ea_code`), matching the paper's baseline inference. Village is the randomization unit, so this is the natural clustering level. Used for all estimate rows.

### Inference variants (written to `inference_results.csv`)

| spec_id | Description | Notes |
|---------|-------------|-------|
| `infer/se/hc/hc1` | Heteroskedasticity-robust SE (no clustering) | Stress test; with only 20 clusters (12T, 8C), clustered SE may be imprecise |
| `infer/ri/fisher/permutation` | Fisher randomization inference (10,000 permutations) | Matches the paper's supplementary inference; seed=760130 for group 1 |

---

## 5. Constraints

- **Controls count**: [0, 6]. Baseline uses 0-1 controls (ANCOVA only). Extended adds 5 pre-treatment household characteristics.
- **No linked adjustment**: Each outcome's ANCOVA control is specific to that outcome (no shared covariate set across outcomes).
- **Functional form**: asinh/log1p applied only to continuous monetary outcomes, not to binary variables. Coefficient interpretation changes from level to approximate semi-elasticity.
- **Season FE**: Always included (part of the design, not varied).
- **Cluster variable**: Always `ea_code` (village) for canonical inference (randomization unit).

---

## 6. Budget and Sampling

### Spec count breakdown

| Category | G1 | G2 | Total |
|----------|----|----|-------|
| Baseline specs | 8 | 8 | 16 |
| design/diff_in_means (3 focal) | 3 | 3 | 6 |
| design/with_covariates (3 focal) | 3 | 3 | 6 |
| rc/controls/none (3 focal) | 3 | 3 | 6 |
| rc/controls/extended (3 focal) | 3 | 3 | 6 |
| rc/sample/trim_1_99 (3 focal) | 3 | 3 | 6 |
| rc/sample/balanced_only (3 focal) | 3 | 3 | 6 |
| rc/form/asinh (2 monetary focal) | 2 | 2 | 4 |
| **Total** | **28** | **28** | **56** |

Full enumeration is feasible (56 specs). No sampling required.

### Rationale for focal-outcome selection

Running all 8 outcomes x 9 RC/design variants x 2 groups = 144 specs would be excessive and redundant. Instead, 3 focal outcomes per group (chosen to span continuous monetary, binary, and productivity families) provide adequate coverage. All 16 baseline specs are run to establish the full results table.

---

## 7. Diagnostics Plan (separate from core universe)

| Diagnostic | Scope | Description |
|-----------|-------|-------------|
| `diag/randomized_experiment/balance/covariates` | baseline_group | Joint F-test and individual balance tests for household characteristics and baseline outcomes (replicates Table 3) |
| `diag/randomized_experiment/attrition/attrition_diff` | baseline_group | Differential attrition between treatment and control groups (replicates Appendix Table 6) |

---

## 8. Notes on Data Structure

- **Data file**: `panel_g1g2_analysis.dta` (merged from `panel_g1_analysis.dta` and `panel_g2_analysis.dta`)
- **Panel structure**: Household (`hhh_id`) x season (`survey_season`). Seasons 1-3 are baseline (pre-treatment), seasons 4-7 are post-treatment for group 1.
- **Group indicator**: `group==1` for Market Access experiment, `group==2` for Extension Service experiment.
- **Post-treatment indicator**: `season_post==1` restricts to post-treatment seasons.
- **ANCOVA control construction**: For each outcome, `var_p3` = mean of the outcome at the last baseline season (or pooled baseline if missing). Created via `xfill` across household within seasons.
- **Currency conversion**: All monetary outcomes are converted from UGX to USD using season-specific exchange rates.
- **Randomization**: 12 of 20 villages assigned to treatment (Market Access). Permutation matrix (`permutations_gr1.dta`) generated with seed 760130.
