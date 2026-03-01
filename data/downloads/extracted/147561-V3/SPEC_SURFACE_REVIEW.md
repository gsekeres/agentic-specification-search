# Specification Surface Review: 147561-V3

**Paper**: Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo (Balan, Bergeron, Tourek, and Weigel)

**Review date**: 2026-02-25

---

## Summary of Baseline Groups

**G1**: Effect of local (chief-led) tax collection on property tax compliance.

- **Outcome**: `taxes_paid` (binary compliance) and `taxes_paid_amt` (continuous revenue)
- **Treatment**: `t_l` (local assignment, tmt==2 vs central tmt==1)
- **Estimand**: ITT
- **Population**: Property owners in Kananga, DRC (excluding villas, pilot polygons, missing rate)
- **Design**: Randomized experiment (stratified by stratum, clustered at polygon a7)
- **Baseline spec**: Table 4, Col 4 -- `reg taxes_paid t_l i.stratum i.house i.time_FE_tdm_2mo_CvL if inlist(tmt,1,2), cl(a7)`

One baseline group is appropriate. The paper has one focal comparison (local vs central). Tables 5-9 explore mechanisms (assessment accuracy, bribes, attitudes, collector visits, chief knowledge), not independent main claims. These are correctly excluded from the core universe.

---

## Changes Made to SPECIFICATION_SURFACE.json

### 1. Added second baseline spec for revenues (was missing from `baseline_specs`)

The surface listed `baseline__revenues` in `baseline_spec_ids` but did not include the corresponding entry in the `baseline_specs` array. Added:
```json
{
  "label": "Table4-Revenues-Col4",
  "outcome_var": "taxes_paid_amt",
  "treatment_var": "t_l",
  ...
}
```
**Rationale**: The revenues baseline was implicitly assumed but not formally specified. The runner needs the explicit outcome variable and sample restriction to emit the correct regression.

### 2. Removed `design/randomized_experiment/estimator/strata_fe` from `design_spec_ids`

This design variant is redundant. The baseline already includes stratum FE. The `strata_fe` design implementation is intended for cases where the baseline does *not* include strata FE. Here, `rc/fe/sets/stratum_only` already captures the stratum-FE-only specification (Table 4, Col 1).

**Rationale**: Avoid double-counting. The `diff_in_means` design variant (no FE at all) is the correct complementary design spec.

### 3. Removed redundant `rc/fe/add/house` and `rc/fe/add/time_FE_tdm_2mo_CvL`

These "add" variants are only meaningful relative to a stripped-down specification. Since the baseline already includes both house and time FE, "adding" them is a no-op. The FE axis is fully captured by the `rc/fe/sets/*` and `rc/fe/drop/*` spec IDs:
- `rc/fe/sets/stratum_only` (Col 1)
- `rc/fe/sets/stratum_month` (Col 2)
- `rc/fe/sets/stratum_month_house` (equivalent to baseline Col 4)
- `rc/fe/drop/house` (drop house from baseline)
- `rc/fe/drop/time_FE_tdm_2mo_CvL` (drop month from baseline)

**Rationale**: Redundant spec IDs inflate the budget and create confusion. The `sets/*` and `drop/*` variants are sufficient and map cleanly to Table 4 columns.

### 4. Added `estimand` and `n_clusters` to `design_audit`

Added `"estimand": "ITT"` and `"n_clusters"` fields as recommended by DESIGN_AUDIT_FIELDS.md for randomized experiments. These ensure results remain interpretable when detached from code.

### 5. Added `functional_form_scope` and `treatment_definition_scope` constraints

The original surface did not explicitly note that:
- `rc/form/outcome/log1p_amt` and `rc/form/outcome/asinh_amt` apply only to the revenues outcome (taxes_paid_amt), not the binary compliance outcome
- `rc/data/treatment/*` variants change the sample restriction (broadening beyond inlist(tmt,1,2))

**Rationale**: Without these scope annotations, the runner could incorrectly apply log/asinh to a binary variable, or forget to adjust the sample filter when adding treatment arms.

### 6. Refined `target_population` description

Added "pilot polygons" and "properties with missing rate information" to the exclusion criteria, matching the actual sample construction in `2_Data_Construction.do` (lines: `drop if house==3`, `drop if rate==.`, `drop if pilot==1`).

### 7. Added `reviewed_at` timestamp

---

## Key Constraints and Linkage Rules

- **Control-count envelope**: [0, 0]. No individual-level covariates in Table 4. The only variation is in fixed effects, not controls. This is correct and verified against Table 4 code.
- **No linkage constraints**: Single-equation OLS/RCT design. No bundled estimator components.
- **Sample restriction**: Always `inlist(tmt,1,2)` unless a treatment-definition RC explicitly broadens this.
- **Data construction invariants (not varied)**: Dropping villas (house==3), missing rate observations, pilot polygons. These are fundamental sample selection rules, not robustness axes.
- **time_FE_tdm_2mo_CvL construction**: Created by `egen cut()` on `today_alt` with fixed cutpoints. This variable is constructed in Table4.do itself (not pre-existing in analysis_data.dta). The runner must replicate this construction.

---

## Budget and Sampling Assessment

- **Budget**: 60 max core specs. Generous for the actual universe size.
- **Estimated total core specs**: ~18-20
  - 2 baseline specs (compliance + revenues)
  - 1 design variant (diff_in_means)
  - 5 FE combinations (stratum_only, stratum_month, stratum_month_house, drop_house, drop_time)
  - 3 sample variants (exclude_exempt, polygon_means, trim_amt)
  - 2 functional form variants (log1p, asinh -- revenues only)
  - 3 treatment definition variants (include_cli, include_cxl, pooled)
  - Total: ~16 unique spec axes
  - With both outcomes (compliance + revenues) for applicable specs, total approaches ~30
- **No sampling needed**: Universe is well within budget.
- **Seed**: 147561 (not needed since no sampling is required)

---

## What's Missing (minor, non-blocking)

1. **ANCOVA design variant**: The paper does not use ANCOVA (controlling for baseline outcome), but this is a standard RCT design variant. Could be added as `design/randomized_experiment/estimator/ancova` if baseline outcome data are available. However, the data construction does not create a pre-treatment compliance measure, so this is likely infeasible. Non-blocking.

2. **Wild cluster bootstrap inference**: Given clustering at the polygon level, wild cluster bootstrap (WCB) would be a natural inference variant. Not included as an inference variant. Could be added as `infer/resampling/wild_cluster_bootstrap`. Non-blocking since the paper does not use it.

3. **Lee bounds for attrition**: The diagnostics plan includes attrition differentials but not Lee bounds. Could be added as a sensitivity analysis (`sens/randomized_experiment/attrition_bounds/lee`) if attrition is nontrivial. Non-blocking.

4. **Interaction of FE sets with treatment definition RCs**: The surface does not specify whether FE variants are crossed with treatment-definition variants. The default interpretation is one-axis-at-a-time (each RC varied from baseline). This is appropriate and consistent with the architecture document.

---

## Final Assessment

**APPROVED TO RUN.**

The surface is well-constructed, conceptually coherent, and faithful to the manuscript. The single baseline group is appropriate for the paper's focal claim. The revealed search space (FE progression in Table 4) is correctly identified as the main robustness dimension. The corrections above are primarily about precision (adding the revenues baseline spec, removing redundant spec IDs, adding scope constraints) rather than fundamental issues. No blocking problems identified.
