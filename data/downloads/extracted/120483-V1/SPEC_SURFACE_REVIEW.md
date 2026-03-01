# Specification Surface Review: 120483-V1

## Paper: Malaria and the Historical Roots of Slavery (Esposito, 2021, AEJ: Applied)

## Summary of Baseline Groups

**G1**: Effect of malaria ecology (MAL) on slavery intensity (slaveratio), cross-sectional OLS with state FE and Conley spatial SEs, at the US county level in 1790 and 1860.

- Single baseline group is appropriate. The claim object (association of malaria ecology with slavery prevalence) is well-defined and consistent across Table 1 columns.
- Two baseline specs are recorded: Table 1 Col 3 (1790, 14 controls) and Table 1 Col 5 (1860, 15 controls). The 1860 spec is designated as the anchor baseline, which is appropriate since it uses the larger dataset.

No changes to baseline group structure.

## Changes Made

### A) Removed redundant spec IDs

1. **Removed `rc/controls/sets/full`**: This is identical to the baseline (all 15 controls for 1860), so it would produce a duplicate row. The baseline already covers this.

2. **Removed `rc/controls/progression/bivariate`**: This is identical to `rc/controls/sets/none` (zero controls with state FE). Keeping `rc/controls/sets/none` as the canonical zero-controls spec.

3. **Removed `rc/sample/restrict/1860_data`**: The baseline already uses the 1860 dataset. This would be a no-op.

### B) Added clarifying notes

1. **Added `inference_note` to design_audit**: The paper uses Conley spatial SEs (via Stata `acreg`) as its primary inference, which is non-standard in pyfixest. The note clarifies that the runner may need to fall back to state-clustered SEs as the canonical choice if Conley SEs are unavailable in the Python environment.

2. **Added `control_set_note` to constraints**: The 1790 and 1860 datasets have slightly different crop control sets (1790 lacks coffee). LOO specs are defined relative to the 1860 full set, so `drop_coffee` is inapplicable when running on 1790 data. The runner needs to handle this gracefully.

3. **Added `mandatory_controls` to constraints**: All controls are optional; there are no mandatory controls. This makes the control-subset sampling policy explicit.

### C) Budget adjustment

Reduced `max_specs_core_total` from 80 to 75 after removing 3 redundant specs. Revised count: 15 LOO + 3 sets + 2 progression + 15 random subsets + 2 sample restricts + 2 trims + 1 FE drop + 2 form transforms + 1 baseline = 43. Budget of 75 allows headroom for additional specs discovered during execution.

## Key Constraints and Linkage Rules

- **No linkage constraints**: Single-equation OLS; no bundled estimator components.
- **Control-count envelope**: [0, 15], matching the paper's range from bivariate (Col 1) to full controls (Col 5).
- **Functional form**: `rc/form/outcome/asinh` and `rc/form/outcome/log1p` change the coefficient interpretation (level vs elasticity-like), but preserve the direction-of-association claim. They are kept as RC but flagged in `functional_form_policy`.

## Budget and Sampling Assessment

- Total enumerated core specs: ~43, well within the budget of 75.
- Controls subset sampling: 15 random draws with seed 120483, stratified by size. This is sufficient for a 15-variable control pool.
- The universe is fully tractable (no need for sampling beyond the control subsets).

## What's Missing

1. **No diagnostics plan**: The surface has an empty `diagnostics_plan`. For cross-sectional OLS, standard diagnostics (e.g., residual distribution, VIF/multicollinearity checks) are optional but would strengthen the audit. This is not blocking.

2. **No exploration specs**: The paper has additional tables (Table 2 crop interactions, Tables 5-7 state-level DiD, Tables 8-10 slave prices) that are correctly excluded from G1. If a second baseline group were added for the state-level DiD evidence, that would expand the surface significantly, but this is out of scope for the current analysis which focuses on the core cross-sectional claim.

3. **Spatial SE implementation uncertainty**: Conley SEs via `acreg` in Stata are not directly available in pyfixest. The runner will likely need to either: (a) implement a custom Conley SE calculator, (b) use a Python spatial econometrics package, or (c) fall back to state-clustered SEs as canonical and record Conley SEs as a separate inference variant if implementable. This practical constraint does not block surface approval but should be noted in the execution plan.

## Verification Against Code

- **Control variable names**: Verified against `1_table_1_2_3.do` lines 9-13. The globals `$crop1790`, `$geo1790`, `$crop1860`, `$geo1860` match the surface's control lists exactly.
- **Dataset files**: `county_1790.dta` and `county_1860.dta` are present in `AEJApp-2019-0372/dta/`.
- **Fixed effects**: `state_g` is the absorbed FE variable, confirmed in the `acreg` and `reghdfe` calls.
- **Slave states restriction**: Cols 6-7 use `if slave_state==1`, confirming the `rc/sample/restrict/slave_states_only` spec.
- **Estimation commands**: `acreg` for Conley SEs, `reghdfe` for state-clustered SEs. Both produce OLS estimates.

## Final Assessment

**Approved to run.** The surface is conceptually coherent, statistically principled, and faithful to the paper's revealed specification search. The three redundant specs have been removed, clarifying notes have been added, and the budget has been adjusted accordingly. No blocking issues remain.
