# Specification Surface Review: 112517-V1

## Paper
Fowlie, Holland & Mansur (2012), "What Do Emissions Markets Deliver and to Whom? Evidence from Southern California's NOx Trading Program," AER.

## Summary of Baseline Groups

- **G1**: ATT of RECLAIM program participation on facility-level NOx emissions change. The paper uses a matching-based DiD design: nearest-neighbor matching on pre-treatment emissions with exact industry matching.

Single baseline group is appropriate. The paper has one central claim (emissions reduction effect), with variations in functional form (levels vs. logs), sample, period, and matching parameters all constituting robustness to the same claim object. The environmental justice analysis (Tables 7-8) is correctly excluded as exploration (heterogeneous effects by demographics).

## Changes Made

### 1. Removed `rc/form/outcome/log_biasadj_quadratic`
The do-file (Table4.do, line 284) shows that the log specification already uses quadratic bias adjustment (`biasadj(lnPRENOX lnPRENOX2)`) by default. This is not a variant of the log specification -- it IS the log specification. Removed to avoid confusion. The linear bias adjustment is only used for the levels specification.

### 2. Removed `rc/form/matching/areg_ols`
This was redundant with `design/difference_in_differences/estimator/twfe` already listed in `design_spec_ids`. The OLS-with-FE estimator is a design alternative, not a matching parameter variation. It remains in the surface via the design_spec_ids and the joint areg specs (areg_levels_nonatt, areg_log_nonatt, areg_levels_drop_elec, areg_log_drop_elec).

### 3. No changes to baseline group definition
The two baseline specs (levels without demographics, levels with demographics) correctly represent the paper's Table 4 Panel A, Rows 1-2. The log versions are captured as `baseline_spec_ids` (baseline__table4_log, baseline__table4_log_demog).

## Key Constraints and Linkage Rules

- **Log-levels linkage**: When using log outcomes (`lnDIFFNOX`), matching variables and bias adjustment must switch to log scale (`lnPRENOX`, `lnPRENOX + lnPRENOX^2`). The surface correctly notes this constraint.
- **Demographics + quartile linkage**: When adding income1/pctminor1 as matching variables, exact matching on PRENOX quartile (PRENOX_Q) is also added. These move together in all paper specifications. Correctly noted.
- **Propensity score trimming**: Always applied as a preprocessing step (`psclean` program in all do-files). Not varied across specifications.
- **Estimator-inference linkage**: The nnmatch estimator uses Abadie-Imbens robust variance; the areg/OLS estimator uses HC1 or cluster(ab). These are correctly separated in the inference plan.

## Variable Verification

All variable names verified against the do-files (Table4.do, Table5.do, Table6.do, TableA2.do) and data files:
- `DIFFNOX`, `PRENOX`, `POSTNOX`: constructed in makedata program from `allyrs6.dta` (noxt variable averaged by period)
- `dumreclaim`: renamed from `RECLAIM1` in allyrs6.dta (confirmed in Table4.do line 84)
- `fsic`, `fsic2`: present in allyrs6.dta (confirmed in column listing)
- `nonattall`: constructed from `nonaOZ1` in allyrs6.dta (confirmed in Table4.do lines 152-154)
- `income1`, `pctminor1`: from demographics90.dta; pctminor1 constructed as `(black1 + hispanic1)/total1` (confirmed in Table4.do line 170)
- `PRENOX_Q`: constructed dynamically as quartiles of PRENOX by fsic (confirmed in Table4.do lines 234-244)
- `ab` (air basin): present in allyrs6.dta (confirmed in column listing)
- `lnDIFFNOX`, `lnPRENOX`, `lnPRENOX2`: constructed in the log specification block of Table4.do (lines 271-274)
- Electric utility flag from `R2009.dta` (r2009 variable): confirmed in dropelec program

**Data file mapping**:
- `allyrs6.dta`: Raw facility-year emissions data (133,535 obs x 19 vars)
- `panel_14.dta`: Pre-built analysis dataset for period 1 vs period 4 (14,038 obs x 15 vars)
- `demographics90.dta`: Census demographics merged by ufacid
- `R2009.dta`: Electric utility identification

## Budget/Sampling Assessment

- 80 max specs. The core universe lists 2 baselines (log versions) + 1 design spec + ~47 RC/joint specs = ~50 explicit specs. With the 2 primary baseline specs (levels, listed in baseline_specs), total reaches ~52 specs. Well within budget.
- Control subsets: exhaustive enumeration is trivial (only 2 optional matching variables: income1, pctminor1). 4 possible states: {neither, income only, minority only, both}.
- Main combinatorial expansion comes from: {levels, logs} x {no demog, with demog} x {full, drop elec} x {pd1-pd4, pd2-pd3, pd1-pd3} x {m=1,2,3,4,5} = 2 x 2 x 2 x 3 x 5 = 120 potential cells. Budget of 80 requires sampling from this space. The surface's explicit joint specs provide good coverage of the most important combinations.
- Seed 112517 is reproducible.

## What's Missing (minor)

- **Table A3-A4**: Appendix tables with additional robustness checks overlap with RC axes already included. Not a gap.
- **Table 7-8 environmental justice**: Correctly excluded as exploration (heterogeneity by neighborhood demographics). These involve match-group fixed effects and a complex post-matching regression structure that changes the estimand.
- **Propensity score trimming sensitivity**: The `psclean` preprocessing step is always applied. Varying the propensity score specification (e.g., different covariates in pscore) could be an additional axis but is not explored in the paper.
- **Table 2 data construction**: The create_data_panelX.do file could reveal additional data construction choices, but the surface focuses on the analysis-stage specifications.

## Design Audit Completeness

The design_audit block includes: estimator (nnmatch_att), treatment timing, estimand (ATT), comparison groups, matching variables, exact match variables, bias adjustment, n_matches, robust variance, and outcome construction. This is sufficiently detailed for a matching-based DiD. The key design-defining parameters (matching variables, exact matching, bias adjustment, number of matches) are all recorded.

## Final Assessment

**APPROVED TO RUN.** The surface is well-constructed for a matching-based DiD design. The single baseline group with rich variation along sample, period, functional form, and matching parameters provides comprehensive coverage. The removal of the redundant areg_ols RC spec and the spurious log_biasadj_quadratic variant improves clarity. The budget is feasible and the specification space is well-scoped.
