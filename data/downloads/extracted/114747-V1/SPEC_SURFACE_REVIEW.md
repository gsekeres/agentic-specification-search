# Specification Surface Review: 114747-V1

## Summary

The surface was reviewed against the paper's do-file and the provided .dta dataset. A major issue is that the provided data contains only a subset of the variables needed for the regressions described in the do-file.

## Baseline Groups

### G1: ADR Counts and Promotion
- **Status**: Correctly defined. The main claim is about promotion expenditures affecting ADR reporting.
- **Design code**: `panel_fixed_effects` is appropriate for Poisson with drug and time FE.
- **Design audit**: Present and accurate. Notes the Poisson estimator, exposure variable, and condition-specific estimation.
- **Data issue**: The key treatment variables (promotion expenditures: q1totalexp, q2q4totalexp, q1rprof_promo, q2q4rprof_promo, q1rtotal_dtca, q2q4rtotal_dtca, q1rcost_of_contact, q2q4rcost_of_contact) are NOT in the provided .dta file. Patient demographic shares ($char, $age) are also missing. Only drug identifiers, ADR counts, FDA labeling changes, and drug approval age dummies (Dappr_cats) are available.

### G2: FDA Labeling Changes
- **Status**: Correctly defined. This is a separate claim about ADR informativeness.
- **Design code**: `panel_fixed_effects` is acceptable for the logit model.
- **Data issue**: The v1-v4 interaction variables can be constructed from available data (veryserious x condition), but the control variables are still missing.

## Checklist Results

### A) Baseline groups
- Two baseline groups -- correct. ADR-promotion relationship (G1) and ADR-FDA labeling relationship (G2) are distinct claims.
- Duration models (countduration) are correctly excluded as a different estimand.

### B) Design selection
- `panel_fixed_effects` is appropriate for both Poisson (G1) and Logit (G2).
- Design variants appropriately include negative binomial, OLS log-rate, LPM, and probit alternatives.

### C) RC axes
- **Controls**: Appropriately structured given available variables. Block-based removal of control families.
- **Sample**: Drug exclusion (withdrawn drugs) is a meaningful robustness check.
- **Missing**: Cannot assess data construction axes since most variables are unavailable.

### D) Controls multiverse policy
- `controls_count_min=4` (G1) and `controls_count_max=17` -- reasonable given the control structure.
- No linked adjustment needed.

### E) Inference plan
- Robust sandwich SE is standard for Poisson and matches the paper.
- Drug-level clustering is a useful addition since the panel unit is drug-month.

### F) Budgets + sampling
- Budget of 55 (G1) + 25 (G2) = 80 total is reasonable given data constraints.
- Seed specified (114747).

### G) Diagnostics plan
- Overdispersion test is appropriate for Poisson.

## Key Constraints and Linkage Rules
- No bundled estimators.
- Condition-specific estimation means each condition has separate drug FE sets.
- The main constraint is data availability.

## What's Missing
- **Promotion expenditure variables**: Critical treatment variables are not in the provided data. This is a blocking issue for G1 execution.
- **Patient demographic controls**: Missing from the provided data.
- The surface correctly identifies these limitations.

## Final Assessment
**Conditionally approved.** The surface correctly identifies the paper's claim objects and specification structure. However, execution of G1 specifications is blocked by the absence of promotion expenditure variables from the provided dataset. G2 specifications may be partially executable since ADR counts and FDA labeling changes are available. The surface should be used as a template; actual execution will require the complete merged dataset.
