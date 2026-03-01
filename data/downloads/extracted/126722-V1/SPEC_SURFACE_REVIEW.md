# Specification Surface Review: 126722-V1

## Summary

Paper: Lopez, Sautmann, Schaner - "Does Patient Demand Contribute to the Overuse of Prescription Drugs?" (AEJ Applied). Field RCT in Mali health clinics with patient voucher and doctor voucher treatment arms.

## Baseline Groups Assessment

### G1: Voucher effects on malaria treatment outcomes
- **Claim object**: Well-defined. The paper's central question is whether patient demand (proxied by patient voucher) vs doctor supply-side incentives (doctor voucher) contribute to antimalarial overuse.
- **Outcome multiplicity**: The paper has 5 outcomes in Table 3. All are main-table outcomes, with `RXtreat_sev_simple_mal` (prescribed any antimalarial) as the lead specification. The surface correctly treats this as one baseline group with multiple outcome columns rather than multiple baseline groups, since the claim object (voucher effects on treatment) is unified.
- **Design code**: `randomized_experiment` -- correct. Within-clinic random assignment at the day level.

## Key Issues Identified and Resolutions

### A) pdslasso approximation
The paper's main specification uses pdslasso (double-selection LASSO) to select controls from 272 candidate variables. Since pdslasso is unavailable in Python, the surface correctly identifies Table B10's manual covariate approach as the baseline replication strategy. This is a faithful approximation because:
1. Table B10 is an official robustness table in the paper
2. The manual covariate set is a reasonable subset of the LASSO candidate pool
3. The "no controls" specification (Table B8) is included as a design variant

### B) Multiple treatment variables
The paper regresses each outcome on both `patient_voucher` and `doctor_voucher` simultaneously. The focal coefficient is on `patient_voucher` (the parameter most directly testing patient-driven demand). The surface should record both treatment variables in each regression but track `patient_voucher` as the focal parameter for the specification curve. This is correctly handled.

### C) Clustering level
Clustering at `cscomnum_OS` (clinic level, ~60 clusters) matches the paper exactly. The randomization unit is clinic-day, but treatment assignment is correlated within clinics, making clinic-level clustering appropriate.

### D) Design audit completeness
The design_audit block includes randomization unit, FE structure, cluster vars, and a note about the pdslasso limitation. This is adequate.

### E) Controls universe
- LOO candidates: The 13 individual covariates from the manual control set are appropriate
- Progression: From bivariate (date FE only) to full controls -- captures the paper's own robustness
- Random subsets: 15 draws with seed 126722 is reproducible and informative
- The control count envelope [0, 20] is correct (from no controls to full manual set)

### F) Missing axes considered
- **Functional form**: Not applicable -- all outcomes are binary (0/1). No log/asinh transforms make sense. Correctly excluded.
- **Weights**: Paper does not use weights. Correctly excluded.
- **Data construction**: No recoding alternatives identified. The `dropme` filter is the only sample restriction.
- **Home survey subsample**: Included as `rc/sample/restriction/home_survey` -- good addition since some outcomes are only available for the home survey subsample.

## Budget Assessment
- ~65 specifications across 5 outcomes is achievable and informative
- For each outcome: 1 baseline + 2-3 design + ~10 RC = ~14 specs per outcome
- Total: ~70 specs -- within the 80-spec budget

## Inference Plan
- Canonical: CRV1 at clinic level -- matches paper
- Variant: HC1 robust -- reasonable stress test given 60 clusters

## Approved to Run
**APPROVED**. The surface is conceptually coherent, faithful to the paper's revealed search space, and the pdslasso limitation is transparently documented with a sensible fallback. No blocking issues.
