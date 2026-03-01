# Specification Surface Review: 116531-V1

## Summary of Baseline Groups
- **G1**: LATE of loan offer on borrowing (IV with random assignment as instrument)
  - Well-defined claim object: effect of receiving a loan offer on take-up
  - Two baseline outcomes (borrowed indicator, accepted amount) measure the same borrowing concept
  - IV structure is appropriate: random assignment instruments for actual loan offer with noncompliance
- **G2**: ITT effect of loan assignment on educational attainment
  - Separate claim object: reduced-form effect on education outcomes
  - Four outcomes (credits attempted, credits earned, GPA, degree) are different measures of attainment
  - Correctly separate from G1 because the estimand and outcome concept differ

## Changes Made
1. No changes to baseline group definitions -- both are well-specified.
2. Verified that G1 uses linked adjustment (controls partialled out in both stages of IV) -- correctly flagged.
3. Confirmed that the stratum-level clustering is appropriate given the stratified randomization design.
4. Verified the sample restriction (enrolled_fall==1 for G2) matches the paper's attainment analysis sample.

## Key Constraints and Linkage Rules
- G1 is a bundled estimator (IV): controls must be varied jointly across stages (linked_adjustment=true)
- G2 is single-equation OLS (ITT): no linkage constraint
- Strata FE are the natural FE for this stratified design; dropping them is a valid robustness check since randomization is valid within strata
- Controls are pre-treatment and serve precision purposes, not identification -- randomization ensures internal validity

## Budget/Sampling Assessment
- G1: ~40 specs planned within 80-spec budget; 10 random control subsets is sufficient
- G2: ~30 specs planned within 50-spec budget
- Combined total ~70 specs is feasible and informative
- LOO covers 7 covariates per group -- adequate for sensitivity analysis

## What's Missing (minor)
- No wild cluster bootstrap variant (small number of clusters ~46 strata) -- could be added but wildboottest not available
- No Lee bounds (correctly excluded as sensitivity, not core)
- No LATE for attainment outcomes (paper does report these in Table 7 Panels C-E) -- could add as additional baselines for G2 but would require IV structure

## Final Assessment
**APPROVED TO RUN.** The surface correctly separates borrowing (G1/IV) from attainment (G2/ITT) as distinct claim objects. The stratified RCT design is well-documented, constraints are appropriate, and budgets are feasible.
