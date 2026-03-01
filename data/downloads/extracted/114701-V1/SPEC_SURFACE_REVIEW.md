# Specification Surface Review: 114701-V1

## Summary

The surface was reviewed against the paper's do-files (table3.do, table4.do, ks1variables.do). This paper studies literacy interventions using DiD with school FE. Data is confidential (National Pupil Database).

## Baseline Groups

### G1: EDRp Program Effects
- **Status**: Correctly defined. The EDRp pilot is the primary intervention studied.
- **Design code**: `difference_in_differences` is correct for treatment-by-cohort interactions with school FE.
- **Design audit**: Present and accurate. Captures the areg structure, school FE, and clustering.
- Multiple test stages (FSP, KS1, KS2) are appropriately included as baseline-like rows within one group, since they represent the same treatment effect measured at different developmental stages.

### G2: CLLD Program Effects
- **Status**: Correctly defined as a separate baseline group. CLLD is a different program from EDRp, though with similar structure.
- Budget appropriately smaller.

## Checklist Results

### A) Baseline groups
- Two baseline groups for two programs (EDRp, CLLD) -- correct. They use different treatment schools and different cohort windows.
- Subgroup heterogeneity (Table 4: by EMT/EAL and FSM) is correctly excluded from core as explore/*.
- Tables 5-8 are correctly excluded as separate program comparisons.

### B) Design selection
- `difference_in_differences` with areg (school FE) is correct.
- Design variants appropriately include pilot-only vs phase1-only analysis.

### C) RC axes
- **Controls**: Well-structured with block-based organization. 30 controls naturally group into 4 blocks (core demographics, ethnicity, language, school means).
- **Sample**: Subgroup restrictions (EMT, EAL, FSM) are informative given the heterogeneity results in Table 4.
- **Missing**: No functional form variants, which is appropriate since outcomes are already standardized.

### D) Controls multiverse policy
- `controls_count_min=15` (individual only) and `controls_count_max=30` (all) -- reasonable. The paper uses the same 30 controls throughout, so the envelope reflects removing the school-mean block entirely as the minimum.
- No linked adjustment needed.

### E) Inference plan
- Canonical clustering at school matches all paper specifications.
- Two-way school-year clustering is a useful robustness check.

### F) Budgets + sampling
- Budget of 70 (G1) + 35 (G2) = 105 total is reasonable.
- Block-based sampling is appropriate for the 30-variable control set.

### G) Diagnostics plan
- Pre-treatment cohort comparison for parallel trends is standard for DiD.

## Key Constraints and Linkage Rules
- No bundled estimators.
- Control set is fixed in the paper; variation comes from subsetting.
- School FE absorbs all time-invariant school characteristics.

## What's Missing
- Nothing material. Data is confidential, limiting executable specifications.

## Final Assessment
**Approved to run (conditional on data access).** The surface correctly identifies both program interventions, handles the large control set with block-based organization, and captures the paper's own robustness dimensions. Data is confidential and not included in the package.
