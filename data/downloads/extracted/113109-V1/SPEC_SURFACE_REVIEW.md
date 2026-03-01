# Specification Surface Review: 113109-V1

## Summary

The surface was reviewed against the paper's Stata do-files (table1_emp_all.do, table3_educ.do, iv.do, ols.do, tableOA_robustness.do, tableOA_robustness2.do, and related files), data files, and the classification. This is Charles, Hurst, & Notowidigdo (2018), studying housing boom effects on employment and college enrollment using IV.

## Baseline Groups

### G1: Housing Boom Effect on Employment (Table 1)
- **Status**: Correctly defined. Primary claim (housing boom -> employment for young workers).
- **Design code**: `instrumental_variables` is correct.
- **Design audit**: Comprehensive. Notes 2SLS, Saiz instrument, state-level clustering, population/employment weights, and the MSA-level long-difference structure.
- **Linked adjustment**: Correctly set to `true` (controls in both stages).

### G2: Housing Boom Effect on College Enrollment (Table 3)
- **Status**: Correctly defined. Second main claim (housing boom -> reduced college enrollment).
- **Both any-college and bachelor's rates captured** as baseline specs -- correct.
- **Separate from G1**: Different outcome concept (education vs employment) warrants a separate baseline group.

### Assessment of baseline group choices
- Two groups for employment and education outcomes -- correct. These are both headline results in the paper.
- Table 2 (relative wages) is correctly excluded as supporting mechanism evidence.
- Housing bust period (2006-2009) is correctly excluded as a separate analysis.

## Checklist Results

### A) Baseline groups
- Two groups for two main outcome families -- correct.
- Could consider a third group for construction/FIRE employment specifically, but the total employment result is the headline finding and construction/FIRE are sectoral decompositions.

### B) Design selection
- `instrumental_variables` is correct for ivreg2 with Saiz elasticity as instrument.
- LIML design variant is included -- important for just-identified IV robustness.
- OLS as a comparison is not included in the core surface (different estimand) -- this is a judgment call. Including it as `explore/*` would be reasonable.

### C) RC axes
- **Controls**: Good progressive build-up from no controls to full model with region FE, matching the paper's robustness table (tableOA_robustness.do). LOO of each baseline control.
- **Instrument variations**: Critical axis for IV. iv_sig, iv_sig2, iv2_poly3, and price_rent_ratio from tableOA_robustness2.do. Excellent coverage of the paper's revealed instrument space.
- **Treatment definition**: deltaP only vs combined shock -- appropriate.
- **Sample**: Outlier trimming, extreme-elasticity exclusion, small-MSA exclusion.
- **Weights**: Weighted vs unweighted is an important check.
- No major high-leverage axes appear missing.

### D) Controls multiverse policy
- G1: `controls_count_min=0`, `controls_count_max=8` -- correct (0 for bivariate IV, 8 for full model with extras).
- G2: Same -- correct.
- `linked_adjustment=true` -- critical for IV and correctly set.
- Mandatory controls: none (bivariate IV is a valid specification) -- correct per the paper.

### E) Inference plan
- Canonical state-level clustering matches the paper -- correct.
- HC1 and HC2 variants are appropriate with ~50 state clusters.

### F) Budgets + sampling
- G1: 70 specs, G2: 50 specs -- reasonable given the instrument and control variation.
- Stratified-size sampling for 8-variable pool is appropriate.
- Seed specified (113109).

### G) Diagnostics plan
- First-stage F-statistic (Kleibergen-Paap) included for both groups -- correct and essential for IV.
- With a single instrument, no overidentification test is possible -- correctly omitted.

## Key Constraints and Linkage Rules
- **Linked adjustment**: Controls must be varied jointly across first and second stages.
- **Single instrument**: Just-identified, no overid tests. Instrument alternatives change the LATE but test robustness of the identification.
- **Instrument variations are not just robustness**: Different instruments (e.g., price-rent ratio vs Saiz elasticity) may identify effects for different subpopulations. This is correctly noted in the surface.

## What's Missing
- Could add `design/instrumental_variables/estimator/control_function` as a design variant (control function approach as alternative to 2SLS).
- Could add Anderson-Rubin confidence intervals as an inference variant (robust to weak instruments).
- The paper's gender-specific results could be additional baseline groups if gender heterogeneity is a headline claim, but the combined result is clearly the primary finding.

## Final Assessment
**Approved to run.** The surface correctly identifies two main claims (employment and college enrollment), includes critical IV-specific RC axes (instrument variations, LIML, first-stage diagnostics), and properly enforces linked adjustment. The coverage of the paper's robustness table instrument variations is excellent. No blocking issues.
