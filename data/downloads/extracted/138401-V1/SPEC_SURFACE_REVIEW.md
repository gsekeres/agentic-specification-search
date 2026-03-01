# Specification Surface Review: 138401-V1

## Summary of Baseline Groups

- **G1**: Effect of measles vaccine exposure on adult labor market outcomes (ln_cpi_income ~ M12_exp_rate + FE, cluster(bplcohort))
  - Well-defined claim object: continuous DiD estimating the effect of childhood measles vaccine exposure on adult outcomes
  - Treatment variable: M12_exp_rate = (avg 12-year pre-vaccine measles incidence rate) x (years of exposure) / 100,000
  - Baseline spec matches Table 2, Column 3 (log CPI-adjusted income) exactly
  - Additional baselines: cpi_incwage, cpi_incwage_no0, poverty100, employed, hrs_worked (all Table 2 outcomes)
  - Rich FE structure: bpl, birthyr, year, ageblackfemale, bpl_black, bpl_female, bpl_black_female
  - Correctly treated as one baseline group since all outcomes test the same underlying claim

## Changes Made

1. **Removed `rc/form/outcome/asinh_income`**: The inverse hyperbolic sine transformation is NOT present in the paper's Stata code. The paper uses level income (cpi_incwage), level income excluding zeros (cpi_incwage_no0), and log income (ln_cpi_income). The asinh transformation would be an analyst addition with no basis in the revealed search space.

2. **Removed `rc/data/treatment_construction/M10_exp_rate_binary_exposure`**: This spec (using M10 with binary exposure rather than continuous 0-16 exposure) is not clearly in the paper's revealed search space. Appendix Table 2 uses M2-M12 with continuous exposure throughout. A binary exposure recoding would change the treatment concept fundamentally.

3. **Added two missing inference variants from Appendix Table 3**:
   - `infer/se/cluster/stateexposure`: Cluster at state-of-residence x exposure level. Present in AppendixTable3.do code (`egen stateexposure=group(statefip exposure)`)
   - `infer/se/cluster/statecohort`: Cluster at state-of-residence x birth-year. Present in AppendixTable3.do code (`egen statecohort=group(statefip birthyr)`)

   The original surface was missing these two clustering variants despite them being explicitly in the paper's Appendix Table 3 code.

4. **Added detailed linkage notes to constraints**: Documented the critical FE-sample linkages:
   - Core DiD FE (bpl, birthyr, year) are design-critical and must never be dropped
   - Gender-specific subsamples require dropping bpl_female and bpl_black_female FE
   - Race-specific subsamples require dropping bpl_black and bpl_black_female FE
   - Joint specs handle these linkages explicitly

## Key Constraints and Linkage Rules

- **Design-critical FE**: Birth state (bpl), birth year (birthyr), and ACS year (year) FE are required for the DiD identification strategy. They must never be dropped independently.
- **Demographic interaction FE linkage**: The FE structure includes i.bpl_black, i.bpl_female, i.bpl_black_female, and i.ageblackfemale. These interact demographics with geography/age. When restricting to a single race or gender, the corresponding interaction FE become collinear and must be dropped.
- **Joint specs**: The `rc/joint/*` specifications correctly handle the sample+FE linkage for race/gender subsamples (e.g., white_men_only drops black, female, and all demographic interaction FE).
- **Treatment construction variants**: M2 through M11 use different pre-vaccine averaging windows (2-year through 11-year). These are data construction choices that preserve the same estimand but change the treatment measure. All are from Appendix Table 2.
- **Minimal control set**: Only 2 non-FE controls (black, female), with most variation absorbed by the rich FE structure. LOO fully covers this space.

## Budget/Sampling Assessment

- 47 planned core specs (after removing asinh_income and binary_exposure) is within the 80-spec budget
- No random control subset sampling needed (only 2 controls, LOO covers the space)
- Full enumeration is feasible for all RC axes
- The FE/add and FE/drop axes are well-grounded in the paper's Tables 3-4 and Appendix Table 4
- Treatment construction variants (M2-M11) from Appendix Table 2 provide 10 additional specs
- Joint specs (6) correctly pair sample restrictions with appropriate FE adjustments

## Inference Plan Assessment

The inference plan is unusually rich, reflecting the paper's Appendix Table 3 which systematically tests 11+ clustering schemes. After the addition of stateexposure and statecohort, the plan now includes all clustering variants from the Stata code:
- Canonical: bplcohort (birth state x birth year)
- Variants: bpl, bplexposure, bpl_region4 (4 clusters -- flagged as questionable), bpl_region9, stateexposure, statecohort, statefip, birthyr, exposure, HC1

This is 10 inference variants, which is large but justified by the paper's revealed search space.

## What's Missing (minor)

- **No Callaway-Sant'Anna or Sun-Abraham estimator**: The staggered DiD literature suggests these as robustness to TWFE heterogeneity bias. However, this paper uses a continuous treatment intensity design (not staggered binary adoption), so the standard staggered-DiD concerns are less directly applicable. The paper's own robustness (mean reversion controls, region x cohort FE) addresses the relevant threats.
- **No wild cluster bootstrap**: Given the relatively large number of clusters (bplcohort has ~2000 clusters), wild bootstrap is unnecessary for the canonical inference. For the few-cluster variants (bpl_region4 with 4 clusters), wild bootstrap would be appropriate but is outside the core surface.
- **Employed-only restriction**: Added as `rc/sample/restriction/employed_only` which is important for wage/hours outcomes (zeros may confound the income effect). This was already in the surface.

## Verification Against Code

Verified against:
- `Table2.do`: `reg outcome M12_exp_rate controls i.year, robust cluster(bplcohort)` with `local controls i.bpl i.birthyr i.ageblackfemale i.bpl_black i.bpl_female i.bpl_black_female black female` -- matches surface exactly
- `Table3.do`: exclude_partial_exposure (`exposure==0 | exposure==16`), narrow_cohort_window (`birthyr<1972`) -- both in surface
- `Table4.do`: breg9_byear FE, mean_reversion_control -- both in surface
- `AppendixTable2.do`: M2 through M12 treatment constructions -- M2-M11 in surface (M12 is baseline)
- `AppendixTable3.do`: 11 clustering variants including stateexposure and statecohort -- now all in surface
- `AppendixTable4.do`: breg4_byear, breg9_byear, bpl_cohort_trend -- all in surface
- `acs_cleaning.do`: Confirms sample restrictions (age 25-59, native-born, black/white only), variable construction (exposure 0-16, ln_cpi_income, poverty100, employed, hrs_worked, cpi_incwage_no0)
- `rates.do`: Confirms treatment construction (avg_Xyr_measles_rate from measles rates 1952-1963, M_exp_rate = avg * exposure / 100000)

## Final Assessment

**APPROVED TO RUN.** The surface is well-structured for this continuous DiD design, faithful to the paper's revealed search space (Tables 2-4, Appendix Tables 2-4), and the budget is feasible. The key corrections were: (1) removing an analyst-added outcome transformation (asinh) not in the code, (2) removing a binary-exposure recoding not in the revealed search space, and (3) adding two missing clustering variants (stateexposure, statecohort) that were present in the paper's Appendix Table 3 Stata code. The linkage constraints for demographic-subsample FE adjustments are critical and are correctly handled by the joint specs.
