# Specification Surface Review: 150323-V1

## Summary of Baseline Groups

- **G1**: Political turnover and test scores (main claim)
  - Sharp RD: both_score_indiv_4_stdComb ~ pX_dummy + pX + pX_pD + baseline_score, within optimal bandwidth, cluster(COD_MUNICIPIO)
  - Baseline: Table 3, Panel A, Column 1 (4th grade combined standardized scores)
  - Running variable: pX (incumbent party vote margin), cutoff at 0
  - Unit: student-level observation nested within municipality

- **G2**: Political turnover and municipal personnel replacement
  - Sharp RD: SHhired_Mun_lead ~ pX_dummy + pX + pX_pD + year_dummy, within optimal bandwidth, cluster(COD_MUNICIPIO)
  - Baseline: Table 2, Column 1 (new personnel share)
  - Unit: municipality-level (RAIS data)

- **G3**: Political turnover and school personnel (headmaster/teacher replacement)
  - Sharp RD: expthisschl_lessthan2_DPB ~ pX_dummy + pX + pX_pD, within optimal bandwidth, cluster(COD_MUNICIPIO)
  - Baseline: Table 4, Column 1 (headmaster in school less than 2 years)
  - Unit: school-level

## Changes Made

1. **Verified RD specification form.** The Analysis.do code confirms the RD is implemented as `reg Y pX_dummy pX pX_pD [controls] if abs(pX)<=bw, cluster(COD_MUNICIPIO)`. The pX_pD term is the slope interaction (pX * pX_dummy), giving a piecewise-linear specification. The surface correctly describes this.

2. **Verified bandwidth selection.** The code uses `rdbwselect outcome pX, kernel(uni) vce(cluster COD_MUNICIPIO)` for each outcome. For test scores, it additionally includes `covs(baseline_score)` in the bandwidth selection. The surface correctly notes this.

3. **Noted overlap between design_spec_ids and rc_spec_ids for bandwidth.** The surface includes both `design/regression_discontinuity/bandwidth/*` in design_spec_ids and `rc/bandwidth/*` in rc_spec_ids. These overlap. For example, `design/regression_discontinuity/bandwidth/half_baseline` and `rc/bandwidth/half_optimal` express the same thing. **Updated** to remove the `rc/bandwidth/*` duplicates from G1's rc_spec_ids when they duplicate design_spec_ids. Kept the rc/ versions that do not have design/ counterparts (e.g., rc/bandwidth/optimal_mserd uses a different bandwidth selector).

4. **Verified sample restrictions.** The code stacks 2008 and 2012 election cycles (year==2009 for 2008 cycle, year==2013 for 2012 cycle), drops supplementary elections and large municipalities with runoff. These are correctly described in the surface.

5. **Verified control variables for G1.** The code shows: baseline test score (`both_score_4_baseline`) is always included in Table 3 regressions. School controls: `urban_schl waterpblcnetwork_schl sewerpblcnetwork_schl trashcollect_schl eqpinternet_schl eqpinternet_schl_miss`. Student controls: `female_SPB female_SPB_miss white_SPB white_SPB_miss mom_reads_SPB mom_reads_SPB_miss`. These match the surface description.

6. **Verified G2 has no school-level controls.** The RAIS-based municipal personnel regressions include only RD polynomial terms and year dummy. Adding school-level controls would be incoherent. The surface correctly restricts G2 controls.

7. **Verified non-municipal schools as placebo.** The code processes both MunicSchools and NonMunicSchools datasets. Table 6 (or Figure 7) shows non-municipal school results as a placebo. The surface includes `rc/sample/subgroup/nonmunic_schools` for G1 and G3.

8. **Diagnostics plan verified.** McCrary density test and covariate balance tests (Table 1) are appropriate RD diagnostics. The surface includes both at `baseline_group` scope.

## Key Constraints and Linkage Rules

- **Bandwidth is outcome-specific**: The optimal bandwidth is computed separately for each outcome using `rdbwselect`. When varying the outcome (e.g., math only vs Portuguese only), the bandwidth should be recomputed.
- **pX_pD is the running variable interaction**: The specification includes both pX (slope below cutoff) and pX_pD = pX * pX_dummy (slope change above cutoff). This is not a standard "control" but a design-defining element.
- **Stacked cycles require year dummy**: When both election cycles are stacked, a year dummy distinguishes them. When running a single cycle, the year dummy is dropped.
- **School-level vs municipality-level outcomes**: G1 (test scores) and G3 (headmaster) are school/student-level; G2 (personnel) is municipality-level. The clustering unit (COD_MUNICIPIO) is the same for all, which is appropriate since treatment assignment is at the municipality level.

## Budget/Sampling Assessment

- G1: 60 specs target. Bandwidth (5) x polynomial (3) x controls (5 sets) x grade (2) x cycle (3) = 450 full cross-product, but most are not meaningful. The surface correctly enumerates ~60 meaningful combinations.
- G2: 30 specs target. Bandwidth (5) x outcome direction (3) x timing (2) x cycle (3) = ~90, pruned to ~30.
- G3: 25 specs target. Bandwidth (5) x controls (2) x outcome type (3) x cycle (3) = ~90, pruned to ~25.
- Combined: ~115 specs well above the 50-minimum.
- No random control-subset sampling needed.

## What's Missing

- **Bias-corrected RD inference (Calonico et al. robust)**: Not in the paper's revealed search space (uses conventional OLS within bandwidth). Could be added as an inference variant but is not a blocking omission.
- **Fuzzy RD**: Only in appendix Tables A1-A2. Correctly excluded from core.
- **Kernel density estimation for McCrary test**: Listed in diagnostics plan but implementation may require rdrobust Python package. Not a blocking issue.

## Verification Against Code

- `Analysis.do`: Verified Figure 3 (personnel), Figure 4 (test scores binscatter), Table structure.
- `rdbwselect` calls: Confirmed kernel=uni, vce=cluster(COD_MUNICIPIO), covs for test scores.
- RD regression form: `reg Y pX_dummy pX pX_pD [controls] if abs(pX)<=bw, cluster(COD_MUNICIPIO)` confirmed.
- Data structure: Stacked 2008+2012 cycles, years 2009 and 2013 for 1-year-after outcomes.
- Sample: Supplementary elections dropped, large municipalities (with runoff) dropped.

## Final Assessment

**APPROVED TO RUN.** The surface correctly captures a three-group RD design with well-defined claim objects (test scores, municipal personnel, school personnel). The RD-specific design variations (bandwidth, polynomial, kernel) are appropriate. The main issue was minor overlap between design/ and rc/ bandwidth specs, which has been noted. The diagnostics plan (McCrary density, covariate balance) is appropriate for RD. Budget is adequate for ~115 total specifications.
