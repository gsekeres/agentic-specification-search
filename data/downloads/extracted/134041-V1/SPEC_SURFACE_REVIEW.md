# Specification Surface Review: 134041-V1

## Summary of Baseline Groups

- **G1**: Effect of information treatment on perceptions (z_mani_index ~ T1 + controls, pweight, HC1)
  - Well-defined claim object: ITT effect of information provision on gender wage gap perceptions
  - Baseline spec matches Table 5, Panel A, Column 6 (perception index)
  - Additional baselines: posterior, zposterior, large, problem, govmore (all Table 5 Panel A outcomes)
  - These are individual components of the perception index -- correctly treated as additional baselines of the same claim

- **G2**: Effect of information treatment on policy demand (z_lmpolicy_index ~ T1 + controls, pweight, HC1)
  - Well-defined claim object: ITT effect of information provision on demand for gender-equity policies
  - Baseline spec matches Table 5, Panel B, Column 7 (policy demand index)
  - Additional baselines: quotaanchor, AAanchor, legislationanchor, transparencyanchor, UKtool, childcare
  - Correctly separated from G1 since the paper frames perceptions and policy demand as distinct outcome families

## Changes Made

1. **Removed duplicate outcome specs from G1**: `rc/form/outcome/posterior_raw` and `rc/form/outcome/posterior_zscored` were redundant with `baseline__posterior` and `baseline__zposterior` already listed in baseline_spec_ids. Running the same outcome at baseline controls is already covered by the baseline specs.

2. **Added block-level LOO specs to both G1 and G2**: The original surface only listed individual-variable LOO drops. However, several controls form natural blocks (region dummies, age dummies, employment dummies) that should be dropped as blocks, not individually. Added:
   - `rc/controls/loo/drop_region_block` (drops midwest, south, west together)
   - `rc/controls/loo/drop_age_block` (drops age1, age2, age3, age4 together)
   - `rc/controls/loo/drop_employment_block` (drops fulltime, parttime, selfemp, unemp, student together)

3. **Added linkage notes to constraints for both groups**: Documented the wave-control linkage (when restricting to a single wave, the wave dummy is collinear and should be dropped), and the block structure of region/age/employment dummies.

## Key Constraints and Linkage Rules

- **No bundled estimator**: Single-equation OLS, no linked adjustment needed
- **Controls are for precision, not identification**: In a well-executed experiment, adding/dropping controls should not change the estimand. Control sensitivity tests precision rather than bias.
- **Wave-control linkage**: When restricting to wave_a_only or wave_b_only, the wave dummy must be dropped from the control set
- **Global controls set**: `$controls` is defined once in 01_Master_Replication.do (line 47): `wave gender prior democrat indep otherpol midwest south west age1 age2 age3 age4 anychildren loghhinc associatemore fulltime parttime selfemp unemp student` -- this matches the surface exactly (21 controls)
- **Probability weights**: pweight is used throughout the paper. The unweighted specification is a legitimate robustness check given individual randomization.
- **Pure control group exclusion**: The analysis sample drops rand==0 (pure control group), keeping only treated (T1=1) and information-only control (rand!=0, T1=0) respondents

## Budget/Sampling Assessment

- **G1**: ~48 core specs (9 LOO individual + 3 LOO block + 5 control sets + 5 progression + 10 random subsets + 3 sample + 1 weights + 2 design + 5 additional baselines + 1 baseline = ~44 unique) is within the 80-spec budget
- **G2**: ~42 core specs (9 LOO individual + 3 LOO block + 5 control sets + 5 progression + 5 random subsets + 3 sample + 1 weights + 2 design + 6 additional baselines + 1 baseline = ~41 unique) is within the 60-spec budget
- Seeds (134041, 134042) are reproducible
- Stratified-size sampler ensures coverage across control-count levels

## What's Missing (minor)

- **No strata FE spec**: The randomization was stratified by wave. The surface includes `design/randomized_experiment/estimator/with_covariates` and `diff_in_means` but not `strata_fe` (i.wave absorbed as FE rather than a dummy). This is minor since `wave` is already in the control set as a dummy, which is equivalent.
- **No prior-belief interaction (excluded correctly)**: Table 9 interacts T1 with perceived policy effectiveness, but this changes the estimand to a conditional treatment effect and is correctly excluded from the core surface.
- **No 2SLS / IV estimates (excluded correctly)**: Table 5 Panel C instruments zposterior with T1, but this changes the estimand from ITT to a causal mediation estimate.
- **Sharpened q-values**: The paper uses FDR-corrected q-values across outcome families. This is a post-processing step, not a specification choice, and is correctly omitted from the specification surface.

## Verification Against Code

Verified against:
- `01_Master_Replication.do` (line 47): global controls definition matches surface exactly
- `08_MainTables.do` (lines 530, 616): Table 5 regressions use `reg outcome T1 $controls [pweight=pweight], vce(r)` -- matches surface
- Table 5 Panel A outcomes: posterior, zposterior, large, problem, govmore, z_mani_index -- all in surface
- Table 5 Panel B outcomes: quotaanchor, AAanchor, legislationanchor, transparencyanchor, UKtool, childcare, z_lmpolicy_index -- all in surface
- Sample restriction: `drop if rand==0` in Table 5 code confirms pure control group exclusion

## Final Assessment

**APPROVED TO RUN.** The surface correctly separates the perception and policy demand outcome families into two baseline groups, the control list matches the code exactly, and the RC axes are well-chosen for a randomized experiment (controls for precision, sample restrictions by wave, weights sensitivity). The key correction was removing duplicate outcome forms that were already covered by additional baseline specs, and adding block-level LOO drops for the multi-variable control blocks.
