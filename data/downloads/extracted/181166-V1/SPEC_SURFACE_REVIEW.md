# Specification Surface Review: 181166-V1

## Summary of Baseline Groups

- **G1**: Effect of occupation-level technological change on post-displacement earnings losses (cross-sectional OLS)
  - Well-defined claim object: effect of a 1-SD increase in occupation-level computer/software requirement change on earnings losses among displaced workers
  - Baseline spec matches Table 3 Col 2 exactly: `reghdfe d_ln_real_earn_win1 d_computer_2017_2007_n_s1 d_share_emp_2017_2007_n_s1 i_computer_2007_n_s1 i_male ${controls} [aw=dwsuppwt] if samp_1 == 1, vce(cluster dwsoc4) absorb(year year_job_loss)` where `${controls} = ln_unemp_dur tenure_lost_job i_ft_current_job i_ft_lost_job educ_num age`
  - Additional baselines: Table 3 Col 1 (no employment share), Col 3 (full-time only), Col 4 (Autor-Dorn codes), Col 5 (AD + full-time)
  - Design code `cross_sectional_ols` is correct: cross-section of displaced workers with absorbed year/year-of-job-loss FE

## Changes Made

1. **Fixed `controls_count_min`**: Changed from 1 to 0 to be consistent with `rc/controls/sets/none` and `rc/controls/progression/bivariate` which include no controls beyond the treatment variable. The paper's minimum reported specification (Table 3 Col 1) has 8 non-treatment regressors, but the specification search should be able to test the bivariate case.

2. **Added normalization note**: The treatment variable and occupation-level controls are normalized to SD units within the analysis sample (samp_1). For the full-time subsample specs, normalization is recomputed within the full-time sample (using `_n_s1_ft` suffix variables). The runner must replicate this sample-specific normalization.

3. **Added AD occupation code note detail**: AD regressions use different variable names (d_computer_2017_2007_AD_n_s1, etc.), cluster at `dwocc1990dd` instead of `dwsoc4`, and re-normalize within the analysis sample.

4. **Added winsorization note**: Outcome is winsorized at 2.5th/97.5th percentile within samp_1 using Stata's `winsor` command. Alternative winsorization rc specs should replicate within-sample winsorization.

## Key Constraints and Linkage Rules

- No bundled estimator: single-equation weighted OLS (reghdfe with absorbed FE)
- Cluster at 4-digit SOC occupation level (dwsoc4) matches paper's `vce(cluster dwsoc4)`
- Analytical weights: `dwsuppwt` (DWS supplement sampling weights) -- always included unless explicitly testing unweighted
- The sample restriction `samp_1 == 1` is computed from the intersection of two initial regressions (one with SOC-4 codes, one with AD codes) plus age/earnings restrictions; the runner must reconstruct this sample flag
- Normalization is sample-specific: the runner must compute mean/SD within the analysis sample and normalize before regression. This is a data construction step, not just a regression option.
- For AD occupation code specs, clustering variable changes to `dwocc1990dd` and all occupation-level variables are recomputed -- this is a joint data/inference change

## Budget/Sampling Assessment

- ~55 planned core specs within the 75-spec budget -- feasible
- 10 random control subset draws with seed=181166 is reproducible
- 9 LOO specs cover all non-treatment controls from Table 3 Col 2 (including the occupation-level controls d_share_emp and i_computer_2007)
- 4 control progression specs provide informative build-up
- 8 sample restriction specs cover a wide range of subsample analyses (full-time, age splits, education, gender) that correspond to the paper's Table A1 heterogeneity analysis
- 2 FE drop specs and 3 data construction variants round out the core universe

## What's Missing (minor)

- No `rc/data/treatment/d_cognitive_2017_2007` variant: the code creates a winsorized cognitive change variable (`d_cognitive_2017_2007_win1`), suggesting the paper also considers cognitive skill requirements as an alternative measure. This could be a useful rc/data variant but is not present in the reported tables.
- The Atalay occupation classification (Appendix C.9) is excluded from the surface, which is reasonable given it requires a different data loading pipeline.
- No interaction between treatment and occupation switching (`i_occ_switch_4`) is included as an explore variant -- this is the Table 4/5 mechanism analysis and is correctly excluded from the main claim.
- The `rc/data/treatment/d_computer_2017_2010` variant (2010-2017 window instead of 2007-2017) is confirmed in the code -- this is used for model calibration quintiles but could also serve as a robustness check on the measurement window.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's Table 3 progression and Table A1 heterogeneity analysis, and the budget is feasible. The normalization and winsorization requirements are now clearly documented for the runner.
