# Specification Surface: 181166-V1

## Paper: Technological Change and the Consequences of Job Loss (Braxton & Taska, 2022)

## Baseline Groups

### G1: Effect of occupation-level technological change on post-displacement earnings losses

- **Outcome**: d_ln_real_earn_win1 (change in log real weekly earnings after displacement, winsorized at 2.5th/97.5th percentile)
- **Treatment**: d_computer_2017_2007_n_s1 (change in computer and software requirements in job postings, 2007-2017, normalized to SD units)
- **Estimand**: Effect of occupation-level technological change on earnings losses among displaced workers
- **Population**: Displaced workers from CPS Displaced Worker Supplement (2010-2018 waves), age 25-65, with pre/post-displacement real weekly earnings >= $100, not top-coded
- **Baseline spec**: Table 3 Col 2 (reghdfe d_ln_real_earn_win1 d_computer_2017_2007_n_s1 + d_share_emp_n_s1 + i_computer_2007_n_s1 + i_male + controls [aw=dwsuppwt], vce(cluster dwsoc4) absorb(year year_job_loss))
- **Focal parameter**: Coefficient on d_computer_2017_2007_n_s1 (how a 1-SD increase in occupation-level tech change affects earnings losses)

### Additional baselines (same claim, different specifications)
- Table 3 Col 1: No employment share control (baseline__table3_col1_soc4)
- Table 3 Col 3: Full-time workers only (baseline__table3_col3_ft)
- Table 3 Col 4: Autor-Dorn occupation codes (baseline__table3_col4_AD)
- Table 3 Col 5: AD codes, full-time only (baseline__table3_col5_AD_ft)

## Design and Identification

The paper studies how occupation-level technological change affects the economic consequences of job loss. The key variation is cross-occupational: displaced workers from occupations that experienced larger increases in computer and software requirements (measured from Burning Glass job posting data, 2007-2017) suffer larger earnings losses.

This is a cross-sectional OLS design (not panel FE). Individual displaced workers are observed at one point in time (in the DWS interview year). The regressions absorb year FE and year-of-job-loss FE. Controls include individual demographics (age, education, gender), job characteristics (unemployment duration, tenure, full-time status), and occupation-level variables (initial computer requirements, change in employment share).

The treatment variable (d_computer_2017_2007) is constructed at the occupation level from Burning Glass vacancy data and varies across occupations but not within occupation. Standard errors are clustered at the occupation level.

## Core Universe

### Controls axes
- **LOO**: 9 specs (drop each control one at a time from the baseline set)
- **Standard sets**: 4 specs (none / demographics only / job characteristics only / full)
- **Progression**: 4 specs (bivariate / demographics / demographics + job / full)
- **Subset search**: 10 budgeted random draws (seed=181166)

### Sample axes
- Full-time workers only (both pre- and post-displacement)
- Age 25-44 only
- Age 45-65 only
- College graduates only (educ_num >= 16)
- Non-college only (educ_num < 16)
- Male only
- Winsorize at 1st/99th percentile (instead of 2.5th/97.5th)
- Winsorize at 5th/95th percentile

### Fixed effects axes
- Drop year-of-job-loss FE
- Drop year FE

### Data construction axes
- Unnormalized treatment variable (raw d_computer_2017_2007 instead of SD-normalized)
- Autor-Dorn occupation codes (alternative classification, cluster at dwocc1990dd)
- Alternative time window: d_computer_2017_2010 (2010-2017 change)

### Weights axes
- Unweighted regression (drop survey weights)

## Inference Plan
- **Canonical**: Cluster SEs at 4-digit SOC occupation level (dwsoc4), matching paper's vce(cluster dwsoc4)
- **Variant 1**: HC1 robust SEs (no occupation clustering)
- **Variant 2**: Cluster at 2-digit SOC level (coarser clustering)

## Constraints
- Control-count envelope: [1, 9]
- No linkage constraints (single-equation cross-sectional OLS)
- Treatment variable is occupation-level (assigned to individuals based on their displaced occupation)
- Sample restricted to: samp_1 == 1 (employed in both pre- and post-displacement jobs, age 25-65, real earnings >= $100, not top-coded, job loss year >= 2007)
- Outcome is winsorized by default at 2.5th/97.5th percentile; alternative winsorization is a robustness check

## Budget
- Max core specs: 75
- Max control subset specs: 10
- Total planned: ~55
- Seed: 181166

## What is excluded and why
- **Table 4 (occupation switching)**: Different outcome (i_occ_switch_4, binary). This is a complementary result about mechanisms, not the main earnings claim.
- **Table 5 (occupation switching to lower-tech occupations)**: Different outcome (i_occ_switch_4_lower). Mechanism analysis.
- **Table A1 (heterogeneity)**: Subsample splits by age, education, gender. These are included as sample restriction rc variants rather than as separate baseline groups, since the main claim is for the full population of displaced workers.
- **Quantitative model (MATLAB)**: Structural calibration and decomposition. Not an empirical regression.
- **SSA-ASEC earnings analysis (Appendix E)**: Uses confidential linked data not available in the replication package.
- **ACS analyses**: Used for constructing occupation-level statistics (employment shares, wages), not for the main displacement regression.
- **Atalay occupation code analysis**: Alternative occupation classification robustness from Appendix C.9. Could be included as an rc variant but requires different data loading pipeline.
- **Displacement rate analysis**: Different outcome (displacement probability by occupation). Not the earnings loss claim.
