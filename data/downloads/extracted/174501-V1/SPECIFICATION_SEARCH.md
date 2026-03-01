# Specification Search Log: 174501-V1

## Paper
Corno, La Ferrara & Burns (2019/2022). "Interaction, Stereotypes and Performance: Evidence from South Africa."

## Surface Summary

| Property | Value |
|----------|-------|
| Paper ID | 174501-V1 |
| Design | randomized_experiment (natural roommate assignment at UCT) |
| Baseline groups | 3 (G1: Race IAT, G2: Academic outcomes, G3: Social outcomes) |
| Budget G1 | 60 core, 30 control-subset |
| Budget G2 | 80 core, 40 control-subset |
| Budget G3 | 60 core, 30 control-subset |
| Sampling seed | 174501 |
| Sampler | full_enumeration |

## Data

- Source: `Data/Clean/uctdata_balanced.dta` (balanced panel, 499 individuals x 2 rounds)
- Follow-up sample (round=="Follow-up"): 499 observations
- White subsample: 117, Black subsample: 332
- Lagged outcomes (L.DscoreraceIAT, L.DscoreacaIAT) constructed from baseline round
- `Female` converted from categorical ("male"/"female") to numeric dummy
- `continue` renamed to `continue_var` to avoid Python keyword conflict
- Note: Data variable `roconsbas_missing` (not `roconsbas_miss` as in surface spec; surface variable name adjusted)

## Execution Summary

### Counts

| Group | Baselines | Design | RC | Total | Success | Failed |
|-------|-----------|--------|-----|-------|---------|--------|
| G1 | 1 | 4 | 14 | 19 | 19 | 0 |
| G2 | 3 | 12 | 32 | 47 | 47 | 0 |
| G3 | 3 | 12 | 27 | 42 | 42 | 0 |
| **Total** | **7** | **28** | **73** | **108** | **108** | **0** |

### Inference variants: 6 (2 per group: HC1 robust, cluster at residence level)

## Baseline Results

| spec_id | Group | Outcome | Coef | SE | p-value | N |
|---------|-------|---------|------|----|---------|---|
| baseline__table3_col2_black_raceiat | G1 | DscoreraceIAT | -0.0944 | 0.0693 | 0.174 | 332 |
| baseline__table4_black_examspassed | G2 | examspassed | 0.6445 | 0.2422 | 0.008 | 324 |
| baseline__table4_black_continue | G2 | continue | 0.1515 | 0.0391 | 0.000 | 324 |
| baseline__table4_black_pcaperf | G2 | PCAperf | 0.4433 | 0.1391 | 0.002 | 324 |
| baseline__table5_white_pcaattitude | G3 | PCAattitude | 0.6703 | 0.2605 | 0.012 | 106 |
| baseline__table5_white_pcacomm | G3 | PCAcomm | 0.4377 | 0.2502 | 0.084 | 94 |
| baseline__table5_white_pcasocial | G3 | PCAsocial | 0.7603 | 0.2937 | 0.012 | 79 |

## Specs Executed

### G1: Race IAT (DscoreraceIAT) on Black subsample

- **Baseline**: Table 3 Col 2 -- mixracebas on DscoreraceIAT, Black students, with lagged IAT + own + roommate controls, Res_base FE, cluster SE at room
- **Design variants** (4): diff_in_means, ancova (lagged outcome only), with_covariates (no FE), strata_fe (FE + lagged outcome)
- **RC controls** (12): drop_roommate_controls; LOO over 6 own control groups (Female, Falseuct2012, Foreign, privateschool_nomiss, durpcabas_nomiss, consbas_nomiss) and 5 roommate control groups (roFalseuct2012, roForeign_bas, roprivschool_nomiss, rodurpcabas_nomiss, roconsbas_nomiss)
- **RC sample** (2): full_sample_with_race_fe, restrict_white_black_only

### G2: Academic Outcomes on Black subsample

- **Baselines** (3): examspassed, continue, PCAperf (Table 4 Black subsample, Res_base + regprogram FE)
- **Design variants** (12): diff_in_means, with_covariates, strata_fe -- for each of 4 outcomes (GPA, examspassed, continue, PCAperf)
- **RC controls** (15): drop_roommate_controls (4 outcomes); LOO over 6 own + 5 roommate control groups (GPA as representative)
- **RC sample** (8): full_sample_with_race_fe (4 outcomes), restrict_white_black_only (4 outcomes)
- **RC FE** (4): drop_program_fe (4 outcomes)
- **RC form** (5): pca_performance_index; second_year_outcomes (GPA2013, examspassed2013, continue2013, PCAperf2013)

### G3: Social Outcomes on White subsample

- **Baselines** (3): PCAattitude, PCAcomm, PCAsocial (Table 5 White subsample, Res_base FE)
- **Design variants** (12): diff_in_means, with_covariates, strata_fe -- for each of 4 outcomes (PCAfriend, PCAattitude, PCAcomm, PCAsocial)
- **RC controls** (15): drop_roommate_controls (4 outcomes); LOO over 6 own + 5 roommate control groups (PCAfriend as representative)
- **RC sample** (8): full_sample_with_race_fe (4 outcomes), restrict_black_only (4 outcomes)
- **RC form** (4): nomiss_pca_indices (4 outcomes using *_nomiss1 variables from Table A15)

## Deviations from Surface

1. **GPA not included as baseline for G2**: The surface lists GPA in `baseline_specs` but not in `baseline_spec_ids`. GPA is still covered via design variants (diff_in_means, with_covariates, strata_fe) and RC specs (drop_roommate_controls, full_sample_with_race_fe, restrict_white_black_only, drop_program_fe).

2. **PCAfriend not included as baseline for G3**: Same as above -- PCAfriend is in `baseline_specs` but not `baseline_spec_ids`. Still covered via design and RC variants.

3. **Variable name mismatch**: Surface specifies `roconsbas_miss` but the actual data variable is `roconsbas_missing`. Script uses the actual data variable name.

4. **PCA nomiss variable names**: Surface references `PCAfriend_nomiss`, `PCAatt_nomiss`, etc. Actual data variables are `PCAfriend_nomiss1`, `PCAatt_nomiss1`, `PCAcomm_nomiss1`, `PCAsocial_nomiss1` (with "1" suffix).

5. **Multicollinearity**: In some regressions (particularly Black subsample with full controls), pyfixest drops `foreign_missing` and `privateschool_miss` due to multicollinearity with the residence FE. This matches Stata behavior where these variables are collinear with the FE structure.

## Software Stack

| Package | Version |
|---------|---------|
| Python | 3.12.7 |
| pyfixest | 0.40.1 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| statsmodels | 0.14.6 |
| scipy | 1.15.1 |

## Outputs

- `specification_results.csv`: 108 rows (7 baseline + 28 design + 73 RC), all run_success=1
- `inference_results.csv`: 6 rows (2 per group: HC1, cluster at residence)
- `scripts/paper_analyses/174501-V1.py`: executable analysis script
