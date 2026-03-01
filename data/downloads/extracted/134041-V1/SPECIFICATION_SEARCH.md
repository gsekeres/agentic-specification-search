# Specification Search Log: 134041-V1

## Paper
"How Do Beliefs About the Gender Wage Gap Affect the Demand for Public Policy?" by Sonja Settele (AEJ: Economic Policy)

## Surface Summary

- **Paper ID**: 134041-V1
- **Design**: Randomized experiment (survey experiment, ITT)
- **Baseline groups**: 2
  - **G1**: Effect of information treatment (T1) on perception index (`z_mani_index`), Table 5 Panel A Col 6
  - **G2**: Effect of information treatment (T1) on policy demand index (`z_lmpolicy_index`), Table 5 Panel B Col 7
- **Budgets**: G1 max 80, G2 max 60
- **Seeds**: G1=134041, G2=134042
- **Sampler**: stratified_size

## Execution Summary

### Counts

| Category | G1 | G2 | Total |
|---|---|---|---|
| Baseline (primary) | 1 | 1 | 2 |
| Additional baselines | 5 | 6 | 11 |
| Design variants | 2 | 2 | 4 |
| RC/controls/loo (singles) | 9 | 9 | 18 |
| RC/controls/loo (blocks) | 3 | 3 | 6 |
| RC/controls/sets | 5 | 5 | 10 |
| RC/controls/progression | 5 | 5 | 10 |
| RC/controls/subset (random) | 10 | 5 | 15 |
| RC/sample/restriction | 3 | 3 | 6 |
| RC/weights | 1 | 1 | 2 |
| **Total** | **44** | **40** | **84** |

### Status

- **Planned**: 84
- **Executed**: 84
- **Succeeded**: 84
- **Failed**: 0

### Inference Variants

- 4 inference variant rows written to `inference_results.csv`
  - IID (classical) SEs for G1 baseline and G2 baseline
  - HC3 robust SEs for G1 baseline and G2 baseline

## Data Cleaning Notes

The package ships raw Stata `.dta` files only -- the final cleaned dataset (`SurveyStageI_AB_final.dta`) is not included. The analysis script replicates the full Stata cleaning pipeline (do-files 03, 04, and 05) in Python:

1. **Wave A cleaning** (03): Convert string-coded survey responses (e.g., "A1" -> 1) for employment, region, age, gender, political orientation, prior beliefs, posterior beliefs, manipulation check outcomes, policy preferences, children, education, household income.
2. **Wave B cleaning** (04): Same as Wave A, plus handles Wave B-specific variables (`UKtool` replaces `transparencyanchor` in Wave B, `extrasame`/`extrachild` for posterior beliefs).
3. **Append and index construction** (05):
   - Append Wave A and B
   - Create probability weights correcting for oversampled age-gender cells in Wave B
   - Z-score all outcome variables using control group (rand==0) mean and SD
   - Build inverse-covariance-weighted indices (Anderson 2008):
     - `z_mani_index` from {large, problem, govmore}
     - `z_lmpolicy_index` from {quotaanchor, AAanchor, legislationanchor, transparencyanchor/UKtool, childcare}
   - Generate interaction terms (T1female, T1democrat, T1indep)
   - Z-score posterior beliefs (zposterior) using treatment-group weighted mean/SD

**Sample**: Treatment sample only (rand != 0), N=3,031 for most specifications.

## Key Baseline Results

### G1: Perception Index
- `z_mani_index ~ T1 + 21 controls [pweight], HC1`: coef=0.4170, se=0.0322, p<0.001, N=3031
- The information treatment strongly shifts perceptions about the gender wage gap.

### G2: Policy Demand Index
- `z_lmpolicy_index ~ T1 + 21 controls [pweight], HC1`: coef=0.0563, se=0.0251, p=0.025, N=3031
- The treatment also increases demand for gender-equity policies, but the effect is smaller and marginally significant.

## Robustness Summary

### G1 (Perceptions)
- All 44 specifications are highly significant (p < 0.001 in all cases).
- Coefficients range from 0.399 (Wave A only) to 0.460 (Wave B only, incentivized prior only).
- Result is highly robust to control set, sample restriction, and weighting choices.

### G2 (Policy Demand)
- Significance varies: most full-sample specs are significant at p < 0.05, but single-wave subsamples (Wave A only, Wave B only) lose significance (p = 0.10-0.12).
- Coefficients range from 0.040 to 0.075.
- The result is directionally robust but less precisely estimated in subsamples.

## Deviations from Surface

1. **G2 additional baseline `baseline__transparencyanchor`**: The `transparencyanchor` variable was replaced by `UKtool` in Wave B (different survey question). For `baseline__transparencyanchor`, I restricted the sample to Wave A only and dropped the wave dummy. This is consistent with the paper's handling.

2. **No `rc/form/*` specifications**: The surface mentioned raw posterior belief and z-scored posterior as alternative outcomes in the spec surface markdown, but these were not included in the JSON core_universe's `rc_spec_ids`. They are already captured as additional baseline specs (`baseline__posterior`, `baseline__zposterior`).

## Software Stack

- Python 3.x
- pyfixest (HC1 robust SEs matching Stata `vce(r)`)
- pandas, numpy
- No Stata required (full cleaning pipeline replicated in Python)
