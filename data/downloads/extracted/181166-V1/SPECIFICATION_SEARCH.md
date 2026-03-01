# Specification Search: 181166-V1
## Braxton & Taska (2023) "Technological Change and the Consequences of Job Loss"

### Surface Summary
- **Paper ID**: 181166-V1
- **Surface hash**: sha256:b42784177274527e863087e42ff07b54bce4faec96949b3dfcec4e562578da8d
- **Design**: Cross-sectional OLS with absorbed FE
- **Baseline groups**: 1 (G1)
  - d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + controls | year + year_job_loss
  - cluster(dwsoc4), weighted by dwsuppwt
  - Budget: 75 core specs, 10 control subsets
- **Seed**: 181166

### Data Construction
- Built from CPS Displaced Worker Supplement raw fixed-width data (cps_00065.dat)
- Merged with Burning Glass occupation-level data (occ_req_all_years_full_samp.dta)
- CPI deflation via cpi_year_current.dta and cpi_year_job_loss.dta
- **Note**: Only 2012-2018 DWS waves used (2010 wave excluded due to occupation code crosswalk
  complexity requiring Stata value labels). This reduces sample size compared to the paper
  but preserves the core analysis structure.
- Occupation mapping: Census 2010 occupation codes mapped directly to SOC-4 codes.
- Winsorization at 2.5th/97.5th percentile (matching paper's winsor command).
- Normalization to SD units within analysis sample (weighted).

### Execution Summary
- **Total specifications executed**: 43
  - Successful: 43
  - Failed: 0
- **Inference variants**: 2

### Specifications Run
- **Baseline**: Table 3 Col 2 (normalized, with employment share control)
- **Additional baselines**: Col 1 (no emp share), Col 3 (full-time only)
  - Cols 4-5 (Autor-Dorn occ codes) skipped: requires full AD data pipeline not available
    without Stata
- **LOO controls**: Drop each of 9 controls individually
- **Control sets**: None, demographics only, job chars only, full
- **Control progression**: Bivariate, demographics, demographics+job, full
- **Random control subsets**: 10 random draws (seed=181166)
- **Sample restrictions**: Full-time only, age 25-44, age 45-65, college, no college, male only
- **Winsorization variants**: 1-99, 5-95 (in addition to baseline 2.5-97.5)
- **FE variants**: Drop year_job_loss, drop year
- **Treatment variants**: Unnormalized, 2017-2010 change
- **Weight variant**: Unweighted
- **Inference variants**: HC1 (no clustering), cluster at 2-digit SOC

### Deviations from Surface
- Autor-Dorn (AD) occupation code specifications (baseline__table3_col4_AD,
  baseline__table3_col5_AD_ft, rc/data/treatment/AD_occ_codes) were **skipped** because
  the AD data merge requires the full Stata pipeline with value-label-based crosswalks.
- The 2010 DWS wave was excluded (mapping CPS 1990 occupation codes to SOC-4 requires
  Stata value labels not accessible from Python). This affects sample size but not the
  core identification strategy.

### Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
- pyreadstat: 1.3.3
