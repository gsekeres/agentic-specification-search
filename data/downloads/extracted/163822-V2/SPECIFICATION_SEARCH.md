# Specification Search: 163822-V2
## Allcott, Gentzkow & Song (2022) "Digital Addiction"

### Surface Summary
- **Paper ID**: 163822-V2
- **Surface hash**: sha256:ab6d5bda350feac4beb4a8cd18470e3c37c24b05b6c34c18dec8d63009951f50
- **Design**: Randomized experiment (RCT)
- **Baseline groups**: 2
  - **G1**: Phone usage (PD-measured screen time) - Budget: 50 specs
  - **G2**: Survey well-being (stacked S3/S4 panel) - Budget: 40 specs
- **Seed**: 163822

### Data Source
- `final_data_sample.dta` downloaded from Harvard Dataverse (doi:10.7910/DVN/GN636M)
- 1,933 observations (Android phone users)
- Treatment: Bonus (financial incentive, B) and Limit (commitment device, L)
- Stratification on baseline usage x addiction x restriction

### Execution Summary
- **Total specifications executed**: 54
  - G1 (phone usage): 38
  - G2 (survey well-being): 16
- **Successful**: 54
- **Failed**: 0
- **Inference variants**: 2

### G1 Specifications (Phone Usage)
- **Baseline**: PD_P2_UsageFITSBY ~ B + L + Strata + PD_P1_UsageFITSBY, robust
- **Additional baselines**: P3, P4, P5, P2-P5 avg FITSBY; P2 total, P2-P5 avg total
- **Design variants**: Diff-in-means, with covariates
- **RC variants**: LOO (drop baseline, drop strata), control sets, progression,
  sample trimming (1-99, 5-95), balanced panels, total usage outcome,
  hours outcome, detailed limit types, log1p transform, arcsinh transform
- **Inference variant**: HC2

### G2 Specifications (Survey Well-Being)
- **Baseline**: index_well_N ~ B4 + L + S + Strata x S + baseline x S, cluster(UserID)
  - Data stacked across S3 (midline) and S4 (endline) survey waves
  - B4 = Bonus x (S==S4); B replaced by B4 to capture S4-specific bonus
- **Additional baselines**: AddictionIndex, SMSIndex, PhoneUseChange, LifeBetter, SWBIndex
- **Design variant**: Diff-in-means (no controls)
- **RC variants**: LOO, control sets, trimming, S3-only, S4-only
- **Inference variant**: HC1 (no clustering)

### Deviations from Surface
- None. All planned specs were executed.

### Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
