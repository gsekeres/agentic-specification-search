# Verification Report: 113783-V1

## Paper Information
- **Title**: Improving the Design of Conditional Transfer Programs: Evidence from a Randomized Education Experiment in Colombia
- **Authors**: Felipe Barrera-Osorio, Marianne Bertrand, Leigh Linden, Francisco Perez-Calle
- **Journal**: AEJ: Applied Economics (2011)
- **Total Specifications**: 89

## Baseline Groups

### G1: T1 (Basic CCT) on Monitored Attendance
- **Claim**: The basic conditional cash transfer increases monitored school attendance by ~3.2 percentage points.
- **Baseline spec**: `baseline_attendance_T1`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0316 (SE: 0.0074, p < 0.0001)
- **Outcome**: `at_msamean`
- **Treatment**: `T1_treat`
- **Table 3, Column 2 (SC panel)**

### G2: T2 (Savings CCT) on Monitored Attendance
- **Claim**: The savings-treatment CCT increases monitored school attendance by ~2.7 percentage points.
- **Baseline spec**: `baseline_attendance_T2`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0270 (SE: 0.0071, p = 0.0002)
- **Outcome**: `at_msamean`
- **Treatment**: `T2_treat`
- **Table 3, Column 2 (same regression as G1, different treatment coefficient)**

### G3: T2 (Savings CCT) on School Enrollment
- **Claim**: The savings-treatment CCT increases administrative school enrollment by ~4.0 percentage points.
- **Baseline spec**: `baseline_enrollment_T2`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0396 (SE: 0.0106, p = 0.0002)
- **Outcome**: `m_enrolled`
- **Treatment**: `T2_treat`
- **Table 4, Column 2**

### G4: T1 (Basic CCT) on School Enrollment (Null Result)
- **Claim**: The basic CCT has no statistically significant effect on enrollment, in contrast to the savings treatment.
- **Baseline spec**: `baseline_enrollment_T1`
- **Expected sign**: Positive (point estimate), but not statistically significant
- **Baseline coefficient**: 0.0107 (SE: 0.0102, p = 0.296)
- **Outcome**: `m_enrolled`
- **Treatment**: `T1_treat`
- **Table 4, Column 2 (same regression as G3)**

This null result is itself a key finding: it demonstrates that the savings incentive (T2) is more effective than the basic transfer (T1) at promoting enrollment, one of the paper's central contributions.

### G5: T3 (Tertiary/Suba) on Attendance
- **Claim**: The tertiary savings treatment increases monitored attendance in the Suba subsample by ~5.6 percentage points.
- **Baseline spec**: `baseline_attendance_T3_suba`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0561 (SE: 0.0199, p = 0.005)
- **Outcome**: `at_msamean`
- **Treatment**: `T3_treat`
- **Table 3, Suba panel. N=930 (separate smaller experiment in Suba locality, grades 9-10 only)**

### G6: T3 (Tertiary/Suba) on Enrollment
- **Claim**: The tertiary savings treatment has a marginally significant positive effect on enrollment.
- **Baseline spec**: `baseline_enrollment_T3_suba`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0370 (SE: 0.0196, p = 0.059)
- **Outcome**: `m_enrolled`
- **Treatment**: `T3_treat`
- **Table 4, Suba panel. N=1735. Marginally significant at 10% but not at 5%.**

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **53** | |
| core_controls | 27 | 6 baselines + no-controls, partial control sets (demo, hh, edu, demo+hh, demo+edu), full controls without FE, no-FE-clustered duplicates |
| core_sample | 16 | Grade subsamples (lower/upper), exclude over-age, non-missing enrollment, baseline observed, include grade 11, T1-only/T2-only samples, pooled 3-way locality samples |
| core_inference | 3 | HC1 robust SE, conventional OLS SE, grade-level clustering |
| core_method | 2 | Probit and Logit marginal effects for binary enrollment |
| core_treatment | 2 | Pooled treatment indicator (any treatment vs control) for attendance and enrollment |
| core_funcform | 2 | Winsorized attendance (duplicate of baseline), z-scored attendance (linear transformation) |
| **Non-core tests** | **36** | |
| noncore_alt_outcome | 12 | Self-reported attendance (2), currently attending (2), graduation (2), tertiary enrollment (1), primary activity study/work (2), hours worked (1), earnings (1), log hours (1) |
| noncore_heterogeneity | 14 | Gender subsamples (4), income subsamples (4), gender interactions (4), income interactions (2), predicted baseline interactions (2), overage interaction (1), age interaction (1) |
| noncore_placebo | 5 | Balance tests on baseline characteristics: SISBEN score, income, utilities, durables, head age |
| **Duplicates noted** | **3** | funcform/winsorize_5_95/attendance_T1 = baseline_attendance_T1; inference/no_fe_clustered/attendance_T1 = ctrl/baseline_controls/attendance_T1; inference/no_fe_clustered/enrollment_T2 = ctrl/baseline_controls/enrollment_T2 |
| **Total** | **89** | |

## Detailed Classification Notes

### Core Tests (53 specs including 6 baselines)

**Baselines (6 specs)**: The six primary baseline specifications correspond to Tables 3 and 4 of the paper. All use OLS with the full set of baseline controls (demographic, household, and education covariates) plus school fixed effects, with standard errors clustered at the school level. The SC baselines (G1-G4) include both T1 and T2 treatment indicators simultaneously. The Suba baselines (G5-G6) include only T3. The attendance baselines use the survey-selected subsample; the enrollment baselines use the full administrative sample.

**Control variations (21 non-baseline core_controls specs)**: Systematic exploration of control sets for the three primary treatment-outcome pairs (T1 on attendance, T2 on attendance, T2 on enrollment):
- No controls (3 specs): treatment effect without any covariates; in a well-randomized RCT, point estimates should be similar to baseline.
- Partial control sets (15 specs): demographic-only, household-only, education-only, demographic+household, demographic+education -- 5 sets x 3 outcome-treatment pairs.
- Full controls without school FE (3 specs): removes school fixed effects while keeping all baseline covariates, testing the importance of the within-school comparison.

The no-controls and partial-control specs confirm that point estimates are very stable across control sets, consistent with successful randomization. Coefficients vary only in the third decimal place.

**Sample restrictions (16 specs)**: Grade range splits (lower/upper), exclusion criteria (over-age, non-missing enrollment, baseline observed), inclusion of grade 11, and alternative sample definitions (T1-only or T2-only comparisons dropping the other treatment arm, pooled 3-way locality-treatment comparisons across SC and Suba).

**Inference variations (3 specs)**: HC1 robust standard errors, conventional (homoskedastic) OLS standard errors, and clustering at grade level instead of school level. All share the same point estimate as the baseline; only SEs change. The HC1 SE (0.0084) is slightly larger than the clustered SE (0.0074), confirming that school-level clustering already addresses the main source of dependence.

**Estimation method (2 specs)**: Probit and Logit marginal effects for the binary enrollment outcome. Marginal effects (~0.046) are slightly larger than the OLS linear probability model estimate (0.040), which is expected.

**Treatment variable (2 specs)**: Pooled treatment indicator (any CCT arm vs. control) for both attendance and enrollment. Tests the average treatment effect of the program as a whole.

**Functional form (2 specs)**: Winsorized attendance (identical to baseline since at_msamean is already bounded) and z-scored attendance (linear transformation with identical t-statistic of 4.253 and p-value of 0.000021).

### Non-Core Tests (36 specs)

**Alternative outcomes (12 specs)**: These measure different dimensions of the CCT's impact beyond monitored attendance and administrative enrollment:
- Self-reported attendance (Table 6): T1 coef = 0.006 (p = 0.152), T2 coef = 0.007 (p = 0.126). Neither significant, suggesting self-reports may be noisy.
- Currently attending at follow-up (Table 6): small and insignificant effects.
- Graduation (Table 7): grade 11 subsample only (N=529), small positive but insignificant effects for both T1 and T2.
- Tertiary enrollment (Table 7): T2 coef = 0.094 (p = 0.005), significant positive effect of savings treatment on college enrollment.
- Labor outcomes (Table 8): primary activity study/work, hours worked (significant negative T1 effect of -0.37 hours, p = 0.020), earnings (insignificant).
- Log hours worked: marginal significance (p = 0.069).

These are non-core because they test fundamentally different outcome constructs from the two main outcomes (monitored attendance and administrative enrollment).

**Heterogeneity (14 specs)**: Split-sample and interaction analyses exploring treatment effect heterogeneity:
- Gender subsamples (4 specs): T1 attendance effect is larger for males (0.049, p < 0.001) than females (0.016, p = 0.184). T2 enrollment effect is similar across genders.
- Income subsamples (4 specs): T1 attendance effect is larger for high-income (0.069, p < 0.001) than low-income (0.016, p = 0.050), somewhat surprising.
- Interaction models (6 specs): gender interactions (T1*girl, T2*girl), income interaction (T1*low_income), predicted baseline interactions (T1*baseline_attendance, T2*baseline_enrollment), overage interaction, age interaction. These test Table 5-style heterogeneity.

These are classified as non-core because they decompose the treatment effect by subgroup, testing heterogeneity hypotheses rather than providing alternative estimates of the average treatment effect.

**Placebo tests (5 specs)**: Balance checks regressing T1 treatment status on baseline characteristics (SISBEN score, total income, utilities index, durables index, age of household head). All coefficients are small and statistically insignificant (p-values: 0.617, 0.322, 0.982, 0.233, 0.500), confirming successful randomization. These validate the research design rather than estimate the treatment effect.

## Duplicates Identified

1. `funcform/winsorize_5_95/attendance_T1` = `baseline_attendance_T1`: Identical coefficient (0.0316), SE (0.0074), t-stat (4.253), and p-value because attendance is a proportion in [0,1] and winsorization at the 5th/95th percentiles does not alter any values.

2. `inference/no_fe_clustered/attendance_T1` = `ctrl/baseline_controls/attendance_T1`: Both use full baseline controls, no school FE, and school-level clustering. Coefficient 0.0316, t-stat 4.144. These are the same specification labeled differently.

3. `inference/no_fe_clustered/enrollment_T2` = `ctrl/baseline_controls/enrollment_T2`: Same duplication for enrollment. Coefficient 0.0460, t-stat 3.066.

After removing duplicates, there are approximately 86 unique specifications.

## Robustness Assessment

### G1: T1 on Attendance (at_msamean)
The T1 attendance effect is **very robust**. Across all core specifications maintaining the same outcome and treatment:
- Coefficient range: 0.010 (baseline-observed sample, which may suffer from selection) to 0.034 (include grade 11 or non-missing enrollment restriction).
- The baseline estimate of 0.032 is in the center of the distribution.
- All core specs except `sample/bl_observed/attendance_T1` (p = 0.043) are significant at p < 0.01.
- Stability across control sets: coefficients range from 0.031 to 0.034 across all control variations, confirming randomization quality.
- Point estimate is identical under HC1, OLS, and cluster-robust SEs; only inference changes.

### G2: T2 on Attendance
Similarly robust. Coefficient ranges 0.026-0.029 across control variations. Significant at p < 0.01 in all core control specifications.

### G3: T2 on Enrollment (m_enrolled)
Robust across specifications. Coefficient ranges 0.029-0.047 across core specs. Probit and Logit marginal effects (0.046 and 0.045) are slightly larger than the OLS baseline (0.040). All core specs significant at p < 0.01. The pooled 3-way specification (grades 9-10, both localities) yields a smaller effect (0.030, p = 0.074), reflecting the different sample composition.

### G4: T1 on Enrollment (Null Result)
The null result for T1 on enrollment is robust: coefficient ranges from 0.001 to 0.011 across specifications, never significant at conventional levels. The pooled 3-way spec yields an essentially zero coefficient (0.0004, p = 0.979).

### G5-G6: T3 (Suba) Effects
T3 attendance effect (0.056, p = 0.005) and marginally significant enrollment effect (0.037, p = 0.059) are based on smaller samples (N=930 and N=1735) and have fewer robustness checks. The pooled 3-way specifications yield similar magnitudes (T3 attendance 0.055, T3 enrollment 0.042).

### Key Observations
1. **Control set stability**: Treatment effects are nearly identical across all control specifications (no controls through full baseline), which is the expected pattern for a well-implemented RCT with good balance.
2. **School FE matters little**: Removing school fixed effects barely changes point estimates, consistent with randomization at the school level.
3. **Grade heterogeneity**: The T1 attendance effect is slightly larger in lower grades (0.036) than upper grades (0.023), but both are significant.
4. **Gender heterogeneity**: The T1 attendance effect is concentrated among males (0.049 vs 0.016 for females). This is one of the paper's findings in Table 5.
5. **Income heterogeneity**: Counter-intuitively, the T1 attendance effect appears larger for higher-income students (0.069 vs 0.016).
6. **Balance tests confirm randomization**: All 5 placebo tests yield statistically insignificant coefficients.
