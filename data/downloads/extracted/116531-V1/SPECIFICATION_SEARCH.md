# Specification Search: 116531-V1

## Paper
Marx & Turner (2019), "Student Loan Nudges: Experimental Evidence on Borrowing
and Educational Attainment," *American Economic Review*, 109(2), 566-592.

## DATA NOTE
The student-level analysis data is **CONFIDENTIAL** (provided by an anonymous
community college). Only `T1_data_packaging_practices.dta` (institution-level
packaging practices, Table 1 only) is included in the replication package.

All specifications were run on **SYNTHETIC DATA** calibrated to the paper's
published summary statistics (Table 3 control group means, reported sample sizes,
approximate coefficient magnitudes). Results validate the specification surface
and estimation pipeline but coefficients will not exactly match published values.

## Surface Summary
- **Paper ID**: 116531-V1
- **Baseline groups**: 2 (G1: IV borrowing, G2: ITT attainment)
- **G1 design**: Stratified RCT, IV (package instruments for offered)
- **G2 design**: Stratified RCT, ITT/OLS (package -> attainment)
- **Canonical inference**: cluster(stratum_code) for both groups
- **Seed**: 116531
- **Surface hash**: sha256:8f388c6b5bb27c68b77759fcbaf8634f35707daf6cd753b1d9b5a3bff2b54673

### G1: Effect of loan offers on borrowing
- Baseline outcomes: borrowed, AcceptedAmount
- Treatment: offered (endogenous, instrumented by package)
- Controls: month_packaged dummies, Prmry_EFC, CumulativeGPA, CumulativeEarnedHours, pell_elig, indep, has_outstanding
- FE: stratum_code
- Budget: max 80 core specs, 10 control subset draws

### G2: ITT effect on educational attainment
- Baseline outcomes: crdattm_total, credits_total, gpa_total, anydeg
- Treatment: package (random assignment)
- Sample: enrolled_fall == 1
- Budget: max 50 core specs

## Execution Summary
- **Total specification rows**: 131
- **Successful**: 131
- **Failed**: 0
- **Inference variant rows**: 6
- **G1 specifications**: 63
- **G2 specifications**: 68

### Spec breakdown by type
| Type | Count |
|------|-------|
| baseline | 6 |
| design/* | 12 |
| rc/controls/loo/* | 42 |
| rc/controls/sets/* | 14 |
| rc/controls/progression/* | 18 |
| rc/controls/subset/* | 20 |
| rc/sample/* | 12 |
| rc/fe/* | 6 |
| rc/form/* | 1 |
| infer/* (separate table) | 6 |

## Deviations
- **Synthetic data**: All results use synthetic data because the student-level
  analysis data is confidential. The Stata do-files reference
  `formatted_analysis_data.dta` which is not included in the replication package.
- The xtivreg2 command in Stata uses the `partial()` option to partial out
  controls before IV estimation. In pyfixest, controls in the IV formula are
  included directly (not partialled), which is algebraically equivalent but may
  produce minor numerical differences.

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
- statsmodels 0.14.6
