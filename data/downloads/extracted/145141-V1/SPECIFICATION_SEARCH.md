# Specification Search: 145141-V1

## Surface Summary

- **Paper ID**: 145141-V1
- **Surface hash**: sha256:8488e93835b0f123ed4bf85c6c055755f78bdc7c62b679ff7fb1e56ddecd01b9
- **Baseline groups**: 3
  - G1: YMCA field experiment (attendance ~ image)
  - G2: Charity real-effort experiment (pts ~ SR)
  - G3: WTP elicitation (wtp ~ visits/interval, quadratic)
- **Design**: randomized_experiment (all groups)
- **Seed**: 145141

## Execution Summary

### specification_results.csv
- **Total rows**: 96
- **Successful**: 96
- **Failed**: 0
- **G1 (YMCA attendance)**: 16 specs
- **G2 (Charity real-effort)**: 43 specs
- **G3 (WTP elicitation)**: 37 specs

### inference_results.csv
- **Total rows**: 7
- **Successful**: 7
- **Failed**: 0

## Specifications Executed

### G1: YMCA Attendance (HC1 robust SEs)
- 2 baseline specs (Table 2 Cols 2-3, coherent sample)
- 1 design variant: diff-in-means
- 3 control sets: none, past only, past+beliefs
- 2 sample definitions: monotonic, robust (full excl BDM)
- 3 outlier treatments: trim 1/99, trim 5/95, winsorize 1/99
- 3 functional forms: log1p, asinh, standardized
- 2 preprocessing: topcode at 22, topcode at 15

### G2: Charity Real-Effort (clustered SEs at individual level)
- 3 baseline specs (Table 5 Cols 1-3: Prolific, Berkeley, BU)
- 3 design variants: diff-in-means (per sample)
- 6 control sets: ownpay-only and ownpay+order (per sample)
- 12 sample definitions: first-round-only, no-attention-check, strict-consistency,
  approx-monotonic (each per sample)
- 6 outlier treatments: trim 1/99, trim 5/95 (per sample)
- 1 pooled sample spec
- 9 functional forms/preprocessing: log1p, standardized, pts-in-hundreds (per sample)
- 3 individual FE specs (per sample)

### G3: WTP Elicitation (clustered SEs at individual level)
- 4 baseline specs (Table 3 Col 2 YMCA; Table 6 Cols 2/4/6 charity)
- 4 OLS linear (YMCA + 3 charity samples)
- 2 Tobit specs (YMCA quadratic + linear, MLE with Hessian SEs)
- 3 YMCA functional form: ln_visits quadratic, interval_idx quadratic/linear
- 10 YMCA sample restrictions: monotonic, close-to-beliefs-4, exact-belief-match,
  close-to-past-4, excl-top-interval, excl-top-2-intervals, WTP trim 1/99, 5/95
- 9 charity sample restrictions: include-top-interval, close-to-score, no-consistency (per sample)
- 6 charity WTP trim: 1/99, 5/95 (per sample)
- 1 pooled charity WTP spec

### Inference variants
- G1: HC3 on baseline
- G2: HC1 (no clustering) on each baseline
- G3: HC1 (no clustering) on each charity baseline

## Skipped / Not Feasible
- Tobit for charity experiments: not listed in core surface rc_spec_ids (charity code does not run Tobit)
- BDM arm analysis: excluded per surface (BDM arm excluded from reduced-form)
- QM221 sample: separate pilot sample, not in the surface specification

## Software Stack
- Python 3.12.7
- pandas 2.2.3
- numpy 2.1.3
- pyfixest 0.40.1
- scipy (for Tobit MLE)

## Data Construction Notes
- YMCA data built from raw CSV files (scans, membership, survey, treatment assignment)
- Attendance = count of unique scan days during June 15 - July 15, 2017
- Past attendance = total scans in pre-period (May 2016 - May 2017) / 13
- Charity data built from raw CSV files (Prolific, Berkeley, BU QM222)
- WTP variables: negative (pay to avoid) or positive (pay for recognition)
- Coherent sample: excludes incoherent WTP respondents (>2 switches, or starts yes with 2 switches)
- Monotonic sample: excludes anyone with WTP decreasing at any point
- avg_att_mainpop = 3.1357 (YMCA full population excluding experiment participants)
