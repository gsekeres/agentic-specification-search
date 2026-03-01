# Verification Report: 113630-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1_mortality**: Effect of treatment on death_year
  - Baseline spec_run_ids: ['113630-V1_run_001', '113630-V1_run_002', '113630-V1_run_003', '113630-V1_run_004', '113630-V1_run_005']
- **G2_weight**: Effect of treatment on zw1
  - Baseline spec_run_ids: ['113630-V1_run_031', '113630-V1_run_032', '113630-V1_run_033', '113630-V1_run_034']

## Counts
- **Total rows**: 48
- **Core**: 48 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 19
- core_funcform: 9
- core_method: 12
- core_sample: 8
