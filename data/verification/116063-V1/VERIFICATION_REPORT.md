# Verification Report: 116063-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of lnwt on lnprocs
  - Baseline spec_run_ids: ['116063-V1_run_001', '116063-V1_run_002', '116063-V1_run_003', '116063-V1_run_004', '116063-V1_run_005', '116063-V1_run_006']
- **G2**: Effect of spread on fracy
  - Baseline spec_run_ids: ['116063-V1_run_052', '116063-V1_run_053']

## Counts
- **Total rows**: 62
- **Core**: 62 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 1
- core_fe: 13
- core_funcform: 9
- core_method: 10
- core_sample: 19
- core_weights: 10
