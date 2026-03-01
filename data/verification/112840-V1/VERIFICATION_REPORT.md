# Verification Report: 112840-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of lag_d_ave_finindex1 on changerealdebt
  - Baseline spec_run_ids: ['112840-V1_spec_001', '112840-V1_spec_002', '112840-V1_spec_003']

## Counts
- **Total rows**: 59
- **Core**: 56 (95%)
- **Non-core**: 0
- **Invalid**: 3

## Category Breakdown
- core_controls: 31
- core_data: 1
- core_fe: 1
- core_funcform: 2
- core_method: 12
- core_preprocess: 1
- core_sample: 8
- invalid_failure: 3
