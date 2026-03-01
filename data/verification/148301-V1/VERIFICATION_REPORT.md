# Verification Report: 148301-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of lfma on ep
  - Baseline spec_run_ids: ['148301-V1__run_001', '148301-V1__run_002', '148301-V1__run_003', '148301-V1__run_004', '148301-V1__run_005', '148301-V1__run_006']
- **G2**: Effect of ep_haven on lprofit
  - Baseline spec_run_ids: ['148301-V1__run_052', '148301-V1__run_053', '148301-V1__run_054']

## Counts
- **Total rows**: 68
- **Core**: 67 (99%)
- **Non-core**: 0
- **Invalid**: 1

## Category Breakdown
- core_controls: 33
- core_fe: 4
- core_funcform: 7
- core_method: 10
- core_sample: 13
- invalid_failure: 1
