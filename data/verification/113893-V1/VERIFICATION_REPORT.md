# Verification Report: 113893-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of radio1 on Nazi_share
  - Baseline spec_run_ids: ['113893-V1_run_001', '113893-V1_run_002', '113893-V1_run_003', '113893-V1_run_004', '113893-V1_run_005']
- **G2**: Effect of radio1 on graffiti
  - Baseline spec_run_ids: ['113893-V1_run_062']

## Counts
- **Total rows**: 71
- **Core**: 62 (87%)
- **Non-core**: 9
- **Invalid**: 0

## Category Breakdown
- core_controls: 32
- core_fe: 1
- core_method: 33
- core_sample: 3
- core_weights: 1
- noncore_other: 1
