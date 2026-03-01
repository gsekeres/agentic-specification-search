# Verification Report: 113577-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1_math**: Effect of peer_tfx_m on m_growth
  - Baseline spec_run_ids: ['113577-V1_run_001']
- **G1_reading**: Effect of peer_tfx_r on r_growth
  - Baseline spec_run_ids: ['113577-V1_run_032']

## Counts
- **Total rows**: 61
- **Core**: 57 (93%)
- **Non-core**: 0
- **Invalid**: 4

## Category Breakdown
- core_controls: 34
- core_fe: 8
- core_funcform: 3
- core_method: 6
- core_sample: 6
- invalid_failure: 4
