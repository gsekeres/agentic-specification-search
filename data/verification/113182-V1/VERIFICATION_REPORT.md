# Verification Report: 113182-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of df_5to11 on total_v2_agcho10
  - Baseline spec_run_ids: ['113182-V1_run_001', '113182-V1_run_002', '113182-V1_run_003', '113182-V1_run_004', '113182-V1_run_005']
- **G2**: Effect of post_event_indicator on total_votes_wins
  - Baseline spec_run_ids: ['113182-V1_run_030', '113182-V1_run_031', '113182-V1_run_032', '113182-V1_run_033']

## Counts
- **Total rows**: 52
- **Core**: 51 (98%)
- **Non-core**: 0
- **Invalid**: 1

## Category Breakdown
- core_controls: 19
- core_method: 30
- core_sample: 2
- invalid_failure: 1
