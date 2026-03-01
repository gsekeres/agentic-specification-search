# Verification Report: 134041-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of T1 on z_mani_index
  - Baseline spec_run_ids: ['S001', 'S002', 'S003', 'S004', 'S005', 'S006']
- **G2**: Effect of T1 on z_lmpolicy_index
  - Baseline spec_run_ids: ['S045', 'S046', 'S047', 'S048', 'S049', 'S050', 'S051']

## Counts
- **Total rows**: 84
- **Core**: 84 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 59
- core_method: 17
- core_sample: 6
- core_weights: 2
