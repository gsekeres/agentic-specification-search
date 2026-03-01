# Verification Report: 112444-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of center on bank
  - Baseline spec_run_ids: ['112444-V1_run_001']
- **G2**: Effect of debt_move on debt
  - Baseline spec_run_ids: ['112444-V1_run_040']

## Counts
- **Total rows**: 60
- **Core**: 60 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 10
- core_fe: 7
- core_funcform: 4
- core_method: 29
- core_sample: 10
