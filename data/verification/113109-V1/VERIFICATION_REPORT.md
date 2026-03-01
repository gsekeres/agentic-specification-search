# Verification Report: 113109-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of hp_growth_real_00_06 on d_emp_18_25_le
  - Baseline spec_run_ids: ['113109-V1_run_001']
- **G2**: Effect of housing_demand_shock on d_any_18_25_a1
  - Baseline spec_run_ids: ['113109-V1_run_053', '113109-V1_run_054']

## Counts
- **Total rows**: 86
- **Core**: 86 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 52
- core_data: 8
- core_funcform: 5
- core_method: 11
- core_sample: 7
- core_weights: 3
