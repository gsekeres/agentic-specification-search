# Verification Report: 163241-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of 1.female#1.treated on ln_salary_annual_rate
  - Baseline spec_run_ids: ['163241-V1_spec_0001', '163241-V1_spec_0002']

## Counts
- **Total rows**: 51
- **Core**: 47 (92%)
- **Non-core**: 0
- **Invalid**: 4

## Category Breakdown
- core_controls: 11
- core_data: 3
- core_fe: 7
- core_funcform: 1
- core_method: 3
- core_sample: 21
- core_weights: 1
- invalid_failure: 4
