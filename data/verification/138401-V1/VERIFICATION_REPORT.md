# Verification Report: 138401-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of M12_exp_rate on ln_cpi_income
  - Baseline spec_run_ids: ['138401-V1_run_001', '138401-V1_run_003', '138401-V1_run_004', '138401-V1_run_005', '138401-V1_run_006']

## Counts
- **Total rows**: 49
- **Core**: 46 (94%)
- **Non-core**: 0
- **Invalid**: 3

## Category Breakdown
- core_controls: 3
- core_data: 10
- core_fe: 8
- core_funcform: 1
- core_method: 12
- core_sample: 12
- invalid_failure: 3
