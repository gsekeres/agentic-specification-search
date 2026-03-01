# Verification Report: 112466-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of Turnover on Price
  - Baseline spec_run_ids: ['112466-V1_run_001', '112466-V1_run_054', '112466-V1_run_055']
- **G2**: Effect of Turnover on Warrant_price
  - Baseline spec_run_ids: ['112466-V1_run_029', '112466-V1_run_056', '112466-V1_run_057']

## Counts
- **Total rows**: 61
- **Core**: 57 (93%)
- **Non-core**: 4
- **Invalid**: 0

## Category Breakdown
- core_controls: 37
- core_funcform: 4
- core_method: 10
- core_sample: 10
