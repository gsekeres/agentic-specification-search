# Verification Report: 113500-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of femaleXsolicited on decision
  - Baseline spec_run_ids: ['113500-V1_run_001']
- **G2**: Effect of femaleXsolicited on decision
  - Baseline spec_run_ids: ['113500-V1_run_021']
- **G3**: Effect of femaleXbacklashXsol on decision
  - Baseline spec_run_ids: ['113500-V1_run_041']

## Counts
- **Total rows**: 60
- **Core**: 60 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 36
- core_funcform: 6
- core_method: 9
- core_sample: 9
