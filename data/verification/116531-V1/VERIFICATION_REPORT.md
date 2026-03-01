# Verification Report: 116531-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of offered on borrowed
  - Baseline spec_run_ids: ['116531-V1__spec_001', '116531-V1__spec_002']
- **G2**: Effect of package on crdattm_total
  - Baseline spec_run_ids: ['116531-V1__spec_064', '116531-V1__spec_065', '116531-V1__spec_066', '116531-V1__spec_067']

## Counts
- **Total rows**: 131
- **Core**: 131 (100%)
- **Non-core**: 0
- **Invalid**: 0

## Category Breakdown
- core_controls: 94
- core_fe: 6
- core_funcform: 1
- core_method: 18
- core_sample: 12
