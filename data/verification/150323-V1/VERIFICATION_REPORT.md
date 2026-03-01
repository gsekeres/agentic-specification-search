# Verification Report: 150323-V1

## Method
Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.

## Baseline Groups
- **G1**: Effect of pX_dummy on both_score_indiv_4_stdComb
  - Baseline spec_run_ids: ['150323-V1_spec_001', '150323-V1_spec_002', '150323-V1_spec_003', '150323-V1_spec_004']
- **G2**: Effect of pX_dummy on SHhired_Mun_lead
  - Baseline spec_run_ids: ['150323-V1_spec_035', '150323-V1_spec_036', '150323-V1_spec_037', '150323-V1_spec_038']
- **G3**: Effect of pX_dummy on expthisschl_lessthan2_DPB
  - Baseline spec_run_ids: ['150323-V1_spec_056', '150323-V1_spec_057', '150323-V1_spec_058']

## Counts
- **Total rows**: 74
- **Core**: 72 (97%)
- **Non-core**: 0
- **Invalid**: 2

## Category Breakdown
- core_controls: 7
- core_funcform: 5
- core_method: 44
- core_sample: 16
- invalid_failure: 2
