# Verification Report: 207766-V1

## Paper Information
- **Paper ID**: 207766-V1
- **Total Specifications**: 149
- **Baseline Specifications**: 4
- **Core Test Specifications**: 87
- **Non-Core Specifications**: 62

## Baseline Groups

### G1: Municipalities aligned with the ruling party receive more government transfers t
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__amount_congr1_3_pcap`
- **Outcome**: `amount_congr1_3_pcap`
- **Treatment**: `alignm > 0`
- **Notes**: RD estimate: 1.39 euros per capita (SE=0.35, p<0.001). Linear polynomial, triangular kernel, MSE-optimal bandwidth (~16.2pp). Primary outcome of the paper.

### G2: Transfers to moderate (neither congruent nor non-congruent) organizations show n
- **Expected sign**: 0
- **Baseline spec(s)**: `baseline__amount_congr2_3_pcap`
- **Outcome**: `amount_congr2_3_pcap`
- **Treatment**: `alignm > 0`
- **Notes**: RD estimate: -0.06 (SE=0.15, p=0.68). Null result expected and found.

### G3: Transfers to non-congruent (opposition-aligned) organizations show no discontinu
- **Expected sign**: 0
- **Baseline spec(s)**: `baseline__amount_congr3_3_pcap`
- **Outcome**: `amount_congr3_3_pcap`
- **Treatment**: `alignm > 0`
- **Notes**: RD estimate: -0.15 (SE=0.19, p=0.42). Null result expected and found.

### G4: Total transfers per capita also show a positive discontinuity, but smaller than 
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__amount_pcap`
- **Outcome**: `amount_pcap`
- **Treatment**: `alignm > 0`
- **Notes**: RD estimate: 1.08 (SE=0.48, p=0.023). Significant at 5% but not 1%.

**Global Notes**: Paper: 'Organized Voters' by Camille Urvoy (AER 2025). Sharp RD design using French municipal elections. The running variable (alignm) is the win margin of the ruling party candidate. Many specifications apply the same spec_id to 4 different outcomes (congr1_3, congr2_3, congr3_3, total), so unique IDs are constructed as spec_id__outcome_var. The primary claim is about congruent organizations (G1); null results for moderate/non-congruent orgs support the specificity of the finding. Specs with different cutoffs (placebo) test RD validity. Heterogeneity specs test where the effect is concentrated.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **87** | |
| core_funcform | 12 | |
| core_inference | 4 | |
| core_method | 16 | |
| core_sample | 55 | |
| **Non-core tests** | **62** | |
| noncore_alt_outcome | 30 | |
| noncore_heterogeneity | 26 | |
| noncore_placebo | 6 | |
| **Total** | **149** | |

## Verification Checks

- Total specs in CSV: 149
- Unique spec_ids: 149
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
