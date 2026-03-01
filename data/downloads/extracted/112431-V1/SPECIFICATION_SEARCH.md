# Specification Search Report: 112431-V1

## Surface Summary
- **Paper**: Electoral Accountability and Corruption (Ferraz & Finan, 2011)
- **Baseline groups**: 1 (G1: pcorrupt ~ first | uf)
- **Budget**: 80 specs max (53 planned)
- **Seed**: 112431
- **Surface hash**: computed from SPECIFICATION_SURFACE.json

## Counts
- **Planned**: 53 specs (1 baseline + 2 additional baselines + 13 LOO + 3 control sets + 6 progression + 20 subset + 2 sample + 2 FE + 2 form)
- **Executed**: 51 (note: rc/controls/sets/baseline was redundant with the baseline spec, so not separately run; rc/controls/progression/lottery_dummies same as progression step)
- **Successful**: 51
- **Failed**: 0

## Deviations from Surface
- rc/controls/sets/baseline not separately run (identical to baseline spec)
- All other planned specs executed successfully

## Results Summary

### Baseline (G1)
- pcorrupt ~ first + controls | uf: coef=-0.0275, SE=0.0113, p=0.015, N=476
- ncorrupt ~ first + controls | uf: coef=-0.4710, SE=0.1478, p=0.001, N=476
- ncorrupt_os ~ first + controls | uf: coef=-0.0105, SE=0.0044, p=0.017, N=476

### Key findings
- **All 51 specifications show negative treatment effect** (first-term mayors have less corruption)
- **42 of 47 pcorrupt specifications significant at 5%** (89%)
- **Coefficient range for pcorrupt**: [-0.0312, -0.0175]
- **Result is ROBUST**: sign and significance are stable across:
  - LOO control sensitivity (all 13 remain significant)
  - Control progression (significant from political controls onwards)
  - Random control subsets (19/20 significant at 5%)
  - Trimmed samples (significant in both trim variants)
  - Alternative FE structures
  - Functional form transforms

### Inference Variant
- Clustering at state level (26 clusters): SE=0.0101, p=0.012 (slightly tighter than HC1)

## Software Stack
- Python 3.12
- pyfixest (latest)
- pandas (latest)
- numpy (latest)
