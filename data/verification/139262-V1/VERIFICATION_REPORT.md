# Verification Report: 139262-V1

## Paper Information
- **Title**: Motivated Beliefs and Anticipation of Uncertainty Resolution
- **Authors**: Christoph Drobner
- **Journal**: AER Papers and Proceedings
- **Total Specifications**: 123

## Baseline Groups

### G1: Belief adjustments track Bayesian benchmark for good news in No-Resolution treat...
- **Expected sign**: +
- **Baseline spec(s)**: baseline
- **Outcome**: belief_adjustment
- **Treatment**: bayes_belief_adjustment
- **Notes**: Coef=0.665, p<0.001. Subjects respond strongly to good news.

### G2: Belief adjustments weakly track Bayesian benchmark for bad news in No-Resolution...
- **Expected sign**: +
- **Baseline spec(s)**: baseline_nores_bad
- **Outcome**: belief_adjustment
- **Treatment**: bayes_belief_adjustment
- **Notes**: Coef=0.076, p=0.672. Subjects largely ignore bad news -- evidence of motivated beliefs.

### G3: Resolution treatment restores responsiveness to good news
- **Expected sign**: +
- **Baseline spec(s)**: baseline_res_good
- **Outcome**: belief_adjustment
- **Treatment**: bayes_belief_adjustment
- **Notes**: Coef=0.530, p=0.015.

### G4: Resolution treatment restores responsiveness to bad news
- **Expected sign**: +
- **Baseline spec(s)**: baseline_res_bad
- **Outcome**: belief_adjustment
- **Treatment**: bayes_belief_adjustment
- **Notes**: Coef=0.645, p=0.010. Key finding: anticipation of resolution eliminates the good/bad news asymmetry.

### G5: Signal affects perceived study performance importance (No-Resolution)
- **Expected sign**: +
- **Baseline spec(s)**: baseline_studyperformance_no-resolution
- **Outcome**: studyperformance
- **Treatment**: signal
- **Notes**: Coef=0.795, p=0.021.

### G6: Signal affects perceived study performance importance (Resolution)
- **Expected sign**: +
- **Baseline spec(s)**: baseline_studyperformance_resolution
- **Outcome**: studyperformance
- **Treatment**: signal
- **Notes**: Coef=0.150, p=0.687. Not significant in Resolution.

### G7: Signal affects perceived job performance importance (No-Resolution)
- **Expected sign**: +
- **Baseline spec(s)**: baseline_jobperformance_no-resolution
- **Outcome**: jobperformance
- **Treatment**: signal
- **Notes**: Coef=1.054, p=0.007.

### G8: Signal affects perceived job performance importance (Resolution)
- **Expected sign**: +
- **Baseline spec(s)**: baseline_jobperformance_resolution
- **Outcome**: jobperformance
- **Treatment**: signal
- **Notes**: Coef=0.231, p=0.512. Not significant in Resolution.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 8 |
| Core tests (non-baseline) | 84 |
| Non-core tests | 31 |
| **Total** | **123** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 40 |
| core_funcform | 6 |
| core_inference | 6 |
| core_method | 8 |
| core_sample | 32 |
| noncore_alt_outcome | 8 |
| noncore_heterogeneity | 20 |
| noncore_placebo | 3 |

## Global Notes

Lab experiment on motivated beliefs with 200 subjects. The 2x2 design (Resolution/No-Resolution x Good/Bad signal) creates 4 cells for belief updating and 4 for secondary outcomes (study/job performance). Many specs split by treatment-signal cell. The key finding is the asymmetry in No-Resolution (strong good, weak bad) that disappears in Resolution.
