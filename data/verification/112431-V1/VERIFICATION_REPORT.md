# Verification Report: 112431-V1

## Paper Information
- **Title**: Electoral Accountability and Corruption: Evidence from the Audits of Local Government
- **Authors**: Ferraz & Finan
- **Journal**: AER (2011)
- **Total Specifications**: 96

## Baseline Groups

### G1: First-Term Effect on Corruption
- **Claim**: First-term mayors who face reelection incentives engage in less corruption than second-term mayors facing term limits.
- **Baseline specs**: `baseline` (no controls), `baseline_fe` (state FE + full controls)
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0188 (SE: 0.0095, p = 0.047) without FE; -0.0193 (SE: 0.0101, p = 0.056) with state FE
- **Outcome**: `pcorrupt` (proportion of audited funds associated with corruption)
- **Treatment**: `first` (first-term mayor indicator)

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **68** | |
| core_controls | 24 | 2 baselines + mayor char, municipality char, political vars, transfers, experience, 14 leave-one-out controls, 4 progressive control additions |
| core_fe | 1 | State fixed effects |
| core_sample | 32 | Drop lottery rounds (9), drop states (10), pop/urban splits (4), winsorize/trim (5), experienced/likely reelect (2), weights (2) |
| core_inference | 6 | Classical SE, HC1/HC2/HC3, state cluster, lottery cluster |
| core_funcform | 3 | Log outcome, IHS outcome, binary LPM |
| core_method | 12 | RDD linear/quad/cubic (3), RDD bandwidths (4), logit, probit, negbin, poisson (4), plus binary outcome LPM counted above |
| **Non-core tests** | **28** | |
| noncore_alt_outcome | 7 | ncorrupt, ncorrupt_os, pmismanagement, dcorrupt, dcorrupt_desvio, dcorrupt_licitacao, dcorrupt_superfat |
| noncore_heterogeneity | 9 | Political competition, media, judiciary, population, urbanization, income, male mayors, PT party, same party governor |
| noncore_placebo | 4 | Resources audited, population, income, urbanization (all predetermined) |
| **Total** | **96** | |

## Robustness Assessment

The main finding shows **moderate** support:
- **Sign consistency**: 93.8% of all specs show negative coefficients, 98.7% for pcorrupt outcome.
- **Significance**: 41.7% significant at 5%. Marginal in baseline with state FE (p=0.056).
- **Inference sensitivity**: Clustering at state level (26 clusters) often pushes above conventional thresholds.
- **Leave-one-out**: Stable across control omissions (range -0.018 to -0.025).
- **Placebo tests pass**: Predetermined characteristics show no relationship with term status.
