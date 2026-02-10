# Verification Report: 114717-V1

## Paper Information
- **Title**: How Punishment Severity Affects Jury Verdicts: Evidence from Two Natural Experiments
- **Authors**: Anna Bindler, Randi Hjalmarsson
- **Journal**: AEJ: Economic Policy
- **Total Specifications**: 69

## Baseline Groups

### G1: Death Penalty Abolition Effect on Conviction Rates
- **Claim**: Abolishing the death penalty for specific offenses increases jury conviction rates.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0764 (SE: 0.0365, p = 0.036)
- **Outcome**: `guilty_jury_dummy`
- **Treatment**: `not_death_punishable`
- **N**: 104,670; R2 = 0.068
- **Table 3, Panel A, Column 2**

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **39** | |
| core_controls | 11 | Baseline + no controls + offense trends (linear, specific) + session count + num cases + criminal history + age (2 variants) + full |
| core_fe | 5 | No FE, year only, offense only, month only, year+offense |
| core_sample | 17 | By offense type (6), time period (5), defendant characteristics (3), bandwidth, reform exclusion |
| core_inference | 3 | Robust SE, cluster by year, cluster by offense x year |
| **Non-core tests** | **30** | |
| noncore_alt_outcome | 24 | Guilty of original charge (5), lesser charge (5), mercy (5), guilty incl pleas (4), plea (1), punishment types (3), plea by offense (1) |
| noncore_heterogeneity | 2 | Treatment x male, treatment x criminal history |
| noncore_placebo | 4 | Transportation experiment (second natural experiment) |
| **Total** | **69** | |

## Robustness Assessment

The main finding is **moderately robust**:

- **100% sign consistency**: The coefficient on not_death_punishable is positive in all 40 main-outcome specifications.
- **67.5% significant at 5%**, 80% at 10% -- decent but not overwhelming given the 26-cluster concern.
- **Key heterogeneity**: Effect is driven by violent/sex offenses (coef ~0.17-0.29, p<0.001) with no significant effect for property offenses alone (coef=0.015, p=0.36).
- **FE stability**: Coefficient is stable (0.076-0.106) across FE configurations.
- **Age sensitivity**: Adding age controls drops sample to 79K and coefficient drops substantially (to 0.005-0.014), suggesting composition sensitivity.
- **Alternative inference**: Robust SE and alternative clustering yield highly significant results (p<0.001), confirming that the 26-cluster baseline is the most conservative approach.
- **Transportation experiment**: The second experiment provides supporting evidence with opposite direction (introducing severe punishment reduces convictions).
