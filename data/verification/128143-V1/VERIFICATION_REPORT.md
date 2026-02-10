# Verification Report: 128143-V1

## Paper Information
- **Title**: Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion
- **Authors**: Thomas Douenne & Adrien Fabre
- **Journal**: AEJ: Economic Policy
- **Total Specifications**: 77

## Baseline Groups

### G1: Self-Interest Beliefs Channel
- **Claim**: Believing one does not lose from a carbon tax with dividend increases tax acceptance by ~36pp.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.363 (p < 0.001)
- **Outcome**: `tax_acceptance`
- **Treatment**: `believes_not_lose`

### G2: Environmental Effectiveness Channel
- **Claim**: Believing the carbon tax is environmentally effective increases tax approval by ~40pp.
- **Baseline spec**: `baseline_ee`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.405 (p < 0.001)
- **Outcome**: `tax_approval`
- **Treatment**: `believes_effective`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **48** | |
| core_controls | 25 | Two baselines + control variations + leave-one-out + control blocks |
| core_sample | 17 | Gender, age, urban/rural, YV support, income, ecologist + EE subsamples + winsorize |
| core_inference | 4 | Classical SE, unweighted, region cluster, region FE |
| core_funcform | 3 | Quadratic gain, YV interaction, income interaction |
| core_method | 6 | IV/2SLS specifications (EE and SI channels) |
| **Non-core tests** | **29** | |
| noncore_alt_outcome | 5 | Tax approval, targeted acceptance/approval, feedback acceptance, EE acceptance |
| noncore_alt_treatment | 3 | Believes wins, simulated winner, continuous gain |
| noncore_placebo | 3 | First-stage check, reduced form info, ecologist placebo |
| noncore_heterogeneity | 6 | Interactions with female, urban, education, ecologist, age |
| noncore_diagnostic | 2 | Cross-outcome checks |
| **Total** | **77** | |

## Robustness Assessment

**STRONG** support. 97.4% positive, 93.5% significant at 5%. IV estimates (0.40-0.60) are larger than OLS (0.36), consistent with attenuation bias. Results stable across all subgroups.
