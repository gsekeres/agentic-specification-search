# Verification Report: 134041-V1

## Paper Information
- **Title**: How Do Beliefs about the Gender Wage Gap Affect the Demand for Public Policy?
- **Journal**: AEJ-Policy
- **Method**: Cross-sectional OLS (survey experiment)
- **Total Specifications**: 93

## Baseline Groups

### G1: Policy Demand Index
- **Claim**: Information about the gender wage gap increases demand for public policies addressing gender inequality.
- **Baseline spec**: `baseline` (z_lmpolicy_index)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.062 (SE: 0.025, p = 0.014)
- **Outcome**: `z_lmpolicy_index`
- **Treatment**: `T1` (information treatment)

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **54** | |
| core_controls | 33 | Baseline + no controls + LOO (20) + incremental addition (10) + control blocks (2) |
| core_sample | 19 | Wave, gender, party, region, age, trimming, employment |
| core_inference | 4 | Unweighted, HC2, HC3, wave clustering |
| core_funcform | 2 | Standardized, quadratic prior |
| **Non-core tests** | **39** | |
| noncore_alt_outcome | 20 | Individual policy components (5 baselines), manipulation indices, no-controls variants for alt outcomes |
| noncore_placebo | 5 | Predetermined characteristics (gender, democrat, age, midwest, republican) |
| noncore_heterogeneity | 6 | Gender, democrat, independent, education, employee, multiple interactions |
| **Total** | **93** | |

## Robustness Assessment

**MODERATE** support. z_lmpolicy_index is significant (p=0.014) and 97.8% of specs positive. However, individual policy components show mixed significance. LOO analysis very stable. Subgroup effects vary (Democrats vs. Republicans).

## Notes on Baseline Multiplicity

The CSV contains 6 rows with spec_id="baseline" for different outcomes. The primary baseline is z_lmpolicy_index. The other 5 (quotaanchor, AAanchor, legislationanchor, transparencyanchor, childcare) are individual policy components that form the index. Only 3 of 6 baseline outcomes are significant at 5%.
