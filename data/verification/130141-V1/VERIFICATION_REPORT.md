# Verification Report: 130141-V1

## Paper Information
- **Title**: News Shocks under Financial Frictions
- **Authors**: Gortz, Tsoukalas, and Zanetti
- **Journal**: AEJ: Macroeconomics
- **Method**: Structural VAR with Cholesky identification
- **Total Specifications**: 58

## Baseline Groups

### G1: TFP News Shock FEVD Share
- **Claim**: TFP news shocks explain ~22% of GDP forecast error variance at 20-quarter horizon.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive (FEVD is always non-negative)
- **Baseline FEVD**: 22.3%
- **Outcome**: GDP FEVD share
- **Treatment**: TFP shock

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **40** | |
| core_controls | 14 | Baseline + variable sets + leave-one-out + variable progression |
| core_sample | 8 | Sample period variations + exclusions |
| core_funcform | 10 | Lag length variations + IC-optimal lags |
| core_method | 6 | Cholesky ordering variations |
| **Non-core tests** | **18** | |
| noncore_alt_outcome | 4 | Investment, hours, consumption, SP500 as main FEVD outcome |
| noncore_heterogeneity | 6 | First/second half + rolling windows |
| noncore_placebo | 2 | Shuffled TFP, lagged TFP |
| **Total** | **58** | |

## Robustness Assessment

**MODERATE** support. 100% positive FEVD (mechanical), 69% above 10%. But highly sensitive to: sample period (excluding 2008-2009 drops to 2.4%), lag selection (7.5% at p=1 vs 42.6% at p=8), and Cholesky ordering (6-22%). Rolling windows show substantial instability.
