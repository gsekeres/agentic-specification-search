# Verification Report: 120568-V1

## Paper Information
- **Title**: Declining Worker Turnover: The Role of Short-Duration Employment Spells
- **Authors**: Michael Pries and Richard Rogerson
- **Journal**: AEJ: Macroeconomics (2021)
- **Total Specifications**: 74

## Baseline Groups

### G1: Time Trend in One-Quarter Hazard Rate
- **Claim**: The one-quarter hazard rate has declined over time
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0745 (p<0.0001)
- **Outcome**: `oneqhazrate`
- **Treatment**: `time_trend`
- **NOTE**: This is a descriptive finding, not a causal estimate

## Key Notes

- VERY ROBUST trend: 93% negative, 88% significant at 5%
- All 11 alternative outcomes show declining trends
- Heterogeneity: young workers show largest declines (-10% to -14%)
- Post-2010 subsample shows near-zero trend (flattening)
- Controlling for hire rate reduces coefficient by ~40% (cyclical component)
- 2 of 19 industries (Information, FIRE) show non-negative trends
- All 5 permutation placebos are insignificant
- HAC inference with various lag structures does not change conclusions
