# Verification Report: 125321-V1

## Paper Information
- **Title**: Can Technology Solve the Principal-Agent Problem? Evidence from China's Pollution Monitoring
- **Authors**: Greenstone, He, Jia, Liu
- **Journal**: American Economic Review
- **Total Specifications**: 70

## Baseline Groups

### G1: Automation Effect on Reported PM10
- **Claim**: Automation increases reported PM10 (reduces under-reporting)
- **Expected sign**: Positive
- **Baseline coefficient**: 32.71 ug/m3 (p<0.001)
- **Outcome**: `pm10`
- **Treatment**: `after` (post-automation indicator)

## Key Notes

- 90% positive, 83% significant at 5%
- Global polynomial specs very robust (orders 1-4 all give ~33 ug/m3)
- Local RD at narrow bandwidths (30-90 days) noisy and often insignificant
- Wider bandwidths (120+ days) give significant results
- Weather placebo tests: temp and rain show no discontinuity (good)
- Wind speed and relative humidity show small discontinuities (some concern)
- Placebo cutoffs: pre-automation cutoffs show smaller effects, post-automation cutoffs show even smaller
- SO2 and NO2 also increase after automation, consistent with manipulation story
- Log and IHS transformations confirm the finding
- Donut hole estimates (excluding 7-30 days around cutoff) very similar to baseline
