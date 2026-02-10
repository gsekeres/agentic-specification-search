# Verification Report: 116347-V1

## Paper Information
- **Title**: Workplace Friendships and Productivity
- **Journal**: AEJ: Applied Economics
- **Total Specifications**: 70 (including ~10 failed/error specs)

## Baseline Groups

### G1: Friend Presence on Productivity
- **Claim**: Working alongside friends increases productivity
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: -0.017 (p=0.84, NOT significant)
- **Outcome**: `logprod`
- **Treatment**: `has_friend_present`

## Classification Summary

The baseline result is null. The treatment variable has almost no within-worker variation (99.3% of observations have a friend present), making the main specification severely underpowered. Specifications without worker FE show significant positive effects, but these reflect selection rather than causation.

## Key Notes

- ~10 specifications failed due to collinearity (marked as ERROR in results)
- The paper's actual main treatment is proximity-based (working alongside friends), not presence-based
- Heterogeneity specs without worker FE show positive effects due to omitted variable bias
- Placebo tests show significant effects, likely reflecting time trends
- First difference estimator shows significant negative effect
