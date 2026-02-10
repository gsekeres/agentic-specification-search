# Verification Report: 116540-V2

## Paper Information
- **Title**: Individual Development Accounts and Homeownership among Low-Income Renters
- **Journal**: AEJ: Policy
- **Total Specifications**: 80

## Baseline Groups

### G1: IDA Treatment on Homeownership
- **Claim**: IDA program increases homeownership
- **Expected sign**: Positive
- **Baseline coefficient**: 0.011 (p=0.77, NOT significant)
- **Outcome**: `own_home_u42`
- **Treatment**: `treat`

## Key Notes

- ROBUST NULL FINDING: 0 out of 80 specifications significant at 5%
- 80% of specs show positive coefficients, but effects are tiny (~1-3 pp)
- Study likely underpowered (N=652, SE~0.04)
- One near-significant heterogeneity: high-income interaction (p=0.063)
- Placebo tests show no effects (as expected for RCT)
- Results remarkably stable across all control sets and sample restrictions
