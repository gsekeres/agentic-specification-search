# Verification Report: 120078-V1

## Paper Information
- **Title**: Can Information Reduce Ethnic Discrimination? Evidence from Airbnb
- **Journal**: Not specified
- **Total Specifications**: 77

## Baseline Groups

### G1: Minority Name Price Penalty
- **Claim**: Minority-name hosts face lower prices on Airbnb
- **Expected sign**: Negative
- **Baseline coefficient range**: -0.034 to -0.169 across Table 2 columns
- **Outcome**: `log_price`
- **Treatment**: `minodummy`

## Key Notes

- VERY ROBUST finding: 98.7% negative, 97.4% significant at 5%
- Only 1 positive coefficient out of 77 specs
- Effect magnitude depends on controls and FE: 3-17%
- City-level clustering weakens significance but result survives (p<0.001)
- Barcelona and Berlin city subsamples show near-zero insignificant effects
- Placebo tests: random treatment shows near-zero effect; picture change shows tiny effect
