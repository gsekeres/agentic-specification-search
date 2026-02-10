# Verification Report: 116433-V1

## Paper Information
- **Title**: Referrals: Peer Screening and Enforcement in a Consumer Credit Field Experiment
- **Authors**: Bryan, Karlan, and Zinman
- **Journal**: American Economic Review
- **Total Specifications**: 236

## Baseline Groups

### G1: Enforcement Effect
- **Claim**: Conditioning referrer reward on borrower repayment reduces defaults
- **Expected sign**: Negative (for repaid: negative means fewer repaid... actually for charged_off: negative = fewer defaults = GOOD)
- **Baseline specs**: 4 (one per outcome: repaid, charged_off, interest, portion)
- **Key result**: charged_off enforcement coefficient = -0.102 (p=0.016)

### G2: Selection Effect
- **Claim**: Asking referrers to screen improves loan performance
- **Expected sign**: Positive (for repaid)
- **Baseline specs**: 4 (one per outcome)
- **Key result**: repaid selection coefficient = 0.039 (p=0.41, NOT significant)

## Key Notes

- Specs come in enforcement/selection pairs for each outcome
- Enforcement effect is robust; selection effect is consistently null
- Small sample (N=243 loans) limits power
- Leave-one-out and control progression specs confirm stability
- Heterogeneity by relationship type (work vs relative) shows interesting patterns
- Balance/placebo tests confirm randomization
