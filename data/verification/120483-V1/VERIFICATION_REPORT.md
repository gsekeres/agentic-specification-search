# Verification Report: 120483-V1

## Paper Information
- **Title**: Malaria Ecology and the Spread of Slavery in the Early United States
- **Journal**: AEJ: Applied Economics
- **Total Specifications**: 76

## Baseline Groups

### G1: Malaria and Slave Ratio
- **Claim**: Higher malaria ecology => higher slave population shares
- **Expected sign**: Positive
- **Key baseline (1860 full)**: coef=0.192 (p<0.001)
- **Key baseline (1790 full)**: coef=0.065 (p=0.156, NOT significant)
- **Outcome**: `slaveratio`
- **Treatment**: `MAL`

## Key Notes

- 95% of specs positive, 67% significant at 5%
- 1860 data much stronger than 1790 data
- Full controls in 1790 render the effect insignificant (p=0.156)
- Leave-one-out controls: dropping tobacco or ELEV pushes toward marginal significance
- Alternative malaria measures all confirm the finding
- Slave state interaction is negative and insignificant -- complex heterogeneity
- Political outcomes (voting) provide supporting evidence
- Cross-country (Americas) analysis also shows positive association
