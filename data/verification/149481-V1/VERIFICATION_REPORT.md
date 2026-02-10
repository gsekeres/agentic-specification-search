# Verification Report: 149481-V1

## Paper Information
- **Title**: Do Thank-You Calls Increase Charitable Giving? A Field Experiment
- **Authors**: Various
- **Journal**: AER
- **Total Specifications**: 147

## Baseline Groups

### G1: Thank-you calls affect donation probability (Experiment 1)
- **Expected sign**: +
- **Baseline spec(s)**: exp1/donated/baseline
- **Outcome**: donated
- **Treatment**: treat
- **Notes**: Coef=0.0002, p=0.908. Null result. N=494,646.

### G2: Thank-you calls affect gift amount conditional on donating (Experiment 1)
- **Expected sign**: +
- **Baseline spec(s)**: exp1/gift_cond/baseline
- **Outcome**: gift_cond
- **Treatment**: treat
- **Notes**: Coef=0.192, p=0.858. Null result.

### G3: Thank-you calls affect donation probability (Experiment 2)
- **Expected sign**: +
- **Baseline spec(s)**: exp2/donated/baseline
- **Outcome**: donated
- **Treatment**: treat
- **Notes**: Coef=-0.0001, p=0.968. Null result. N=57,643.

### G4: Thank-you calls affect gift amount conditional on donating (Experiment 2)
- **Expected sign**: +
- **Baseline spec(s)**: exp2/gift_cond/baseline
- **Outcome**: gift_cond
- **Treatment**: treat
- **Notes**: Coef=-0.261, p=0.888. Null result.

### G5: Thank-you calls affect donation probability (Experiment 3)
- **Expected sign**: +
- **Baseline spec(s)**: exp3/donated/baseline
- **Outcome**: donated
- **Treatment**: treat
- **Notes**: Coef=0.013, p=0.134. Null result (marginal). N=24,348.

### G6: Thank-you calls affect gift amount conditional on donating (Experiment 3)
- **Expected sign**: +
- **Baseline spec(s)**: exp3/gift_cond/baseline
- **Outcome**: gift_cond
- **Treatment**: treat
- **Notes**: Coef=2.760, p=0.466. Null result.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 6 |
| Core tests (non-baseline) | 105 |
| Non-core tests | 36 |
| **Total** | **147** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 62 |
| core_fe | 4 |
| core_funcform | 9 |
| core_inference | 18 |
| core_sample | 18 |
| noncore_heterogeneity | 34 |
| noncore_placebo | 2 |

## Global Notes

Three RCTs testing thank-you calls on charitable giving. Paper finds robust null result (no significant effect). 147 specs across 3 experiments x 2 outcomes. Only 1.4% significant at 5%. The null finding is itself the key result and is highly robust.
