# Specification Search Run Log: 113561-V1

**Paper**: Fong & Luttmer (2009), "What Determines Giving to Hurricane Katrina Victims?"

**Date**: 2026-02-15

## Surface Summary

- **Design**: Randomized experiment
- **Baseline groups**: 4 (G1: giving, G2: hypgiv_tc500, G3: subjsupchar, G4: subjsupgov)
- **Treatment**: picshowblack (randomly assigned photo showing Black victims)
- **Population**: White respondents who passed soundcheck (N=915)
- **Canonical inference**: HC1 robust SE (pyfixest vcov="hetero")
- **Seed**: 113561

## Counts

| Metric | Count |
|--------|-------|
| Total specs planned | 90 |
| Total specs executed | 90 |
| Successful | 90 |
| Failed | 0 |
| Inference variants | 1 |

### By Baseline Group

| Group | Outcome | Planned | Executed | Successful |
|-------|---------|---------|----------|------------|
| G1 | giving | 23 | 23 | 23 |
| G2 | hypgiv_tc500 | 23 | 23 | 23 |
| G3 | subjsupchar | 22 | 22 | 22 |
| G4 | subjsupgov | 22 | 22 | 22 |

### By Spec Type

| Spec type | Count |
|-----------|-------|
| baseline | 4 |
| design/* | 4 |
| rc/controls/sets/* | 12 |
| rc/controls/loo/* | 28 |
| rc/controls/progression/* | 16 |
| rc/sample/restriction/* | 16 |
| rc/weights/* | 8 |
| rc/form/* | 2 |

## Key Results

### G1: Actual Giving (giving)
- **Baseline**: coef = -4.20, SE = 4.68, p = 0.370 (N=915)
- Treatment effect is negative but not statistically significant across all 23 specifications.
- Range of coefficients: -8.87 (Biloxi only) to +1.46 (Slidell only)
- All p-values > 0.18

### G2: Hypothetical Giving (hypgiv_tc500)
- **Baseline**: coef = -2.18, SE = 4.06, p = 0.591 (N=913)
- Treatment effect is negative but far from significance across all 23 specifications.
- Range of coefficients: -3.97 to -0.53
- All p-values > 0.24

### G3: Charity Support (subjsupchar)
- **Baseline**: coef = -0.22, SE = 0.16, p = 0.168 (N=907)
- Consistently negative but not significant at 5% level.
- Range of coefficients: -0.28 to +0.09
- p-values range from 0.095 to 0.694

### G4: Government Support (subjsupgov)
- **Baseline**: coef = -0.44, SE = 0.20, p = 0.026 (N=913)
- This is the only outcome where the baseline is significant at 5%.
- Coefficient is robustly negative across most specifications.
- 13 of 22 specs have p < 0.05; range extends from -0.50 to -0.14.
- Significant in most control variations, but loses significance in Biloxi-only and race-shown-only subsamples.

## Deviations from Surface

None. All planned specifications were executed successfully.

## Software Stack

- Python 3.x
- pyfixest (vcov="hetero" for HC1, vcov="HC3" for HC3 inference variant)
- pandas, numpy
- Data: katrina.dta (Stata format, loaded via pd.read_stata)
