# Specification Search Report: 114098-V1

**Paper:** Eden (2015), "Excessive Financing Costs in a Representative Agent Framework", AEJ: Macroeconomics 7(2)

## Baseline Specification

- **Design:** Structural calibration (representative agent model with intermediation costs)
- **Outcome:** Welfare gain from implementing optimal intermediation tax
- **Key parameter:** tau* (optimal intermediation tax rate)
- **Calibration:** theta=0.015, alpha=0.18, alphatil=0.08, beta=1/1.03, delta=0.10
- **Labor supply:** Inelastic (baseline), elastic (appendix)
- **Grid:** 100 points on [0, 0.1]

| Statistic | Value |
|-----------|-------|
| Welfare gain | 0.00877455 |
| tau* | 0.031313131313131314 |
| Finance share change | -0.015396361252698684 |
| Output change (%) | -0.04201268527052344 |
| Consumption change (%) | 0.008813156744878163 |
| Capital change (%) | -0.21215079330039321 |

## Specification Counts

- Total specifications: 64
- Successful: 64
- Failed: 0
- Inference variants: 1

## Category Breakdown

| Category | Count | Coef Range (welfare_gain) |
|----------|-------|---------------------------|
| Baseline | 1 | [0.00877455, 0.00877455] |
| Elastic Labor | 3 | [0.00966340, 0.01541374] |
| Theta | 15 | [0.00449345, 0.01402765] |
| Alpha | 4 | [0.00737290, 0.01071083] |
| Alphatil | 4 | [0.00298657, 0.02382829] |
| Alpha Share (const total) | 10 | [0.00000000, 0.03030152] |
| Beta | 4 | [0.00269431, 0.02708999] |
| Delta | 4 | [0.00579720, 0.01105702] |
| Intermediated Assets | 2 | [0.00871002, 0.00884004] |
| Grid Resolution | 3 | [0.00877124, 0.00877557] |
| Grid Range | 2 | [0.00877386, 0.00877559] |
| Combined Parameters | 6 | [0.00423849, 0.02083797] |
| Alternative Outcomes | 6 | (other outcomes: [-0.81322935, 0.03131313]) |

## Inference Variants

This is a calibration/structural model. Results are deterministic given parameters.
No statistical inference (p-values, confidence intervals) applies.

## Overall Assessment

- **Welfare gain specs:** 58 specifications
- **Sign consistency:** Mixed signs across specifications
- **Direction:** Median welfare gain is positive (0.00880782)
- **Range:** [0.00000000, 0.03030152]
- **Positive welfare gain:** 57/58 specifications
- **Robustness assessment:** MODERATE

Surface hash: `sha256:a6bad6cc415a11c29f59aa17e20f78ca76ed775d880255de0fd4c4fc26075b8e`

## Notes

- This is a purely theoretical/calibration paper with no empirical data.
- The MATLAB calibration code was re-implemented in Python (scipy.optimize).
- Specification search varies model parameters (theta, alpha, alphatil, beta, delta, eta)
  and numerical choices (grid resolution, grid range) to assess robustness of the
  key finding: implementing the optimal intermediation tax yields positive welfare gains.
- The paper's comparative statics (varying theta and alpha_share) are replicated exactly.
- For calibration papers, 'robustness' means the qualitative conclusion (positive welfare gain)
  holds across a wide range of parameter values.
