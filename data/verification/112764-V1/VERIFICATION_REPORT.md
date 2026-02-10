# Verification Report: 112764-V1

## Paper
**Title**: Sovereign Debt Booms in Monetary Unions
**Authors**: Mark Aguiar, Manuel Amador, Emmanuel Farhi, and Gita Gopinath
**Journal**: AER Papers and Proceedings, 2014
**Method**: Structural calibration (continuous-time sovereign debt model)

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Monetary union membership leads to sovereign overborrowing: equilibrium debt (bstar) exceeds inflation threshold (bPi) | + (bstar > bPi) | baseline |

This is a purely theoretical paper with no empirical data. The "coefficient" is the equilibrium debt level bstar from the calibrated model. Robustness is assessed via parameter sensitivity rather than statistical inference.

- **Baseline bstar**: 1.677 (paper: 1.679, ~0.12% difference from ODE discretization)
- **Baseline bPi**: 0.198
- **Baseline overborrowing**: bstar - bPi = 1.479 (bstar is 8.5x the inflation threshold)
- **Baseline SE / p-value**: Not applicable (theoretical model)
- **Parameters**: chi=0.1547, rho=0.07, r=0.06, psi=0.2, piBar=0.2, log utility

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **80** |
| Baselines | 1 |
| Core tests (non-baseline) | 74 |
| Non-core tests | 5 |
| Converged | 78 |
| Failed to converge | 2 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 1 | Exact paper replication |
| core_param_sensitivity | 60 | Variations of chi, rho, r, psi, piBar, income, spread, joint, extreme |
| core_funcform | 7 | CRRA utility with sigma in {0.5, 0.75, 1.5, 2.0, 3.0, 4.0, 5.0} |
| noncore_counterfactual | 3 | No union, high inflation, strict targeting regimes |
| noncore_placebo | 2 | r > rho configurations (different dynamic regime) |

## Classification Decisions

### Core Test Classifications

**Parameter sensitivity (60 specs)**: All single-parameter variations (chi, rho, r, psi, piBar, income), spread variations, joint parameter variations, and extreme parameter combinations are classified as core. In a calibration paper, varying calibration parameters is the direct analog of robustness checks in an empirical paper -- each tests whether the qualitative claim (overborrowing) holds under alternative plausible parameterizations.

- **Default cost chi (11 specs)**: Chi is the most influential parameter. bstar ranges from 0.19 (chi=0.05) to the baseline 1.68. The overborrowing result (bstar > bPi) holds in all converging specs.
- **Discount rate rho (10 specs)**: bstar ranges from 0.43 (rho=0.09) to 1.17 (most other values). The overborrowing result holds in all specs.
- **Interest rate r (9 specs)**: bstar ranges from 0.60 (r=0.12) to 3.10 (r=0.02). Lower interest rates allow much more borrowing.
- **Inflation cost psi (9 specs, 1 failed)**: bstar ranges from 0.14 (psi=0.05) to 1.17 (psi=0.15). At psi=0.75, overborrowing_abs is negative (-0.21), indicating the qualitative result may break under very high inflation costs. psi=1.00 did not converge.
- **Inflation rate piBar (7 specs)**: bstar ranges from 0.30 (piBar=0.05) to 1.22 (piBar=0.15). The overborrowing result holds in all specs.
- **Spread rho-r (6 specs)**: 5 of 6 are exact duplicates of rho variation specs (same parameter values, same results). Only spread_0.050 (rho=0.11) is unique.
- **Income y (5 specs)**: bstar scales nearly linearly with income. Overborrowing holds in all specs.
- **Joint variations (6 specs)**: Joint chi-piBar and joint r-rho variations. All exhibit overborrowing.
- **Extreme parameters (4 specs, 1 failed)**: Very patient/impatient government and very costly/cheap default. costly_default is a duplicate of chi_0.50. cheap_default (chi=0.02) did not converge.

**Functional form (7 specs)**: CRRA utility generalizations with sigma from 0.5 to 5.0. These test whether the overborrowing mechanism depends on the log utility assumption. The result holds across all sigma values, confirming it is not an artifact of log utility. bstar ranges from 0.98 (sigma=5.0) to 1.43 (sigma=1.5).

### Non-Core Classifications

**Counterfactuals (3 specs)**: These test fundamentally different economic regimes rather than robustness of the baseline calibration:
- **No monetary union (piBar=0)**: Eliminates the inflation channel entirely. This is a placebo-like test -- it removes the mechanism to verify the model nesting. bstar=1.388 in this case, but bmax=bmaxNoPi, confirming the monetary union channel is shut off. Not a robustness test of the baseline claim.
- **High inflation, low cost (piBar=0.5, psi=0.1)**: Tests a qualitatively different monetary union regime with very high inflation and low inflation cost. This is a counterfactual policy scenario, not a robustness check.
- **Strict inflation targeting (piBar=0.02, psi=1.0)**: Tests a regime where inflation targeting is very strict. Again, a policy counterfactual rather than a calibration robustness check.

**Placebo (2 specs)**: These set r > rho (interest rate exceeding discount rate), which places the model in a fundamentally different dynamic regime from the baseline. Note that both are duplicates of rho variation specs (rho=0.04 and rho=0.05), which are classified as core -- the distinction is that these are labeled as placebos by the specification search itself because the r > rho regime is economically different.

## Notable Issues

### 1. Extensive duplication
At least 7 specifications are exact duplicates of other specifications:
- 5 spread specs duplicate rho specs (spread_0.005 = rho_0.065, spread_0.015 = rho_0.075, spread_0.020 = rho_0.080, spread_0.030 = rho_0.090, spread_0.040 = rho_0.100)
- extreme/costly_default duplicates chi_0.50
- placebo/r_gt_rho duplicates rho_0.040; placebo/r_slightly_gt_rho duplicates rho_0.050

After deduplication, there are approximately 71 unique specifications.

### 2. Suspicious invariance in rho specifications
Multiple rho specifications (rho=0.030, 0.040, 0.050, 0.060, 0.100, 0.120) all produce the identical bstar=1.1005030007087973. This is suspicious -- one would expect different discount rates to produce different equilibrium debt levels. The rho=0.075, 0.080, and 0.090 specs produce different values, but the clustering of identical values at rho=0.030-0.060 and rho=0.100-0.120 suggests a possible bug in the ODE solver or a flat region of the solution. This warrants investigation.

### 3. Qualitative result generally robust
The overborrowing result (bstar > bPi) holds in 76 of 78 converging specifications. The one clear exception is psi=0.75, where overborrowing_abs = -0.21 (bstar < bPi). This suggests that very high inflation costs can eliminate the overborrowing result, which is economically sensible: if inflation is sufficiently costly, the inflation channel becomes too expensive to use, and the monetary union behaves more like autarky.

### 4. No statistical inference
As a calibration paper, there are no standard errors, p-values, or confidence intervals. The entire "robustness" assessment is parameter sensitivity. This is appropriate for the methodology but means that the specification search provides a fundamentally different kind of evidence than for empirical papers.

### 5. Two specifications did not converge
- psi=1.00 (extreme inflation cost)
- chi=0.02 (very cheap default)

Both are extreme parameter values where the ODE solver failed. The failure at chi=0.02 is notable because very low default costs represent an interesting economic case where the default option is not very punitive.

## Recommendations

1. **Investigate the rho invariance**: The identical bstar values across multiple rho settings is the most concerning finding and may indicate a solver issue in the specification search code.

2. **Flag psi=0.75 as a boundary case**: This is the only converging specification where the qualitative overborrowing result fails. It may be useful for bounding the parameter space where the model's predictions hold.

3. **Remove duplicates from analysis**: 7 duplicate specifications should be consolidated, leaving approximately 71 unique specs.

4. **Treat the 5 non-core specs as supplementary**: The counterfactual and placebo specs provide useful economic context but should not be included in a specification curve analysis of the baseline claim's robustness.
