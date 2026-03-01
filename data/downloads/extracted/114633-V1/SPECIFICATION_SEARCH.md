# Specification Search Report: 114633-V1

**Paper:** Lawson (2017), "Liquidity Constraints, Fiscal Externalities and Optimal Tuition Subsidies", AEJ: Economic Policy 9(1)

## Baseline Specification

- **Design:** Structural calibration (education choice model with liquidity constraints)
- **Outcome:** dW/db (marginal welfare gain from increasing tuition subsidy at current policy)
- **Key formula:** dW/db = S * (L - eps_Sb + (1 + G/(Sb)) * eps_Yb)
- **Calibration:** S_hat=0.388, eps_Sb=0.2, dsda=0.0021, ETI=0.4, tau=0.23, r=0.12
- **Model:** Baseline with fiscal externalities

| Statistic | Value |
|-----------|-------|
| dW/db at b_hat | 0.437029 |
| Optimal subsidy b* | 8.3552 |
| Welfare gain | 0.867427 |
| Welfare gain (% of S*b) | 1.117818 |
| L_hat (liquidity param) | 0.057221 |
| eps_Yb (fiscal externality) | 0.014195 |
| G/(Sb) ratio | 88.409730 |

## Specification Counts

- Total specifications: 75
- Successful: 75
- Failed: 0
- Inference variants: 1

## Category Breakdown

| Category | Count | dW/db Range |
|----------|-------|-------------|
| Baseline | 1 | [0.437029, 0.437029] |
| NoFE Model | 14 | [-0.122147, 0.006383] |
| GEHLT Model | 1 | [0.414827, 0.414827] |
| GE Spillovers | 2 | [0.414827, 0.414827] |
| NoLiq Model | 1 | [0.414827, 0.414827] |
| eps_Sb | 13 | [0.171431, 0.893096] |
| dsda/L_hat | 10 | [0.414827, 0.472224] |
| ETI | 5 | [0.435324, 0.441241] |
| tau_hat | 5 | [0.319508, 0.618410] |
| S_hat | 4 | [0.343323, 0.503142] |
| r | 4 | [0.229595, 0.634147] |
| w1_factor | 4 | [-0.039003, 0.968997] |
| Combined Parameters | 5 | [0.181089, 0.731759] |
| Alternative Outcomes | 4 | (other outcomes: [0.001650, 8.355235]) |
| Grid Resolution | 2 | [0.437029, 0.437029] |

## Inference Variants

This is a calibration/structural model. Results are deterministic given parameters.
No statistical inference (p-values, confidence intervals) applies.

## Overall Assessment

- **dW/db specs:** 71 specifications
- **Sign consistency:** Mixed signs across specifications
- **Direction:** Median dW/db is positive (0.425092)
- **Range:** [-0.122147, 0.968997]
- **Positive dW/db:** 58/71 specifications
- **Robustness assessment:** MODERATE

Surface hash: `sha256:fe25d740fded6c5b72b87311405604379a1b70a2b47c83f0cd06f142b4d84464`

## Notes

- This is a purely theoretical/calibration paper with no empirical data.
- The MATLAB sufficient statistics and calibration code was re-implemented in Python.
- Key claim: current tuition subsidies are below optimal (dW/db > 0 at b=b_hat).
- The specification search varies the key sufficient statistics parameters
  (eps_Sb, dsda/L_hat, ETI, tau, S_hat, r, wage premium) and model variants
  (NoFE, GEHLT, GE Spillovers, NoLiq) to assess robustness.
- Positive dW/db means subsidies should be increased from current levels.
- For calibration papers, 'robustness' means the qualitative conclusion holds
  across a wide range of parameter values and model specifications.
