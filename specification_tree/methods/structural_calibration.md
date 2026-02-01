# Structural Calibration Specifications

## Spec ID Format: `structural/{category}/{variation}`

This method applies to papers that calibrate structural economic models to match data moments, rather than estimating reduced-form relationships through regression.

## Key Characteristics

- Model parameters are chosen to match observed data (CDS spreads, asset prices, etc.)
- Results derived from model simulations/solutions rather than regression coefficients
- "Treatment effects" emerge from counterfactual model exercises
- Standard errors often computed via bootstrap or moment uncertainty

---

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main calibrated model
- Record: Calibrated parameters, matched moments, model fit statistics

---

## Core Variations

### Calibration Targets

| spec_id | Description |
|---------|-------------|
| `structural/targets/baseline` | Paper's baseline moment conditions |
| `structural/targets/alternative_moments` | Different data moments targeted |
| `structural/targets/extended` | Additional moments included |
| `structural/targets/subset` | Subset of moments |

### Model Parameters

| spec_id | Description |
|---------|-------------|
| `structural/params/fixed` | Subset of parameters fixed externally |
| `structural/params/all_estimated` | All parameters estimated from data |
| `structural/params/alternative_priors` | Different prior/starting values |

### Functional Form Assumptions

| spec_id | Description |
|---------|-------------|
| `structural/functional/baseline` | Paper's baseline functional forms |
| `structural/functional/alternative_distributions` | Different distributional assumptions |
| `structural/functional/alternative_preferences` | Different preference specifications |

### Sample Period

| spec_id | Description |
|---------|-------------|
| `structural/sample/full` | Full time period |
| `structural/sample/pre_crisis` | Pre-crisis period |
| `structural/sample/post_crisis` | Post-crisis period |
| `structural/sample/rolling` | Rolling window calibration |

### Counterfactual Exercises

| spec_id | Description |
|---------|-------------|
| `structural/counterfactual/baseline` | Paper's main counterfactual |
| `structural/counterfactual/alternative_policy` | Different policy scenario |
| `structural/counterfactual/parameter_variation` | Sensitivity to key parameters |

### Bootstrap/Uncertainty

| spec_id | Description |
|---------|-------------|
| `structural/uncertainty/point` | Point estimates only |
| `structural/uncertainty/bootstrap` | Bootstrap standard errors |
| `structural/uncertainty/bayesian` | Bayesian posterior uncertainty |

---

## Robustness Checks for Structural Models

### Model Fit

| spec_id | Description |
|---------|-------------|
| `structural/robust/overid_test` | Over-identification test |
| `structural/robust/moment_sensitivity` | Sensitivity to individual moments |
| `structural/robust/parameter_bounds` | Check at parameter boundaries |

### External Validity

| spec_id | Description |
|---------|-------------|
| `structural/robust/out_of_sample` | Out-of-sample predictions |
| `structural/robust/holdout_moments` | Moments not used in calibration |
| `structural/robust/cross_section` | Cross-sectional fit |

---

## Output Format

For structural calibration papers, record:

```json
{
  "calibrated_parameters": {
    "param1": {"value": 0.05, "se_bootstrap": 0.01},
    "param2": {"value": 0.20, "se_bootstrap": 0.03}
  },
  "moment_fit": [
    {"moment": "avg_cds_spread", "data": 150, "model": 148, "t_stat": 0.5},
    {"moment": "equity_vol", "data": 0.25, "model": 0.24, "t_stat": 0.8}
  ],
  "counterfactual_results": {
    "treatment_effect": 0.15,
    "se": 0.05,
    "ci_lower": 0.05,
    "ci_upper": 0.25
  },
  "model_fit_statistics": {
    "j_stat": 5.2,
    "j_pval": 0.39,
    "n_moments": 10,
    "n_params": 6
  }
}
```

---

## Notes on Specification Variation

For structural models, systematic specification search involves:

1. **Varying calibration targets**: Which moments to match
2. **Varying fixed parameters**: Which parameters to calibrate vs. fix externally
3. **Varying functional forms**: Distribution assumptions, preference specs
4. **Varying sample periods**: Time windows for calibration
5. **Counterfactual variations**: Different policy experiments

Unlike reduced-form regressions, standard "leave-one-out" covariate analysis does not apply. Instead, perform:
- Leave-one-out moment analysis (drop each target moment)
- Parameter sensitivity analysis (vary each fixed parameter)
- Model specification analysis (alternative functional forms)

---

## Checklist

Before completing a structural calibration analysis, verify:

- [ ] Baseline model replication
- [ ] Report all calibrated parameters with uncertainty
- [ ] Report moment fit (data vs. model)
- [ ] Run at least one alternative sample period
- [ ] Run at least one parameter sensitivity check
- [ ] Report counterfactual results with uncertainty bounds
