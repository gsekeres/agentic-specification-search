# DSGE Bayesian Estimation Specifications

## Spec ID Format: `dsge/{category}/{variation}`

This method applies to papers that estimate Dynamic Stochastic General Equilibrium (DSGE) models using Bayesian methods (MCMC, Maximum Likelihood with state-space models, etc.).

## Key Characteristics

- Structural macroeconomic model with forward-looking agents
- Parameters estimated via Bayesian inference (MCMC, Metropolis-Hastings)
- State-space representation with Kalman filtering
- Posterior distributions reported for structural parameters
- Model comparison via marginal likelihood or Bayes factors

## Why Standard Specification Search Does Not Apply

DSGE papers differ fundamentally from reduced-form econometric papers:

1. **No regression coefficients**: Results are structural parameters (preference, technology, policy rule coefficients), not treatment effects from exogenous variation

2. **Identification through model structure**: Parameters are identified through cross-equation restrictions implied by the theoretical model, not through treatment/control comparisons

3. **Computational intensity**: Each "specification" requires full Bayesian estimation (typically 50,000-500,000 MCMC draws), taking hours to days of computation

4. **Parameter interpretation**: Structural parameters have specific economic interpretations within the model context, making simple sign/significance comparisons meaningless

---

## Alternative Robustness Framework for DSGE Papers

If robustness analysis is desired, consider the following (each requiring substantial computation):

### Prior Sensitivity

| spec_id | Description |
|---------|-------------|
| `dsge/prior/baseline` | Paper's prior specification |
| `dsge/prior/diffuse` | More diffuse (uninformative) priors |
| `dsge/prior/tight` | Tighter priors around calibrated values |
| `dsge/prior/alternative_distribution` | Different prior distribution families |

### Sample Period Variations

| spec_id | Description |
|---------|-------------|
| `dsge/sample/full` | Full estimation sample |
| `dsge/sample/great_moderation` | 1984-2007 (low volatility period) |
| `dsge/sample/post_crisis` | Post-2008 period |
| `dsge/sample/rolling` | Rolling window estimation |
| `dsge/sample/pre_[year]` | Sample ending before specific year |
| `dsge/sample/post_[year]` | Sample starting after specific year |

### Observable Variables

| spec_id | Description |
|---------|-------------|
| `dsge/observables/baseline` | Paper's baseline observables |
| `dsge/observables/extended` | Additional observables |
| `dsge/observables/reduced` | Subset of observables |
| `dsge/observables/alternative_measure` | Different measures of same variable |

### Model Specification

| spec_id | Description |
|---------|-------------|
| `dsge/model/baseline` | Paper's baseline model |
| `dsge/model/no_[feature]` | Model without specific feature |
| `dsge/model/alternative_[mechanism]` | Alternative mechanism |
| `dsge/model/nested` | Nested model comparison |

### Fixed vs. Estimated Parameters

| spec_id | Description |
|---------|-------------|
| `dsge/fixed/baseline` | Paper's fixed parameter calibration |
| `dsge/fixed/alternative_[param]` | Different value for fixed parameter |
| `dsge/fixed/estimate_[param]` | Estimate previously fixed parameter |
| `dsge/fixed/fix_[param]` | Fix previously estimated parameter |

### Regime Switching (if applicable)

| spec_id | Description |
|---------|-------------|
| `dsge/regime/baseline` | Baseline regime specification |
| `dsge/regime/single_regime` | Single regime (no switching) |
| `dsge/regime/alternative_states` | Different number of states |
| `dsge/regime/alternative_transition` | Different transition probabilities |

---

## Computational Requirements

- **Typical MCMC estimation**: 50,000-500,000 draws
- **Typical runtime**: 2-24 hours per specification on modern hardware
- **Software**: Dynare (MATLAB), IRIS (MATLAB), DSGE.jl (Julia), dsge (R)

---

## Output Format for DSGE Papers

For papers that can be analyzed, record:

```json
{
  "posterior_parameters": [
    {"param": "rho_a", "mean": 0.90, "median": 0.91, "std": 0.02, "ci_5": 0.86, "ci_95": 0.94},
    {"param": "sigma_a", "mean": 0.01, "median": 0.01, "std": 0.001, "ci_5": 0.008, "ci_95": 0.012}
  ],
  "model_fit": {
    "log_marginal_likelihood": -450.2,
    "log_data_density": -448.5,
    "rmse_output": 0.5,
    "rmse_inflation": 0.3
  },
  "prior_posterior_comparison": {
    "prior_mean_rho_a": 0.85,
    "posterior_mean_rho_a": 0.90,
    "prior_sd_rho_a": 0.10,
    "posterior_sd_rho_a": 0.02
  },
  "convergence_diagnostics": {
    "geweke_stat": 0.8,
    "raftery_lewis_dependence_factor": 3.2,
    "acceptance_rate": 0.28
  }
}
```

---

## Recommendation for Specification Search Framework

Papers using DSGE Bayesian estimation should generally be marked as:
- **Status**: `not_applicable`
- **Reason**: "Structural DSGE model - standard specification search framework not applicable"

If partial analysis is possible (e.g., reduced-form regressions in appendices), those can be analyzed separately.

---

## Example Papers Using This Method

- Smets & Wouters (2007) AER - Medium-scale DSGE for US
- Christiano, Eichenbaum, Evans (2005) JPE - Nominal rigidities and monetary policy
- Guerron-Quintana, Hirano, Jinnai (2023) AEJ:Macro - Bubbles and growth (173441-V1)
- An & Schorfheide (2007) ER - Bayesian Analysis of DSGE Models

---

## Checklist for DSGE Papers

Before marking a DSGE paper as not_applicable, verify:

- [ ] Paper uses Bayesian/ML estimation of structural DSGE model
- [ ] Results are structural parameters, not regression coefficients
- [ ] No appendix with reduced-form regressions that could be analyzed
- [ ] Computational requirements preclude rapid specification search
- [ ] Document the model type and estimation method used
