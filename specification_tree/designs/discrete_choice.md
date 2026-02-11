# Design: Discrete Choice / Discrete Outcomes

This design file enumerates **within-design estimator implementations** for papers whose primary evidence object is a model for a discrete outcome (binary, ordered, multinomial), typically interpreted under maintained selection-on-observables or randomized-assignment assumptions embedded in the paper’s design.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical discrete-outcome estimate for the claim object.

**Focal-parameter rule (important)**: papers differ in what they treat as the headline object:

- raw index coefficients (log-odds / latent-index units),
- odds ratios / relative-risk ratios,
- average marginal effects (AME).

The scalar `coefficient/std_error/p_value` must correspond to the paper’s focal object, and the full set of coefficients and any marginal-effects outputs must be stored in `coefficient_vector_json`.

## Design estimator implementations (`design/discrete_choice/*`)

Spec ID format:

- `design/discrete_choice/{axis}/{variant}`

### A) Binary-outcome model family

| spec_id | Description |
|---|---|
| `design/discrete_choice/model/logit` | Logistic regression |
| `design/discrete_choice/model/probit` | Probit regression |
| `design/discrete_choice/model/lpm` | Linear probability model (comparison) |
| `design/discrete_choice/model/cloglog` | Complementary log-log |

### B) Multinomial / ordered model family (paper-relevant only)

| spec_id | Description |
|---|---|
| `design/discrete_choice/model/mlogit` | Multinomial logit |
| `design/discrete_choice/model/ordered_logit` | Ordered logit |
| `design/discrete_choice/model/ordered_probit` | Ordered probit |
| `design/discrete_choice/model/nested_logit` | Nested logit (when implementable) |

### C) High-dimensional fixed effects for discrete outcomes (optional)

Adding FE via absorption is not always available for nonlinear MLE. When the paper uses a conditional-likelihood approach, treat it as a within-design estimator:

| spec_id | Description |
|---|---|
| `design/discrete_choice/estimator/conditional_logit` | Conditional logit with group FE (when applicable) |
| `design/discrete_choice/estimator/correlated_random_effects` | Random-effects + Mundlak (comparison; assumption change) |

## Standard discrete-choice diagnostics (record when applicable)

These are **diagnostics**, not estimates of the focal estimand. They are not part of the default core surface, but should be computed/recorded when the paper relies on them.

- `diag/discrete_choice/fit/pseudo_r2` (McFadden or paper-specific)
- `diag/discrete_choice/fit/ll_aic_bic`
- `diag/discrete_choice/iia/test` (multinomial logit only; if implementable)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls / adjustment set: `specification_tree/modules/robustness/controls.md`
- Sample rules: `specification_tree/modules/robustness/sample.md`
- Fixed effects (when linear approximations are used): `specification_tree/modules/robustness/fixed_effects.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`
- Functional form (transformations of continuous covariates): `specification_tree/modules/robustness/functional_form.md`
- Weights: `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- SE and clustering (sandwich, clustered, etc.): `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- General regression diagnostics: `specification_tree/modules/diagnostics/regression_diagnostics.md`

### Exploration (`explore/*`)

- Alternative estimands (e.g., distributional effects): `specification_tree/modules/exploration/alternative_estimands.md`
- Heterogeneity / subgroup effects: `specification_tree/modules/exploration/heterogeneity.md`

