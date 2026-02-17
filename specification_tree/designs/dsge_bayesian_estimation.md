# Design: DSGE Bayesian Estimation (Structural Macro)

This design file enumerates **within-design implementation choices** for papers that estimate DSGE models using Bayesian methods (state-space + Kalman filtering + MCMC / posterior simulation).

Important practical note: full “specification search” is often computationally infeasible for DSGE estimation (hours to days per run). The specification surface should therefore set explicit budgets and may choose to mark many DSGE packages as out-of-scope for execution, while still recording the *menu of relevant axes*.

Universal RC, inference, diagnostics, sensitivity, and exploration menus live in `specification_tree/modules/*` and should be applied separately where meaningful.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical DSGE estimation:
  - model equations/features,
  - observables,
  - priors,
  - sampling scheme (MCMC details),
  - and focal reported quantities (posterior moments, IRFs, marginal likelihood).
- Record design-defining metadata under `coefficient_vector_json.design.dsge_bayesian_estimation` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design implementation variants (`design/dsge_bayesian_estimation/*`)

Spec ID format:

- `design/dsge_bayesian_estimation/{axis}/{variant}`

### A) Prior sensitivity

| spec_id | Description |
|---|---|
| `design/dsge_bayesian_estimation/prior/baseline` | Paper’s priors |
| `design/dsge_bayesian_estimation/prior/diffuse` | More diffuse priors |
| `design/dsge_bayesian_estimation/prior/tight` | Tighter priors |
| `design/dsge_bayesian_estimation/prior/alternative_family` | Alternative prior families |

### B) Sample window

| spec_id | Description |
|---|---|
| `design/dsge_bayesian_estimation/sample/full` | Full estimation sample |
| `design/dsge_bayesian_estimation/sample/pre_crisis` | Pre-crisis window (paper-defined) |
| `design/dsge_bayesian_estimation/sample/post_crisis` | Post-crisis window (paper-defined) |
| `design/dsge_bayesian_estimation/sample/rolling` | Rolling windows (paper-defined) |

### C) Observables / measurement equations

| spec_id | Description |
|---|---|
| `design/dsge_bayesian_estimation/observables/baseline` | Paper’s observables |
| `design/dsge_bayesian_estimation/observables/reduced` | Reduced observables set |
| `design/dsge_bayesian_estimation/observables/extended` | Extended observables set |
| `design/dsge_bayesian_estimation/observables/alternative_measure` | Alternative measure of a key observable |

### D) Model features (nested comparisons; paper-relevant only)

| spec_id | Description |
|---|---|
| `design/dsge_bayesian_estimation/model/baseline` | Paper’s baseline model |
| `design/dsge_bayesian_estimation/model/drop_feature` | Drop a key feature/mechanism (paper-defined) |
| `design/dsge_bayesian_estimation/model/alternative_mechanism` | Alternative mechanism (paper-defined) |

## Standard DSGE diagnostics (record when applicable)

These are **diagnostics**, not estimates of the focal DSGE quantities. They are not part of the default core surface, but they should be computed and recorded when the paper relies on them.

- `diag/dsge_bayesian_estimation/mcmc/convergence` (trace/R-hat-like summaries, acceptance rates)
- `diag/dsge_bayesian_estimation/fit/marginal_likelihood` (or log posterior, paper-specific)
- `diag/dsge_bayesian_estimation/filter/kalman_ok` (filter likelihood computed; no numerical failures)

## Output contract (JSON-heavy)

DSGE outputs are design-specific and typically do not fit cleanly into a flat coefficient vector.
Store them under `coefficient_vector_json.design.dsge_bayesian_estimation`, including:

- priors + posteriors for key parameters,
- sampler configuration (draws, burn-in, chains),
- focal objects (IRFs, variance decompositions, marginal likelihood),
- convergence summaries.

If integrating into a scalar evidence pipeline, the surface must declare a focal scalar summary (e.g., a particular IRF response at a horizon, or a log marginal likelihood difference), and that choice must be recorded in JSON.

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Sample rules (window, exclusions): `specification_tree/modules/robustness/sample.md`
- Pre-processing/coding (deflators, scaling, standardization): `specification_tree/modules/robustness/preprocessing.md`
- Data construction (aggregation, measurement choices): `specification_tree/modules/robustness/data_construction.md`

### Inference (`infer/*`)

- Resampling / uncertainty procedures (when representing DSGE outputs with scalar summaries): `specification_tree/modules/inference/resampling.md`

### Post-processing (`post/*`)

- Specification-curve / multiverse summaries: `specification_tree/modules/postprocess/specification_curve.md`

### Exploration (`explore/*`)

- Alternative scalar summaries / estimands (IRF horizons, welfare/policy objects): `specification_tree/modules/exploration/alternative_estimands.md`
- Variable definitions / measurement choices: `specification_tree/modules/exploration/variable_definitions.md`
