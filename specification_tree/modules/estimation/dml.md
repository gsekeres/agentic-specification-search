# Double / Debiased Machine Learning (DML) (Estimation Wrapper)

## Positioning (important)

DML is **not** a primary design/identification strategy. It is a *nuisance-learning + orthogonalization wrapper* that can be used **conditional on** a baseline claim object and a maintained identification assumption (typically unconfoundedness / selection-on-observables, sometimes IV via orthogonal scores).

In this repo:

- DML belongs under `rc/estimation/dml/*` **only when it preserves the baseline claim object**.
- CATE and policy learning belong in `explore/*` (see `specification_tree/modules/exploration/cate_estimation.md` and `specification_tree/modules/exploration/policy_learning.md`).

## Spec ID format

Use:

- `rc/estimation/dml/{score_family}/{variant}`

Examples:

- `rc/estimation/dml/plr/default`
- `rc/estimation/dml/irm/aipw_trim_01`
- `rc/estimation/dml/iv/default`

## Applicability (when to run)

Run DML specs only when the baseline estimand can be written in a score form with nuisance functions under maintained assumptions.

Typical cases:

- cross-sectional ATE/ATT under unconfoundedness with many covariates,
- panel FE with rich time-varying covariates (with careful inference),
- IV settings when the baseline estimand is IV-LATE and an orthogonal IV score is implementable.

Do **not** force DML when it changes the estimand (e.g., replacing a TWFE DiD coefficient with an unconfoundedness ATE).

## A) PLR score family (continuous treatment; partialling-out)

Target: \(Y = \theta D + g(X) + \varepsilon\).

| spec_id | Description |
|---|---|
| `rc/estimation/dml/plr/default` | PLR-DML with cross-fitting; default learners |
| `rc/estimation/dml/plr/linear_nuisance` | Linear nuisance models for \(E[Y\mid X]\), \(E[D\mid X]\) |
| `rc/estimation/dml/plr/flexible_nuisance` | Flexible ML learners for nuisances (if available) |

## B) IRM score family (binary treatment; AIPW)

Target: ATE/ATT under unconfoundedness with binary \(D\).

| spec_id | Description |
|---|---|
| `rc/estimation/dml/irm/aipw_default` | AIPW/DML for ATE with cross-fitting |
| `rc/estimation/dml/irm/aipw_trim_01` | AIPW with propensity trimming to [0.01, 0.99] |
| `rc/estimation/dml/irm/aipw_trim_05` | AIPW with trimming to [0.05, 0.95] |
| `rc/estimation/dml/irm/att_default` | ATT variant (reweighting + outcome regression), when appropriate |

## C) IV-DML (optional; orthogonal IV score)

Use only when the baseline estimand is IV-LATE and you can implement an orthogonal IV score.

| spec_id | Description |
|---|---|
| `rc/estimation/dml/iv/default` | DML-IV with cross-fitting |
| `rc/estimation/dml/iv/weak_first_stage_guardrail` | Adds explicit first-stage diagnostics + trimming guardrails |

## Default configuration (record in outputs)

Unless a paper requires something else, record:

- cross-fitting folds: `K=5`
- seed: `42`
- learners: explicitly listed (even if linear)
- trimming rules (if any)
- inference method (cluster-robust where feasible)

If clustered/panel dependence matters, prefer:

- cluster-aware fold assignment (split at cluster level),
- cluster-robust SE or a cluster bootstrap.

## Output requirements

Scalar fields:

- `coefficient`: DML estimate of focal estimand
- `std_error`: corresponding SE
- `p_value`: two-sided
- `n_obs`: effective N after trimming

`coefficient_vector_json` must include:

- `estimand` (`plr_theta` / `ate` / `att` / `late`)
- `score_family` (`plr` / `irm` / `iv`)
- `crossfit` (`K`, seed, fold assignment, cluster fold var if any)
- `learners` (names + key hyperparameters)
- `trimming` (if any)
- `diagnostics` (propensity support summaries; first-stage strength for IV-DML)

Example:

```json
{
  "dml": {
    "spec_id": "rc/estimation/dml/irm/aipw_trim_01",
    "estimand": "ate",
    "score_family": "irm",
    "crossfit": {"K": 5, "seed": 42, "cluster_fold_var": "state"},
    "learners": {"model_y": "lasso", "model_t": "logit_l1"},
    "trimming": {"ps_min": 0.01, "ps_max": 0.99},
    "diagnostics": {"ps_min_raw": 0.001, "ps_min_trim": 0.01}
  }
}
```
