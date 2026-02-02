# Difference-in-Differences Specifications

## Spec ID Format: `did/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main DiD result
- Record: treatment indicator, FE structure, clustering, all controls

---

## Core Variations

### Fixed Effects

| spec_id | Description |
|---------|-------------|
| `did/fe/unit_only` | Unit FE only |
| `did/fe/time_only` | Time FE only |
| `did/fe/twoway` | Unit + Time FE (standard TWFE) |
| `did/fe/unit_x_time` | Unit × Time interactions |
| `did/fe/region_x_time` | Region × Time interactions (if applicable) |
| `did/fe/industry_x_time` | Industry × Time interactions (if applicable) |
| `did/fe/none` | No fixed effects (pooled OLS) |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `did/controls/none` | No controls (treatment only) |
| `did/controls/minimal` | Essential controls only (demographics) |
| `did/controls/baseline` | Paper's baseline control set |
| `did/controls/full` | All available controls |
| `did/controls/saturated` | Full + all reasonable interactions |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `did/sample/full` | Full sample |
| `did/sample/pre_treatment` | Pre-treatment placebo test |
| `did/sample/early_period` | First half of sample period |
| `did/sample/late_period` | Second half of sample period |
| `did/sample/exclude_always_treated` | Drop always-treated units |
| `did/sample/exclude_never_treated` | Drop never-treated units |
| `did/sample/balanced_panel` | Balanced panel only |
| `did/sample/drop_outliers` | Drop outliers (top/bottom 1%) |

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `did/method/twfe` | Standard two-way FE |
| `did/method/sun_abraham` | Sun & Abraham interaction-weighted estimator |
| `did/method/callaway_santanna` | Callaway & Sant'Anna group-time ATT |
| `did/method/borusyak` | Borusyak et al. imputation estimator |
| `did/method/dechaisemartin` | de Chaisemartin & D'Haultfoeuille |
| `did/method/wooldridge` | Wooldridge extended TWFE |

### Dynamic Effects (Event Study)

| spec_id | Description |
|---------|-------------|
| `did/dynamic/leads_lags` | Event study with leads/lags |
| `did/dynamic/leads_only` | Pre-treatment trends only |
| `did/dynamic/lags_only` | Post-treatment effects only |
| `did/dynamic/bucketed` | Bucketed post-treatment periods |

### Treatment Definition

| spec_id | Description |
|---------|-------------|
| `did/treatment/binary` | Binary treatment indicator |
| `did/treatment/intensity` | Treatment intensity (continuous) |
| `did/treatment/staggered` | Staggered adoption timing |

### Weighting and Matching

| spec_id | Description |
|---------|-------------|
| `did/weight/population` | Population-weighted regression |
| `did/weight/inverse_propensity` | IPW-DiD estimator |
| `did/weight/entropy` | Entropy balancing weights |
| `did/matched/psm` | Propensity score matching + DiD |
| `did/matched/cem` | Coarsened exact matching + DiD |
| `did/matched/mahalanobis` | Mahalanobis matching + DiD |

### Decomposition and Diagnostics

| spec_id | Description |
|---------|-------------|
| `did/decomp/bacon` | Bacon decomposition of TWFE |
| `did/decomp/goodman_bacon` | Goodman-Bacon decomposition |
| `did/diagnostic/parallel_trends` | Parallel trends test |
| `did/diagnostic/pretrend_joint` | Joint test of pre-trends |

### Synthetic Control Comparisons

| spec_id | Description |
|---------|-------------|
| `did/synthetic/control` | Synthetic control comparison |
| `did/synthetic/did` | Synthetic DiD (Arkhangelsky et al.) |
| `did/augsynth` | Augmented synthetic control |

### Alternative Inference

| spec_id | Description |
|---------|-------------|
| `did/inference/permutation` | Permutation test |
| `did/inference/randomization` | Randomization inference |
| `did/inference/wild_bootstrap` | Wild cluster bootstrap |

---

## Python Implementation Notes

```python
# Standard TWFE
import pyfixest as pf
model = pf.feols("y ~ treat | unit + time", data=df)

# Callaway & Sant'Anna (requires did package in R)
# Sun & Abraham
model = pf.feols("y ~ sunab(cohort, time) | unit + time", data=df)
```

---

## Coefficient Vector Format

```json
{
  "treatment": {
    "var": "policy_dummy",
    "coef": 0.05,
    "se": 0.02,
    "pval": 0.01,
    "ci_lower": 0.01,
    "ci_upper": 0.09
  },
  "controls": [
    {"var": "age", "coef": 0.1, "se": 0.05, "pval": 0.04},
    {"var": "income", "coef": -0.02, "se": 0.01, "pval": 0.02}
  ],
  "fixed_effects_absorbed": ["unit_id", "year"],
  "diagnostics": {
    "first_stage_F": null,
    "pretrend_pval": null
  },
  "n_obs": 10000,
  "n_clusters": 50,
  "r_squared": 0.45,
  "r_squared_within": 0.12
}
```

---

## Checklist

Before completing a DiD analysis, verify you have run:

- [ ] Baseline replication
- [ ] At least 3 FE variations
- [ ] At least 3 control set variations
- [ ] At least 2 sample restrictions
- [ ] Pre-treatment placebo (if time series)
- [ ] At least 1 modern DiD estimator (if staggered)
- [ ] Event study / dynamic effects
