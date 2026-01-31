# Event Study Specifications

## Spec ID Format: `es/{category}/{variation}`

## Baseline (REQUIRED)

- **spec_id**: `baseline`
- Exact replication of paper's main event study result
- Record: event window, reference period, all lead/lag coefficients

---

## Core Variations

### Event Window

| spec_id | Description |
|---------|-------------|
| `es/window/symmetric` | Symmetric window around event |
| `es/window/short` | Shorter window (e.g., -3 to +3) |
| `es/window/long` | Longer window (e.g., -10 to +10) |
| `es/window/asymmetric` | Asymmetric (more lags than leads) |
| `es/window/trim_endpoints` | Bin endpoints to avoid composition |

### Reference Period

| spec_id | Description |
|---------|-------------|
| `es/reference/t_minus_1` | t-1 as reference (standard) |
| `es/reference/t_minus_2` | t-2 as reference |
| `es/reference/first_lead` | First available lead as reference |
| `es/reference/average_pre` | Average of all pre-periods |

### Fixed Effects

| spec_id | Description |
|---------|-------------|
| `es/fe/unit_time` | Unit + Time FE |
| `es/fe/unit_only` | Unit FE only |
| `es/fe/region_time` | Region × Time FE |
| `es/fe/cohort_time` | Cohort × Time FE |

### Control Sets

| spec_id | Description |
|---------|-------------|
| `es/controls/none` | No controls |
| `es/controls/baseline` | Paper's baseline controls |
| `es/controls/time_varying` | Time-varying controls |
| `es/controls/full` | All available controls |

### Sample Restrictions

| spec_id | Description |
|---------|-------------|
| `es/sample/full` | Full sample |
| `es/sample/treated_only` | Treated units only |
| `es/sample/clean_controls` | Clean control group only |
| `es/sample/balanced` | Balanced panel |
| `es/sample/no_switchers` | Exclude treatment switchers |

### Estimation Method

| spec_id | Description |
|---------|-------------|
| `es/method/twfe` | Standard TWFE event study |
| `es/method/sun_abraham` | Sun & Abraham (heterogeneity-robust) |
| `es/method/callaway_santanna` | Callaway & Sant'Anna |
| `es/method/borusyak` | Borusyak et al. imputation |
| `es/method/interaction_weighted` | Interaction-weighted estimator |

### Pre-trend Tests

| spec_id | Description |
|---------|-------------|
| `es/pretrend/joint_test` | Joint F-test of pre-period coefficients |
| `es/pretrend/linear` | Test for linear pre-trend |
| `es/pretrend/rambachan_roth` | Rambachan & Roth sensitivity bounds |

---

## Python Implementation Notes

```python
import pyfixest as pf

# Standard event study
model = pf.feols("y ~ i(rel_time, ref=-1) | unit + time", data=df)

# Sun & Abraham
model = pf.feols("y ~ sunab(cohort, rel_time) | unit + time", data=df)
```

---

## Coefficient Vector Format

```json
{
  "event_time_coefficients": [
    {"rel_time": -3, "coef": 0.01, "se": 0.02, "pval": 0.62},
    {"rel_time": -2, "coef": 0.00, "se": 0.02, "pval": 0.98},
    {"rel_time": -1, "coef": null, "se": null, "pval": null, "note": "reference"},
    {"rel_time": 0, "coef": 0.05, "se": 0.02, "pval": 0.01},
    {"rel_time": 1, "coef": 0.08, "se": 0.02, "pval": 0.00},
    {"rel_time": 2, "coef": 0.07, "se": 0.03, "pval": 0.02},
    {"rel_time": 3, "coef": 0.06, "se": 0.03, "pval": 0.05}
  ],
  "controls": [
    {"var": "age", "coef": 0.1, "se": 0.05, "pval": 0.04}
  ],
  "fixed_effects_absorbed": ["unit", "time"],
  "diagnostics": {
    "pretrend_joint_F": 1.23,
    "pretrend_pval": 0.31,
    "reference_period": -1
  },
  "n_obs": 10000,
  "n_clusters": 50,
  "r_squared": 0.45
}
```

---

## Checklist

Before completing an event study analysis, verify you have run:

- [ ] Baseline replication
- [ ] At least 2 window variations
- [ ] At least 2 reference period choices
- [ ] Pre-trend joint test
- [ ] At least 1 modern estimator (if staggered treatment)
- [ ] Report ALL lead/lag coefficients in coefficient vector
