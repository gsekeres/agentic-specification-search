# Placebos & Falsification (Diagnostics)

## Spec ID format

Use:

- `diag/placebo/{family}/{variant}`

Examples:

- `diag/placebo/time/fake_treatment_lead1`
- `diag/placebo/assignment/permuted_treatment`
- `diag/placebo/outcome/unaffected_outcome`

## Purpose

Placebos are **diagnostics**: they assess credibility of the maintained identification story by testing for effects where none should exist.

They are not estimand-preserving robustness checks, and they should not be counted as “core RC” for the baseline claim object.

## A) Temporal / timing placebos (panel/time-series designs)

| spec_id | Description |
|---|---|
| `diag/placebo/time/fake_treatment_lead1` | Shift treatment 1 period earlier |
| `diag/placebo/time/fake_treatment_lead2` | Shift treatment 2 periods earlier |
| `diag/placebo/time/fake_treatment_lag1` | Shift treatment 1 period later |
| `diag/placebo/time/fake_treatment_lag2` | Shift treatment 2 periods later |
| `diag/placebo/time/pre_period_only` | Estimate effect using pre-treatment period only |

## B) Assignment / permutation placebos

Only valid when assignment is plausibly exchangeable under the null (or as a stress test, clearly labeled).

| spec_id | Description |
|---|---|
| `diag/placebo/assignment/random_treatment` | Randomly reassign treatment (unit level) |
| `diag/placebo/assignment/permuted_treatment` | Permute treatment labels across units |
| `diag/placebo/assignment/permuted_treatment_cluster` | Permute treatment labels at cluster level |

## C) Outcome placebos

| spec_id | Description |
|---|---|
| `diag/placebo/outcome/lagged_outcome` | Use lagged outcome as DV (should not respond) |
| `diag/placebo/outcome/predetermined_outcome` | DV is predetermined characteristic |
| `diag/placebo/outcome/unaffected_outcome` | DV should be unaffected by treatment (paper-justified) |

## D) Regression-discontinuity / threshold placebos

| spec_id | Description |
|---|---|
| `diag/placebo/regression_discontinuity/fake_cutoff_above` | Estimate RD at fake cutoff above true cutoff |
| `diag/placebo/regression_discontinuity/fake_cutoff_below` | Estimate RD at fake cutoff below true cutoff |
| `diag/placebo/regression_discontinuity/multiple_fake_cutoffs` | Sweep a small set of plausible fake cutoffs |

## Required audit fields (`coefficient_vector_json`)

Placebo rows should include a `diagnostic` block:

```json
{
  "diagnostic": {
    "spec_id": "diag/placebo/time/fake_treatment_lead2",
    "family": "time",
    "description": "Shifted treatment 2 periods earlier",
    "placebo_expected_effect": "zero",
    "notes": "Matched baseline spec exactly except for timing shift."
  }
}
```

If a placebo produces a scalar coefficient/SE, those can be stored in the standard numeric fields, but the row must still be typed as `diag/*`.
