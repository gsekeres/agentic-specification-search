# Multiple Testing & Familywise Corrections (Post-process)

## Spec ID format

This module defines **set-level post-processing** operations for multiple testing and multiplicity control.

Use:

- `post/mht/{family}/{variant}`

Examples:

- `post/mht/family_outcomes/bh`
- `post/mht/family_heterogeneity/romano_wolf`

## Purpose

Multiplicity corrections are **not** new regressions/specifications. They are operations on a *family of tests* produced by the spec search. Treat them as:

- `post/*` objects computed after estimates exist, and
- explicitly parameterized by a *family definition* (what set of hypotheses is being corrected).

## Family definitions (must be explicit)

Common families:

- outcomes within a baseline claim (e.g., index + components),
- treatments within a “definition” family,
- heterogeneity interactions (many subgroups),
- event-study leads/lags (multiple coefficients),
- placebo tests (multiple falsification checks),
- RC (“robustness check”) specs within a baseline group (default for this project).

**Project default recommendation**: apply MHT **within baseline groups** and only across *estimand-preserving estimate rows* (`baseline`, `design/*`, `rc/*`). Do **not** pool across:

- `explore/*` (concept/estimand changes),
- `diag/*` (diagnostics/placebos),
- `sens/*` (sensitivity-analysis bounds; not ordinary p-values).

## Standard procedures

| spec_id | Description |
|--------|-------------|
| `post/mht/{family}/bonferroni` | Bonferroni adjusted p-values / CI |
| `post/mht/{family}/holm` | Holm step-down (FWER) |
| `post/mht/{family}/sidak` | Šidák (FWER under independence) |
| `post/mht/{family}/bh` | Benjamini–Hochberg FDR |
| `post/mht/{family}/by` | Benjamini–Yekutieli FDR (arbitrary dependence) |
| `post/mht/{family}/romano_wolf` | Romano–Wolf step-down (bootstrap) |
| `post/mht/{family}/westfall_young` | Westfall–Young (resampling) |

## Output contract

Store results as a JSON object keyed by the corrected family and procedure. Example:

```json
{
  "postprocess": {
    "spec_id": "post/mht/family_outcomes/bh",
    "family": "family_outcomes",
    "procedure": "BH",
    "n_hypotheses": 12,
    "q": 0.05,
    "summary": {
      "n_rejected_raw_05": 7,
      "n_rejected_adj_05": 3
    },
    "adjusted": [
      {"spec_row_id": 123, "p_raw": 0.003, "p_adj": 0.018, "reject": true},
      {"spec_row_id": 124, "p_raw": 0.041, "p_adj": 0.091, "reject": false}
    ]
  }
}
```

Implementation note: this can be computed in the estimation pipeline once specs are assembled and baseline groups are known.
