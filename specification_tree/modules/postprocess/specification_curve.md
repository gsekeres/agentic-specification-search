# Specification Curves / Multiverse Summaries (Post-process)

## Spec ID format

Use:

- `post/speccurve/{family}/{variant}`

Examples:

- `post/speccurve/family_core_rc/default`
- `post/speccurve/family_exploration/heterogeneity`

## Purpose

Specification curves are **set-level evidence objects**: they summarize a distribution of estimates across a family of specifications.
They are not individual regressions and should be computed after the spec search finishes.

This module standardizes the families and outputs so the pipeline can:

- report robustness summaries without double-counting,
- define families consistent with baseline groups and core gating,
- support later dependence/mixture estimation with clear “core” subsets.

## Family definitions (must be explicit)

Recommended families:

- `family_core_rc`: within a baseline group, include `baseline`, `design/*`, `rc/*` (estimate rows under canonical inference).
- `family_exploration`: within a baseline group, include `explore/*` rows (or a subtype family such as heterogeneity).
- `family_placebos`: include `diag/*` placebo rows.

Avoid pooling across baseline groups unless the paper’s claim object is explicitly pooled.

## Standard summaries to report

For a family of scalar estimates \(\{\hat\beta_s, \widehat{se}_s\}\):

- number of specs
- fraction with consistent sign
- fraction significant at 5% (raw)
- median and IQR of \(\hat\beta\)
- median and IQR of \(|t|\) or `Z_logp`
- min/max of \(\hat\beta\)
- “robustness interval” (e.g., 10th–90th percentile)

## Spec IDs

| spec_id | Description |
|---|---|
| `post/speccurve/family_core_rc/default` | Core RC family summary within baseline group |
| `post/speccurve/family_exploration/definition` | Alt definitions family summary |
| `post/speccurve/family_exploration/heterogeneity` | Heterogeneity/CATE family summary |
| `post/speccurve/family_placebos/default` | Placebo family summary |

## Output contract (`coefficient_vector_json`)

Store as a `postprocess` block:

```json
{
  "postprocess": {
    "spec_id": "post/speccurve/family_core_rc/default",
    "baseline_group_id": "G1",
    "family": "family_core_rc",
    "n_specs": 72,
    "beta": {
      "median": 0.08,
      "p10": 0.03,
      "p90": 0.12,
      "min": -0.01,
      "max": 0.19
    },
    "t_abs": {
      "median": 2.3,
      "p10": 0.9,
      "p90": 3.8
    },
    "sign_consistency": 0.94,
    "sig_05_raw": 0.61
  }
}
```

If the pipeline produces a plot, store the plot artifact separately (e.g., as a PDF) and reference it in the JSON.
