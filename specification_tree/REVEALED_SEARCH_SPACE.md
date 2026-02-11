# Revealed Search Space & Linkage Constraints

This document formalizes a central policy for this project: **what search space is “revealed” by the manuscript**, and how that affects which specification combinations are treated as natural robustness checks versus expansive exploration.

It complements:

- `specification_tree/ARCHITECTURE.md` (typed axes and node kinds),
- `specification_tree/CLAIM_GROUPING.md` (baseline groups and core vs exploration),
- `specification_tree/CONTRACT.md` (what agents must emit).

## 1) Two different “spaces”

We distinguish:

1) **Potential search space**: all forks a researcher could, in principle, try.
2) **Revealed search space**: the forks that are exposed by the manuscript’s surface (main tables/figures + interpreted appendices), and therefore become *prima facie* “reasonable” to audit systematically.

The verification and search protocols should be keyed to the **revealed** space, not the unconstrained potential space.

Clarification: the project can still run *standardized stress-test batteries* that go beyond what a given paper reports (especially for covariate inclusion). The “revealed space” concept is used to:

- decide what constitutes a baseline claim object (how many core claims),
- enforce linkage constraints for bundled estimators (avoid incoherent mix-and-match),
- interpret how much the manuscript itself opened the forking-paths surface.

## 2) Bundled estimators (multi-equation / multi-component implementations)

Many baseline estimates are not a single regression. They are **bundles** with multiple components that must be logically consistent:

- **IV**: first stage + reduced form + second stage.
- **AIPW / doubly robust**: propensity model + outcome regression(s) + AIPW combining formula.
- **DML**: nuisance learners + cross-fitting + orthogonal score regression.
- **Synthetic control / SDID**: donor weights + fit window + predictor set + placebo distribution.

These bundles often share a key degree of freedom: the **covariate adjustment set** (and/or fixed effects).

## 3) Linkage rule for adjustment sets (the main policy)

For a bundled baseline estimator, define whether covariate adjustment is:

- **linked/shared**: the manuscript (or code) uses the *same* adjustment set across components, or clearly intends them to be the same; or
- **unlinked/component-specific**: the manuscript uses different covariate sets in different components (or explicitly allows them to differ).

### Default search policy

- If adjustment sets are **linked/shared**, then robustness search should vary controls **jointly**, enforcing that linkage.
  - Example: if the IV first stage and second stage share controls, then `rc/controls/*` changes apply to both stages together.
  - Example: if an AIPW estimator uses the same controls/features for the propensity and outcome models (as is common in replication code), then changes to the covariate set apply jointly.

- If adjustment sets are **unlinked/component-specific**, then robustness search may vary them **independently** (mix-and-match), but only within a constrained, auditable menu (avoid a full factorial unless the paper itself exposes that combinatorial structure).

This policy prevents the search agent from inventing a combinatorial space that the paper did not reveal, while still allowing rich robustness when the paper actually uses multiple control strategies.

### Control-count envelope as an additional surface constraint

Even when we intentionally explore covariate inclusion combinatorially, it is often useful to bound complexity by the manuscript’s surface.
A practical default is to constrain sampled control subsets to have size within the **main-spec control-count envelope** for the baseline group (min/max number of controls across main-table canonical specs). See `specification_tree/SPECIFICATION_SURFACE.md`.

## 4) Analogous rule for “multiple core claims” (subgroups and outcomes)

The same logic applies to **baseline claim objects**:

- If the manuscript presents and interprets subgroup estimates as headline claims (distinct target populations), then it has revealed **multiple baseline groups**.
- If subgroup results are presented as heterogeneity exploration, they do not expand the core by default; they live in `explore/*`.

Implication: the size of the *reasonable replication* space expands sharply when the paper itself spans many outcomes/subpopulations as interpreted claims.

## 5) What the search agent must record (auditability)

For any bundled estimator, the search agent should record a `bundle` block in `coefficient_vector_json`:

```json
{
  "bundle": {
    "bundle_type": "iv | aipw | dml | synth | other",
    "linked_adjustment": true,
    "components": {
      "second_stage": {"controls": ["x1", "x2"], "fixed_effects": ["unit", "time"]},
      "first_stage": {"controls": ["x1", "x2"], "fixed_effects": ["unit", "time"]}
    },
    "notes": "Controls and FE are shared across stages per paper code."
  }
}
```

If controls are unlinked, record per-component control lists and set `linked_adjustment=false`.

## 6) How verification should use this

Verification should (when possible) record, per baseline group:

- whether the baseline estimator is bundled,
- whether adjustment is linked,
- whether the manuscript reveals multiple baseline groups (multiple outcomes/subpopulations as claims).

This should influence:

- how RC specs are generated (joint vs component-wise),
- how “reasonable search space” is described in the verification report,
- how post-processing families (e.g., MHT) are defined (usually within baseline groups).

## 7) Practical guardrails

- Prefer **joint variation** as the default for bundles; allow independent variation only when the manuscript clearly motivates it.
- Avoid full factorial expansion unless the paper’s own specification tables reveal a cross-product structure.
- When independent variation is allowed, keep it to a small menu (e.g., baseline propensity covariates vs expanded; baseline outcome covariates vs expanded) and label it clearly in the spec IDs and JSON audit blocks.

## 8) Controlled expansion: budgeted multiverse batteries

Some axes are so central (covariate inclusion) that a standardized protocol may intentionally explore beyond the paper’s reported variations.
When doing so:

- keep the estimand concept fixed (stay in `rc/*`),
- respect linkage constraints for bundled estimators (shared controls across components unless the paper reveals otherwise),
- use budgeted, reproducible subset generation (`rc/controls/subset/*` in `specification_tree/modules/robustness/controls.md`),
- record the generation rule + seed + realized included/excluded sets in `coefficient_vector_json`.
