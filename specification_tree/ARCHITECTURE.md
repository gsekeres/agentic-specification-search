# Specification Tree Architecture (Typed, Orthogonal, Auditable)

This document defines the **conceptual contract** for the specification tree used in this repository. The goal is a *statistically principled*, *typed*, and *largely orthogonal* decomposition of empirical variation, so that (i) replication robustness is coherent and auditable, (ii) exploration is explicit and separable, and (iii) downstream estimation can filter specifications mechanically.

## 1) Core object: the baseline claim (what must be pinned down)

The verification stage should define one or more **baseline claim objects** per paper (baseline groups). A claim object is the invariant for “core” replication robustness.

Minimum fields (conceptual, not tied to any one file format):

- **Outcome concept**: what quantity is being measured (including construction choices that matter conceptually).
- **Treatment/exposure concept**: what intervention/exposure is being varied.
- **Estimand concept**: what causal/statistical parameter is targeted (e.g., ITT, ATT, LATE, RD local effect at cutoff, dynamic effect path, elasticity, etc.).
- **Target population**: the population/subsample for which the estimand is defined.
- **Direction / sign expectation** (optional but useful for orientation and auditing).

**Core = estimand-preserving**: a core robustness specification is one that (to first order) preserves these concepts while varying defensible implementation choices.

### Baseline vs RC (robustness checks)

For each baseline claim object (baseline group), we distinguish:

- **Baseline spec(s)**: the paper’s canonical estimate(s) for that claim object.
- **Robustness checks (RC)**: alternative *estimand-preserving* implementations (controls, sample rules, coding, estimator implementation, inference choices) that are intended to test the same claim object.

This separation matters because many papers have **multiple baseline claims** (multiple baseline groups), and the meaningful “how much robustness” question is:

- *how many baseline groups?* and
- *how many RC per baseline group?*

## 2) Orthogonal axes (what can vary, and what kind of object it is)

Empirical variation falls into a small number of axes with different statistical meanings. The tree should not mix these as siblings without typing.

### A. Design / Identification family (method-specific)

This is the identifying strategy and its assumptions (e.g., RCT, DiD, RD, IV, synthetic control, panel FE, etc.). It determines:

- what the estimand typically is,
- what diagnostics are meaningful,
- which estimator implementations are “within-design” alternatives.

### B. Estimator implementation (within a fixed design/estimand)

Alternative estimators that (aim to) estimate the *same* estimand under the *same* design assumptions (e.g., modern staggered-adoption DiD estimators; RD bandwidth/kernel choices; LIML vs 2SLS when instruments are fixed).

### C. Robustness-check modules (RC; ceteris paribus re-specification)

Variations intended to preserve the baseline claim object while changing **reasonable implementation choices** under maintained design assumptions:

- Controls / adjustment set
- Sample restrictions / trimming / outliers
- Functional form / transformations (when treated as “same concept”)
- Measurement / missingness / weights
- Data pre-processing & coding choices (top-coding, standardization, index construction, categorical encoding)

These are the project’s main “specification search” objects for observational work.

### D. Inference modules (uncertainty quantification)

Inference-only recomputations or alternative uncertainty methods:

- Cluster/robust/HAC SE choices
- Bootstrap / randomization inference (where applicable)
- Small-sample corrections

These are not “new estimands”; they are changes to the uncertainty model.

### E. Diagnostics (assumption checks, falsification)

Outputs that assess credibility but are not alternative estimates of the baseline estimand:

- Pre-trends / placebo timing tests
- McCrary density (RD), balance checks, first-stage strength, etc.

Diagnostics can be essential for replication quality but should be typed separately from estimates.

### F. Sensitivity analysis (assumption relaxations / partial identification)

This axis is **not** “another robustness check battery.” It is a different statistical object: it explicitly relaxes key causal assumptions and reports how conclusions move.

Examples:

- unobserved confounding sensitivity (Oster/AET, Rosenbaum bounds, E-values),
- IV exclusion restriction sensitivity (Conley-style bounds; imperfect-IV bounds),
- DiD parallel-trends sensitivity (honest DiD / bounded deviations),
- RCT attrition bounds (Lee/Manski),
- synth donor-pool / fit sensitivity summaries.

Sensitivity analysis outputs are often **bounds/intervals** or **breakdown parameters**, not ordinary regression p-values.

### G. Post-processing (set-level transforms)

Operations defined on a *family* of estimates (not a single regression):

- Multiple-testing adjustments (BH/BY, Romano–Wolf, etc.)
- Specification-curve summaries, sensitivity envelopes, etc.

These should not be treated as “a regression spec”; they are **set-level** objects.

### H. Exploration (concept / estimand changes)

Exploration is valuable but different: it changes the baseline claim object or its estimand.

Examples (non-core by default):

- Alternative outcomes or treatments (concept changes)
- Heterogeneity / CATE / subgroup targeting when the baseline estimand is an average effect
- Policy learning / welfare optimization (requires extra assumptions)
- Alternative estimands (quantile effects, distributional effects, MTE, etc.)

Exploration should be **explicitly labeled** and should not silently contaminate “core”.

## 3) DML positioning (important)

**Double/debiased machine learning (DML)** is **not** a primary empirical method/design. It is a *nuisance-learning + orthogonalization layer* that can be used under multiple designs/assumptions (e.g., unconfoundedness / partialling-out; IRM; IV-DML).

In this project it should live as an **estimation module** (or robustness wrapper) that is invoked *conditional on a fixed baseline claim/design*, not as its own “method”.

## 4) Node typing (“minimal type system”)

Every node in the tree (and every recorded output row) must have an unambiguous **node kind**:

- `estimate`: re-estimate the target parameter (point estimate may change)
- `inference`: recompute uncertainty for (nominally) the same point estimate
- `sensitivity`: assumption-relaxation object (bounds, breakdown points, robust intervals)
- `diagnostic`: assumption check / falsification output
- `postprocess`: set-level transform (not a single regression)
- `explore`: concept/estimand change (non-core by default)

### Practical encoding

We use typed **spec_id namespaces** so the kind is mechanically recoverable:

- `baseline` (reserved)
- `design/{design_code}/...`  (within-design estimator/implementation)
- `rc/{axis}/{variant}`        (robustness checks; ceteris paribus re-specification)
- `infer/{axis}/{variant}`     (inference modules)
- `sens/{family}/{variant}`    (sensitivity analysis; assumption relaxations / partial-ID)
- `diag/{family}/{axis}/{variant}` (diagnostics)
- `post/{axis}/{variant}`      (post-processing)
- `explore/{axis}/{variant}`   (exploration)

This avoids “mixed object types under one namespace” and reduces ambiguity in verification.

## 5) Canonical home per axis (de-duplication rule)

To avoid axis duplication and double counting:

- **Design files** contain *design-specific* estimator/implementation variations and design-specific diagnostics.
- **Modules** contain *universal* sensitivity/inference/diagnostic/exploration variants.
- Design files may **reference** modules but should not duplicate them.

## 5.1) Revealed vs potential search space (important workflow policy)

A full factorial over all degrees of freedom is neither interpretable nor faithful to what a paper actually discloses. In this project we distinguish:

- **Potential search space**: everything a researcher could have tried.
- **Revealed search space**: what the manuscript’s surface (main tables/figures + interpreted appendices) exposes as actively varied.

The default search protocol should be keyed to the **revealed** space:

- enforce linkage constraints for bundled estimators (IV, AIPW/DML, synth),
- avoid inventing combinatorial cross-products unless the paper itself reveals them.

See `specification_tree/REVEALED_SEARCH_SPACE.md`.

## 6) What “coverage” means (and what it does not)

The tree should aim for:

- broad coverage of *common identification families* in applied econ,
- robust coverage of *universal sensitivity and inference axes*,
- explicit and well-typed *exploration* options.

But “full coverage” is impossible if interpreted as a full factorial over all axes. The goal is a disciplined, auditable **menu** that:

- emphasizes one-axis-at-a-time robustness for interpretability,
- includes a small set of high-value interactions when justified,
- cleanly separates exploration from estimand-preserving replication.

## 7) Output requirements (what every recorded estimate must contain)

At minimum for estimate-like rows:

- `spec_id` (typed namespace)
- `spec_tree_path` (file + section anchor)
- `outcome_var`, `treatment_var`
- `coefficient`, `std_error`, `p_value`, `n_obs`
- `coefficient_vector_json` (full coefficient vector + diagnostics fields when relevant)

Diagnostics/postprocess rows may not have a meaningful `(coef, se)` pair; they should either:

1) be stored separately from the regression-results table, or
2) be included with an explicit node kind and with missing numeric fields allowed.

## 8) Core vs exploration gating (mechanical default)

Default rule (for the **verified core used in mixture/dependence estimation**):

- Core-eligible namespaces: `baseline`, `design/*`, `rc/*`, `infer/*`
- Not in verified core by default: `sens/*`, `diag/*`, `post/*`, `explore/*`

The verification stage can override this *only with an explicit reason*, e.g., when the baseline claim itself is heterogeneous (so some heterogeneity nodes become core for that baseline group).
