# Claim Objects, Baseline Groups, and Core vs Exploration

This document complements `specification_tree/ARCHITECTURE.md` by giving **operational rules** for:

- how to define baseline claim objects (“baseline groups”),
- how to decide whether a specification is an estimand-preserving **robustness check (RC)** or a **concept/estimand change**,
- how to handle the hard cases: *sample restrictions / subpopulations* and *alternative outcomes*.

This is intended to guide both:

- the **spec-search agent** (what to enumerate and how to label it), and
- the **verification agent** (how to classify rows mechanically and conservatively).

## 1) Baseline group = one claim object

A baseline group corresponds to a single claim object:

- outcome concept
- treatment concept
- estimand concept
- target population

**Key rule**: if any of these concepts changes, you are no longer in the same baseline group.

### Multi-design papers (important)

A single paper can have multiple “main” specifications that rely on different **design/identification families** (e.g., IV for one claim, RD for another, or DiD in one section and RCT in another).

This is well-posed: treat each main claim object as its own baseline group, and record the relevant design family for that baseline group in the specification surface (`design_code`).

Within a baseline group, changing to a different design family is **not** an RC move by default: it typically changes assumptions and often changes the estimand (e.g., IV LATE vs OLS average effect; RD local-at-cutoff vs DiD ATT). Treat it as a separate baseline group unless the paper explicitly argues equivalence.

## 2) When do multiple outcomes imply multiple baseline groups?

Many papers report multiple outcomes. These can mean very different things:

### Case A: multiple outcomes are **distinct core claims**

Example patterns:

- “We estimate effects on earnings, employment, and health.”
- Separate main-table panels with distinct outcomes, each interpreted as a headline result.

**Action**: define **multiple baseline groups**, one per outcome concept (or per tight outcome family if explicitly framed that way).

### Case B: multiple outcomes are **components / proxies** for one concept

Example patterns:

- a summary index + many components (components are supporting evidence, not separate hypotheses),
- multiple operationalizations of the *same* construct used as a measurement check.

**Action**: define one baseline group for the canonical outcome concept (often the index) and treat the rest as:

- `rc/*` if truly measurement/coding robustness *within the same concept*, or
- `explore/*` if they are substantively different outcome concepts.

### Practical default (conservative)

Unless the manuscript explicitly treats an outcome as a main claim, treat it as `explore/*` and explain in verification.

## 3) Sample restrictions vs “population changes”

Sample restrictions span two qualitatively different objects:

### A: “Within-population” robustness checks (RC)

These change implementation while (arguably) targeting the same population concept:

- trimming/winsorization/outlier rules,
- dropping low-quality / flagged observations,
- balanced vs unbalanced panel when the estimand is intended for the underlying panel population,
- alternative but equivalent eligibility filters stated in the replication code.

These are typically `rc/sample/*` (or `rc/preprocess/*` when it is a coding operation).

### B: Subpopulation estimands (concept changes)

These change the **target population**, hence the estimand:

- “male only”, “urban only”, “firms above 50 employees”, “high income tercile”,
- cohort-only effects, region-only effects, etc.

These are **not RC by default**. They should be:

- separate baseline groups **if the paper’s headline claim is explicitly about that subpopulation**, or
- `explore/population/*` (or `explore/heterogeneity/*`) otherwise.

### The key verifier question

When you see a subsample spec: *is the paper claiming this as the target population, or is it probing effect heterogeneity?*

If it’s heterogeneity, it is exploration unless heterogeneity itself is the baseline estimand.

## 4) Alternative outcomes and subpopulations “open forking paths”

If a manuscript reports many distinct outcomes/subpopulations as “main analyses”, it has effectively disclosed a wider specification space.

Implication for this project:

- baseline groups should reflect the paper’s disclosed claim objects, even if there are many;
- RC should be run **within each baseline group** (when feasible) rather than pooling everything as “alt outcomes”.

This matters for:

- counting “how many core claims does the paper make?” vs “how much robustness per claim?”,
- later modeling choices that treat each baseline group as a distinct hypothesis vs a single hypothesis with many robustness checks.

## 5) Mechanical classification defaults (for verification)

Given a baseline group definition:

- If `outcome_var` changes in a way that changes the outcome concept → `explore/*` (alt outcome) unless it is part of the baseline group’s outcome set.
- If `treatment_var` changes in a way that changes the treatment concept → `explore/*` (alt treatment) unless explicitly equivalent per the paper.
- If the sample restriction implies a new target population (subgroup) → `explore/*` unless the baseline group’s population is that subgroup.
- If it changes controls/coding/inference but preserves the claim object → `rc/*` / `infer/*`.

Always be conservative: if unsure whether a change preserves the claim object, classify as `explore/*` and record why.

## 6) How the spec-search agent should label things

Search agents should try to emit explicit labels that make verification easier:

- Baselines: `baseline` plus `baseline__{slug}` for additional baseline claims.
- RC: `rc/{axis}/{variant}`.
- Exploration: `explore/{axis}/{variant}`.

When generating subsample and alternative-outcome specs:

- If clearly a main claim in the paper, label as another baseline and run RC around it.
- Otherwise, label as exploration and do not let it contaminate “core RC counts”.

## 7) Revealed search space (linkage and combinatorics)

Whether a paper’s disclosed analysis opens a large “reasonable replication” space depends not only on *how many outcomes/subpopulations* it treats as claims, but also on whether its main estimators are **bundles** with **linked** implementation choices (e.g., shared covariate sets across AIPW nuisance models or IV stages).

Policy:

- If the manuscript reveals *linked* adjustment across estimator components, robustness search should vary controls **jointly** (do not mix-and-match across components).
- If it reveals *unlinked* component-specific adjustments, controlled mix-and-match is allowed (but avoid full factorials unless the manuscript reveals them).

See `specification_tree/REVEALED_SEARCH_SPACE.md`.
