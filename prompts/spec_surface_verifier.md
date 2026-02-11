# Specification Surface Verifier Agent Instructions (Pre-Run Audit)

Use this prompt to **critique and revise** a candidate specification surface *before any models are run*.

This is the pre-run “verifier intervention” stage.

---

## Inputs

- **Paper/package**: `{PAPER_ID}`
- **Package directory**: `{EXTRACTED_PACKAGE_PATH}`
- **Surface file (required)**: `{EXTRACTED_PACKAGE_PATH}/SPECIFICATION_SURFACE.json`
- **Spec tree**: `specification_tree/` (typed designs + modules)

Do not run regressions in this stage.

---

## Outputs (REQUIRED)

Write to `{EXTRACTED_PACKAGE_PATH}`:

1) Update `SPECIFICATION_SURFACE.json` in place (edit it to reflect your critique).
2) `SPEC_SURFACE_REVIEW.md` (short audit report: issues found + changes made + rationale).

---

## What you are verifying

Your job is to ensure the surface is:

1) **Conceptually coherent** (baseline claim objects are well-defined).
2) **Statistically principled** (RC vs exploration is not mixed).
3) **Faithful to the revealed manuscript surface** (constraints + linkage rules match what the paper/code implies).
4) **Auditable** (budgets/sampling/linkage are explicit and reproducible).

Diagnostics (`diag/*`) and sensitivity (`sens/*`) should not contaminate the core universe; if present, they must be clearly scoped and linked.

---

## Checklist (systematic)

### A) Baseline groups

- Does each `baseline_group_id` correspond to a single claim object (outcome/treatment/estimand/population)?
- Are there missing baseline groups (multiple main outcomes, multiple main subpops, multiple main horizons)?
- Are there baseline groups that should be *exploration* instead (heterogeneity-only, placebo-only, alternative treatments/outcomes not framed as claims)?

### B) Design selection

- Is each baseline group’s `design_code` correct for the paper’s identification strategy?
- Are design variants in `core_universe.design_spec_ids` appropriate and not over-expansive?

### C) RC axes (core robustness)

- Are the included `rc/*` axes the right ones for this design and paper?
- Are any high-leverage axes missing (especially data construction + preprocessing/coding)?
- Are any axes incorrectly included as RC when they change the claim object?

### D) Controls multiverse policy

- Are `controls_count_min/max` correctly derived from the baseline specs?
- Are “mandatory controls” vs “optional pool” implicitly assumed? If so, make it explicit in the surface.
- For bundled estimators: is `linked_adjustment` correct? If yes, ensure the surface does not allow component-wise mixing.

### E) Budgets + sampling

- Are budgets large enough to be informative but small enough to be feasible?
- Is the sampling plan reproducible (seed, sampler type)?
- If the universe is huge: does the surface clearly define it and sample from it instead of pretending to enumerate it?

### F) Diagnostics plan (if present)

- Are standard design diagnostics listed when appropriate (e.g., IV first-stage strength, DiD pretrends)?
- Is the diagnostic `scope` correct (spec vs baseline_group) given dependence on controls?
- Is linkage clearly defined (so it is auditable which estimate(s) each diagnostic supports)?

---

## Report format: `SPEC_SURFACE_REVIEW.md`

Include:

- Summary of baseline groups (and any changes you made)
- Key constraints + linkage rules (esp. bundled estimators)
- Budget/sampling assessment
- “What’s missing” list (if any)
- Final “approved to run” note (or “not approved” with blocking issues)
