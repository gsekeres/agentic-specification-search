# Specification Surface Review: 112431-V1 (Ferraz & Finan, AER 2011)

**Reviewer**: Verifier agent (pre-run audit)
**Date**: 2026-02-13
**Surface version reviewed**: Initial candidate (2026-02-13)
**Status**: APPROVED TO RUN (with changes applied)

---

## 1. Summary of Baseline Groups

The surface defines three baseline groups, all sharing the same treatment (first-term mayor indicator `first`) and target population (476 CGU-audited municipalities with `esample2==1`):

| Group | Outcome | Baseline spec | Coefficient | p-value | N |
|-------|---------|---------------|-------------|---------|---|
| G1 | `pcorrupt` (share of resources involving corruption) | Table 4, Col 6 | -0.0275 | 0.015 | 476 |
| G2 | `ncorrupt` (count of corruption violations) | Table 5A, Col 2 | -0.471 | 0.002 | 476 |
| G3 | `ncorrupt_os` (violations per audited item) | Table 5B, Col 2 | -0.0105 | 0.017 | 476 |

**Assessment**: The three groups are well-defined and correspond to the three distinct corruption measures that the paper treats as headline results in Tables 4-5. The grouping correctly excludes Table 9 (matching grants, different dataset/estimand), Table 10 (heterogeneity/exploration), Table 8 (placebo/diagnostic), and Table 11 (manipulation checks/diagnostics) from baseline claims.

No changes to baseline group definitions were needed.

---

## 2. Changes Made to the Surface

### Change 1: Added `sign_expectation` to all claim objects

The original surface lacked explicit sign expectations. Added `"sign_expectation": "Negative"` to all three groups, since the paper's hypothesis is that reelection-eligible (first-term) mayors are less corrupt. This is useful for orientation and auditing.

### Change 2: Clarified treatment concept description (G1)

The original description "cannot run again vs. can run again" was ambiguous about which direction maps to `first=1`. Replaced with explicit coding: "first=1 means first term, eligible for reelection; first=0 means second term, term-limited."

### Change 3: Added `mandatory_controls` field to all groups

This was the most consequential structural change. The original surface noted in prose that `lrec_fisc` is mandatory for G2 and both `lrec_fisc` and `lfunc_ativ` are mandatory for G3, but did not encode this as a machine-readable field. Added:

- **G1**: `mandatory_controls: []` (no mandatory controls; pcorrupt is a share, not scale-dependent)
- **G2**: `mandatory_controls: ["lrec_fisc"]` (audit scale determines opportunity for violations)
- **G3**: `mandatory_controls: ["lrec_fisc", "lfunc_ativ"]` (outcome is a ratio of violations to audited items)

This has operational implications:
- For G2/G3, `rc/controls/sets/none` was renamed to `rc/controls/sets/mandatory_only` to make clear that mandatory controls are always included.
- LOO block drops must not drop mandatory controls. Added `controls_loo_blocks_note` to G2 and G3.
- Controls count minimums were adjusted: G2 min = 1, G3 min = 2 (previously both were 0).

### Change 4: Added experience-control RC specs to G1

Table 7 of the paper addresses a key identification concern (confounding by political experience) by adding experience controls. Cols 3-4 add `exp_prefeito` and `nexp + nexp2` respectively. These are revealed robustness checks that were missing from the original surface. Added:

- `rc/controls/add/exp_prefeito`
- `rc/controls/add/nexp_nexp2`

This increases the G1 spec count by 2 (from ~107 to ~109).

### Change 5: Added revealed estimator variants to G2/G3

The original surface included Tobit for G1 but omitted paper-revealed estimator variants for the secondary outcomes:

- **G2**: Added `rc/estimation/nbreg` (negative binomial, Table 5A Col 4). Noted implementation caveat (NaN SEs in replication).
- **G3**: Added `rc/estimation/tobit_ll0` (Tobit left-censored at 0, Table 5B Col 4).

### Change 6: Renamed `full_with_lfunc_ativ` to `full_with_audit_scale` for G1

The original name was misleading: the "full" control set used in Tables 6-11 adds both `lfunc_ativ` and `lrec_fisc` to the Table 4 baseline, not just `lfunc_ativ`. Renamed the spec ID and updated the description to "Baseline + lfunc_ativ + lrec_fisc (42 vars)".

### Change 7: Added control-set documentation notes

Added `controls_sets_notes` sub-object to G1 and G2/G3 to make each control set's composition explicit (variable counts, which blocks are included). This supports auditability.

### Change 8: Updated controls_count_max for G1

Changed from 41 to 42 to correctly reflect the maximum control count when both audit scale variables are added (as in Tables 6-11).

### Change 9: Added RDD polynomial constraint documentation

Added `functional_form_notes` sub-object to G1 documenting that RDD polynomial specs are inherently joint (sample restriction + added controls). This is a constraint the spec-search agent must enforce.

### Change 10: Added cluster SE caveat

Added `infer_notes` to G1 documenting that state-level clustering (26 clusters) may produce unreliable asymptotic SEs per the few-cluster guardrails in the inference module.

### Change 11: Clarified block-combination overlap in sampling notes

Noted that the 64 exhaustive block combinations include both the empty set (= `rc/controls/sets/none`) and the full set (= baseline), which should be tagged as overlapping specs to avoid double-counting.

### Change 12: Added total_estimated counts and notes to all groups

Added explicit estimated spec counts and breakdowns to the budgets sections of all three groups to support feasibility assessment.

---

## 3. Checklist Assessment

### A) Baseline groups

**Pass**. Three baseline groups are well-defined. Each corresponds to a single claim object with a distinct outcome concept. The treatment, estimand, and population are shared. No missing baseline groups (the paper's three headline corruption measures are all covered). Tables 8-11 are correctly classified as diagnostics/exploration, not baseline claims.

### B) Design selection

**Pass**. `cross_sectional_ols` is the correct design code for all three groups. The paper uses OLS with state FE as its primary identification strategy. Table 6's polynomial-in-running-variable approach is correctly classified as a functional-form RC (not a design change to RD) since the paper treats it as parametric controls, not formal RDD estimation.

### C) RC axes

**Pass with additions**. The original surface covered controls, FE, sample, functional form, estimator, and inference axes. The main gap was the missing experience-control RC (Table 7), which has been added. The axes are appropriate for this design and paper:

- **Controls**: Exhaustive block enumeration (2^6 = 64) plus LOO at block and variable levels is thorough.
- **FE**: Drop/replace state FE is appropriate.
- **Sample**: Paper-revealed restrictions (running nonmissing, pmismanagement nonmissing) plus standard outlier checks.
- **Functional form**: RDD polynomial controls and asinh transformation are well-motivated.
- **Estimator**: Tobit (G1, G3), negative binomial (G2) are paper-revealed.
- **Inference**: HC1/HC2/HC3 plus state clustering provide a reasonable range.

No axes are incorrectly included as RC when they change the claim object.

### D) Controls multiverse policy

**Pass with corrections**. Key changes:
- Mandatory controls are now explicitly flagged for G2/G3.
- The `controls_count_min` has been adjusted for G2 (min=1) and G3 (min=2) to reflect mandatory controls.
- LOO blocks are documented as operating only on the optional pool (mandatory controls never dropped).
- The block-level exhaustive enumeration (64 combinations) is a sound approach given 6 blocks.
- No bundled estimator issues (single OLS equation).

### E) Budgets and sampling

**Pass**. All surfaces are within budget:
- G1: ~109 specs against a 150-spec budget. Full enumeration is feasible.
- G2: ~15 specs against a 30-spec budget.
- G3: ~14 specs against a 30-spec budget.

No random sampling is needed. The seed (112431) is defined for reproducibility but will not be used since enumeration is complete.

### F) Diagnostics plan

**Pass**. The diagnostics plan is minimal but appropriate:
- G1 includes a balance test diagnostic (Table 3 of the paper).
- Table 8 (pmismanagement placebo) is correctly classified as a diagnostic, not a core estimate.
- Table 11 manipulation tests are correctly excluded from core.

---

## 4. What Is Not Changed (and Why)

### Table 7 experience subsamples (Cols 1-2, 5-6)

Table 7 Cols 1-2, 5-6 restrict the sample to various experience-defined subpopulations (e.g., `experience2==1`, `experience1==1`). These are NOT included in the G1 surface because they change the target population and are closer to heterogeneity exploration than estimand-preserving RC. Only the "add experience controls" variants (Cols 3-4) are included, since they preserve the full sample.

### Table 11 interaction robustness (Cols 1-3)

Table 11 Cols 1-3 add specific interaction terms (first x sorteio_electyear, first x PT, first x samepartygov98) to test manipulation/confounding concerns. These could in principle be RC, but they test very specific threats rather than general robustness dimensions. Excluding them keeps the surface focused on standard specification-search axes.

### Alternative estimators (IPW, AIPW, matching)

These are available in the design file but are not in the paper's revealed search space for the main OLS claims. The matching estimator (Table 4 Col 7) could not be replicated due to software constraints (Abadie-Imbens estimator). IPW/AIPW are interesting but not the paper's revealed approach.

### Exploration axes (Table 9, Table 10)

Table 9 (matching grants) uses a different dataset and outcome concept. Table 10 (heterogeneous effects) tests moderators. Both are correctly classified as exploration, not core.

---

## 5. Potential Concerns

1. **Small sample size (N=476)**: With only 476 observations and 40+ controls, many specifications are pushing the limits of reliable inference. Some block combinations in the exhaustive search will have very few controls (bivariate) while others have many, spanning a wide range of effective degrees of freedom. This is an inherent feature of the paper's setting, not a surface design flaw.

2. **State FE with 26 states**: Absorbing 26 state fixed effects in a sample of 476 uses substantial degrees of freedom. The `infer/se/cluster/uf` spec (clustering at the same level as the absorbed FE) should be interpreted with caution given the small cluster count.

3. **Tobit and negative binomial implementation**: These non-OLS estimators require separate implementations (scipy/statsmodels) that may have optimization differences compared to Stata. The replication report notes numerical issues with negative binomial SEs. Results from these estimator variants should be flagged for extra scrutiny.

4. **RDD polynomial specs as joint variation**: The polynomial-in-running-variable specs necessarily change both the sample and the control set simultaneously. They are documented as joint specs in the surface, but the spec-search agent must implement this correctly (restrict to N=328 AND add polynomial terms).

---

## 6. Final Assessment

**APPROVED TO RUN**.

The specification surface is conceptually coherent, statistically principled, faithful to the paper's revealed search space, and auditable. The changes made during this review (mandatory controls, experience controls, estimator variants, naming clarifications, documentation improvements) strengthen the surface without expanding it beyond what the paper reveals. Total estimated specs across all groups: ~138 (109 + 15 + 14), well within the combined budget.
