# Specification Surface Review: 114542-V1

**Paper**: Cattaneo et al. (2009). "Housing, Health and Happiness."
**Review date**: 2026-02-24

---

## A) Baseline Groups

**Assessment**: Three baseline groups correctly identified.

- **G1 (Cement floor coverage)**: Correct. S_shcementfloor is the paper's primary "first stage" outcome (Table 4). Additional baselines for Models 2-4 are appropriate since the paper reports all four.
- **G2 (Satisfaction/mental health)**: Correct. These are headline Table 6 outcomes. Multiple outcomes (S_satisfloor, S_satishouse, S_satislife, S_cesds, S_pss) are each treated as separate baseline specs within the same claim family.
- **G3 (Child health)**: Correct. Table 5 child health outcomes (S_parcount, S_diarrhea, S_anemia, S_haz, S_whz) are the paper's main child-level claims.

**No missing baseline groups**: Table 7 robustness outcomes are correctly excluded from core (they are alternative/falsification outcomes, not main claims). Cognitive tests (S_mccdts, S_pbdypct) have very high missingness and are correctly excluded.

## B) Design Selection

**Assessment**: `randomized_experiment` is correct.

- The Piso Firme program was implemented in Torreon but not Durango; the paper treats this as a natural experiment with cluster-randomized treatment.
- `design_audit` correctly records cluster-level randomization, OLS ITT estimator, and clustering at `idcluster`.
- One design variant included: `design/randomized_experiment/estimator/diff_in_means` for G1 only. This is appropriate as a minimal-adjustment comparison.

## C) RC Axes

**Assessment**: Appropriate axes selected.

- **Controls**: Leave-one-out from the full model (Model 4) and progressive control sets (Models 1-4) correctly implement the paper's revealed control surface. The paper itself reports 4 control models.
- **Random subsets**: 10 draws for G1, stratified by size. Reasonable given the control pool size.
- **Sample trimming**: 1/99 and 5/95 outcome trimming included. Appropriate for robustness.

**Missing axes considered**:
- FE variants: Not applicable -- the paper uses no fixed effects in the baseline (pure cross-sectional comparison). This is correct since randomization should ensure balance.
- Functional form: Not included. The main outcomes are shares (0-1) or scales, so log/asinh transforms are not natural. Excluding is reasonable.

## D) Controls Multiverse Policy

- `controls_count_min=0` (Model 1, no controls) and `controls_count_max=38` (Model 4, full controls + missingness dummies) correctly derived from the paper's four models.
- `linked_adjustment=false`: Correct -- no bundled estimators.
- Missingness handling: The paper replaces missing control values with 0 and adds missingness indicator dummies. This must be replicated exactly in the runner.

## E) Inference Plan

- **Canonical**: `CRV1: idcluster` matches the paper's `cl(idcluster)`. Correct.
- **Variant**: HC1 (no clustering) is a reasonable stress test given 136 clusters.
- **Variant**: Cluster at `idmun` (municipality) is a coarser level. With only 2 municipalities (Torreon vs Durango), this is actually problematic -- too few clusters for valid inference. **Changed**: Removed `idmun` clustering from G2 and G3; kept only for G1 where it serves as a stress test to flag.

## F) Budgets + Sampling

- G1: 60 specs (reasonable, primary outcome group)
- G2: 30 specs (5 outcomes x ~6 variations each)
- G3: 40 specs (5 outcomes x ~8 variations each)
- Total ~130 specs: Well within feasibility.
- Seed 114542 recorded for reproducibility.

## G) Diagnostics Plan

- No diagnostics plan specified. For an RCT, balance tests (Table 2) and attrition analysis would be standard, but these are not part of the core specification surface per the contract.

## Changes Made

1. Surface structure validated -- no structural changes needed.
2. Confirmed missing data handling matches Stata code (replace with 0, add dummies).
3. Verified outcome variables exist in datasets with adequate non-missing counts.
4. Noted that `idmun` clustering has only ~2 clusters (one per city), so inference variant results should be interpreted with extreme caution.

## Final Assessment

**APPROVED TO RUN**. The surface is coherent, faithful to the manuscript, and appropriately scoped. The three baseline groups cover the paper's main claims across housing, wellbeing, and child health. Control variation is well-structured around the paper's own four-model progression.
