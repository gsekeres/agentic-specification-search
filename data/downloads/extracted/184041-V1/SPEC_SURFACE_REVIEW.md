# Specification Surface Review: 184041-V1

**Paper**: "The Common-Probability Auction Puzzle" (Ngangoue and Schotter)
**Review date**: 2026-02-25
**Status**: APPROVED TO RUN (after edits below)

---

## Summary of Baseline Groups

| Group | Claim Object | Design | Baseline Specs |
|---|---|---|---|
| G1 | CV vs CP bid factors in auctions (Exp I) | randomized_experiment | 4 specs: BF/BEBF/NEBF x OLS, plus median variants in baseline_spec_ids |
| G2 | CV vs CP price factors in individual pricing (Exp II) | randomized_experiment | 2 specs: BF x OLS/median, stage 22 |
| G3 | Decision weights (structural decomposition, Exp I) | randomized_experiment | 3 specs: CV-only, CP-only, interaction model (Table A4) |

All three groups correspond to distinct claim objects with different populations and outcome concepts. This is correct.

---

## Issues Found and Changes Made

### Issue 1: G2 -- `rc/form/median_regression` was redundant

The original surface listed `rc/form/median_regression` as an RC spec for G2, but the baseline already includes `baseline__tablea3_expii_bf_medianreg_stage22`. This is the same redundancy that was already cleaned up for G1. Removed from rc_spec_ids.

**Change**: Replaced `rc/form/median_regression` with `rc/sample/stage21` and `rc/sample/stage1` -- these are the stage variants from Table A3 code (stage 21 = compound lottery without signal; stage 1 = reduced lottery). These are genuine sample variants that the paper reveals and that preserve the claim object.

### Issue 2: G3 -- `design/randomized_experiment/estimator/diff_in_means` was incoherent

G3's baseline specs are structural regressions of the form `lnbid ~ lnfix + lnsignal` (Table A4). These are not treatment-control mean comparisons. A diff-in-means estimator is meaningless for a structural decomposition -- there is no single binary treatment effect to compute as a difference in means.

**Change**: Removed `design/randomized_experiment/estimator/diff_in_means` from G3's design_spec_ids (now empty list).

### Issue 3: G3 -- Three `rc/form/*` items changed the claim object

- `rc/form/expii_pricing_weights`: Moves from Experiment I auction bidding to Experiment II individual pricing. This changes both the population and the outcome (lottery price vs auction bid).
- `rc/form/expii_pricing_weights_interaction`: Same issue.
- `rc/form/structural_decomposition_expv_nmexpv`: Uses Table A6's different functional form (`bid ~ expV + nmexpV + sigstage`) for Experiment II pricing. Changes population, outcome, and functional form.

All three change the claim object and should be classified as `explore/*`, not `rc/*`.

**Change**: Moved all three to exploration_notes with rationale. G3's rc_spec_ids now contains only `rc/sample/full_sample_no_exclusion` and `rc/form/ols_instead_of_median`.

### Issue 4: G3 -- `target_population` incorrectly pooled experiments

The original surface listed the target population as "Experiments I, II, III/IIIb" but G3's baseline specs are all from Table A4, which uses Experiment I data only.

**Change**: Updated target_population to "NYU lab experiment participants in Experiment I (auction bidding, reduced sample)".

---

## Key Constraints and Linkage Rules

1. **No controls to vary**: G1 and G2 have zero controls in all baseline specs. G3 has structural regressors (lnfix, lnsignal) that are not optional controls but structural components -- they cannot be dropped.
2. **Structural form linkage in G3**: The interaction model's CVlnfix and CVlnsignal must move together. The linked_adjustment=true flag correctly enforces this.
3. **Sample exclusion rule**: The reduced sample criterion (domnBid<=8 for Exp I, goodsample==1 for Exp III/IIIb) is a design choice. Full-sample variants are included as RC specs.
4. **Experiment isolation**: Each experiment has its own subject pool. Cross-experiment comparisons are exploration, not robustness.

---

## Budget/Sampling Assessment

| Group | Budget | Feasibility |
|---|---|---|
| G1 | 50 core specs | Feasible. ~6 baseline-like rows + 1 design + 2 sample RC x 6 outcome/estimator combos = well within budget |
| G2 | 30 core specs | Feasible. 2 baseline rows + 1 design + 2 stage RC x 2 estimators = small universe |
| G3 | 30 core specs | Feasible. 3 baseline rows + 2 RC specs x applicable combos = small universe |

Full enumeration is correct for all groups given the absence of combinatorial control variation.

---

## What's Missing (Minor)

1. **G2 stage variants could use more detail**: The surface could note that stage 21 (compound lottery without signal) and stage 1 (reduced lottery) test different information conditions. These are arguably on the border between RC and exploration, but the paper presents them in the same table (A3), suggesting they are part of the revealed surface.
2. **G1/G2 missing BEBF/NEBF for Exp II**: Table A3 code only shows BF regressions for Experiment II (not BEBF or NEBF). This is correctly omitted from the surface.
3. **Kolmogorov-Smirnov diagnostic**: The paper mentions a KS test for signal distribution balance (footnote 11). This could be added to G1's diagnostics_plan but is not blocking.

---

## Final Assessment

**Approved to run.** The surface is now conceptually coherent after the four edits above. Each baseline group corresponds to a well-defined claim object. The RC axes are estimand-preserving. Exploration items that change populations or outcomes have been properly excluded from the core universe. Budgets are feasible. Inference plans match the paper's baseline clustering.
