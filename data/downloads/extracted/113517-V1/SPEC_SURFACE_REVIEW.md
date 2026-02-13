# Specification Surface Review: 113517-V1

**Paper**: Moscarini & Postel-Vinay (AER P&P 2017) -- "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth"
**Reviewer**: Verifier agent (pre-run audit)
**Date**: 2026-02-13
**Status**: APPROVED TO RUN (with changes made)

---

## 1. Summary of Baseline Groups

The surface defines 4 baseline groups, each corresponding to a different outcome variable in the paper's single regression table (Table 1). This is appropriate because:

- The paper's main claim is that EE reallocation predicts nominal earnings growth but not hourly wage growth.
- The sign and significance patterns differ qualitatively across the 4 outcome variables.
- Each outcome is a genuinely different concept (earnings vs. hourly wages; nominal vs. real).

| Group | Outcome | Direction | Status |
|-------|---------|-----------|--------|
| G1 | Nominal earnings growth (xdlogern_nom) | Positive | Retained |
| G2 | Real earnings growth (xdlogern) | Positive | Retained |
| G3 | Nominal hourly wage growth (xdloghwr_nom) | Weak/null | Retained |
| G4 | Real hourly wage growth (xdloghwr) | Null/negative | Retained |

**No changes to baseline group structure.** The 4-group decomposition is well-motivated and faithful to the manuscript.

---

## 2. Changes Made to SPECIFICATION_SURFACE.json

### 2.1 Reclassification: Specs 2 and 3 (UE-only, UR-only) moved to explore/

**Issue**: The original surface listed all 9 Table 1 specifications as `rc/controls/sets/*` for the EE claim. However, Specs 2 (UE only) and 3 (UR only) drop xee entirely from the second-stage regression. They do not estimate an EE coefficient and therefore cannot serve as robustness checks for the EE-wage claim. They estimate a different treatment-outcome association.

**Change**: Reclassified as `explore/treatment/ue_only` and `explore/treatment/ur_only`. These are alternative treatment concepts, not estimand-preserving robustness checks.

**Rationale**: Per ARCHITECTURE.md, core RC must be estimand-preserving. Dropping the focal treatment variable changes the claim object entirely.

### 2.2 Reclassification: Spec 8 (job stayers) moved from rc/controls to rc/sample

**Issue**: Spec 8 was listed under `rc/controls/sets/all_flows_stayers` in the original surface, but it is a sample restriction (eetrans_i==0 & lagemp>0), not a change in the control set. The second-stage regressors are identical to Spec 6 (all flows).

**Change**: Reclassified as `rc/sample/restriction/job_stayers`. Remains core-eligible.

**Rationale**: This is a sample restriction that changes who enters the regression, not what regressors are included. The correct axis is `rc/sample`, not `rc/controls`.

### 2.3 Reclassification: Spec 9 (EE interaction) moved to explore/

**Issue**: Spec 9 adds xee_i = xee * eetrans_i (market-level EE rate interacted with individual-level EE transition indicator). This changes the interpretation of the coefficient on xee from "average EE-wage association" to "EE-wage association for non-movers (eetrans_i=0)." The focal estimand changes.

**Change**: Reclassified as `explore/estimand/ee_interaction`.

**Rationale**: Adding a treatment-by-moderator interaction changes the focal estimand interpretation. This is exploration, not estimand-preserving RC.

### 2.4 Reclassification: rc/fe/drop/market_fe moved to explore/

**Issue**: The original surface flagged this as "borderline explore/" and included it as `rc/fe/drop/market_fe`. Dropping market FE changes the identifying variation from within-market (the paper's stated approach) to pooled. This is a fundamental change in identification strategy.

**Change**: Reclassified as `explore/identification/pooled`.

**Rationale**: The paper explicitly relies on within-market variation (market FE absorption). Removing this changes the identifying variation, not just the implementation.

### 2.5 Market definition: finer_markets moved to explore/

**Issue**: Adding state to the market definition would create ~5,000 market groups and ~880,000 mkt_t cells. This is computationally speculative, the paper never considers geographic disaggregation, and many mkt_t cells would be extremely thin (potentially 0-1 observations), making FE extraction unreliable.

**Change**: Moved to `explore/data/aggregation/finer_markets`. Only `rc/data/aggregation/coarser_markets` remains in core.

**Rationale**: This is beyond the paper's revealed search space and computationally speculative with potentially invalid thin cells.

### 2.6 First-stage controls: u_controls_extended now defined consistently

**Issue**: The `core_universe.rc_spec_ids` in G1 listed `rc/controls/first_stage/u_controls_extended` as a variant, but it was not defined in the `shared_rc_axes.first_stage_control_variants` section.

**Change**: Added explicit definition for `rc/controls/first_stage/u_controls_extended` with u_controls = ["lagstate", "laguni"]. This is an intermediate variant that adds laguni to the unemployed controls.

### 2.7 Controls count max corrected from 7 to 6

**Issue**: The original surface set `controls_count_max = 7` based on Spec 9 (all flows + EE interaction = 7 regressors). Since Spec 9 has been reclassified to explore/, the maximum within-core second-stage regressor count is 6 (all flows: EE+UE+UR+NE+EN+EU).

**Change**: `controls_count_max` set to 6 for all baseline groups.

### 2.8 G3/G4 first-stage eligibility explicitly documented

**Issue**: The original surface noted the lagphr==1 restriction in the markdown (section 10.4) but did not make it explicit in the JSON constraints for G3 and G4, nor did it flag that the first-stage extraction must be re-run for G3/G4 (not shared with G1/G2).

**Change**: Added `g3_g4_eligibility_notes` in the two_stage_procedure section and explicit warnings in G3/G4 constraints and core_universe notes that the first-stage extraction is NOT shared with G1/G2.

### 2.9 Budgets adjusted

**Issue**: The original budget of 120 per group was based on 9 baseline flow specs. With reclassification, the core flow specs drop to 5. Budget arithmetic has been corrected.

**Change**: Budget reduced to 100 per group. New breakdown: 5 (flow specs) + 15 (5 first-stage variants x 3 anchors) + 18 (6 sample variants x 3 anchors) + 6 (2 FE variants x 3 anchors) + 6 (2 weight variants x 3 anchors) + 3 (1 market def variant x 3 anchors) + 12 (4 inference variants x 3 anchors) + 15 (cross-axis) = ~80 core + headroom to 100.

### 2.10 Diagnostics plan: Wooldridge serial correlation test added to G2-G4

**Issue**: The original surface included the Wooldridge serial correlation diagnostic only for G1. G2-G4 have the same panel structure and should also be tested.

**Change**: Added `diag/panel_fixed_effects/serial_corr/wooldridge` to all four baseline groups' diagnostics plans.

### 2.11 Direction expectations added

**Issue**: The original surface did not include `direction_expectation` in the claim objects.

**Change**: Added direction expectations based on the paper's findings: positive for G1/G2 (earnings), weak_positive_or_null for G3 (nominal hourly wage), null_or_negative for G4 (real hourly wage).

### 2.12 First-stage dependent variable details clarified

**Issue**: The original surface documented the first-stage as using `e_controls` uniformly, but the do-file shows that `lag{depvar}` enters only the EE transition and wage growth first-stage regressions (lines 165, 197), not the EU/EN regressions (lines 180-188). The wage growth first-stage also includes eetrans_i as a regressor (line 197).

**Change**: Updated the first-stage dependent_variables documentation to specify which regressions use lag(depvar) and which do not. Added note about eetrans_i in the wage growth first-stage.

---

## 3. Key Constraints and Linkage Rules

### 3.1 Two-stage linkage (CRITICAL)

The paper's two-stage procedure creates a fundamental linkage constraint:

- **First-stage FE extraction determines second-stage generated regressors.** Changing first-stage controls (e_controls, u_controls, lag(depvar)) changes xee, xue, xur, etc.
- **e_controls are shared** across EE, EU, EN transition regressions and the wage growth regression.
- **u_controls are shared** across UE, NE, UR regressions.
- When varying first-stage controls, the change must apply jointly to all regressions of the same type.

The surface correctly enforces `linked_adjustment = true` and treats first-stage control variants as a separate RC axis from second-stage flow composition.

### 3.2 G1/G2 vs G3/G4 first-stage sharing

- G1 and G2 share the same first-stage extraction (same eligibility: EZeligible = lagemp > 0, DWeligible = lagemp > 0 & emp > 0).
- G3 and G4 share a different first-stage extraction (eligibility additionally requires lagphr == 1).
- This means 2 distinct first-stage extractions, not 4. The surface now documents this clearly.

### 3.3 Control-count envelope

After reclassification, the second-stage regressor count ranges from 1 (EE only) to 6 (all flows). Year_month_num is always present as a continuous control and is not counted.

---

## 4. Budget and Sampling Assessment

### 4.1 Feasibility

Full enumeration is feasible. The universe is small:
- 5 core flow specs + 5 first-stage variants + 6 sample variants + 2 FE variants + 2 weight variants + 1 market definition variant + 4 inference variants = modest.
- One-axis-at-a-time from 3 anchor specs keeps the per-group total around 80-95.
- No random sampling is needed.

### 4.2 Computational cost

The main computational bottleneck is first-stage FE extraction (~10M observations, ~17,500 FE groups). The first stage must be re-run for:
- Each first-stage control variant (5 variants)
- Each market definition variant (1 variant)
- Each sample variant that affects the first-stage (5 temporal/quality variants)
- G3/G4 vs G1/G2 (different eligibility)

Second-stage regressions (flow composition, FE structure, weights, inference) are cheap once the first-stage FE are cached.

### 4.3 Total across groups

4 groups x ~95 specs = ~380 total specs. With first-stage caching between G1/G2 and between G3/G4, the effective computational cost is substantially lower.

---

## 5. What Is Missing

### 5.1 No missing baseline groups

The 4 outcomes in Table 1 are all represented. The paper does not present other tables or headline claims.

### 5.2 Potential additions (not blocking)

- **rc/data/coding/agegroup_cuts**: The agegroup variable is constructed with arbitrary age cutoffs (<=25, 26-35, 36-45, 46-60, >60). Alternative binnings could be tested. This is a data construction degree of freedom. Not added because the paper does not reveal this as varied.

- **rc/data/quality/dicey_dates**: The do-file drops observations where `dicey==1` (extreme dates in each panel). This is a data construction choice. Testing sensitivity to including these observations could be informative, but the do-file treats this as a data quality filter, not a specification choice. Not added.

- **Bootstrap SE for generated regressors**: The two-stage generated regressor problem means second-stage SE are inconsistent. A bootstrapped SE (resampling both stages) would be the gold-standard inference approach. This is listed in diagnostics but not in core inference. Consider promoting if computationally feasible.

---

## 6. Final Assessment

**APPROVED TO RUN.**

The specification surface is now:

1. **Conceptually coherent**: Each baseline group corresponds to a single claim object (EE-wage association for a specific wage concept). RC axes preserve the estimand. Exploration is explicitly separated.

2. **Statistically principled**: Core RC does not mix estimand-changing specifications. The reclassification of Specs 2, 3, 9 and the pooled regression ensures clean separation.

3. **Faithful to the revealed manuscript surface**: The paper reveals a single table with 9 specs x 4 outcomes. We retain the core-eligible subset (5 specs x 4 outcomes = 20 baseline/flow specs) and clearly classify the remainder.

4. **Auditable**: Budgets are explicit, full enumeration is feasible, linkage constraints are documented, first-stage sharing rules are clear.

No blocking issues remain.
