# Specification Surface Review: 138922-V1

**Paper**: Marcus, Siedler & Ziebarth, "The Long-Run Effects of Sports Club Vouchers for Primary School Children"

**Review date**: 2026-02-24

---

## A) Baseline Groups

**Assessment**: PASS

One baseline group (G1) is appropriate. The paper's main claim is the effect of C2SC vouchers on sports club membership. Table 2 Column 3 is the clear preferred specification. The paper also reports effects on other outcomes (program knowledge, voucher receipt/redemption, sport hours, overweight), but these are secondary outcomes reported in the same regression framework. They are appropriately included as `rc/form/outcome/*` rather than separate baseline groups.

No changes made.

---

## B) Design Selection

**Assessment**: PASS

`difference_in_differences` is correct. The paper uses a standard DiD with repeated cross-sections (school health exam cohorts). Treatment state: Saxony (C2SC program from 2008/09). Control states: Brandenburg and Thuringia.

`design_audit` is complete: records the TWFE estimator, treatment timing, comparison groups, FE structure, and clustering.

Note: The paper is a repeated cross-section DiD, not a panel DiD. The `panel_unit` field correctly notes "repeated_cross_section" rather than naming an entity identifier. This is important for interpretation: there is no within-unit variation; identification comes from state x cohort variation.

No changes made.

---

## C) RC Axes

**Assessment**: PASS with minor notes

All included RC axes are estimand-preserving:

1. **Controls**: The LOO family starts from the full control set (9 controls from Table 5 s12) and drops each one. This is appropriate. The baseline has 0 controls, so adding the full set is itself an RC, and LOO variants from the full set are clean ceteris paribus checks.

2. **Sample window**: All four window variants are taken directly from Table 4 of the paper (r1-r4). Appropriate.

3. **Sample composition**: Five variants from Tables 4-5. The `drop_nonmiss_restriction` spec (r7) is valid -- it relaxes the sample restriction to include observations with some missing covariates, which is a legitimate data construction choice.

4. **Fixed effects**: Three variants. Dropping city FE is meaningful because city FE absorb a lot of variation and the treatment is at the state level. Dropping state FE when city FE are present is also valid (city subsumes state).

5. **Treatment definition**: Three variants from Table 5. The `treat_v2` alternative coding and the first-wave/current-state treatments are legitimate data construction alternatives that preserve the same estimand concept.

6. **Alternative outcomes**: Five outcomes from Table 2. These preserve the treatment concept while varying the outcome concept. This is borderline between RC and explore -- the outcomes `kommheard`, `kommgotten`, `kommused` measure program uptake rather than the final outcome. However, since they are reported in the same table and frame, they are appropriate as `rc/form/outcome/*` for a broad specification search.

7. **Joint specs**: 14 joint combinations of controls with other axes. These are well-motivated because the paper itself reports many of its robustness checks without controls, and adding controls is a natural stability check.

No changes made.

---

## D) Controls Multiverse Policy

**Assessment**: PASS

- `controls_count_min=0` (baseline has no individual controls)
- `controls_count_max=9` (full set from Table 5 s12)
- No `linked_adjustment` needed (single-equation OLS)
- The 9 controls are: female, siblings, born_germany, parent_nongermany, newspaper, art_at_home, academictrack, sportsclub_4_7, music_4_7

These are all pre-determined individual characteristics, so adding/removing them is appropriate for robustness.

No changes made.

---

## E) Inference Plan

**Assessment**: PASS

- **Canonical**: CRV1 at cityno matches the paper's baseline exactly.
- **Variants**: HC1, cluster at state, cluster at cohort. These are all from Table B5.

Note: State-level clustering (only 3 clusters) is extremely aggressive and may not produce reliable inference. This is noted in the surface.

No changes made.

---

## F) Budgets and Sampling

**Assessment**: PASS

Full enumeration is feasible with ~52 planned specs. The control pool is small (9 controls), so LOO is tractable without sampling. No seed-based sampling needed.

---

## G) Diagnostics Plan

**Assessment**: PASS (empty)

No formal pre-trend test is included in the diagnostics plan. The paper does not have a traditional panel pre-trend test because this is a repeated cross-section. The paper's Figure B3 shows outcome differences by year, but this is more of a visual diagnostic than a testable spec. The decision to omit diagnostics is reasonable.

---

## Summary of Changes Made

No changes were made to the surface. The surface is well-structured, faithful to the paper's revealed search space, and ready for execution.

---

## Final Verdict

**APPROVED TO RUN**

The surface is conceptually coherent, statistically principled, and faithful to the manuscript. All ~52 specifications are estimand-preserving and tractable. Proceed to execution.
