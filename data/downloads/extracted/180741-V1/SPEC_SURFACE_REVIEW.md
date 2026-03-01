# Specification Surface Review: 180741-V1

**Paper**: "Enabling or Limiting Cognitive Flexibility? Evidence of Demand for Moral Commitment" (Saccardo & Serra-Garcia)
**Reviewed**: 2026-02-24
**Surface file**: SPECIFICATION_SURFACE.json (created 2026-02-24)

---

## Summary of Baseline Groups

The surface proposes three baseline groups across two datasets:

| ID | Claim | Table | Dataset |
|----|-------|-------|---------|
| G1 | Choice experiment — effect of preference (`choicebefore`) on recommendation (`recommendincentive`) | Table 3 | `choice_experiments.dta` |
| G2 | NoChoice experiment — effect of random assignment (`seeincentivefirst`) on recommendation (`recommendincentive`) | Table C.1 | `nochoice.dta` |
| G3 | Choice experiment — effect of treatment condition on preference (`choicebefore`) | Table 2 | `choice_experiments.dta` |

These are three genuinely distinct claim objects. The grouping is appropriate.

---

## Issues Found and Changes Made

### Issue 1: G1 baseline_specs — Col 2 (getyourchoice==0) is missing from JSON specs but referenced in core_universe

**Severity: MODERATE**

The JSON `baseline_specs` array for G1 contains only two specs (Table3-Col1 with `getyourchoice==1` and Table3-Col3 the full pooled sample). However, `core_universe.baseline_spec_ids` lists three IDs: `baseline__table3_col1`, `baseline__table3_col2`, and `baseline__table3_col3`.

The `baseline__table3_col2` object (corresponding to the `getyourchoice==0` subsample — "not assigned preference") is referenced but never defined in the `baseline_specs` array. The code at line 582 of `analysis.do` confirms this specification exists:

```stata
eststo: reg recommendincentive choicebefore choicebeforenoconflict ///
    noconflict incentiveB $covariates2 if Highx10==0 & Highx100==0 & getyourchoice==0, vce(hc3)
```

This spec uses the same 14 covariates as Col 1 but a different sample filter (`getyourchoice==0`). **Action taken**: Added the missing baseline spec to the JSON.

### Issue 2: G2 baseline_specs — Col 2 (conflict==0) is missing from JSON specs but referenced in core_universe

**Severity: MODERATE**

Same structure as Issue 1. G2's `baseline_specs` only defines Col1 (`conflict==1`) and Col3 (full sample with interaction), but `core_universe.baseline_spec_ids` references `baseline__tablec1_col2` which is never defined. The code at line 142 confirms:

```stata
eststo:reg recommendincentive seeincentivefirst noconflict incentiveB female age stdalpha if missingalpha==0 & conflict==0, vce(hc3)
```

This is the no-conflict subsample, 5 covariates, `missingalpha==0`. **Action taken**: Added the missing baseline spec to the JSON.

### Issue 3: G2 sample filter — `missingalpha==0` not `alphavaluefinal!=.`

**Severity: MINOR**

The `design_audit.sample_restriction` says `missingalpha==0 (attentive participants only)`, which is correct. The `baseline_specs` correctly use `missingalpha==0` in their `sample_filter` fields. However, `SPECIFICATION_SURFACE.md` (line 93) describes the inattentive RC as "drop the `alphavaluefinal!=.` filter" which is semantically equivalent but uses different variable syntax. The `missingalpha` variable is defined in `datacleaning.do` as `replace missingalpha=1 if alphavaluefinal==.`, so these are the same. No change needed but noted for coder clarity.

### Issue 4: G3 baseline_specs — Col 3 (stdalpha + selfishness interactions) missing from JSON specs

**Severity: MODERATE**

G3's `baseline_specs` defines two specs (Table2-Col1 and Table2-Col2), but `core_universe.baseline_spec_ids` references `baseline__table2_col3` which is never defined. The code at line 284 confirms:

```stata
eststo:reg choicebefore $covariates2 stdalpha selfishseeincentivecostly selfishseequalitycostly if Highx10==0 & Highx100==0, vce(hc3)
```

This is the full sample, 13 covariates (covariates2 + stdalpha + selfishseeincentivecostly + selfishseequalitycostly). **Action taken**: Added the missing baseline spec to the JSON.

### Issue 5: G1 controls_count_min inconsistency

**Severity: MINOR**

The `constraints.controls_count_min` for G1 is set to 14, which is correct for Col 1 and Col 2. However, Col 3 uses 17 controls, and an unspecified Col 2 also uses 14. The notes field describes this correctly. No change needed.

### Issue 6: G2 controls_count_min set to 3 but baseline minimum is 5

**Severity: MINOR**

Both G2 baseline specs define 5–6 controls. The `controls_count_min` of 3 would allow specifications with only 3 controls, which is too permissive and not grounded in any actual specification. The true minimum from the baseline specs is 5 (Col 1 and Col 2 with `noconflict, incentiveB, female, age, stdalpha`). The note says "Minimal controls: incentiveB + demographics. seeincentivefirst_noconflict is a structural interaction term for the full-sample specification." So the floor of 3 appears to imagine dropping `noconflict` and `stdalpha`, but `noconflict` is structural (defines the interaction). **Action taken**: Updated `controls_count_min` from 3 to 4 (allowing drop of stdalpha only from the structural baseline, while keeping noconflict + incentiveB + female + age as minimum floor).

### Issue 7: G1 rc/controls/loo items include structural interaction terms

**Severity: MINOR**

The LOO list for G1 includes `seeincentivecostly` and `seequalitycostly`. The surface's own constraint notes say "structural interaction terms... are NOT optional controls but define the estimand" but only lists `choicebeforenoconflict`, `noconflict`, etc. In fact, `seeincentivecostly` and `seequalitycostly` are treatment arm indicators (ChoiceFree is the omitted reference category). Dropping them from a pooled Choice experiment regression changes the treatment composition comparison and arguably changes the estimand. These are border cases — they are legitimate LOO checks to assess sensitivity of treatment arm pooling, but should be understood as design-type variations rather than pure control LOO. The surface is acceptable as written with this caveat documented.

### Issue 8: G3 rc/sample/include_high_stakes uses umbrella label but code uses Highx10/Highx100 separately

**Severity: MINOR**

The RC `rc/sample/include_high_stakes` bundles two distinct high-stakes conditions (`Highx10==1` and `Highx100==1`) into one RC label, whereas the paper (Tables C.18 and C.19) analyzes them separately. For precision, these could be split into `rc/sample/include_high_stakes_10x` and `rc/sample/include_high_stakes_100x` (matching G1 which already uses this split). **Action taken**: Split in the JSON for G3 to be consistent with G1.

---

## Verified Correct Elements

1. **Variable names**: All key variable names verified against `datacleaning.do` and `analysis.do`:
   - `recommendincentive`, `choicebefore`, `getbefore`, `getyourchoice`, `noconflict`, `seeincentivefirst`, `seeincentivefirst_noconflict`, `notgetyourchoice`, `choicebeforenoconflict`, `choicebeforenotgetyourchoice`, `notgetyourchoicenoconflict` — all confirmed constructed in cleaning.
   - `stdalpha`, `selfishseeincentivecostly`, `selfishseequalitycostly` — confirmed constructed.
   - `professionalsfree`, `seeincentivecostly`, `seequalitycostly` — confirmed from datacleaning.

2. **Global covariate set**: `$covariates2` is confirmed at line 229 of `analysis.do`:
   ```stata
   global covariates2 "professionalsfree seeincentivecostly seequalitycostly wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female"
   ```
   This matches exactly what the surface describes.

3. **Structural terms correctly identified**: `choicebeforenoconflict`, `noconflict`, `notgetyourchoice`, `choicebeforenotgetyourchoice`, `notgetyourchoicenoconflict` are all present in the Table 3 regression and correctly identified as non-optional.

4. **Inference: HC3 throughout**: Confirmed at lines 141–143, 282–284, 576–587 of `analysis.do`. No clustering used. Correct.

5. **Sample filter G1**: `Highx10==0 & Highx100==0` confirmed at all Table 3 regressions. `alphavaluefinal` attentiveness filter is applied via `drop if study!=1 & alphavaluefinal==.` at the top of the Choice section (line 428), which covers all non-professional participants.

6. **Dataset assignment**: G1/G3 use `choice_experiments.dta`, G2 uses `nochoice.dta`. Confirmed.

7. **Excluded experiments**: Belief regressions (Table 4, `logitbelief`, no-constant regression), stakes experiment (`stakes.dta`), Information Architect (`InformationArchitect.dta`), Choice Deterministic (`Choice_Deterministic.dta`), NoChoice Simultaneous (`nochoice1_2_merged.dta`) — all excluded and explained. Correct.

8. **Design code**: All three groups correctly coded as `randomized_experiment`. Correct.

9. **Diagnostics plan G1**: Balance check on `age`, `female` across `getbefore` — correct (this is the random assignment in Choice experiment).

---

## Budget and Sampling Assessment

- **G1 budget (60 specs)**: Reasonable. With 3 baselines, 2 design variants, 12 LOO controls, 3 additions, 7 sample restrictions, and 2 functional forms, estimated ~31–40 specs. Full enumeration is feasible.
- **G2 budget (30 specs)**: With 3 baselines (after fix), 2 design variants, 4 LOO, 1 addition, 5 sample restrictions, 2 functional forms, estimated ~20 specs. Adequate.
- **G3 budget (30 specs)**: With 3 baselines (after fix), 2 design variants, 9 LOO, 3 additions, 3 sample restrictions (now 4 after splitting), 2 functional forms, estimated ~24 specs. Adequate.
- **Seed**: 180741 — matches paper ID. Reproducible.
- **Sampler**: Full enumeration is appropriate given the small control pools.

---

## What Is Missing

1. **G1 and G3 missing diagnostics plan**: G3 has an empty `diagnostics_plan`. G1 has one entry (balance check on demographics). For a Choice experiment, a natural diagnostic is whether the randomization of `getbefore` is balanced conditional on `choicebefore` — this is a design validity check. However, since this is a stated-preference then randomized assignment design, any imbalance in `getbefore` by `choicebefore` would reflect the design's intent (advisors who prefer incentive-first may be differentially assigned to incentive-first). So the existing diagnostic is adequate.

2. **No cross-experiment comparison baseline group**: The paper includes Table C.16/C.17 comparing NoChoice and Choice directly (merged dataset). The surface correctly notes this is excluded since "different datasets and different treatment variables." This is appropriate — the merged analysis is a separate claim object that would require its own surface.

3. **G1 does not have `incentiveB` in LOO list**: `incentiveB` is listed as a "structural regressor" that is "always included" for G1 (in the constraint notes). This is correct — `incentiveB` varies across advisors (which product is incentivized) and defines potential incentive conflict interactions. It should not be in the LOO pool.

---

## Changes Made to SPECIFICATION_SURFACE.json

1. **Added missing `baseline_specs` entries**:
   - G1: Added `Table3-Col2` (getyourchoice==0, 14 controls, same as Col1)
   - G2: Added `TableC1-Col2` (conflict==0, 5 controls)
   - G3: Added `Table2-Col3` (stdalpha + selfishseeincentivecostly + selfishseequalitycostly, 13 controls)

2. **Fixed G2 `controls_count_min`**: Changed from 3 to 4 (noconflict + incentiveB + female + age minimum)

3. **Split G3 `rc/sample/include_high_stakes`**: Into `rc/sample/include_high_stakes_10x` and `rc/sample/include_high_stakes_100x` to match G1 and enable the two appendix tables separately

---

## Final Assessment

**APPROVED TO RUN** with the modifications documented above.

The core surface is sound: design identification is correct (pure randomized experiment), variable names are verified against the code, the global covariate macro `$covariates2` is confirmed, structural vs. optional controls are correctly distinguished, and the exclusion list is well-justified. The three missing baseline spec definitions were the primary structural issue; they are now resolved. The inference plan (HC3 throughout, no clustering) is verified. Budget targets are achievable with full enumeration.
