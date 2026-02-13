# Verification Report: 113561-V1

**Paper**: Fong & Luttmer (2009), "What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty," *AEJ: Applied Economics*, 1(2), 64-87.

**Verified**: 2026-02-13

---

## Baseline Groups Found

| Group | Outcome | Treatment | Baseline spec_run_id | Baseline coef | Baseline p | N |
|-------|---------|-----------|----------------------|---------------|------------|---|
| G1 | giving (dictator game, 0-100) | picshowblack | 113561-V1_0001 | -4.198 | 0.370 | 915 |
| G2 | hypgiv_tc500 (hypothetical giving, topcoded $500) | picshowblack | 113561-V1_0038 | -2.181 | 0.591 | 913 |
| G3 | subjsupchar (charity support, 1-7) | picshowblack | 113561-V1_0072 | -0.221 | 0.167 | 907 |
| G4 | subjsupgov (government support, 1-7) | picshowblack | 113561-V1_0104 | -0.435 | 0.026 | 913 |

All four baseline groups match the `SPECIFICATION_SURFACE.json` plan. No missing or spurious groups. Each group has exactly one baseline spec. Baseline coefficients match the replication script to within rounding tolerance (< 0.001 as documented in SPECIFICATION_SEARCH.md).

---

## Row Counts

| Metric | Count |
|--------|-------|
| Total rows | 135 |
| Valid rows | 135 |
| Invalid rows | 0 |
| Baseline rows | 4 |
| Core test rows | 135 |
| Non-core rows | 0 |
| Unclear rows | 0 |

---

## Category Counts

| Category | Count |
|----------|-------|
| core_controls | 90 |
| core_sample | 20 |
| core_inference | 9 |
| core_method | 8 |
| core_weights | 4 |
| core_preprocess | 4 |
| **Total** | **135** |

### Breakdown by group

| Category | G1 | G2 | G3 | G4 |
|----------|----|----|----|----|
| core_method (baseline + design) | 2 | 2 | 2 | 2 |
| core_controls (sets + progression + loo + manipulation_coding) | 24 | 22 | 22 | 22 |
| core_sample (subpopulation) | 5 | 5 | 5 | 5 |
| core_weights (unweighted) | 1 | 1 | 1 | 1 |
| core_preprocess (outcome) | 2 | 2 | 0 | 0 |
| core_inference (SE type) | 3 | 2 | 2 | 2 |
| **Total** | **37** | **34** | **32** | **32** |

---

## Issues and Observations

### 1. Numerically identical specification pairs (12 total, documented)

Each group has three pairs of specs that produce identical numerical results:

| Pair | Spec A | Spec B | Reason |
|------|--------|--------|--------|
| 1 | `baseline` | `rc/controls/progression/manipulation_plus_demographics_plus_charity` | On white subsample, black/other race controls are collinear zeros and dropped, making baseline and this progression step identical |
| 2 | `design/.../diff_in_means` | `rc/controls/sets/none` | Both are regressions with treatment dummies only (no controls); same model by construction |
| 3 | `rc/controls/sets/extended` | `rc/controls/progression/full` | Both represent nraud + demographics + charitable + extra controls; identical control set |

These are **documented in SPECIFICATION_SEARCH.md** (Deviations notes 1-3) and represent different conceptual paths arriving at the same numerical result. They are **valid** specs but inflate the apparent diversity of results. This reduces the effective unique specification count by 12 (3 duplicated pairs x 4 groups).

**Effective unique specifications**: 135 - 12 = 123

### 2. Outcome variable drift in preprocessing specs (valid, expected)

G1 has 2 specs with transformed outcomes: `giving_tc99` (topcode at 99) and `giving_winsor` (winsorize 1-99). G2 has 2 specs with transformed outcomes: `hypgiv_tc250` (topcode at 250) and `hypgiv_notc` (no topcoding). These are legitimate outcome preprocessing RCs that transform the same outcome concept without changing it. The `no_topcode` variant in G2 is notable because it flips the coefficient sign (from -2.181 to +1.201, p=0.836), illustrating sensitivity to extreme outliers in hypothetical giving.

### 3. Estimand shift in race_shown_only subpopulation (borderline)

The `race_shown_only` (picobscur==0) subsample creates perfect collinearity between `picraceb` and `picshowblack`, since when the obscured condition is excluded, these become identical. One treatment arm is automatically dropped. The coefficient on `picshowblack` becomes a black-vs-white comparison excluding the obscured condition, which is a **slightly different estimand** than the baseline (which compares black-vs-all-others including obscured). Classified as `core_sample` with reduced confidence (0.80). This is documented in SPECIFICATION_SEARCH.md (note 6).

### 4. Full sample pools different target population

The `full_sample` spec pools white and black respondents, changing the target population from "white respondents" to "all respondents who passed audio check." Since black respondents show the opposite treatment effect pattern, this dilutes the effect. For G4 (the significant group), full_sample renders the result insignificant (p=0.171). Classified as `core_sample` with slightly reduced confidence (0.85) since the paper's claim is specifically about white respondents.

### 5. No execution failures

All 135 planned specs executed successfully. No NaN values in any numeric field. No skipped specs.

### 6. Surface alignment is perfect

All spec_ids in `specification_results.csv` match exactly the plan in `SPECIFICATION_SURFACE.json`. No missing specs, no spurious specs, and budgets match exactly (G1=37, G2=34, G3=32, G4=32).

### 7. G1 has extra specs vs other groups (expected)

G1 has 37 specs vs 32-34 for other groups because:
- G1 has 2 additional control progression steps (manipulation_only, manipulation_plus_demographics) not needed for G2-G4 (since these steps are already documented as part of the G1 progression)
- G1 has 2 outcome preprocessing specs (topcode_giving_at_99, winsor_1_99) while G3/G4 have none (bounded 1-7 scale outcomes)
- G1 has classical SE as a third inference variant (G2-G4 have only HC2/HC3)

---

## Summary Assessment

This is a **clean, well-executed specification search** with no invalid rows, no execution failures, and perfect alignment with the pre-run surface plan. The 4 baseline groups are coherent and well-defined claim objects. All 135 specifications are classified as core tests.

Key substantive finding: **Only G4 (government spending support) has a significant baseline result** (p=0.026), and this result is robust across 87.5% of specifications (28/32 at p<0.05). The 4 insignificant G4 specs are all subpopulation splits that reduce power. G1-G3 are consistently insignificant across all specifications.

The main quality concern is the 12 numerically identical specification pairs (3 per group), which are documented and expected but reduce the effective unique specification count from 135 to 123.

---

## Recommendations

1. **Consider flagging numerically identical spec pairs** in future surface designs. While conceptually distinct, they do not add informational value to the specification curve and could be deduplicated or annotated with a `is_duplicate_of` field.

2. **The race_shown_only estimand shift** could be more explicitly flagged in the surface plan with a note that the estimand changes when the obscured condition is excluded. Future surfaces could mark such specs as `estimand_note: "black vs white only, excluding obscured"`.

3. **No diagnostics for G2-G4**: Only G1 has a balance check diagnostic. Since the randomization is the same across all outcomes, one balance check is sufficient, but this could be noted in the surface plan.
