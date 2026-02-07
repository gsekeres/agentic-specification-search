# Verification Report: 157781-V1

## Paper: Canal Closure and Rebellions in Imperial China (AER)

### Baseline Groups

**G1**: The closure of the Grand Canal in 1825 increased rebellions in counties along the canal.
- Baseline spec IDs: `baseline`, `baseline_with_controls`
- Outcome: `ashonset_km2` (asinh of rebellion onsets per 100km^2)
- Treatment: `interaction1` (Along Canal x Post-1825)
- Expected sign: positive
- Both baselines differ only by controls (none vs. larea_after + rug_after). Grouped as a single baseline claim.

---

### Classification Summary

| Category | Count |
|----------|-------|
| **Total specifications** | **65** |
| Baselines (is_baseline=1) | 2 |
| Core tests (is_core_test=1) | 41 |
| Non-core tests | 17 |
| Invalid | 7 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_fe | 5 |
| core_sample | 20 |
| core_inference | 2 |
| core_funcform | 4 |
| noncore_placebo | 7 |
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 1 |
| noncore_heterogeneity | 6 |
| noncore_diagnostic | 1 |
| invalid | 7 |
| **Total** | **65** |

---

### Top 5 Most Suspicious Rows

1. **robust/sample/canal_counties_only** (spec_id: robust/sample/canal_counties_only)
   - Treatment variable is `reform` instead of `interaction1`. Coefficient is ~1e-17 (numerically zero). Clearly a failed estimation where restricting to canal counties caused the treatment variable to be collinear or constant. Marked **invalid**.

2. **robust/sample/non_canal_counties** (spec_id: robust/sample/non_canal_counties)
   - Same issue as above. Treatment is `reform` rather than `interaction1`, and coefficient is ~1e-19. Non-canal counties have no variation in the along-canal indicator, so the original interaction cannot be estimated. Marked **invalid**.

3. **robust/sample/winsorize_1pct / winsorize_5pct / winsorize_10pct**
   - All three winsorization specs produce exactly 0.0 for both coefficient and SE. The outcome variable names are modified (`ashonset_km2_winsorized_Xpct`) but the estimation clearly failed -- the winsorized variables appear to contain no variation (possibly all zeros after winsorization of a sparse outcome). Marked **invalid**.

4. **robust/sample/early_period** and **robust/sample/late_period**
   - Both produce coefficient = 0.0 (or ~1e-21 for late_period). The early period (pre-1825) has no post-treatment variation, so the interaction is identically zero. The late period subsample is entirely post-treatment, making the interaction collinear with the along-canal indicator. Both are **invalid** estimations.

5. **robust/treatment/distance_canal** (spec_id: robust/treatment/distance_canal)
   - Changes the treatment variable from a binary along-canal indicator to continuous distance (`distcanal_after`). This fundamentally changes the causal object being estimated (from ATE on canal counties to a distance-response gradient). Classified as **noncore_alt_treatment**. The negative sign is expected since greater distance should reduce the effect, but this is a different estimand.

---

### Notes on Specific Classifications

**Outcome variants**: 
- `robust/outcome/log` (lonset_km2), `robust/outcome/binary` (onset_any), `robust/outcome/count` (onset_all), and `robust/outcome/per_km2` (onset_km2) are classified as **core_funcform** because they measure the same underlying phenomenon (rebellion onset) with different functional forms or scaling.
- `robust/outcome/attacks` (ashattack) and `robust/outcome/defend` (ashdefend) are classified as **noncore_alt_outcome** because attacks and defensive actions are conceptually different from rebellion onsets -- they measure different aspects of conflict intensity.

**Heterogeneity specs**:
- All 6 heterogeneity specs are classified as **noncore_heterogeneity** because the coefficient of interest is the triple-interaction (or interaction with a moderator), which tests whether the effect *differs* across subgroups rather than testing the baseline claim itself. The main effect (interaction1) is still present in these regressions but the reported coefficient is for the heterogeneity term.

**Pre-treatment trend**:
- `did/sample/pre_treatment` uses treatment variable `pretrend` on a pre-1825 subsample (N=101,200). This is a diagnostic test for parallel trends, not a test of the main claim. Classified as **noncore_diagnostic**.

**did/controls/none**:
- This spec is identical to the `baseline` spec (same coefficient, SE, p-value, N, FE, cluster). It appears to be a duplicate. Classified as core_controls for consistency.

---

### Recommendations for the Spec-Search Script

1. **Fix winsorization**: The winsorized outcome variables produce all-zero coefficients. The winsorization likely needs to be applied before the asinh transformation, or the variable construction has a bug.

2. **Fix early/late period splits**: These splits create subsamples where the treatment interaction has no variation. Consider using event-study windows instead (which are already implemented as window_10yr through window_50yr).

3. **Fix canal/non-canal subsamples**: The script uses `reform` as the treatment variable instead of `interaction1` for these subsamples. Within canal-only counties, the along-canal indicator is constant, so the original interaction cannot be estimated. These should either use a different identification strategy or be removed.

4. **Heterogeneity coefficient extraction**: The heterogeneity specs correctly report the interaction/triple-interaction coefficient, but this means they are not testing the same estimand as the baseline. The script could optionally also extract the main treatment effect from these regressions.

5. **Distance treatment interpretation**: The distance_canal spec changes the causal object. Consider keeping it but flagging it as an alternative identification strategy rather than a robustness check of the baseline.
