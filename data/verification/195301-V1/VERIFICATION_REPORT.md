# Verification Report: 195301-V1

## Paper: Enemies of the People (Toews and Vezina, AEJ-Applied)

### Baseline Groups

**G1: Enemy share -> wages**
- Claim: Higher share of political prisoners (enemies of the people) in nearby 1952 Gulag camps causes higher wages in modern Russian firms.
- Expected sign: positive
- Baseline spec_ids: baseline, baseline_w_controls_1, baseline_full_controls, baseline_no_moscow, baseline_no_moscow_full
- Outcome: lnwage (log average wages)
- Treatment: share_enemies_1952_100 (share of enemies in camps within 100km)
- Method: WLS with Oblast FE, clustered at Gulag cluster level, employee-weighted

### Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 64 |
| Baselines | 5 |
| Core tests (non-baseline) | 39 |
| Non-core tests | 20 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 11 (5 baseline + 6 robustness) |
| core_sample | 14 (2 baseline + 12 robustness) |
| core_inference | 3 |
| core_fe | 3 |
| core_funcform | 5 |
| core_method | 13 |
| noncore_alt_outcome | 5 |
| noncore_heterogeneity | 14 |
| noncore_placebo | 1 |
| noncore_alt_treatment | 0 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

### Core Test Summary

Of the 44 core tests (including 5 baselines):
- 39 have positive coefficients on the treatment variable
- 5 have negative or near-zero coefficients (no_fe flips sign; unweighted yields ~0; quadratic linear term is smaller)
- The median coefficient among core tests with the baseline treatment variable is approximately 0.53
- 30 of 44 are significant at the 5% level

### Top 5 Most Suspicious Rows

1. **robust/control/none** (spec_id: robust/control/none)
   - Issue: Exact duplicate of the baseline specification. Coefficient (0.728), SE (0.238), and n_obs (699226) are identical. This is redundant and inflates the spec count without adding information.
   - Recommendation: Could be removed or flagged as duplicate.

2. **robust/estimation/no_fe** (spec_id: robust/estimation/no_fe)
   - Issue: Coefficient flips to -0.751 (negative and significant) when Oblast FE are dropped. This indicates the positive relationship is entirely driven by within-Oblast variation and reverses sign across Oblasts. While this is informative, it raises questions about the generalizability of the result.
   - Classification: Kept as core_fe because it tests the same estimand with a different FE structure.

3. **robust/weights/unweighted** (spec_id: robust/weights/unweighted)
   - Issue: Coefficient drops to 0.036 (p=0.847) when employee weighting is removed. This means the result is driven entirely by large firms or by the weighting scheme. This is a critical sensitivity finding.
   - Classification: Kept as core_method.

4. **robust/placebo/1939_controlling_1952** (spec_id: robust/placebo/1939_controlling_1952)
   - Issue: The 1939 enemy share is strongly significant (coef=0.973, p<0.001) even when controlling for the 1952 enemy share. Moreover, the 1952 share coefficient is -1.12 (negative and significant) conditional on the 1939 share. This complicates the causal story since the 1939 treatment should be absorbed if the 1952 share is the true mechanism.
   - Classification: noncore_placebo.

5. **robust/heterogeneity/large_firm_interact** (spec_id: robust/heterogeneity/large_firm_interact)
   - Issue: The main effect of treatment is -0.031 (essentially zero, p=0.90) while the interaction with large_firm is 0.791 (p=0.010). This confirms the effect is entirely concentrated in large firms, consistent with the unweighted result being null.
   - Classification: noncore_heterogeneity.

### Recommendations for Spec-Search Script

1. **Remove or flag the duplicate**: robust/control/none is identical to baseline. The spec-search script should check for duplicate coefficient+SE+n_obs combinations.

2. **Clarify treatment variable variations**: The spatial radius specs (10km-90km) change the treatment variable name but test the same concept. The script could annotate these more clearly as bandwidth/radius robustness rather than treatment changes.

3. **Separate interaction main effects from interaction terms**: For the heterogeneity interaction specs (large_firm_interact, moscow_distance_interact), the extracted coefficient is the main effect of treatment, which is not directly comparable to the baseline (it represents the treatment effect for the omitted category only). The script should either extract the full marginal effect or clearly note this distinction.

4. **Consider adding IV specifications**: The paper likely has instrumental variable specifications (using historical camp locations as instruments) that are not captured in this specification search.

5. **Winsorized outcome variables**: The specs robust/sample/winsorize_1pct and winsorize_5pct use slightly different outcome variables (lnwage_wins1, lnwage_wins5). The script correctly handles these but they could be more clearly labeled as functional form variations rather than sample restrictions.
