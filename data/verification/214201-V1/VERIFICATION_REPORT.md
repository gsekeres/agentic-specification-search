# Verification Report: 214201-V1

**Paper**: Khan, Khwaja, Olken -- "Mission vs. Financial Incentives for Community Health Workers" (AER)
**Paper ID**: 214201-V1
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## 1. Baseline Groups

### G1: Mission incentive effect on household visits
- **Claim**: Mission-based incentives (no bonus) increase the probability that a household is visited by a Lady Health Worker.
- **Expected sign**: Positive
- **Baseline spec_id(s)**: `baseline`
- **Outcome**: `lhw_visit` (binary: household visited by LHW)
- **Treatment**: `treat_mission_nobonus`
- **Baseline coefficient**: 0.051 (SE = 0.012, p < 0.001)
- **Design**: Block-randomized controlled trial; panel regression with block + wave FE, household probability weights, SEs clustered at LHW level. Pooled waves 2, 3, 4.

There is a single baseline group. The paper studies multiple treatment arms (mission, financial bonus, combined, social recognition), but the specification search focuses on the mission-no-bonus treatment as the primary object of interest. This is appropriate: the paper's title and main contribution center on whether mission-based incentives affect effort.

---

## 2. Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **55** |
| Baseline | 1 |
| Core tests (including baseline) | 39 |
| Non-core tests | 16 |
| Invalid | 0 |
| Unclear | 0 |

### Core Test Breakdown

| Core Category | Count |
|---------------|-------|
| core_controls | 11 |
| core_sample | 19 |
| core_fe | 3 |
| core_inference | 3 |
| core_method | 3 |

### Non-Core Test Breakdown

| Non-Core Category | Count |
|-------------------|-------|
| noncore_alt_outcome | 8 |
| noncore_alt_treatment | 3 |
| noncore_placebo | 2 |
| noncore_heterogeneity | 3 |

---

## 3. Classification Rationale

### Core tests (39 specs)

**FE variations (3)**: Specs `panel/fe/none`, `panel/fe/block_only`, `panel/fe/wave_only` vary the fixed effects structure while keeping the same outcome, treatment, sample, and weights. These directly test whether the baseline result is sensitive to the FE specification. In an RCT context, treatment is randomized conditional on block, so the FE choice should not materially change point estimates (and it does not: coefficients range from 0.051 to 0.055).

**Sample restrictions (19)**: Includes wave-specific estimates (waves 2, 3, 4), baseline performance splits (high/low), outlier trimming (top/bottom 5% community sizes, block sizes), and leave-one-out block analysis (10 blocks). All use the same outcome (`lhw_visit`) and treatment (`treat_mission_nobonus`). These test whether the effect is driven by particular time periods, subgroups, or geographic clusters.

**Inference variations (3)**: Specs `robust/cluster/iid_se`, `robust/cluster/block`, and `robust/inference/crv3` change the standard error calculation (iid, block-clustered, CRV3 small-sample correction) without changing the point estimate. These test whether statistical significance is robust to inference choices.

**Controls variations (11)**: Includes 4 pure control additions (`add_baseline_perf`, `add_iq`, `add_psm`, `full_controls`) and 6 heterogeneity interaction specs (`health_diploma`, `years_school`, `tenure`, `psm`, `iq_score`, `baseline_perf`) plus the baseline. The heterogeneity specs are classified as core_controls because they report the main treatment coefficient after adding interaction terms as controls. Adding controls in an RCT should not affect the treatment estimate if randomization is balanced, and indeed the coefficient remains stable (0.050-0.059 across all control specs).

**Method variations (3)**: `robust/weights/unweighted` removes household weights. `robust/method/tabA4_unweighted` is a duplicate of the same specification (identical coefficients). `robust/treatment/mission_all` uses a broader treatment definition pooling all mission arms (with and without bonus) but tests the same fundamental hypothesis about mission incentives.

### Non-core tests (16 specs)

**Alternative outcomes (8)**: Five specs use different outcome variables (`were_preg_served`, `were_child_served`, `tb_check`, `diarrhea_incidence`, `vac_sch`) that measure downstream health outcomes rather than the primary claim about household visits. Three conditional-on-visit versions (`pregnant_served_cond`, `child_served_cond`, `tb_check_cond`) further condition on the endogenous outcome (visit), changing the estimand.

**Alternative treatments (3)**: `robust/treatment/financial_incentive` reports the coefficient on financial bonuses (`treat_bonus_pr`), `robust/treatment/mission_plus_bonus` reports the combined arm (`treat5`), and `robust/treatment/public_vs_private` decomposes mission into public vs. private recognition. These all change the causal object being estimated.

**Placebo tests (2)**: `robust/treatment/placebo_social` reports the coefficient on the social recognition treatment (designed as a placebo with expected null effect). `robust/placebo/baseline_period` tests for pre-treatment effects in wave 0 (expected null). Both test validity rather than the main claim.

**Heterogeneity subsamples (3)**: `robust/sample/hh_tercile_small`, `hh_tercile_medium`, `hh_tercile_large` split by community size tercile. The paper itself highlights differential effects by community size. These are heterogeneity analyses, not tests of the overall average effect.

---

## 4. Top 5 Most Suspicious Rows

1. **`robust/method/tabA4_unweighted`** (spec 47): This is an exact duplicate of `robust/weights/unweighted` (spec 16). Both have coefficient = 0.04756, SE = 0.01156, p = 0.0000435, N = 21299. The only difference is the spec_id and sample_desc label. This inflates the specification count without adding information. **Recommendation**: Remove one of the two duplicates in the spec-search script.

2. **`robust/outcome/pregnant_served_cond`** (spec 52): Appears identical to `robust/outcome/pregnant_served` (spec 17) -- same coefficient (0.0599), same SE, same N (1920). The "conditional on visit" label should produce a different subsample, but the numbers match exactly. This suggests the conditioning may not have been applied correctly, or the unconditional and conditional samples coincide for this subgroup. **Recommendation**: Verify that the conditioning on `lhw_visit == 1` actually reduces the sample for pregnant-women households.

3. **`robust/outcome/child_served_cond`** (spec 53): Same concern as above -- identical to `robust/outcome/child_served` (spec 18). Coefficient = 0.0336, N = 3352 in both. **Recommendation**: Same as above.

4. **`robust/outcome/tb_check_cond`** (spec 54): Same concern -- identical to `robust/outcome/tb_check` (spec 19). Coefficient = 0.0499, N = 8605 in both. **Recommendation**: Same as above.

5. **`robust/sample/hh_tercile_medium`** (spec 50): This is the only core-outcome spec (lhw_visit, treat_mission_nobonus) that is not statistically significant at 5% (p = 0.105, coef = 0.040). Together with `hh_tercile_small` (p = 0.057), these show the effect is concentrated in larger communities. This is noteworthy but correctly classified as noncore_heterogeneity.

---

## 5. Recommendations for the Spec-Search Script

1. **Remove duplicate specification**: `robust/method/tabA4_unweighted` is identical to `robust/weights/unweighted`. The script runs the same regression twice. Remove one.

2. **Fix conditional-on-visit outcomes**: Specs 52-54 (`*_cond`) appear to produce identical results to their unconditional counterparts (specs 17-19). The script conditions on `lhw_visit == 1` but the alternative outcome variables (`were_preg_served`, `were_child_served`, `tb_check`) may only be defined for visited households anyway, making the conditioning vacuous. Verify whether these outcomes have meaningful variation outside of visited households. If not, drop the conditional specs as uninformative duplicates.

3. **Community size tercile specs**: These are correctly implemented but should be labeled more clearly as heterogeneity analyses rather than sample restrictions in the `spec_tree_path`.

4. **Consider adding**: The spec search does not include a probit/logit specification (the outcome is binary), which would be a natural functional form variation. Also absent: wild cluster bootstrap inference, which is relevant given the relatively small number of clusters (blocks).

---

## 6. Overall Assessment

The specification search is well-constructed and comprehensive. The baseline claim is clearly identified, and 39 of 55 specs (71%) are genuine core tests of the same estimand. The mission treatment effect is remarkably stable across all core specifications, with coefficients ranging from 0.043 to 0.059 and all 39 core specs statistically significant at the 5% level. The main concern is 4 apparent duplicate specifications (1 exact duplicate + 3 conditional-outcome duplicates) that should be investigated and potentially removed.
