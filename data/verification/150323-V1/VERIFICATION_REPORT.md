# Verification Report: 150323-V1

**Paper**: "Political Turnover, Bureaucratic Turnover and the Quality of Public Services"
**Authors**: Mitra Akhtari, Diana Moreira, Laura Trucco
**Journal**: American Economic Review
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## Baseline Groups

### G1: 4th Grade Test Scores
- **Claim**: Political turnover (party change in close elections) negatively affects 4th grade standardized test scores in municipal schools.
- **Expected sign**: Negative
- **Baseline spec_ids**: `baseline`, `rd/bandwidth/optimal_both_score`, `robust/outcome/both_score_4_std`
- **Outcome variable**: `both_score_4_std`
- **Note**: All three baseline-tagged specs report identical coefficients (-0.042, p=0.101). They are exact duplicates.

### G2: Teacher Turnover
- **Claim**: Political turnover increases the share of new teachers in municipal schools, indicating patronage-driven bureaucratic turnover.
- **Expected sign**: Positive
- **Baseline spec_ids**: `rd/bandwidth/optimal_newtchr`, `robust/outcome/newtchr`
- **Outcome variable**: `newtchr`
- **Note**: No spec is explicitly tagged "baseline" for this outcome. Both specs report identical coefficients (0.128, p<0.001).

### G3: Headmaster Replacement
- **Claim**: Political turnover increases headmaster replacement (probability headmaster has less than 2 years at current school).
- **Expected sign**: Positive
- **Baseline spec_ids**: `rd/bandwidth/optimal_expthissch`, `robust/outcome/expthisschl_lessthan2_DPB`
- **Outcome variable**: `expthisschl_lessthan2_DPB`
- **Note**: No spec is explicitly tagged "baseline" for this outcome. Both specs report identical coefficients (0.274, p<0.001).

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **93** |
| **Baselines** | **7** (3 for G1 + 2 for G2 + 2 for G3; includes duplicates) |
| **Core tests** | **76** |
| **Non-core tests** | **17** |
| **Invalid** | **0** |
| **Unclear** | **0** |

### Core Test Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 20 |
| core_sample | 48 |
| core_method | 4 |
| core_inference | 2 |
| core_funcform | 2 |

### Non-Core Category Breakdown

| Category | Count |
|----------|-------|
| noncore_placebo | 5 |
| noncore_alt_outcome | 9 |
| noncore_heterogeneity | 3 |

---

## Core Tests by Baseline Group

| Baseline Group | Core Specs (incl. baselines) | Non-Core |
|----------------|------------------------------|----------|
| G1 (Test scores) | 49 | -- |
| G2 (Teacher turnover) | 16 | -- |
| G3 (Headmaster replacement) | 11 | -- |
| None (unassigned) | -- | 17 |

---

## Top 5 Most Suspicious / Noteworthy Rows

1. **`rd/poly/order1_both_score`** (coef=+0.014, p=0.76): This is tagged as a linear polynomial spec for test scores, but the coefficient *flips sign* to positive and becomes completely insignificant. The baseline has "Baseline scores + school controls" while this has "School controls" only, meaning the baseline test score control was dropped. This is likely a controls difference rather than purely a polynomial difference. The sign flip may be driven by the absence of the lagged score control, not the polynomial order. **Classified as core_method but should be interpreted with caution.**

2. **`rd/controls/full_both_score`** (coef=+0.023, p=0.70): Full controls specification for test scores shows a sign flip to positive. The "full controls" include teacher-level variables that may be post-treatment (teacher characteristics could change after political turnover), introducing bad-control bias. **Classified as core_controls but the sign flip warrants caution.**

3. **`robust/het/school_size_interaction`** (coef=-0.190, p=0.027): The main treatment coefficient in this heterogeneity spec is much larger in magnitude (-0.19 vs -0.04 baseline) and significant. However, this is an interaction model where the main effect coefficient has a different interpretation (effect for reference category). **Classified as noncore_heterogeneity -- the coefficient represents effect for one subgroup, not the overall average.**

4. **`robust/loo/drop_*` specs** (6 specs, all with sign flip to positive ~0.00--0.02): All six leave-one-out control specifications for test scores show tiny positive coefficients, contrasting with the baseline negative coefficient. However, these LOO specs use "Controls minus X" but the baseline uses "Baseline scores + school controls" -- it appears the LOO specs may have dropped the lagged baseline score control while also dropping one other control. If this is a mis-specification (unintended removal of baseline scores), all 6 are questionable. **Classified as core_controls but may reflect an implementation issue.**

5. **`robust/funcform/log_lefttchr`** (coef=+0.358, p<0.001): Log of left-teacher share. This is a functional form variant of a variable (`lefttchr`) that is not itself a baseline claim outcome. It measures teachers who *left* (vs. `newtchr` which measures teachers who arrived). While conceptually related to G2, it is a different estimand. **Classified as noncore_alt_outcome.**

---

## Recommendations

1. **Fix leave-one-out implementation**: The LOO specifications appear to have different base control sets than the baseline (missing the lagged baseline score). They should drop one control from the exact baseline control set. Currently they all produce near-zero, positive coefficients, suggesting a systematic issue rather than sensitivity to individual controls.

2. **Separate polynomial and controls variations**: `rd/poly/order1_both_score` seems to differ from baseline in both polynomial order AND controls. The spec search should hold controls constant when varying polynomial order.

3. **Create explicit baselines for G2 and G3**: The spec search only designates one `baseline` spec (for test scores). It should create separate `baseline_newtchr` and `baseline_expthissch` specs to make the multi-outcome structure explicit.

4. **Consider whether dropout rate (tx_abandono_primary) should be a separate baseline group**: The paper discusses dropout as an additional outcome, but it never finds significant effects. If the paper does not make a strong claim about dropout effects, it is correctly classified as noncore.

5. **Heterogeneity specs report main-effect coefficients**: The interaction specifications report the coefficient on `pX_dummy`, which in an interaction model represents the effect for the reference category only. This is not directly comparable to the baseline average effect. The spec search should either extract the total average marginal effect or clearly note the interpretation difference.

6. **Inference specs may have controls mismatch**: `robust/inference/robust_hc1` and `robust/inference/robust_hc3` use "Year dummy" as controls rather than the baseline "Baseline scores + school controls." This means they differ from baseline in both inference method AND controls. The coefficient (-0.019) is different from baseline (-0.042), confirming they are not pure inference variations.
