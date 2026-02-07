# Verification Report: 140161-V1

## Paper Summary

**Title**: Sharing Fake News: Survey Experiment on Fact-Checking Effects
**Journal**: AEJ-Policy
**Method**: Cross-sectional OLS (survey experiment with random assignment)
**Core Claim**: Exposure to imposed fact-checking (Survey 2) reduces French Facebook users stated intention to share fake news (Alt-Facts) on Facebook, relative to a control group that sees only the Alt-Facts (Survey 1).

---

## Baseline Groups

### G1: Imposed Fact-Check Effect on Sharing Intent

- **Baseline spec_id**: baseline
- **Outcome**: want_share_fb (binary: intent to share Alt-Facts on Facebook)
- **Treatment**: survey (Imposed Fact-Check) -- coefficient on C(survey)[T.2] in OLS
- **Expected sign**: Negative (fact-checking should reduce sharing)
- **Baseline coefficient**: -0.0448, p = 0.005, N = 2,546

Only one baseline group is warranted. The paper canonical claim centers on the Imposed Fact-Check arm (Survey 2) vs. the control (Survey 1). The Voluntary Fact-Check arm (Survey 3) is a secondary treatment classified as non-core.

---

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **66** |
| Baseline specs | 1 |
| Core test specs (incl baseline) | 49 |
| Non-core specs | 17 |
| Invalid specs | 0 |
| Unclear specs | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 31 |
| core_sample | 13 |
| core_inference | 3 |
| core_method | 2 |
| noncore_alt_treatment | 5 |
| noncore_alt_outcome | 1 |
| noncore_heterogeneity | 7 |
| noncore_placebo | 3 |

### Core tests (49 specs)
- **core_controls** (31): Baseline + 7 control progressions + 23 leave-one-out + 1 redundant outcome spec
- **core_sample** (13): Subgroup analyses on demographic splits (gender, age, education, location, politics, religion)
- **core_inference** (3): Classical SE, HC2, HC3 standard error variations
- **core_method** (2): Logit and Probit estimators for the binary outcome

### Non-core tests (17 specs)
- **noncore_alt_treatment** (5): Voluntary Fact-Check treatment arm (different causal object) and Imposed vs Voluntary comparison
- **noncore_alt_outcome** (1): want_share_others -- sharing with others rather than on Facebook
- **noncore_heterogeneity** (7): Interaction terms testing differential treatment effects by subgroup
- **noncore_placebo** (3): Pre-treatment characteristics (altruism, reciprocity, image) used as placebo outcomes

---

## Top 5 Most Suspicious Rows

1. **robust/build/bivariate**: Exact duplicate of baseline (identical coefficients, same formula, same sample). Should arguably be removed or merged. Not harmful but redundant.

2. **robust/outcome/want_share_fb**: Labeled as an alternative outcome but uses the same outcome (want_share_fb) as baseline. It is actually a controls variation (strata + socioeconomic), identical to robust/build/socioeconomic. Mislabeled in spec_tree_path as measurement when it is really a control progression spec.

3. **robust/treatment/imposed_vs_voluntary**: Uses a different sample (Surveys 2-3 only, excluding control) and a different treatment contrast (Imposed vs Voluntary). The baseline claim is about Imposed vs Control. This is clearly a different estimand. Coefficient is near zero (-0.006, p=0.68), which is expected since both treatments have similar effects.

4. **robust/het/interaction_religious**: Reports the interaction coefficient (survey x religious = -0.138, p=0.005) which is statistically significant. However, this is a heterogeneity test, not a main effect test. The treatment_var label correctly identifies this, but the spec search summary counts it among positive and significant specs, which could be misleading.

5. **robust/sample/high_education**: Reports a positive coefficient (+0.011, p=0.79) -- the only sample restriction with a sign reversal and non-significant result. This is a legitimate subgroup analysis but the small subgroup size likely drives the imprecision.

---

## Recommendations for the Specification Search Script

1. **Remove or flag duplicates**: robust/build/bivariate is identical to baseline, and robust/outcome/want_share_fb is identical to robust/build/socioeconomic. Consider adding deduplication logic.

2. **Fix outcome labeling**: robust/outcome/want_share_fb should not be in the alternative outcomes category since it uses the same outcome as baseline. It should be reclassified as a control variation.

3. **Clarify heterogeneity extraction**: The heterogeneity specs extract the interaction term coefficient, not the main treatment effect. This is correct behavior for heterogeneity analysis, but the spec search should clearly distinguish these from main-effect robustness checks in its summary statistics.

4. **Consider adding the paper preferred specification**: The paper preferred specification appears to include stratification + socioeconomic controls (matching robust/build/socioeconomic). The baseline was chosen as the no-controls version, which is appropriate for a specification curve, but the paper may consider the controlled version as its primary estimate.

5. **Add marginal effects for Logit/Probit**: The Logit (-0.483) and Probit (-0.253) coefficients are on different scales than OLS (-0.045). For comparability in a specification curve, average marginal effects should be computed.
