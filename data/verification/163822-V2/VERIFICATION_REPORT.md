# Verification Report: 163822-V2

## Paper
- **Title**: Digital Addiction
- **Authors**: Allcott, Gentzkow, and Song (2022)
- **Journal**: American Economic Review
- **Paper ID**: 163822-V2

## Baseline Groups

### G1: Bonus Treatment Effect on FITSBY Usage
- **Claim**: Bonus incentives reduce FITSBY app usage (minutes/day) during the treatment period (Period 3).
- **Baseline spec_id**: `baseline`
- **Treatment**: B (Bonus)
- **Outcome**: PD_P3_UsageFITSBY
- **Baseline coefficient**: -56.10 (SE = 3.08, p < 0.001)
- **Expected sign**: Negative

### G2: Limit Treatment Effect on FITSBY Usage
- **Claim**: Screen-time limits (commitment devices) reduce FITSBY app usage (minutes/day) during the treatment period (Period 3).
- **Baseline spec_id**: `baseline_limit`
- **Treatment**: L (Limit)
- **Outcome**: PD_P3_UsageFITSBY
- **Baseline coefficient**: -22.79 (SE = 2.82, p < 0.001)
- **Expected sign**: Negative

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 65 |
| Baselines | 2 |
| Core tests (is_core_test=1) | 37 |
| Non-core tests (is_core_test=0) | 28 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 18 | Control set variations (add/drop controls, leave-one-out, progressive addition) |
| core_sample | 13 | Sample restrictions (demographics, winsorization, trimming) |
| core_funcform | 3 | Functional form changes (log, IHS, quadratic baseline control) |
| core_inference | 3 | Inference variations (classical, HC2, HC3 standard errors) |
| noncore_alt_outcome | 13 | Alternative outcomes (total usage, individual apps, other periods, well-being indices) |
| noncore_heterogeneity | 12 | Heterogeneity analyses (interaction terms and conditional main effects) |
| noncore_placebo | 3 | Placebo/balance tests (pre-treatment outcomes) |

## Classification Rationale

### Core Tests (37 specs)
- **Control variations (18)**: These specs vary the control set while keeping the same outcome (PD_P3_UsageFITSBY) and treatment (B or L) on the full sample. This includes bivariate regressions, progressive control addition, leave-one-out on stratification variables, adding demographics, and a model with both treatments. These all estimate the same causal object with different conditioning sets.
- **Sample restrictions (13)**: These restrict the sample by gender, age, education, baseline usage level, or apply winsorization/trimming. They preserve the same outcome and treatment and estimate the treatment effect on different subpopulations or with different outlier handling. Although demographic subsamples could be viewed as heterogeneity tests, they estimate the unconditional subgroup ATE (not an interaction coefficient), so they function as sample restrictions.
- **Functional form (3)**: Log, IHS transformations, and quadratic baseline control. These change how the outcome is measured but preserve the directional interpretation of the treatment effect.
- **Inference (3)**: Classical, HC2, and HC3 standard errors. Same point estimates, different inference method.

### Non-Core Tests (28 specs)

- **Alternative outcomes (13)**: These change the outcome variable to a different concept:
  - Total usage (all apps, not just FITSBY) -- broader measure
  - Usage in different periods (P4, P432, P5432) -- measures persistence/spillover, different time window
  - Subjective well-being, addiction, and SMS indices -- entirely different constructs
  - Individual app usage (Facebook, Instagram, Twitter, Snapchat, Browser, YouTube) -- subcomponents, not the aggregate FITSBY measure

- **Heterogeneity (12)**: Six pairs of (main effect, interaction term) for gender, age, baseline usage, education, addiction level, and restriction index. The "main" effects from interaction models are conditional on the moderator equaling zero, which changes the estimand. The interaction terms test for differential effects, not the average treatment effect. Neither is directly comparable to the baseline unconditional ATE.

- **Placebo (3)**: Effect of Bonus treatment on pre-treatment usage, education, and income. These are randomization balance checks, not tests of the core causal claim.

## Top 5 Most Suspicious Rows

1. **robust/heterogeneity/baseline_usage_main** (spec #51): The treatment variable is listed as "B (Bonus) x baseline_usage" and the coefficient is -37.09 -- substantially smaller than the baseline (-56.10). This is the conditional main effect when High_Usage=0, not the unconditional ATE. The naming in treatment_var is somewhat misleading; it should be clear this is a conditional effect.

2. **robust/heterogeneity/baseline_usage_interaction** (spec #52): Coefficient of -38.03 (p < 0.001) suggests significant heterogeneity by baseline usage. This is the only interaction term that is statistically significant. While correctly classified as non-core, this might warrant further investigation if it changes the interpretation of the main effect.

3. **robust/outcome/usage_total** (spec #15): Uses outcome PD_P3_Usage (total usage, all apps) rather than PD_P3_UsageFITSBY. The coefficient (-51.76) is close to the baseline but the controls still use PD_P1_UsageFITSBY as baseline control, not PD_P1_Usage. This may be a specification error -- the baseline control should arguably match the outcome.

4. **robust/sample/high_usage** (spec #33): Coefficient of -74.91 is notably larger than the full-sample baseline (-56.10). The stratification variable strat_3 has a near-zero coefficient (1.53e-14) with a very small SE, which looks numerically suspicious and may indicate a collinearity or convergence issue in this subsample.

5. **robust/outcome/swb_index** (spec #19): Coefficient of 0.04 (p = 0.26) for the effect on subjective well-being. While correctly classified as non-core, this is arguably one of the paper's important secondary claims (does reducing phone use improve well-being?). Its non-significance is a substantively important finding but does not test the primary usage-reduction claim.

## Recommendations for the Spec Search Script

1. **Separate heterogeneity clearly**: The heterogeneity specs report both the conditional main effect and the interaction coefficient as separate rows. The conditional main effect (treatment_var labeled as "B (Bonus) x moderator") is confusing because it is not the same estimand as the unconditional ATE. Consider either (a) not including the conditional main effects as separate specs, or (b) labeling them more clearly.

2. **Match baseline controls to outcome**: For alternative outcome specs (e.g., usage_total), the baseline control variable should arguably match the outcome measure. Currently, PD_P1_UsageFITSBY is used as the baseline control even when the outcome is PD_P3_Usage (total usage). This is a minor concern but worth flagging.

3. **More Limit treatment robustness**: Only 4 specs (baseline_limit, limit_nocontrols, limit_fullcontrols, both_limit) test the Limit treatment. Consider adding sample restrictions, functional form, and inference variations for the Limit treatment as well.

4. **Consider including Period 3 outcomes as core**: The Period 4 and averaged-period outcomes (usage_p4, usage_p432, usage_p5432) could arguably be considered core tests if the paper's claim is about the general effect of the bonus, not specifically about Period 3. However, since the baseline is clearly Period 3, these are conservatively classified as non-core alternative outcomes.

5. **Individual app outcomes**: The individual app usage outcomes (Facebook, Instagram, etc.) are subcomponents of the FITSBY aggregate. They test a different granularity of the claim. If the paper treats these as robustness checks (decomposing the aggregate effect), they could potentially be included as core. However, since each measures a different outcome variable and the baseline claim is about the aggregate, they are classified as non-core.
