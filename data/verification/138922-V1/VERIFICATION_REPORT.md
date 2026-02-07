# Verification Report: 138922-V1

## Paper
**Title**: The Long-Run Effects of Sports Club Vouchers for Primary School Children
**Authors**: Marcus, Siedler, Ziebarth
**Journal**: AEJ: Policy (2022)

## Baseline Groups

### G1: Sports club membership effect
- **Claim**: Sports club vouchers for primary school children in Saxony increase long-run sports club membership relative to children in Brandenburg and Thuringia.
- **Baseline spec_ids**: baseline
- **Outcome**: sportsclub (binary: sports club membership)
- **Treatment**: treat (Saxony post-2008 indicator)
- **Expected sign**: Positive (+)
- **Baseline result**: coef = 0.0089, SE = 0.0187, p = 0.635 (null finding)
- **FE structure**: year + state (bula) + city
- **Clustering**: City level (cityno)

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **84** |
| **Baseline** | **1** |
| **Core tests** | **40** |
| **Non-core tests** | **44** |
| **Invalid** | **0** |
| **Unclear** | **0** |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 21 | Control variable variations (baseline + full/minimal + leave-one-out + cumulative) |
| core_fe | 5 | Fixed effects structure variations (none, time-only, unit-only, twoway, city-only) |
| core_sample | 10 | Sample restriction variations (year drops, geography, data quality, siblings) |
| core_inference | 3 | Inference variations (robust HC, state cluster, cohort cluster) |
| core_method | 1 | Estimation method (simple DiD with dummies) |
| core_funcform | 0 | No functional form specs for the membership outcome |
| noncore_alt_outcome | 27 | Different outcome variables (program awareness, age-specific membership, sport hours, health/smoking/alcohol, obesity) |
| noncore_alt_treatment | 2 | Different treatment definitions (treat_v2, t_tcoh_bula) |
| noncore_heterogeneity | 12 | Subgroup analyses (by gender, urban/rural, newspaper, art_at_home, academic track, prior sports) |
| noncore_placebo | 3 | Placebo outcomes (health1, eversmoked, everalc) |
| noncore_diagnostic | 0 | None |
| invalid | 0 | None |
| unclear | 0 | None |

## Classification Rationale

### Core tests (40 specs)
All core tests share the same outcome (sportsclub) and treatment (treat) as the baseline. They vary only in:
- **Controls** (21 specs): The baseline uses no controls. The spec search adds full controls, minimal controls, leave-one-out variations (dropping each control one at a time), and cumulative control additions.
- **Fixed effects** (5 specs): Vary the FE structure from none through various combinations to the baseline year+state+city.
- **Sample** (10 specs): Vary the time window (drop years, extend pre-period), geography (drop a control state), data quality filters, and sibling restrictions.
- **Inference** (3 specs): Same point estimate as baseline with different standard errors (robust HC, state-level clustering, cohort clustering).
- **Method** (1 spec): Simple DiD with state/year dummies instead of absorbed FE.

### Non-core: Alternative outcomes (27 specs)
The largest non-core category. These specifications use outcome variables that measure fundamentally different concepts:
- **Program awareness/first-stage** (6 specs: kommheard, kommgotten, kommused, and parent-reported versions): These measure whether families knew about/received/used vouchers. They are first-stage outcomes, not the behavioral outcome of interest.
- **Age-specific membership** (7 specs: ll6-ll12): Membership at specific ages. While related to the baseline outcome, these are distinct dependent variables measuring recall of membership at particular ages rather than current membership.
- **Sport hours and variants** (10 specs: sport_hrs, sport1hrs, sport2hrs, sport3hrs, sport_alt2, log_sport_hrs, ihs_sport_hrs, winsorized 1%/5%, trimmed 1%): These measure hours of sports participation rather than club membership. While conceptually related, they are a different outcome variable.
- **Health/behavioral** (4 specs: oweight, obese, currentsmoking, alclast7): Health and behavioral outcomes that are downstream or unrelated to the primary sports club membership outcome.

### Non-core: Alternative treatments (2 specs)
Two specifications use different treatment variable definitions:
- **treat_v2**: An alternative treatment definition with different sample restrictions (N=6117 vs baseline N=13333).
- **t_tcoh_bula**: Treatment based on current location rather than cohort-based assignment. This changes the causal object from intent-to-treat to a different identification strategy.

### Non-core: Heterogeneity (12 specs)
Subgroup analyses splitting the sample by observable characteristics (gender, urban/rural, newspaper access, art at home, academic track, prior sports club participation). These are not core tests because heterogeneity is not part of the baseline claim and each subsample tests a different (conditional) estimand.

### Non-core: Placebo (3 specs)
Tests using outcomes (health, smoking, alcohol) that should not be affected by the voucher program. These serve as falsification tests, not direct tests of the main claim.

## Top 5 Most Suspicious Rows

1. **robust/sample/winsorize_1pct** (spec_id: robust/sample/winsorize_1pct): Tagged as a sample restriction but the outcome is sport_hrs_wins_1pct (winsorized sport hours), not sportsclub. This is effectively a different outcome, not just a sample restriction. Classified as noncore_alt_outcome.

2. **robust/sample/winsorize_5pct** (spec_id: robust/sample/winsorize_5pct): Same issue as above -- outcome is sport_hrs_wins_5pct rather than sportsclub. Classified as noncore_alt_outcome.

3. **robust/sample/trim_1pct** (spec_id: robust/sample/trim_1pct): Tagged as sample restriction but uses sport_hrs as outcome (not sportsclub). The trimming is applied to sport_hrs, making this a different outcome. Classified as noncore_alt_outcome.

4. **robust/funcform/log_sport_hrs** and **robust/funcform/ihs_sport_hrs**: Tagged as functional form variations but they transform sport_hrs (hours), not sportsclub (binary membership). A log/IHS transform of a binary variable would be nonsensical, so these are clearly testing a different outcome. Classified as noncore_alt_outcome.

5. **did/fe/none** (spec_id: did/fe/none): No fixed effects at all. While classified as core (it tests the same outcome/treatment), it has a very different point estimate (0.047 vs 0.009) suggesting confounding. This spec may not be a credible test of the causal claim, though it technically satisfies the "same estimand" criterion. Kept as core_fe but noted.

## Recommendations for the Specification Search Script

1. **Separate first-stage from second-stage outcomes**: The 6 program-awareness outcomes (kommheard, kommgotten, kommused and parent versions) should be clearly marked as first-stage/mechanism outcomes, not robustness checks of the main claim. They test a different hypothesis (program implementation) than the baseline (behavioral change).

2. **Distinguish outcome variations from sample/functional-form variations**: The winsorized/trimmed sport_hrs specs are filed under sample_restrictions but should be classified as outcome variations since the dependent variable changes. Similarly, the log/IHS transforms are not functional form variants of the baseline (binary) outcome.

3. **Consider whether age-specific membership is truly "alternative outcomes"**: The ll6-ll12 variables may be viewed as alternative measurements of the same concept (sports club participation at different ages). However, since the baseline uses a contemporaneous measure and these are retrospective age-specific measures, they should remain classified as alternative outcomes.

4. **Treatment variation clarity**: The two alternative treatment definitions (treat_v2, t_tcoh_bula) could be more clearly documented in terms of what exactly changes in the identification strategy.

5. **Heterogeneity vs. sample restriction**: The heterogeneity specs and some sample restriction specs (e.g., no_older_siblings) are conceptually similar -- both split the sample. The distinction is that sample restrictions are data-quality motivated while heterogeneity is for testing effect modification. This distinction is maintained here.
