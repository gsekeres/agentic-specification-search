# Verification Report: 139262-V1

**Paper**: "Motivated Beliefs and Anticipation of Uncertainty Resolution" by Christoph Drobner
**Journal**: AER Papers and Proceedings
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Belief updating and asymmetric responsiveness (Table 2)
- **Claim**: Belief adjustments track Bayesian benchmarks but show asymmetric updating (stronger response to good vs bad news) that is attenuated by the Resolution treatment.
- **Expected sign**: Positive (belief adjustment correlates positively with Bayesian benchmark)
- **Baseline spec_ids**: `baseline`, `baseline_nores_bad`, `baseline_nores_did`, `baseline_res_good`, `baseline_res_bad`, `baseline_res_did`
- **Notes**: These six baselines correspond to the four treatment-signal cells (No-Resolution/Good, No-Resolution/Bad, Resolution/Good, Resolution/Bad) plus two DID-style pooled regressions (one per treatment) that include signal and signal*bayes interaction terms.

### G2: Ex-post rationalization -- study performance (Table 3)
- **Claim**: Good news increases perceived importance of IQ for study performance in the No-Resolution treatment but not in Resolution.
- **Expected sign**: Positive
- **Baseline spec_ids**: `baseline_studyperformance_no-resolution`, `baseline_studyperformance_resolution`

### G3: Ex-post rationalization -- job performance (Table 3)
- **Claim**: Good news increases perceived importance of IQ for job performance in the No-Resolution treatment but not in Resolution.
- **Expected sign**: Positive
- **Baseline spec_ids**: `baseline_jobperformance_no-resolution`, `baseline_jobperformance_resolution`

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | 123 |
| **Baselines** | 10 |
| **Core tests** | 100 |
| **Non-core** | 23 |
| **Invalid** | 0 |
| **Unclear** | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 42 |
| core_sample | 30 |
| core_funcform | 20 |
| core_inference | 6 |
| core_method | 2 |
| noncore_heterogeneity | 20 |
| noncore_placebo | 3 |

---

## Classification Rationale

### Core tests (100 specs)

**Control variations (42)**: The spec search systematically adds controls (rank, sumpoints, prior, demographics) to baseline bivariate regressions and DID regressions across all treatment-signal cells, plus leave-one-out from full-control models. These all preserve the same estimand (effect of Bayesian benchmark on belief adjustment, or effect of signal on performance ratings) while varying covariate sets.

**Sample restrictions (30)**: Includes exclusion of "wrong" belief adjustments (subjects who updated in the wrong direction), exclusion of zero adjustments, restriction to middle ranks (2 and 3), leave-one-session-out stability checks, and winsorization (5% and 10%). The winsorized specifications use slightly different outcome variable names (`belief_adjustment_wins_5`, `belief_adjustment_wins_10`) but are still measuring the same concept with outlier treatment, making them core sample/functional form robustness checks.

**Functional form (20)**: Includes absolute value of belief adjustment (collapsing good/bad into magnitude), quadratic terms, quantile regressions (25th, 50th, 75th percentiles), and rank-specific probability beliefs (rang1-4posteriorbelief). The rang1-4 outcomes decompose the expected-rank summary measure into individual rank probabilities -- these test the same underlying belief-updating hypothesis through different measurement.

**Inference (6)**: Session-clustered SEs, group-clustered SEs, and classical (non-robust) SEs, each for No-Resolution and Resolution DID specifications.

**Method (2)**: Pooled regression with treatment dummy, and fully interacted model with all treatment interactions. These are alternative estimation strategies for the same claim.

### Non-core tests (23 specs)

**Heterogeneity (20)**: Subgroup analyses by age (young/old), prior optimism (optimistic/pessimistic), IQ (high/low), and rank (1/2/3/4), each split by treatment condition. These are heterogeneity analyses -- they test whether the effect varies by subgroup, not the main claim itself. The paper does not make the heterogeneity a primary claim, so these are classified as non-core.

**Placebo (3)**: Two specs regress profit on signal (profit should not be affected by the signal since it is determined by IQ test performance, not beliefs), and one regresses prior beliefs on signal (checking randomization balance). These are diagnostic/validity checks, not tests of the core hypothesis.

---

## Top 5 Most Suspicious Rows

1. **`robust/placebo/prior_on_signal`**: This placebo test shows a statistically significant result (coef=-0.228, p=0.002), meaning prior beliefs are correlated with signal assignment. In a properly randomized experiment, this should be zero. The significant placebo result may indicate a randomization issue or a mechanical relationship through the Bayesian updating structure. This is concerning for the validity of all specifications but is correctly classified as non-core.

2. **`robust/sample/drop_session_2`, `drop_session_4`, `drop_session_6`, `drop_session_8`, `drop_session_9`**: These five "leave-one-session-out" specifications all produce identical coefficients and standard errors (coef=0.0762, se=0.1798, p=0.672). This strongly suggests that sessions 2, 4, 6, 8, and 9 were Resolution-only sessions and therefore dropping them does not change the No-Resolution subsample at all. The session drop is only applied to No-Resolution data, so these are effectively duplicates of `baseline_nores_did`. While not invalid, these redundant specifications inflate the specification count without adding information.

3. **`robust/funcform/quadratic_resolution_bad`**: Produces a large negative coefficient (-0.697, p=0.26) that reverses the expected sign. The quadratic term dominates, suggesting nonlinearity may invert the relationship in this small subsample (N=50). The result is not significant, so it is not alarming, but the sign reversal is notable.

4. **`robust/heterogeneity/age_young_no-resolution`**: Shows a significant negative coefficient (-0.415, p=0.19) with controls. The negative sign is opposite to the expected direction, though the p-value is not significant at conventional levels. The small subsample (N=39) limits power.

5. **`robust/estimation/fully_interacted`**: The treatment coefficient in this fully interacted model (0.076, p=0.67) is identical to the `baseline_nores_did` specification because the No-Resolution/Bad Signal group serves as the reference. The coefficient on `resolution_bayes` (0.569, p=0.064) captures the treatment heterogeneity but is only marginally significant, making the evidence for the Resolution attenuation of asymmetry weaker in this combined specification.

---

## Recommendations

1. **Remove redundant session-drop specs**: Five of the ten session-drop specifications are exact duplicates (sessions 2, 4, 6, 8, 9). The spec-search script should detect when dropping a session does not change the subsample and skip those duplicates, or alternatively run session drops on the full sample rather than only on the No-Resolution subsample.

2. **Investigate the prior-on-signal placebo**: The significant correlation between prior beliefs and signal assignment (p=0.002) is unexpected in a randomized experiment. This could be a mechanical artifact of how the Bayesian benchmark is constructed. The spec-search script should flag this as a concern rather than just including it as a standard placebo.

3. **Clarify heterogeneity vs sample restriction boundary**: The "middle ranks only" specifications (ranks 2 and 3) are classified as sample restrictions and core tests, while individual rank subgroups (rank 1, 2, 3, 4 separately) are heterogeneity. This distinction is reasonable -- ranks 2&3 is a trimming rule to exclude extreme ranks, whereas individual rank analysis is subgroup heterogeneity -- but the spec-search script could make this distinction more explicit in naming.

4. **Consider adding Resolution-side session drops**: The session-drop robustness is only applied to No-Resolution data. For completeness, the spec-search script could also run leave-one-session-out for the Resolution subsample.

5. **The rang1-4 outcome specifications could be grouped more explicitly**: These decompose the summary belief measure into rank-specific probabilities. The spec-search documentation could note that these are alternative outcome measurements rather than separate outcome concepts, which is how they are classified here (core_funcform for G1).
