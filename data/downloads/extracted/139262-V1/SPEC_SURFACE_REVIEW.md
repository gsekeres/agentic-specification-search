# Specification Surface Review: 139262-V1

**Paper**: "Motivated Beliefs and Anticipation of Uncertainty Resolution" by Christoph Drobner
**Reviewer**: Automated verifier
**Date**: 2026-02-24

---

## A) Baseline Groups

**Assessment: PASS**

One baseline group (G1) is appropriate. The paper makes a single unified claim about asymmetric belief updating across Resolution/No-Resolution treatments and good/bad news signals. The 6 regression columns in Table 2 are different views of the same claim object, not separate claims.

The claim object is well-defined:
- Outcome: belief_adjustment (posterior - prior expected rank)
- Treatment: bayes_belief_adjustment (Bayesian benchmark adjustment)
- Estimand: OLS slope coefficient measuring responsiveness to the Bayesian benchmark
- Population: 200 lab subjects

**No missing baseline groups**: Table 3 (ordered logit on study/job performance) is a secondary outcome about ex-post rationalization. It uses a different model (ordered logit) and different outcome concept. Correctly excluded from core.

## B) Design Selection

**Assessment: PASS**

`randomized_experiment` is correct. This is a between-subject randomized experiment conducted in the lab using z-Tree.

The `design_audit` block is complete and includes:
- Estimator (OLS)
- Randomization unit (individual, between-subject)
- Treatment structure (binary: Resolution vs No-Resolution)
- Blocking (session)
- Sample size (200 subjects, 10 sessions)

Design variants are appropriate:
- `diff_in_means`: Minimal model comparison
- `with_covariates`: Matches Appendix Tables 3-4

## C) RC Axes

**Assessment: PASS with minor notes**

Controls: The single-add controls (rank, sumpoints, age, gender, prior) match the paper's appendix robustness tables. The full set is the union of these.

Sample restrictions:
- `exclude_wrong_adjustments`: Matches Appendix Table 1. These are subjects who moved their beliefs in the opposite direction from the signal. Legitimate quality restriction.
- `exclude_wrong_and_zero_adjustments`: Matches Appendix Table 2. Adds subjects with zero adjustment despite non-zero Bayesian benchmark.
- `exclude_extreme_ranks`: Matches Appendix Table 5 (keep only rank 2 and 3). This is a sample restriction, not an estimand change, since the claim is about the general pattern.
- `trim_y_5_95`: Standard outlier trimming.

Session FE: Appropriate since randomization is within-session. This is a standard precision improvement.

**Note**: `rc/controls/sets/none` is effectively the same as the single-signal baselines (which already have 0 controls). For the DiD specifications, it means dropping signal and interaction term, which changes the specification structure. This should be interpreted as "no additional controls beyond the baseline specification's regressors."

## D) Controls Multiverse Policy

**Assessment: PASS**

Controls count range [0, 4] is correct. The baseline single-signal regressions have 0 controls; the DiD regressions have 2 (signal + interaction). Adding up to 2 additional pre-treatment covariates stays within the paper's revealed complexity.

No linked adjustment needed (simple OLS).

## E) Inference Plan

**Assessment: PASS**

Canonical: HC1 (robust) matches the paper's use of `, robust` in all Stata regressions.

Variants are well-chosen:
- Classical: Reference for comparison
- HC3: Appropriate given small cell sizes (N=50)
- Session clustering: Relevant since treatment was assigned at the session level (all subjects in a session get the same treatment)

**Note on session clustering**: This is an important variant. Treatment assignment is at the session level (sessions 1-5 are one treatment, sessions 6-10 are the other, or similar). With only 10 sessions (5 per treatment), clustered SEs may be quite different from robust SEs. This is a meaningful stress test.

## F) Budgets + Sampling

**Assessment: PASS**

Budget of 80 is sufficient. Full enumeration is feasible:
- 6 baselines
- ~12 RC variants per cell for the focal cell (NoRes-Bad)
- Similar variants for other cells
- Design variants

No sampling needed given the small combinatorial space.

## G) Diagnostics Plan

**Assessment: N/A**

No diagnostics plan is included. For this lab experiment, standard RCT diagnostics (balance test, attrition) are mentioned in the paper (ttest for balance on prior beliefs) but are not needed as formal diagnostic rows since the paper provides them.

---

## Changes Made

No changes were made to the surface. The surface is well-constructed and faithful to the paper's revealed specification space.

---

## What's Missing (minor, non-blocking)

1. The paper's Chow tests (comparing coefficients across treatments) could be included as a diagnostic but are not standard RC.
2. Table 3 (ordered logit on study/job performance) could be a separate baseline group but is correctly treated as secondary/exploration.

---

## Final Assessment

**APPROVED TO RUN**

The surface is conceptually coherent, faithful to the manuscript, and well-scoped. The 6-baseline structure correctly captures the paper's main table, and the RC axes cover the paper's own appendix robustness checks plus standard additions.
