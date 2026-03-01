# Specification Surface: 139262-V1

**Paper**: "Motivated Beliefs and Anticipation of Uncertainty Resolution" by Christoph Drobner
**Design**: Randomized experiment (laboratory, between-subject)
**Created**: 2026-02-24

---

## 1. Paper Summary

This is a laboratory experiment studying motivated reasoning in belief updating. 200 university students take an IQ test, are ranked in groups of 4, and report prior beliefs about their rank (as a probability distribution over ranks 1-4). They then receive a binary signal: "good news" (better than median) or "bad news" (worse than median). Subjects are randomly assigned to one of two treatments:

- **Resolution**: subjects will learn their true rank at the end
- **No-Resolution**: subjects will NOT learn their true rank

After receiving the signal, subjects report posterior beliefs. The paper computes Bayesian belief adjustments and compares subjects' actual belief adjustments to the Bayesian benchmark.

The main finding (Table 2): In the No-Resolution treatment, subjects update strongly in response to good news but barely react to bad news (the coefficient on `bayes_belief_adjustment` is ~0.67 for good news but only ~0.08 for bad news). In the Resolution treatment, updating is more symmetric (~0.53 for good and ~0.64 for bad). This asymmetry is interpreted as motivated reasoning: when subjects will not face reality (No-Resolution), they can afford to ignore bad news.

---

## 2. Baseline Group: G1

### Claim Object

- **Outcome concept**: Asymmetric belief updating (motivated reasoning)
- **Treatment concept**: Slope of belief_adjustment on bayes_belief_adjustment, measuring responsiveness to Bayesian benchmark
- **Estimand concept**: OLS coefficient, split by treatment (Resolution/No-Resolution) and signal type (good/bad)
- **Target population**: Laboratory subjects (200 university students)

### Baseline Specifications (Table 2)

The paper presents 6 regressions in Table 2, organized as a 3x2 grid:
- Columns: Good news only | Bad news only | DiD (pooled with interaction)
- Rows: No-Resolution | Resolution

Each regression is: `belief_adjustment ~ bayes_belief_adjustment [+ signal + signal_bayesbeliefadj]`

The focal specification is **Column 2 (No-Resolution, Bad news)** where the coefficient is near zero and insignificant, demonstrating underreaction to bad news. Column 3 (DiD) captures the same asymmetry via the interaction term.

All 6 columns are emitted as baseline rows since the paper treats them as a unified main result.

### Why a single baseline group?

The paper presents one unified claim about asymmetric updating across treatments and signal types. The 6 regressions are different perspectives on the same claim object, not separate claims. Treatment (Resolution vs No-Resolution) is the randomized variable, and signal type is a within-subject experimental feature.

---

## 3. Core Universe

### Design variants

- **diff_in_means**: Simple difference-in-means test comparing belief adjustments across signal types within each treatment. Applied to the 6 cell structure.
- **with_covariates**: Add pre-treatment covariates (as in Appendix Tables 3-4) for precision.

### RC axes

**Controls (single-add)**:
- `rank`: Actual rank in the group (Appendix Table 3)
- `sumpoints`: IQ test score (Appendix Table 4)
- `age`, `gender`, `prior`: Demographics and prior beliefs
- `sets/full`: All available controls together
- `sets/none`: Bivariate (already the baseline for single-signal regressions)

**Sample restrictions** (from Online Appendix):
- `exclude_wrong_adjustments`: Drop subjects with belief adjustments in the wrong direction (Appendix Table 1)
- `exclude_wrong_and_zero_adjustments`: Drop wrong + zero adjustments (Appendix Table 2)
- `exclude_extreme_ranks`: Drop rank 1 and rank 4 subjects, keeping only middle ranks (Appendix Table 5)
- `trim_y_5_95`: Trim extreme belief adjustments

**Fixed effects**:
- `session`: Add session fixed effects (since randomization is within-session)

### Specification count

For each of the 6 baseline cells:
- 1 baseline
- ~12 RC variants (5 single-add controls + full controls + 3 sample restrictions + 1 trim + 1 session FE + 1 none/bivariate)

Plus 2 design variants applied to 6 cells = 12.

Total planned: ~84 specifications. Well within budget.

---

## 4. Inference Plan

**Canonical**: HC1 (robust) standard errors, matching the paper's use of `, robust` throughout.

**Variants**:
- Classical (homoskedastic) SEs
- HC3 (small-sample leverage correction, relevant with N=50 per cell)
- Cluster at session level (randomization was at session level for treatment assignment)

---

## 5. Constraints

- Controls count: 0-4 (baseline uses 0 for single-signal, 2 for DiD)
- No linkage constraints (simple OLS, no bundled estimator)
- All controls are pre-treatment (rank, sumpoints, prior, age, gender), so adding them preserves the experimental estimand
- Session FE is pre-treatment (randomization within sessions)

---

## 6. What is excluded and why

- **Table 3 (ordered logit on study/job performance)**: This is a separate outcome concept (ex-post rationalization) and uses ordered logit, not the core belief-updating claim. Could be an exploration group but is not the headline result.
- **Chow tests**: These are hypothesis tests comparing coefficients across treatments, not separate estimates. They are diagnostic/inferential objects.
- **Exploration of heterogeneity**: The paper does not report heterogeneity analyses beyond the treatment x signal structure.
