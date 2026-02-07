# Verification Report: 184041-V1

## Paper: The Common-Probability Auction Puzzle
**Authors**: M. Kathleen Ngangoue and Andrew Schotter  
**Journal**: AER: Insights (2023)  
**Verified**: 2026-02-03  
**Verifier**: verification_agent

---

## Baseline Groups

### G1: CV vs CP Auction Overbidding
- **Claim**: Subjects bid significantly higher in Common Value (CV) auctions than in economically equivalent Common Probability (CP) auctions, as measured by bid factor deviations from theoretical benchmarks.
- **Expected sign**: Positive (CV coefficient > 0)
- **Baseline spec_ids**: `baseline` (3 rows, one per outcome: BF, BEBF, NEBF)
- **Treatment variable**: CV (=1 for Common Value, =0 for Common Probability)
- **Outcome variables**: BF (Bid Factor), BEBF (Break-Even Bid Factor), NEBF (Nash Equilibrium Bid Factor)
- **Baseline sample**: Experiment 1, reduced sample (exp==1 & rrange==1 & domnBid<=8), N=5817
- **Baseline coefficient**: ~15.1 (all three outcomes nearly identical)
- **Baseline p-value**: ~1.2e-10

**Notes**: The three baseline outcomes are all bid deviations from different theoretical benchmarks (naive bid, break-even bid, Nash equilibrium bid). They produce nearly identical results because the benchmarks differ by a constant across treatment conditions. The paper treats all three interchangeably.

---

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 133 |
| Baseline | 3 |
| Core (non-baseline) | 106 |
| Non-core | 24 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 39 | Control progressions, leave-one-out, single covariates, full controls, plus 3 baselines and 3 outcome duplicates |
| core_sample | 42 | Early/late rounds, trimming, winsorizing, lottery subsets, winners only, cross-experiment (Exp2-4), pooled, drop lottery rounds |
| core_funcform | 9 | Log, IHS, standardized transformations; raw bid and log bid outcomes |
| core_inference | 8 | HC1 robust SE, clustered by subject, clustered by session |
| core_method | 11 | Quantile regressions at 10th, 25th, 50th, 75th, 90th percentiles |
| noncore_heterogeneity | 19 | Subgroup splits by gender, age, cognitive ability, risk preference, ambiguity aversion; interaction terms |
| noncore_alt_outcome | 1 | Payoff outcome (different concept from overbidding) |
| noncore_placebo | 4 | Balance tests on male, age, correctAns, signal |

---

## Top 5 Most Suspicious Rows

1. **robust/outcome/BF, robust/outcome/BEBF, robust/outcome/NEBF** (rows 61-63): These are exact duplicates of the three baselines. They have identical coefficients, p-values, and sample descriptions. They appear to be redundantly entered under the "outcome" robustness branch despite being identical to baselines. Not invalid, but inflates the spec count.

2. **robust/het/interact_age** (row 101): The CV coefficient here is 27.32 with p=0.097, not significant at conventional levels. However, this is the main effect of CV evaluated at age=0, which is not meaningful. The interaction specification reports the main CV coefficient (at the zero-point of age) rather than an average treatment effect. The coefficient is misleading as a test of the main claim.

3. **robust/outcome/payoff** (row 60): The coefficient is -4.29 (negative), which is the only specification with a genuinely negative and significant coefficient on CV. However, this is expected: higher bids in CV auctions reduce payoffs. This is not a test of the same claim (overbidding) but rather a consequence of it. Correctly classified as noncore_alt_outcome.

4. **robust/cluster/subject_NEBF and robust/cluster/subject_BF** (rows 67, 69): These are identical to the baselines since the baseline already clusters by subject. They redundantly confirm the same result and inflate the spec count.

5. **robust/sample/exclude_last_round** (row 39): The coefficient and standard error are exactly identical to the NEBF baseline (coef=15.099, se=2.345). This suggests the "last round" was not in the baseline sample to begin with, so excluding it has no effect. The row is valid but uninformative.

---

## Assessment of Specification Search Quality

The specification search is well-structured and comprehensive. Key observations:

1. **Strengths**: The search covers a wide range of robustness dimensions (controls, samples, functional forms, inference, methods, heterogeneity, placebos) in a systematic way. The use of three parallel outcome measures across many robustness checks is thorough.

2. **Redundancy**: The search produces many near-duplicate specifications because it separately runs BF, BEBF, and NEBF variants for many checks. Since these three outcomes produce nearly identical results (differing by <0.1 in coefficient), this triples the effective spec count without adding information. The "outcome" branch (rows 61-63) also duplicates baselines exactly.

3. **Heterogeneity classification**: The 19 heterogeneity specifications are correctly separated from the main robustness checks. They test whether the effect varies by subgroup, not whether the main effect exists.

4. **Cross-experiment replications**: The Exp2-4 and pooled analyses (13 specs) are valuable as independent replications. They are classified as core_sample because they test the same claim on different experimental populations, though one could argue they are closer to independent replications than robustness checks of a single baseline.

## Recommendations

1. **Reduce redundancy**: Consider selecting one primary outcome (e.g., NEBF) and reporting BF/BEBF only for the baseline, rather than tripling all robustness checks across all three outcomes.

2. **Separate cross-experiment from within-experiment**: The cross-experiment replications (Exp2-4, pooled) could be given their own baseline group since they have different sample definitions and parameters than Experiment 1.

3. **Flag exact duplicates**: The outcome branch rows (robust/outcome/BF, BEBF, NEBF) and the subject clustering rows are exact duplicates of baselines and should be flagged or removed.
