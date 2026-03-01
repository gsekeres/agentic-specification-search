# Specification Surface: 184041-V1

**Paper**: "The Common-Probability Auction Puzzle" (Ngangoue and Schotter)

**Design**: Randomized experiment (lab experiment with random assignment to CV vs CP auction/lottery sessions)

---

## Overview

This paper studies bidding behavior in common-value (CV) vs common-probability (CP) auctions through a series of lab experiments at NYU. The main finding is a "common-probability auction puzzle": subjects bid differently in CV and CP auctions despite the auctions being strategically equivalent. The paper uses four experiments to isolate the source of this puzzle.

The paper's analysis is primarily descriptive/comparative, using simple regressions of bid factors on a CV treatment indicator with clustering at the subject level. The analysis also includes structural median regressions decomposing bids into responses to different uncertainty components (decision weight analysis).

There are three baseline groups:
1. **G1**: CV vs CP bid factor comparisons in auction settings (Experiment I)
2. **G2**: CV vs CP price factor comparisons in individual pricing settings (Experiment II)
3. **G3**: Decision weight analysis (structural decomposition of bids, Tables A4-A6)

---

## Baseline Groups

### G1: Bid Factors in Auctions (Experiment I)

**Claim object**: Subjects bid significantly differently in CV vs CP auctions. The bid factor (bid minus naive/break-even/Nash equilibrium benchmark) is significantly higher in CV than CP, indicating overbidding in CV auctions relative to theory.

**Baseline specs** (Table A1, Experiment I section):
- `BF ~ CV`, clustered at subject level, reduced sample (rrange==1 & domnBid<=8)
- Same for BEBF (break-even bid factor) and NEBF (Nash equilibrium bid factor)
- Both OLS and median regression (qreg2)

**Key features**:
- No control variables. The regressions are simple treatment-control comparisons.
- Sample restriction: "reduced sample" excludes subjects who made more than 10% dominated bids and restricts to the "relevant range" of signals.
- Three outcome measures (BF, BEBF, NEBF) represent the same bid relative to three different normative benchmarks.

**Revealed robustness**: Table A2 restricts to winning bids only (dWin==1). Tables A1 rows show results across Experiments I, III, IIIb, IV.

### G2: Price Factors in Individual Decision Making (Experiment II)

**Claim object**: The CV-CP difference persists even in individual decision making (no strategic interaction), ruling out strategic uncertainty as the sole explanation. Subjects price CV lotteries differently from CP lotteries.

**Baseline specs** (Table A3):
- `BF ~ CV`, stage==22 (compound lottery with signal), clustered at subject level
- Both OLS and median regression

**Key features**:
- No control variables. Simple mean/median comparison.
- Three stages in Experiment II: stage 22 (compound with signal), stage 21 (compound without signal), stage 1 (reduced lottery). The baseline is stage 22 which most closely parallels the auction setting.
- No reduced sample exclusion in Experiment II (all 104 subjects).

**Revealed robustness**: Table A3 shows results for all three stages. Table A5 shows structural decision weight regressions for Exp II.

### G3: Decision Weights (Structural Decomposition)

**Claim object**: Subjects weight the signal component and the fixed component of lottery value differently in CV vs CP. In CV, subjects overweight the fixed component (probability) relative to the signal (value). The marginal rate of substitution (alpha/beta) differs from 1 and differs across CV and CP.

**Baseline specs** (Tables A4-A6):
- Table A4 (Exp I): `lnbid ~ lnfix + lnsignal` separately for CV and CP, and pooled with CV interactions
- Table A5 (Exp II): Same structural form for pricing
- Table A6 (Exp II): `bid ~ expV + nmexpV + sigstage` decomposition with structural parameters

**Key features**:
- These are structural regressions with a specific functional form (log-log for Tables A4-A5, linear decomposition for A6).
- Median regression (qreg2) is the primary estimator for Tables A4-A5.
- The key tests are coefficient equality tests (lnfix = lnsignal) and interaction significance (CVlnfix, CVlnsignal).

---

## Design and Identification

**Randomization**: Between-subject random assignment to CV or CP sessions. Within each session, subjects play multiple rounds with varying lottery parameters and signals. Random assignment occurs at the session level (subjects within a session all face the same auction type).

**Sample selection**: The "reduced sample" for Experiment I excludes subjects with excessive dominated bids (>10% of rounds). This is a pre-registered-style quality filter. Experiments III-IV use a similar but separately defined `goodsample` indicator.

**Clustering**: Standard errors clustered at the subject level throughout, since each subject provides multiple round-level observations.

---

## Core Universe (what will be run)

### Design variants
- **Difference-in-means**: Simple mean comparison of BF across CV and CP (equivalent to baseline OLS with no controls)

### RC axes

**Sample**:
- Full sample (no exclusion of dominated bidders) vs reduced sample
- Winners only (dWin==1) for auction experiments (Table A2)
- CV-only and CP-only intercept regressions (test whether mean BF differs from 0 within each treatment)
- Stage variants for Experiment II (stage 22 vs 21 vs 1)

**Functional form / Estimator**:
- OLS vs median regression (qreg2) -- both are used throughout the paper
- Outcome measure variants: BF vs BEBF vs NEBF (three normative benchmarks)
- Structural decision weight decomposition (log-log form)
- Exp II structural decomposition (expV, nmexpV form from Table A6)

**Cross-experiment**:
- Same specification run across Experiments I, III, IIIb, IV (as in Table A1)

---

## What is Excluded (and Why)

- **Figures 2-9**: Kernel density plots and CDFs showing distributional differences between CV and CP. These are descriptive visualizations, not regression estimates.
- **Probit models (adverse selection analysis)**: `probit Hsig dWin#CV CV` in tables.do analyzes adverse selection properties. This is a different estimand (probability of having highest signal conditional on winning) and belongs in `explore/*`.
- **Table A7 (balance)**: Balance tests of demographics (male, age, correctAns, RP, CRP, AP) across CV/CP. These are diagnostics, not treatment effect estimates.
- **Experiments III/IIIb/IV bid factors**: These replicate the same CV-CP comparison in different experimental contexts (estimating uncertainty, strategic uncertainty). While Table A1 shows these, they represent different populations/experimental contexts. The primary claim is from Experiment I; others are supporting evidence. Cross-experiment comparisons could be run as `explore/*` variants.
- **Within-treatment regressions** (09_qreg.do): Structural models within CV or CP sessions that add lagged outcomes, experience variables, risk preferences. These are mechanism exploration, not the main treatment effect.

---

## Inference Plan

**Canonical**: Cluster-robust SEs at the subject level (`subject`), matching the paper's approach. Each subject provides multiple round-level observations, so clustering accounts for within-subject correlation.

**Variants**:
- HC1 robust SEs (no clustering) -- treating rounds as independent
- Cluster at session level (`numSession`) -- reflects the level of treatment assignment

---

## Budgets and Sampling

| Baseline Group | Max Core Specs | Max Control Subset | Sampling |
|---|---|---|---|
| G1 (Bid Factors) | 50 | 0 | Full enumeration |
| G2 (Price Factors) | 30 | 0 | Full enumeration |
| G3 (Decision Weights) | 30 | 0 | Full enumeration |

Full enumeration is trivially feasible because:
- There are no optional control variables to vary combinatorially
- Variation comes from a small set of outcome measures, estimator choices, and sample restrictions
- The paper's analysis is compact (all tables use simple regressions with the same basic structure)

---

## Diagnostics Plan

- **Balance tests** (Table A7): Check that demographics (male, age, correctAns, risk/ambiguity preferences) are balanced across CV and CP sessions within each experiment.
- **Kolmogorov-Smirnov test**: Distribution of signals across treatments (footnote 11 in the paper).

---

## Key Linkage Constraints

1. **No controls to link**: The main regressions contain no covariates beyond the treatment indicator. This eliminates the most common source of specification search (control set variation).
2. **Structural form is fixed for decision weights**: In G3, the log-log functional form (`lnbid ~ lnfix + lnsignal`) is structural. The interaction terms (CVlnfix, CVlnsignal) must move together.
3. **Sample exclusion rule is pre-specified**: The reduced sample criterion (domnBid<=8, i.e., no more than 10% dominated bids) is a design choice, not a data-driven selection. The full-sample variant is the natural robustness check.
4. **Experiment-specific samples**: Each experiment has its own subject pool, and samples should not be mixed across experiments (except in the merged ExpAll dataset for cross-experiment comparisons in Table A1).

---

## Special Notes on This Paper

This is a lab experiment paper with a very different structure from typical field experiment or observational studies:
- The treatment is between-subjects at the session level but the unit of observation is the round (repeated measures within subject).
- The primary analysis uses both OLS (for means) and median regression (for medians), reflecting the lab tradition of reporting both.
- The "bid factor" outcomes are already transformations (bid minus benchmark), so the outcome variable choice (BF vs BEBF vs NEBF) is itself a specification choice.
- The paper has no controls in its main regressions, making the specification surface much narrower than typical applied micro papers. The main variation comes from outcome measures, estimator choice, and sample restrictions.
