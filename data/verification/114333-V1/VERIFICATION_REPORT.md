# Verification Report: 114333-V1

## Paper: Team versus Individual Play in Finitely Repeated Prisoner Dilemma Games

**Paper ID**: 114333-V1  
**Journal**: AER  
**Verification Date**: 2026-02-03  
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Team effect on cooperation rate
- **Claim**: Teams cooperate more (or differently) than individuals in finitely repeated Prisoner's Dilemma games, as measured by the cooperation rate.
- **Baseline spec_id**: `baseline`
- **Outcome**: `cooperate` (binary: 1=cooperate, 0=defect)
- **Treatment**: `is_team` (binary: 1=team, 0=individual)
- **Baseline coefficient**: 0.0146, p=0.736 (not significant)
- **Expected sign**: Unknown (paper documents both directions depending on game phase)
- **Baseline FE/controls**: Block FE + Round FE, clustered at player level
- **N**: 9,127

---

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 84 |
| Baselines | 1 |
| Core tests (incl. baseline) | 40 |
| Non-core tests | 44 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 6 | Control set variations (no controls, lag controls, etc.) |
| core_fe | 4 | Fixed effects structure variations (none, Block only, Round only, both) |
| core_sample | 21 | Sample restrictions (session drops, block/round subsets, trimming) |
| core_inference | 4 | Clustering variations (robust, session, player, block) |
| core_method | 2 | Estimation method (Logit, Probit) |
| core_funcform | 3 | Functional form (quadratic round, log period, linear trend) |
| noncore_heterogeneity | 33 | Heterogeneity: single-block, single-round, interaction, and history-conditional subsamples |
| noncore_alt_outcome | 9 | Alternative outcomes (mutual_coop, mutual_defect, defect, got_suckered, temptation, payoff, coop_end_round) |
| noncore_placebo | 2 | Placebo tests (round 1 only, randomized fake treatment) |

---

## Classification Rationale

### Core tests (40 specs)
These specifications all share the same outcome (`cooperate`) and treatment (`is_team`) as the baseline, and vary along dimensions that preserve the estimand (the average effect of team play on cooperation):
- **Control variations** (6): Progressive addition of controls (none, round, block FE, round FE, both FE, lags)
- **Fixed effects** (4): Variations on which FE to include
- **Sample restrictions** (21): Leave-one-session-out (10 sessions), broad subsample splits (early/late blocks, early/late rounds, middle blocks/rounds, first/last 3 blocks), and trimming (exclude first/last round, exclude first super-game)
- **Inference** (4): Different clustering levels
- **Method** (2): Logit and Probit as alternatives to the linear probability model
- **Functional form** (3): Alternative parameterizations of round effects

### Non-core tests (44 specs)

#### Heterogeneity (33 specs)
The largest non-core category. These include:
- **Single-block subsamples** (10): `robust/het/block_1` through `robust/het/block_10` -- each restricts to a single super-game. These are heterogeneity analyses (how does the team effect evolve over super-games?), not tests of the average effect.
- **Single-round subsamples** (10): `robust/het/round_1` through `robust/het/round_10` -- each restricts to a single round within super-games. Again, heterogeneity analysis.
- **Interaction models** (5): Interactions with early/late blocks, late rounds, final round, round, and block. These estimate heterogeneous effects, not the average effect.
- **History-conditional subsamples** (4): After mutual cooperation, after getting suckered, after opponent cooperated, after opponent defected. These condition on lagged outcomes, which changes the estimand to a conditional effect.
- **Block-half subsamples** (2): `robust/het/early_blocks_only` and `robust/het/late_blocks_only` are duplicates of `robust/sample/early_blocks` and `robust/sample/late_blocks` but tagged under heterogeneity.
- **Single super-game** (1): `robust/sample/first_supergame_only` restricts to block 1 only.
- **Final round only** (1): `robust/sample/only_final_round` restricts to the last round only.

**Note on boundary cases**: Some sample splits (early_blocks, late_blocks, early_rounds, late_rounds) appear under both `robust/sample/` and `robust/het/`. I classified the broader splits (first/last half) under `robust/sample/` as core sample restrictions (they still test the average effect on a broad subsample), but classified the narrower single-block/single-round splits and the duplicated het versions as non-core heterogeneity. The early/late block splits under `robust/sample/` are classified as core because they represent meaningful pre-registered-style sample restrictions (first half vs. second half of the experiment), while the individual block/round splits are pure heterogeneity.

#### Alternative outcomes (9 specs)
- **mutual_coop, mutual_defect, defect, got_suckered, temptation, outcome/payoff** (6): Each uses a different dependent variable. The baseline claim is specifically about the cooperation rate, so these test different hypotheses.
- **coop_end_round** (3): The defection timing analysis uses a different dataset ("Defection patterns data") and measures when cooperation ends within a super-game, which is a fundamentally different outcome concept.

#### Placebo (2 specs)
- **round1_only**: Tests cooperation in round 1 only (before strategic interaction matters), serving as a placebo.
- **randomized_treatment**: Uses a fake (randomly permuted) team indicator. Classic placebo test.

---

## Top 5 Most Suspicious Rows

1. **`robust/het/early_blocks_only`** (coef=−0.0012, p=0.98) and **`robust/het/late_blocks_only`** (coef=0.0376, p=0.49): These appear to be exact duplicates of `robust/sample/early_blocks` and `robust/sample/late_blocks` respectively (same coefficients, same p-values, same sample sizes). The spec search generated the same regression twice under different tree paths.

2. **`robust/outcome/defect`** (coef=−0.0146, p=0.74): This is mechanically the negative of `cooperate` (defect = 1 − cooperate), so its coefficient is exactly −1 times the baseline coefficient. It provides no independent information.

3. **`robust/placebo/round1_only`** (coef=0.081, p=0.29): Tagged as a placebo test but could also be interpreted as a heterogeneity check (round 1 subsample). The placebo interpretation (initial cooperation before strategy kicks in) is reasonable, but the `robust/het/round_1` spec is numerically identical, creating redundancy.

4. **`robust/control/full_fe`** (coef=0.0146, p=0.74): This spec has Block FE and Round FE -- exactly the same as the baseline. The coefficient is identical, suggesting it is a mechanical duplicate of the baseline spec.

5. **`robust/het/by_early_late_blocks`** (coef=0.0377, p=1.00): This interaction model has a p-value of exactly 1.0 for the main effect, which is unusual. The p=1.0 likely reflects the coefficient on `is_team` when the interaction term absorbs all the variation, but it warrants inspection.

---

## Recommendations

1. **Remove duplicate specifications**: `robust/het/early_blocks_only` / `robust/het/late_blocks_only` duplicate `robust/sample/early_blocks` / `robust/sample/late_blocks`. `robust/outcome/defect` is mechanically identical to `cooperate` (negated). `robust/control/full_fe` appears identical to `baseline`. `robust/het/round_1` duplicates `robust/placebo/round1_only`. These duplicates inflate the spec count without adding information.

2. **Clarify the baseline claim direction**: The paper documents that teams cooperate MORE in later super-games but LESS in the first super-game and in late rounds. The spec search should explicitly state that the baseline claim is about the average (unconditional) effect and note that the sign is ambiguous.

3. **Consider wild cluster bootstrap**: With only 10 sessions (5 per treatment condition), cluster-robust SEs at the session level may be unreliable. A wild cluster bootstrap would provide more reliable inference for the core specifications.

4. **Separate defection timing into its own baseline group**: If the paper treats defection timing as a separate main result (which the SPECIFICATION_SEARCH.md suggests), it should have its own baseline group rather than being classified purely as an alternative outcome. Currently there is no baseline spec for the defection timing claim.

5. **Reduce heterogeneity specs**: The 20 single-block and single-round subsample regressions are highly correlated and provide limited additional information. A more parsimonious approach would use interaction models only.
