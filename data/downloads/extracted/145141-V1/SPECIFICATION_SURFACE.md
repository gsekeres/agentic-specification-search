# Specification Surface: 145141-V1

## Paper: Measuring the Welfare Effects of Shame and Pride (Butera, Metcalfe, Morrison, Taubinsky, 2022)

## Paper Summary

This paper studies the welfare effects of public recognition ("shame and pride") through two types of experiments:

1. **YMCA field experiment**: YMCA members are randomly assigned to have their gym attendance publicly posted (public recognition, PR) or not. The paper estimates (a) the causal effect of PR on attendance and (b) the willingness-to-pay (WTP) for PR as a function of hypothetical attendance levels (elicited via BDM mechanism).

2. **Charity real-effort experiments**: Online participants (Prolific, Berkeley, BU) complete a real-effort task across three within-subject conditions (anonymous, public recognition, financial incentives) in randomized order. The paper estimates (a) the effect of PR on points scored and (b) the WTP for PR as a function of hypothetical performance levels.

The key structural parameter is the curvature of the reputational return function R(a), measured by the ratio -R''/R'(a_bar), which determines whether reputation concerns create welfare gains or losses.

---

## Baseline Groups

### G1: Effect of public recognition on YMCA attendance (ITT)

- **Outcome**: attendance (gym visits during 30-day experiment period)
- **Treatment**: image (randomly assigned to public recognition)
- **Estimand**: ITT effect of PR assignment on attendance
- **Population**: YMCA members who completed the survey (coherent sample, excluding BDM arm)
- **Baseline spec**: Table 2, Column 2 (coherent sample): `reg attendance image past, r`
- **Design**: Individual-level randomized experiment

This is the paper's primary reduced-form result establishing that PR causally affects behavior. The paper reports three columns building up controls: (1) no controls, (2) past attendance, (3) past attendance + beliefs. Results are shown for both the "coherent" and "monotonic" subsamples.

### G2: Effect of public recognition on real-effort task performance

- **Outcome**: pts (points scored in real-effort task)
- **Treatment**: SR (public recognition round indicator, within-subject)
- **Estimand**: Within-subject effect of PR on charitable giving/effort
- **Population**: Online experiment participants across three samples (Prolific, Berkeley, BU) passing attention and consistency checks
- **Baseline spec**: Table 5, Columns 1-3: `reg pts SR ownpay o1 o2, cluster(id)` (one per sample)
- **Design**: Within-subject randomized experiment (3 rounds in random order)

This replicates the YMCA finding in a controlled lab setting. The within-subject design means each person provides three observations (one per round condition). The paper estimates separate regressions per sample and also tests for round-order effects.

### G3: WTP function for public recognition (shape of reputational returns)

- **Outcome**: wtp ($ willingness to pay for public recognition at each performance level)
- **Treatment**: visits (YMCA) or interval (charity experiments) -- hypothetical performance level
- **Estimand**: Shape of the function R(a) - R(a_bar): how reputational payoff varies with attendance/performance. Key parameter is curvature ratio -R''/R'(a_bar).
- **Population**: YMCA coherent sample and charity experiment samples
- **Baseline spec**: Table 3, Col 2 (YMCA Coh): `reg wtp visits visits2, cluster(id)`; Table 6, Col 2/4/6 (charity by sample): `reg wtp c.interval##c.interval, cluster(id)`
- **Design**: Within-subject stated preference via BDM mechanism

This is the paper's central contribution: measuring the curvature of reputational returns. Data is individual-by-interval panel (each person provides WTP at 10-18 hypothetical attendance/performance levels). The quadratic specification recovers the structural curvature parameter.

---

## Core Universe

### G1: YMCA attendance (ITT)

**Controls axes (3 specs)**:
- `rc/controls/sets/none`: No controls (Table 2 Col 1)
- `rc/controls/sets/past_only`: Control for past attendance only (baseline)
- `rc/controls/sets/past_and_beliefs`: Control for past attendance + beliefs (Table 2 Col 3)

**Sample axes (6 specs)**:
- `rc/sample/definition/monotonic_sample`: Stricter consistency criterion (monotonic WTP)
- `rc/sample/definition/robust_sample`: Full sample (all non-BDM, including incoherent)
- `rc/sample/outliers/trim_y_1_99`: Trim attendance at 1st/99th percentile
- `rc/sample/outliers/trim_y_5_95`: Trim attendance at 5th/95th percentile
- `rc/sample/outliers/winsorize_y_1_99`: Winsorize attendance at 1st/99th percentile

**Functional form axes (2 specs)**:
- `rc/form/outcome/log1p`: log(1 + attendance)
- `rc/form/outcome/asinh`: asinh(attendance)
- `rc/form/outcome/standardized`: Standardized attendance (z-score)

**Preprocessing axes (2 specs)**:
- `rc/preprocess/coding/attendance_topcode_22`: Top-code attendance at 22 visits
- `rc/preprocess/coding/attendance_topcode_15`: Top-code attendance at 15 visits

**Design estimator (1 spec)**:
- `design/randomized_experiment/estimator/diff_in_means`: Raw difference in means (no controls)

**Rationale**: The control pool is very small (only past attendance and beliefs are available pre-treatment covariates). This is an RCT, so controls serve for precision, not identification. The main specification variation comes from sample definition (three nested coherence criteria are revealed by the paper) and outcome transformations.

### G2: Real-effort task performance

**Controls axes (2 specs)**:
- `rc/controls/sets/ownpay_only`: Include only ownpay (financial incentives indicator), drop order dummies
- `rc/controls/sets/ownpay_and_order`: Include ownpay + order dummies (baseline)

**Sample axes (8 specs)**:
- `rc/sample/definition/first_round_only`: Restrict to first-round observations only (Table A10)
- `rc/sample/definition/no_attention_check_filter`: Include attention-check failures
- `rc/sample/definition/strict_consistency`: Use strict consistency criterion (consistent==1 vs consistent_b==1)
- `rc/sample/definition/approx_monotonic`: Use approximate monotonicity criterion
- `rc/sample/outliers/drop_high_pts_3000`: Drop observations with pts > 3000 (already applied in baseline)
- `rc/sample/outliers/trim_y_1_99`: Trim points at 1st/99th percentile
- `rc/sample/outliers/trim_y_5_95`: Trim points at 5th/95th percentile
- `rc/sample/pooled/all_three_samples`: Pool Prolific + Berkeley + BU with sample FE

**Functional form axes (2 specs)**:
- `rc/form/outcome/log1p_pts`: log(1 + pts)
- `rc/form/outcome/standardized_pts`: Standardize points within sample

**Other axes (2 specs)**:
- `rc/preprocess/coding/pts_in_hundreds`: Rescale points to hundreds (matches Table 6 coding)
- `rc/fe/add/individual_fe`: Add individual fixed effects (within-person variation only)

**Design estimator (1 spec)**:
- `design/randomized_experiment/estimator/diff_in_means`: Mean SR pts - mean anonymous pts (no controls)

**Rationale**: The within-subject design provides substantial precision from individual fixed effects. The paper itself varies across samples (Prolific, Berkeley, BU) as separate regressions, which we treat as separate baseline specs. Round-order controls (o1, o2) are included as the paper tests for order effects. The main revealed variation is across samples and consistency restrictions.

### G3: WTP function (reputational returns shape)

**Functional form axes (6 specs)**:
- `rc/form/estimator/ols_linear`: Linear OLS (Table 3 Col 1 / Table 6 Col 1,3,5)
- `rc/form/estimator/tobit_quadratic`: Tobit with quadratic (Table 3 Col 4)
- `rc/form/estimator/tobit_linear`: Tobit linear (Table 3 Col 3)
- `rc/form/outcome/ln_visits_quadratic`: Quadratic in ln(1+visits) instead of visits (YMCA)
- `rc/form/outcome/interval_idx_quadratic`: Use interval index number instead of midpoint (Table A4)
- `rc/form/outcome/interval_idx_linear`: Linear in interval index (Table A4)

**Sample axes -- YMCA (6 specs)**:
- `rc/sample/definition/monotonic_sample_ymca`: Monotonic subsample (Table A8)
- `rc/sample/definition/close_to_beliefs_4_ymca`: Restrict to intervals within 4 of beliefs (Table 4)
- `rc/sample/definition/close_to_beliefs_exact_ymca`: Restrict to exact belief interval (Table 4)
- `rc/sample/definition/close_to_past_4_ymca`: Restrict to intervals within 4 of past attendance (Table A2)
- `rc/sample/definition/excl_top_interval_ymca`: Exclude top attendance interval (Table A3)
- `rc/sample/definition/excl_top_two_intervals_ymca`: Exclude top two intervals (Table A3)

**Sample axes -- Charity (4 specs)**:
- `rc/sample/definition/include_top_interval_charity`: Include the top points interval (>= 1700) with averaged midpoint (Table A12)
- `rc/sample/definition/close_to_score_charity`: Restrict to intervals close to realized score (Table A11)
- `rc/sample/definition/no_consistency_filter_charity`: Drop consistency filter
- `rc/sample/pooled/all_charity_samples`: Pool all three charity samples

**Outlier axes (2 specs)**:
- `rc/sample/outliers/trim_wtp_1_99`: Trim WTP at 1st/99th percentile
- `rc/sample/outliers/trim_wtp_5_95`: Trim WTP at 5th/95th percentile

**Rationale**: This is the paper's most specification-rich claim. The paper itself systematically varies: (1) OLS vs Tobit, (2) linear vs quadratic, (3) sample restrictions (coherent vs monotonic, proximity to beliefs/past, excluding top intervals), (4) alternative x-axis coding (visits vs interval index vs ln visits). The WTP data is individual-by-interval panel. Tobit is used because WTP is bounded by slider limits. The quadratic functional form is essential because the curvature ratio is the key parameter.

---

## Inference Plan

### G1 (YMCA attendance ITT)
- **Canonical**: HC1 robust SEs (Stata `robust`)
- **Variant**: HC3 small-sample robust SEs

### G2 (Real-effort task performance)
- **Canonical**: Cluster SEs at individual level (Stata `cluster(id)`) -- required because within-subject design produces 3 observations per person
- **Variant**: HC1 robust SEs (ignoring within-person correlation)

### G3 (WTP function)
- **Canonical**: Cluster SEs at individual level (Stata `cluster(id)`) -- required because individual-by-interval panel produces 10-18 observations per person
- **Variant**: HC1 robust SEs (ignoring within-person correlation)

---

## Constraints

### G1
- Control-count envelope: [0, 2]. Only two pre-treatment covariates available (past attendance, beliefs about attendance with PR).
- No linkage constraints (single-equation OLS).
- All specifications maintain the same treatment concept (image indicator from random assignment).

### G2
- Control-count envelope: [1, 3]. ownpay is always included as the second treatment arm indicator. o1 and o2 are round-order dummies.
- No linkage constraints.
- Treatment is always the SR round indicator. Financial incentives indicator (ownpay) is always included as it is the other experimental condition.

### G3
- Control-count envelope: [0, 1]. The quadratic term (visits2 or c.interval#c.interval) is the only "control" -- but conceptually it is part of the functional form, not a confounder.
- Functional-form policy: The paper treats both linear and quadratic as valid; the quadratic is needed to recover the curvature parameter. Tobit vs OLS is an explicit estimator choice revealed by the paper.
- No linkage constraints.

---

## Budget

| Group | Max core specs | Max control subset | Total planned (approx) |
|-------|---------------|-------------------|----------------------|
| G1    | 50            | 0                 | ~20                  |
| G2    | 80            | 0                 | ~50                  |
| G3    | 80            | 0                 | ~65                  |

Full enumeration is feasible for all groups. No random subset sampling needed.
- **Seed**: 145141 (if any stochastic steps arise)

---

## What is excluded and why

### Structural estimation (Tables 7, 9)
The welfare estimates (Tables 7, 9) involve structural model parameters (gamma1, gamma2, rho, c) recovered via bootstrapped nonlinear combinations of reduced-form coefficients. These are not standard regression specifications -- they are post-processing of the reduced-form estimates. They would belong in `post/*` or `explore/*`, not the core surface.

### Model selection (Tables A18-A20)
Bayesian model selection (BIC-based polynomial selection) is a diagnostic/model-comparison exercise, not an estimate of the baseline estimand.

### Individual differences analysis (Tables A21-A22)
These examine heterogeneity in structural parameters by demographic characteristics. They change the target population (subgroups) and thus belong in `explore/*`.

### Heterogeneity tables (Tables A1, A6)
Tables examining heterogeneity along past attendance or WTP motivation (interaction terms with visits). These are exploration of treatment effect heterogeneity, not the average effect, and belong in `explore/*`.

### Group size interactions (Table 5 Col 4, Table A15)
Prolific group-size interactions test whether the effect of PR varies by group size. This is heterogeneity exploration.

### BDM arm subjects
The BDM arm (treatment==2) is excluded from reduced-form analysis because these subjects endogenously chose whether to receive PR. Including them would change the estimand.

### Demand curves and CDF plots (Figures 7-8, A1, A8)
These are descriptive visualizations, not regression specifications.

### QM221 sample (BU supplementary)
The BU QM221 sample (Table A16) is treated as a supplementary replication sample. It could be added as an additional baseline spec for G2/G3 if desired, but the paper does not present it as a main result.
