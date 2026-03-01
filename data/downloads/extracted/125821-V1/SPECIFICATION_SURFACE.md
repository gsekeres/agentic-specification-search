# Specification Surface: 125821-V1

## Paper: School Spending and Student Outcomes: Evidence from Revenue Limit Elections in Wisconsin (Baron, AEJ: Economic Policy 2022)

## Baseline Groups

### G1: Effect of operational referendum passage on student outcomes (dynamic RD)

- **Outcome**: Student academic outcomes (primarily math proficiency rates grade 10, also dropout rate, WKCE scale scores, postsecondary enrollment)
- **Treatment**: Passage of operational revenue limit referendum (school spending election passing the 50% threshold)
- **Estimand**: ITT effect of referendum passage on student outcomes, averaged over 10 post-election years. This is a dynamic RD local effect at the 50% vote share cutoff, using the Cellini et al. (2010, QJE) one-step estimator.
- **Population**: Wisconsin public school districts holding revenue limit referenda, 1996-97 through 2014-15
- **Design**: Dynamic panel regression discontinuity following Cellini et al. (2010)

### Baseline specification

The paper employs two complementary RD approaches:

1. **One-step dynamic RD estimator** (Cellini et al. 2010, Tables 4-5, 7-8): This is the primary specification. It uses `areg outcome op_win_prev1-10 bond_win_prev1-10 + polynomial_controls_in_vote_share + year_dummies, absorb(district_code) cluster(district_code)`. The focal parameter is the 10-year post-election average: `lincom .10*(op_win_prev1 + ... + op_win_prev10)`.

2. **Cross-sectional RD** (Figures 3-4, Tables B2, C2-C4): Uses `rdrobust outcome perc` in a repeated cross-section framework (election-level panel, one year at a time). This provides visual evidence and alternative point estimates.

The headline claim is Table 5 Panel A, which shows the 10-year average effect of operational referendum passage on student outcomes using the cubic specification.

### Polynomial order variation

The paper explicitly varies the polynomial order of the vote share control function:
- **Linear** (Table 5 Panel C)
- **Quadratic** (Table 5 Panel B)
- **Cubic** (Table 5 Panel A -- baseline)

This is a revealed search dimension and the primary "controls" variation axis.

### Multiple outcome baselines

The paper presents results for multiple outcomes as headline claims:
- **advprof_math10**: % Advanced or Proficient on grade 10 math test
- **dropout_rate**: high school dropout rate
- **wkce_math10**: WKCE math scale score grade 10
- **log_instate_enr**: log in-state postsecondary enrollment
- **rev_lim_mem**: revenue limits per member (first-stage spending effect, Table 4)
- **tot_exp_mem**: total expenditures per member (Table 4)

These are all included as additional baseline rows within the same G1 group since they share the same treatment concept and identification strategy.

### Weights

The paper uses analytic weights:
- Test score outcomes: `[aw=num_takers_math10]`
- Dropout rate: `[aw=student_count]`
- Enrollment and spending: unweighted

Weight variation is a revealed axis.

## Core Universe

### Design axes (polynomial order)
- **Linear**: vote share polynomial of order 1
- **Quadratic**: vote share polynomial of order 2
- **Cubic**: vote share polynomial of order 3 (baseline)

These correspond directly to the paper's Table 5 Panels A-C and are the primary within-design variation for this estimator.

### Sample axes
- **Tried both**: restrict to districts that proposed both operational and bond referenda (Table 7)
- **Passed both**: restrict to districts that passed both types (Table 7)
- **Cross-section post**: rdrobust cross-sectional RD on post-election years (dyear > 0)
- **Cross-section pre placebo**: rdrobust RD on pre-election periods (dyear == -2) as falsification
- **Trim outcome 1/99 and 5/95**: winsorize outcome at percentile bounds

### Functional form axes
- **Log outcome**: log transformation where applicable (log_instate_enr is already logged)

### Weights axes
- **Unweighted**: drop analytic weights
- **Weighted by num_takers**: use aw=num_takers (baseline for test scores)

### Focal parameter variation
- **Five-year average**: lincom .20*(op_win_prev1 + ... + op_win_prev5)
- **Ten-year average**: lincom .10*(op_win_prev1 + ... + op_win_prev10) (baseline)

### Joint / cross-sectional RD
- **rdrobust post-election**: cross-sectional RD using rdrobust on post-election observations
- **rdrobust pre-election placebo**: cross-sectional RD on pre-election observations (falsification)

## Inference Plan

- **Canonical**: Cluster SEs at district level (district_code), matching paper
- **Variant**: HC robust only (no clustering)

## Constraints

- **Control-count envelope**: not applicable in the traditional sense. The Cellini estimator has a fixed structure; the main variation is polynomial order (1, 2, or 3), not individual covariate inclusion/exclusion.
- **Linked adjustment**: Yes. Operational and bond referendum polynomial controls must vary together (both use the same order). The paper always specifies both op_percent_prev and bond_percent_prev at the same polynomial degree.
- **Analytic weights**: some outcomes use enrollment/test-taker weights; variation between weighted and unweighted is a revealed axis.
- **Focal parameter**: the scalar summary is a linear combination of 10 (or 5) lag coefficients, not a single regression coefficient.

## Budget

- Max core specs: 70
- Max control subset specs: 0 (controls are structured, not individually variable)
- Seed: 125821

## What is excluded and why

- **Bond referendum effects** (Table 7 Panel B): different treatment concept (capital spending vs operational spending). The paper treats these as separate effects. Including as explore/treatment variant.
- **Table 8 (interaction of operational and bond effects)**: tests whether operational and bond spending are complements/substitutes. This is a moderation analysis, not the main ITT claim.
- **Reading test scores** (Table B1): alternative subject within the same testing framework. Could be explore/outcome but kept out of core to focus on math (headline).
- **Teacher outcomes** (Tables B3-B4): different outcome concept (staffing ratios, salaries, turnover). These are mechanism analyses.
- **Table 6**: descriptive comparison of districts passing operational vs bond referenda (not a causal estimate).
- **External validity analyses** (using NCES data): different population scope.
- **Figures B13-B15** (event-study dynamic effects): these are the visual counterparts of the one-step estimates, not separate specifications.

## Diagnostics plan

- **McCrary density test**: at 50% vote share cutoff. DCdensity.ado is included in the replication package (Figure B1).
- **Covariate continuity**: Table 3 tests balance between winning and losing districts on pre-election observables and changes from t-2 to t-1.
