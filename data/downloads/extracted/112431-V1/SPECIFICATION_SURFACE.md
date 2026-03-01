# Specification Surface: 112431-V1

## Paper: Electoral Accountability and Corruption (Ferraz & Finan, 2011)

## Baseline Groups

### G1: Effect of reelection incentives on corruption
- **Outcome**: pcorrupt (share of audited resources involving corruption)
- **Treatment**: first (first-term mayor indicator)
- **Estimand**: ATE of reelection eligibility on corruption
- **Population**: Brazilian municipalities audited by CGU lottery
- **Baseline spec**: Table 4 Col 6 (areg pcorrupt first + controls | uf, robust)
- **Baseline coefficient**: -0.0275 (SE=0.0113)

### Additional baselines (same claim, different outcome measures)
- ncorrupt (number of corruption violations)
- ncorrupt_os (share of audited items involving corruption)

## Core Universe

### Controls axes
- **LOO**: 14 specs (drop each meaningful control one at a time)
- **Standard sets**: 4 specs (none/minimal/baseline/extended)
- **Progression**: 6 specs (build up controls step by step)
- **Subset search**: 20 budgeted random draws (seed=112431)

### Sample axes
- Trim pcorrupt at 1st/99th percentile
- Trim pcorrupt at 5th/95th percentile

### Fixed effects axes
- Drop state FE (no absorption)
- Add lottery-number FE (nsorteio)

### Functional form axes
- asinh(pcorrupt): handles zeros and outliers
- log(1+pcorrupt): alternative zero-handling transform

## Inference Plan
- **Canonical**: HC1 robust SEs (matching Stata 'robust')
- **Variant**: Cluster SEs at state level (uf) -- 26 clusters

## Constraints
- Control-count envelope: [0, 41]
- No linkage constraints (single-equation OLS)
- All specifications maintain the same treatment concept (first-term indicator)

## Budget
- Max core specs: 80
- Max control subset specs: 20
- Total planned: 53
- Seed: 112431

## What is excluded and why
- Table 9 (panel convenios regressions): different dataset and design, not same claim object
- Table 6 RDD-style controls: kept as an exploration variant, not core (changes functional form substantially)
- Table 7 experience subsamples: sample restriction changes target population
- Matching estimator (match command): not available in Python
- Tobit/nbreg: alternative estimators for different outcomes
