# Specification Surface: 112415-V1

## Paper: Attention and Choice (Lab Experiment on Fixations and Set Size)

## Baseline Groups

### G1: Effect of choice set size on attention efficiency
- **Outcome**: Fixation efficiency (how well initial fixations target higher-rated items relative to set mean/max)
- **Treatment**: Set size (4, 9, or 16 alternatives) -- within-subject manipulation
- **Estimand**: ITT effect of set size on attention allocation quality
- **Population**: Lab subjects (N~39) performing snack food choice tasks with eye-tracking
- **Baseline spec**: Stats-2A: within-subject efficiency means by set size, paired t-tests
- **Analysis method**: Collapse fixation-level data to (subject x condition) means, then paired t-tests across conditions

### Additional baselines (same claim, different outcome measures)
- Stopping probability (probability of stopping search at a given fixation) -- Stats-4A, Stats-4B
- Fixation duration (time spent looking at items) -- Stats-6A
- Choice quality (whether best-seen item is chosen) -- Stats-3B

## Core Universe

### Design alternatives
- **Diff-in-means**: Direct paired t-test (paper's approach)
- **OLS with covariates**: Regression of outcome on set-size dummies with subject FE

### Sample axes
- Trim reaction times at 2000ms (as paper does in some analyses)
- Trim reaction times at 1500ms (stricter)
- Include vs exclude refixations (paper filters to initial fixations in most analyses)
- Keep all fixation types (including refixations)

### Data construction / outcome axes
- Efficiency_New vs original Efficiency variable
- CurFixef vs Efficiency
- Rank-based efficiency measure

### Condition comparisons
- 4 vs 9 (pairwise)
- 9 vs 16 (pairwise)
- 4 vs 16 (full range)

## Inference Plan
- **Canonical**: Paired t-test on within-subject condition means
- **Variant 1**: OLS with subject-clustered SEs
- **Variant 2**: HC1 robust SEs on pooled regression

## Constraints
- Control-count envelope: [0, 0] -- within-subject design with no observational controls
- No linkage constraints (simple experimental comparisons)
- All specifications maintain the same within-subject comparison structure
- ~39 subjects limits power for detecting small effects

## Budget
- Max core specs: 50
- Total planned: ~30-40
- Seed: 112415

## What is excluded and why
- Figure-producing code (visualization only, no statistical tests)
- Some figures use the same statistical tests as others (duplicates)
- Structural models of attention (not part of the reduced-form evidence)
- Dynamic models of search termination (these would be explore/* variants)
