# Specification Search: 112415-V1

## Paper
Arieli, Ben-Ami & Rubinstein (2011), "Tracking Decision Makers under Uncertainty"
American Economic Journal: Microeconomics, 3(4), 68-76.

## Surface Summary
- **Baseline groups**: 1 (G1: Effect of choice set size on attention efficiency)
- **Design**: Randomized within-subject experiment (N=41 subjects, 3 conditions: 4/9/16 items)
- **Budget**: 50 core specs max
- **Seed**: 112415
- **Canonical inference**: Paired t-test on within-subject condition means

## Execution Summary
- **Planned specifications**: 57 core + 7 inference
- **Executed successfully**: 57 core + 7 inference
- **Failed**: 0 core + 0 inference

## Specification Breakdown

### Baselines (4 specs)
1. `baseline`: Efficiency (4 vs 16) paired t-test -- paper's Stats-2A main comparison
2. `baseline__stopping_prob`: Stopping probability (4 vs 16)
3. `baseline__fixation_duration`: Fixation duration (4 vs 16)
4. `baseline__choice_quality`: Best-seen-chosen (4 vs 16)

### Design Alternatives (9 specs)
- OLS with subject FE for: efficiency, stopping_prob, fix_duration, best_seen_chosen,
  efficiency_new, curfix_ef, rt, num_looked, pct_looked
- Pairwise OLS (4v9, 9v16 subsets)

### Robustness: Sample Restrictions (31 specs)
- Pairwise condition comparisons (4v9, 9v16, 4v16) across outcomes
- RT trimming (< 2000ms, < 1500ms)
- All-fixations sample (include refixations)
- Initial-fixations-only for duration

### Robustness: Data Construction (13 specs)
- Efficiency_New (alternative efficiency definition)
- CurFixef (current fixation efficiency)
- CurFixefNew
- Rank-based efficiency
- Elapsed time at search termination

### Inference Variants (7 specs)
- HC1 robust SEs (ignoring within-subject correlation)
- CRV1(subject) cluster SEs

## Deviations from Surface
- Surface noted N~39 subjects; data contains 41 unique subjects
- No structural model estimation attempted (correctly excluded per surface)
- Rank-based efficiency is approximate (rank within fixation-level data, not within full set)

## Software Stack
- Python 3.12.7
- pandas 2.2.3
- numpy 2.1.3
- pyfixest 0.40.1
- scipy 1.15.1
