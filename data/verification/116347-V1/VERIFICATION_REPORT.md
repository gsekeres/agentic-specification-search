# Verification Report: 116347-V1

## Paper: Workplace Friendships and Productivity (AEJ-Applied)

---

## Baseline Groups

### G1: Effect of friend presence on worker productivity

- **Claim**: Working alongside friends increases worker productivity
- **Expected sign**: Positive
- **Baseline spec IDs**: baseline, baseline/continuous_treatment
- **Outcome**: logprod (log productivity)
- **Treatments**: has_friend_present (binary), num_friends_present (continuous)
- **FE**: Worker (id_str) + group-date (groupdate)
- **Clustering**: Worker (id_str)
- **Baseline result**: coef = -0.017, p = 0.84 (binary); coef = -0.023, p = 0.31 (continuous)
- **Note**: Both baselines are statistically insignificant and have the wrong sign relative to the paper claim. The binary treatment has almost no within-worker variation (99.3% of observations have at least one friend present).

---

## Counts

| Category | Count |
|----------|-------|
| Total specifications | 70 |
| Baselines | 2 |
| Core tests | 35 |
| Non-core tests | 22 |
| Invalid (errors/artifacts) | 11 |
| Unclear | 0 |

---

## Category Breakdown

| Category | Count |
|----------|-------|
| core_fe | 6 |
| core_inference | 4 |
| core_sample | 21 |
| core_funcform | 3 |
| core_method | 1 |
| noncore_placebo | 3 |
| noncore_alt_outcome | 4 |
| noncore_alt_treatment | 4 |
| noncore_heterogeneity | 8 |
| noncore_diagnostic | 3 |
| invalid | 11 |

Note: 2 baselines (core_controls) + 35 core + 22 non-core + 11 invalid = 70 total.

---

## Top 5 Most Suspicious Rows

### 1. robust/cluster/group -- Numerical artifact
- **Issue**: SE near zero (5.41e-17), yielding t-stat of -3.13e14 and p=0.0.
- **Diagnosis**: Clustering at group level absorbed by group-date FE. Numerical artifact.
- **Action**: Marked invalid.

### 2. robust/heterogeneity/* specs -- Estimand mismatch
- **Issue**: All 6 personality/age heterogeneity specs use only groupdate FE without worker FE, yielding large positive significant main effects (0.07 to 0.67).
- **Diagnosis**: Dropping worker FE changes estimand to cross-sectional variation reflecting selection.
- **Action**: Marked noncore_heterogeneity.

### 3. robust/control/* specs -- Estimand mismatch
- **Issue**: Control-progression specs drop worker FE, yielding positive significant effects (0.14 to 0.24).
- **Diagnosis**: Without worker FE, captures between-worker selection, not causal effect.
- **Action**: Marked noncore_diagnostic.

### 4. robust/treatment/at_least_1_friends -- Redundant
- **Issue**: Coefficient and SE identical to baseline. has_1_friends = has_friend_present.
- **Diagnosis**: Redundant duplicate.
- **Action**: Marked noncore_alt_treatment.

### 5. robust/sample/late_period and experienced -- Anomalously significant
- **Issue**: late_period shows coef=0.147, p=0.0; experienced shows coef=0.130, p=0.0; baseline is null.
- **Diagnosis**: May reflect time-varying confounders or structural change during experiment.
- **Action**: Kept as core_sample but flagged.

---

## Recommendations

1. **Treatment variable**: Reconstruct original proximity-based treatment (along_fr) instead of presence-based (has_friend_present) which has 99.3% = 1.
2. **FE consistency**: Keep worker FE in heterogeneity/control specs to maintain consistent estimand.
3. **Clustering fix**: Skip clustering at levels absorbed by FE structure.
4. **Collinearity handling**: Detect insufficient within-group treatment variation before estimation.
5. **Placebo design**: Use randomized treatment permutation rather than fake timing cutoffs.
