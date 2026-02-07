# Verification Report: 114372-V1

## Paper
**Title**: Social Networks and Economic Behavior: Dictator Game Experiments
**Journal**: American Economic Review (AER)
**Paper ID**: 114372-V1

## Baseline Groups

### G1: Height -> Dictator Game Giving
- **Claim**: Height (deviation from grade mean) affects the amount given in dictator game experiments.
- **Baseline spec_id**: baseline
- **Outcome**: amount (proportion of endowment given)
- **Treatment**: height (deviation from within-grade mean)
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0018 (SE=0.0038, p=0.64)
- **Baseline N**: 629
- **Notes**: Not statistically significant. The paper main hypothesis is not supported.

### G2: Popularity -> Earnings
- **Claim**: Popularity (number of friend nominations) increases the total amount received from others in dictator games.
- **Baseline spec_id**: earnings/table6_popular
- **Outcome**: receivesum (total amount received)
- **Treatment**: popular (number of friend nominations)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0195 (SE=0.0034, p < 1e-8)
- **Baseline N**: 330
- **Notes**: Highly significant. Different sample, outcome, and treatment from G1.

### G3: Same-race -> Network Formation
- **Claim**: Same-race pairs are more likely to form social network links (homophily).
- **Baseline spec_id**: network/table3_samerace
- **Outcome**: link_dr (binary: friends or not)
- **Treatment**: samerace (same-race indicator)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0098 (SE=0.0015, p < 1e-10)
- **Baseline N**: 55666
- **Notes**: Linear probability model. Entirely different estimand, sample, and model from G1 and G2.

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | 94 |
| **Core test specifications** | 73 |
| **Non-core specifications** | 21 |
| **Invalid** | 0 |
| **Unclear** | 0 |

### Core test breakdown
| Category | Count |
|----------|-------|
| core_controls | 40 |
| core_sample | 21 |
| core_inference | 4 |
| core_funcform | 8 |
| core_fe | 0 |
| core_method | 0 |

### Non-core breakdown
| Category | Count |
|----------|-------|
| noncore_alt_treatment | 10 |
| noncore_alt_outcome | 2 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 3 |
| noncore_diagnostic | 0 |

### By baseline group
| Group | Total specs | Core specs | Non-core specs |
|-------|------------|------------|----------------|
| G1 (height->giving) | 57 | 46 | 11 |
| G2 (popular->earnings) | 25 | 20 | 5 |
| G3 (samerace->links) | 9 | 7 | 2 |
| Unassigned | 3 | 0 | 3 |

## Top 5 Most Suspicious Rows

1. **robust/placebo/randomized_height** (row 88): The coefficient (-0.001789) and SE (0.003834) are IDENTICAL to the baseline. This strongly suggests the placebo randomization was not actually implemented -- the permuted variable appears to be the original height. This is a clear implementation error in the specification search script.

2. **robust/control/add_sameheight vs robust/loo/drop_sameconf**: These two specs have identical coefficients (-0.001814) and identical SEs (0.003814). The add_sameheight step adds controls through sameheight (all except sameconf), and drop_sameconf from LOO also drops sameconf. They are identical specifications counted twice.

3. **robust/control/add_sameconf vs baseline**: These have identical coefficients and SEs. This is expected since add_sameconf is the final step of the control progression (full controls = baseline), but it means one spec is a duplicate.

4. **earnings/table5 and earnings/table6_height**: Both report Height as treatment on receivesum but from different table specifications. Table 5 has fewer controls and a different coefficient (-0.000405 vs -0.001341). Classified as non-core for G2 since they change the treatment from popular to Height.

5. **robust/sample/diff_conf**: The sameconf control has coefficient 0.0 and SE 0.0 (NaN p-value), expected since in different-confidence pairs sameconf is identically 0. Not an extraction error but reflects collinearity in the restricted subsample.

## Recommendations

1. **Fix the placebo implementation**: The robust/placebo/randomized_height spec appears to use the original height variable rather than a permuted version. The script should ensure np.random.permutation() is actually applied. The randomized_height_all placebo does show a different coefficient (-0.000586), suggesting the global randomization works but the within-population version has a bug.

2. **Deduplicate control progression endpoint**: robust/control/add_sameconf produces an identical result to baseline because the full control set is the same. Consider removing this duplicate.

3. **Clarify baseline claims**: The specification search treats all three analyses as one search, but these are fundamentally different hypotheses. Future searches should run separate specification searches per claim, especially since G1 (height -> giving) is a null result while G2 (popular -> earnings) is strongly significant.

4. **Consider subgroup classification**: Some sample restriction specs (e.g., same_race_only, asian_only, high_shy) could alternatively be viewed as heterogeneity analyses. Classified as core_sample here because they test the same treatment-outcome pair on a subsample.
