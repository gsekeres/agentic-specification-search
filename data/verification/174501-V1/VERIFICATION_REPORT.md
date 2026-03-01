# Verification Report: 174501-V1

## Paper
Lowe (2021), "Types of Contact: A Field Experiment on Collaborative and Adversarial Caste Integration", AER

## Baseline Groups Found

### G1: Race IAT (racial stereotypes) -- Table 3
- **Baseline spec_run_id**: 174501-V1_run_001
- **Baseline spec_id**: baseline__table3_col2_black_raceiat
- **Claim**: ITT effect of mixed-race roommate on Black students' Race IAT D-score at UCT (higher = less prejudice).
- **Baseline coefficient (mixracebas)**: -0.094 (SE=0.070, p=0.174, N=332, R2=0.097)
- **Expected sign**: Negative (mixed-race room reduces racial bias)
- **Note**: Baseline is NOT statistically significant.

### G2: Academic performance -- Table 4
- **Baseline spec_run_ids**: 174501-V1_run_020, 174501-V1_run_021, 174501-V1_run_022
- **Baseline spec_ids**: baseline__table4_black_examspassed, baseline__table4_black_continue, baseline__table4_black_pcaperf
- **Claim**: ITT effect of mixed-race roommate on Black students' academic outcomes (exams passed, continuation, PCA performance index).
- **Baseline coefficients**:
  - examspassed: 0.644 (SE=0.242, p=0.008, N=324)
  - continue: 0.152 (SE=0.037, p=0.0001, N=324)
  - PCAperf: 0.443 (SE=0.139, p=0.002, N=324)
- **Expected sign**: Positive (mixed-race room improves academic performance)

### G3: Social outcomes -- Table 5
- **Baseline spec_run_ids**: 174501-V1_run_067, 174501-V1_run_068, 174501-V1_run_069
- **Baseline spec_ids**: baseline__table5_white_pcaattitude, baseline__table5_white_pcacomm, baseline__table5_white_pcasocial
- **Claim**: ITT effect of mixed-race roommate on White students' social outcomes (racial attitudes, cross-racial communication, pro-social behavior).
- **Baseline coefficients**:
  - PCAattitude: 0.670 (SE=0.262, p=0.012, N=106)
  - PCAcomm: 0.438 (SE=0.252, p=0.084, N=94)
  - PCAsocial: 0.760 (SE=0.296, p=0.012, N=79)
- **Expected sign**: Positive (mixed-race room improves social/attitudinal outcomes)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 108 |
| Valid (run_success=1) | 108 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 90 |
| Non-core (population change) | 18 |
| Baseline rows | 7 |
| Inference variants (inference_results.csv) | 6 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 35 |
| core_controls | 42 |
| core_funcform | 9 |
| core_fe | 4 |
| noncore_population_change | 18 |

## Robustness Assessment

### G1: Race IAT (17 core specs)

#### Sign consistency
- **17 of 17** core specifications (100%) produce a negative coefficient, consistent with the baseline sign.
- Coefficient range: [-0.117, -0.082]. Very stable point estimate.

#### Statistical significance
- **0 of 17** core specs are significant at p < 0.05.
- **4 of 17** core specs are significant at p < 0.10 (diff-in-means, ANCOVA, strata FE, drop roommate controls).
- The result is directionally consistent but statistically insignificant across all specifications.

#### Controls sensitivity
- LOO control variations produce coefficients from -0.086 to -0.106. The result is insensitive to any individual control.
- Dropping all roommate controls slightly increases the magnitude (-0.115 vs -0.094) and borderline significance (p=0.089).

#### Population sensitivity
- Full sample with race FE: coef=-0.008, p=0.891 -- effect disappears entirely.
- White+Black subsample: coef=0.001, p=0.994 -- effect disappears.
- The IAT result is specific to the Black subsample and does not generalize.

### G2: Academic performance (39 core specs)

#### Sign consistency
- **39 of 39** core specifications (100%) produce a positive coefficient.
- Coefficient range varies across outcomes but is consistently positive.

#### Statistical significance
- **27 of 39** core specs (69.2%) are significant at p < 0.05.
- **32 of 39** core specs (82.1%) are significant at p < 0.10.
- Insignificant specs are primarily diff-in-means/with-covariates designs (no program FE) and GPA with certain LOO drops.

#### Key vulnerability
- Dropping program FE substantially weakens results: GPA p=0.12, examspassed p=0.14, continue p=0.054. Program FE are important for academic outcomes.
- Second-year outcomes: GPA2013 is insignificant (p=0.45), but examspassed2013, continue2013, and PCAperf2013 remain significant, suggesting persistence of the academic gains.

### G3: Social outcomes (34 core specs)

#### Sign consistency
- **34 of 34** core specifications (100%) produce a positive coefficient.

#### Statistical significance
- **28 of 34** core specs (82.4%) are significant at p < 0.05.
- **30 of 34** core specs (88.2%) are significant at p < 0.10.
- PCAcomm is the weakest outcome: diff-in-means p=0.72, with-covariates p=0.24, strata FE p=0.17. Only the full-controls baseline (p=0.084) approaches significance.

#### Population sensitivity
- Full sample with race FE: PCAfriend, PCAattitude, PCAsocial remain significant; PCAcomm does not (p=0.22).
- Black-only subsample: None of the four social outcomes are significant (p=0.14-0.66). The social outcome effects are specific to White students.

### Inference sensitivity (from inference_results.csv)

**G1 (baseline Race IAT):**
- Cluster(room): SE=0.070, p=0.174 (baseline)
- HC1 (robust): SE=0.069, p=0.171 -- similar
- Cluster(residence): SE=0.080, p=0.275 -- even less significant

**G2 (baseline examspassed):**
- Cluster(room): SE=0.242, p=0.008 (baseline)
- HC1 (robust): SE=0.244, p=0.009 -- similar
- Cluster(residence): SE=0.290, p=0.062 -- still significant at 10%

**G3 (baseline PCAattitude):**
- Cluster(room): SE=0.262, p=0.012 (baseline)
- HC1 (robust): SE=0.262, p=0.013 -- similar
- Cluster(residence): SE=0.314, p=0.077 -- still significant at 10%

Clustering at the residence level (coarser) inflates standard errors by ~20-30% but does not eliminate significance for G2/G3.

## Top Issues

1. **G1 baseline is insignificant**: The Race IAT result (Table 3 Col 2) has p=0.174 at baseline. No specification achieves p<0.05. The claim about racial stereotypes reduction is not supported by conventional statistical criteria.

2. **G2/G3 depend on program/residence FE**: Dropping program FE for academic outcomes (G2) substantially weakens results. The strata FE are important for precision in this small-sample experiment.

3. **Population specificity**: Academic effects (G2) are specific to Black students. Social effects (G3) are specific to White students. Full-sample and cross-race specifications show attenuated or null effects, suggesting the findings do not generalize across race groups.

4. **PCAcomm weakness**: Cross-racial communication (PCAcomm) is the weakest social outcome, with most non-baseline specifications being insignificant. This specific outcome drives the marginal significance in G3.

5. **Small samples**: N ranges from 79 (PCAsocial) to 332 (IAT). The very small samples for social outcomes limit power and may contribute to instability.

## Recommendations

1. G1 should not be interpreted as providing evidence for IAT effects -- the baseline is insignificant and no robustness check achieves significance.
2. The runner could have included LOO specs across all outcomes in G2/G3 (currently only GPA for G2 and PCAfriend for G3), though the budget constraint is reasonable.
3. Cluster-at-residence inference should be reported as a stress test given the stratified randomization design.

## Conclusion

**G1 (Race IAT)**: **NO support**. The baseline is statistically insignificant (p=0.174) and no specification achieves significance at p<0.05. The sign is consistent (100% negative) but the effect is imprecise.

**G2 (Academic performance)**: **STRONG support**. All 39 core specs are positive, 69% significant at p<0.05. The result is robust to control variations, most design alternatives, and persists into the second year. Main vulnerability is sensitivity to program FE.

**G3 (Social outcomes)**: **STRONG support** for PCAfriend, PCAattitude, PCAsocial; **WEAK support** for PCAcomm. All 34 core specs are positive, 82% significant at p<0.05. Results are specific to White students and weaken for the cross-racial communication outcome.

Overall assessment: **STRONG support** for G2 (academic gains for Black students) and **STRONG support** for G3 (social gains for White students), but **NO support** for G1 (IAT effects).
