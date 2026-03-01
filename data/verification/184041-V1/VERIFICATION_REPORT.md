# Verification Report: 184041-V1

## Paper
Esponda & Vespa, "Hypothetical Thinking and Information Extraction in the Laboratory" (AEJ: Microeconomics)

## Baseline Groups Found

### G1: Bid factor CV-CP difference, Experiment I (Table A1)
- **Baseline spec_run_ids**: 184041-V1_run_001 through run_006 (6 baselines)
- **Baseline spec_ids**: baseline, baseline__tablea1_expi_bf_medianreg, baseline__tablea1_expi_bebf_ols, baseline__tablea1_expi_bebf_medianreg, baseline__tablea1_expi_nebf_ols, baseline__tablea1_expi_nebf_medianreg
- **Claim**: CV auction participants bid significantly more than CP participants (bid factor = bid minus benchmark)
- **Primary baseline coefficient (BF-OLS)**: 15.13 (p = 5.5e-09, N = 5,817)
- **Expected sign**: Positive

### G2: Price factor CV-CP difference, Experiment II (Table A3)
- **Baseline spec_run_ids**: 184041-V1_run_033, run_034 (2 baselines)
- **Baseline spec_ids**: baseline, baseline__tablea3_expii_bf_medianreg_stage22
- **Claim**: CV lottery participants price higher than CP participants in individual decision-making (no strategic interaction)
- **Primary baseline coefficient (BF-OLS)**: 6.16 (p = 0.002, N = 8,256)
- **Expected sign**: Positive

### G3: Decision weights (lnfix coefficient), Experiment I (Table A4)
- **Baseline spec_run_ids**: 184041-V1_run_044, run_045, run_046 (3 baselines)
- **Baseline spec_ids**: baseline, baseline__tablea4_expi_cp_decisionweights, baseline__tablea4_expi_interaction
- **Claim**: Subjects put positive weight on the fixed/value component (lnfix) when forming bids, with different weights in CV vs CP
- **Primary baseline coefficient (lnfix, CV)**: 0.244 (p = 4.1e-15, N = 3,253)
- **Expected sign**: Positive

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 51 |
| Valid (run_success=1) | 51 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 51 |
| Non-core | 0 |
| Baseline rows | 11 |
| Inference variants (inference_results.csv) | 5 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 15 |
| core_sample | 30 |
| core_funcform | 6 |

## Robustness Assessment

### G1: Bid factor, Experiment I (30 specs)

- **Sign consistency**: 30/30 (100%) produce positive coefficients.
- **Statistical significance**: 27/30 (90%) significant at 1%.
- **Coefficient range**: [4.58, 17.40] (baseline BF-OLS = 15.13).
- The 3 non-significant specs (runs 022-024) are the full-sample + winners-only intersection (N=2,155), which produces smaller coefficients (~4.6) with p~0.13. This is a small, specific subsample.
- OLS and median regression produce broadly consistent results across all outcomes (BF, BEBF, NEBF).
- Full sample (N=8,560, no exclusion restrictions) yields smaller but still significant effects (~9.5 for OLS, ~13.4 for median).
- Winners-only subsample (N=1,158) yields coefficients close to the baseline (~15), highly significant.

### G2: Price factor, Experiment II (9 specs)

- **Sign consistency**: 6/9 (67%) produce positive coefficients; 3 are negative.
- **Statistical significance**: 4/9 (44%) significant at 5%.
- **Coefficient range**: [-3.11, 6.16].
- The stage 1 subsample (runs 038-039, N=416) produces NEGATIVE coefficients (-1.56 OLS, -3.11 median), reversing the baseline sign. Stage 1 is the simple lottery (before signal), so the treatment effect reversal is plausible and potentially informative.
- Stage 2.1 subsample (runs 036-037, N=832) shows a weak positive OLS effect (3.75, p=0.090) and a near-zero median regression effect (0.00001, p=1.000). The median result effectively finds no treatment difference.
- G2 is the weakest of the three groups, with sign reversals in subsamples.

### G3: Decision weights (12 specs)

- **Sign consistency**: 12/12 (100%) produce positive coefficients.
- **Statistical significance**: 10/12 (83%) significant at 5%; 12/12 significant at 10%.
- **Coefficient range**: [0.224, 1.092].
- OLS (rc/form/ols_instead_of_median) produces larger coefficients than median regression, ranging from 0.45 to 1.09.
- Full sample without exclusions preserves the result across all specifications.
- Two borderline specs (runs 054-055): OLS on full sample with CP-only or interaction samples yields p~0.08.

### Inference sensitivity (from inference_results.csv)
- **G1 baseline (BF-OLS)**: HC1 (p < 0.001), session-cluster (p = 0.005). Highly robust.
- **G2 baseline (BF-OLS)**: HC1 (p < 0.001), session-cluster (p = 0.012). Robust.
- **G3 baseline (lnfix-OLS)**: HC1 (p < 0.001). Only one OLS inference variant available (OLS variant of median baseline).

## Top Issues

1. **Non-unique spec_ids**: The same spec_id (e.g., `rc/sample/full_sample`, `design/randomized_experiment/estimator/diff_in_means`) appears multiple times, differentiated only by outcome variable and/or estimator (OLS vs median). While spec_run_id is unique, this makes spec_id unreliable as a within-group identifier. The runner should use compound spec_ids (e.g., `rc/sample/full_sample__bf_ols`, `rc/sample/full_sample__bf_medianreg`) to disambiguate.

2. **Design spec_id used for non-baseline samples**: Runs 025-030 (G1) and 040-041 (G2) use `design/randomized_experiment/estimator/diff_in_means` but are applied to full_sample and winners_only subsamples, not the baseline sample. The spec_id should include the sample modifier (e.g., `design/.../diff_in_means__full_sample`).

3. **G2 sign reversal in stage 1 subsample**: Runs 038-039 produce negative coefficients, reversing the baseline sign. This is a legitimate finding (the treatment effect is heterogeneous across experiment stages) but should be noted as a robustness concern.

4. **High baseline count (11 of 51 rows)**: 21.6% of rows are baselines, reflecting the crossed outcome-x-estimator structure. This is appropriate for this paper but reduces the effective number of non-baseline robustness checks.

5. **No control variations**: The paper does not use controls in the main regressions (lab experiment with no covariates), so the specification search is limited to sample and estimator variations. This is appropriate but limits the dimensionality of the robustness check.

## Structural Checks

- **spec_run_id uniqueness**: Confirmed unique across all 51 rows.
- **baseline_group_id present**: Yes, all rows have G1, G2, or G3.
- **No infer/* rows in specification_results.csv**: Confirmed.
- **coefficient_vector_json structure**: All rows contain required keys (coefficients, inference, software, surface_hash). Design key present.
- **Inference canonical match**: All rows use infer/se/cluster/subject (canonical), though median regression rows note that clustered SEs were not available and kernel SEs were used instead.
- **Numeric fields finite**: All finite for all run_success=1 rows.

## Recommendations

1. Use compound spec_ids to disambiguate outcome-estimator combinations within each sample variant.
2. Tag design spec_ids applied to non-baseline samples with the sample modifier.
3. Consider adding random control subsets with demographic covariates (age, gender, correctAns) as additional robustness checks, even though the paper omits them.
4. The Stata qreg2 clustered SEs could not be replicated in Python; only kernel SEs were available. This inference discrepancy should be noted.

## Conclusion

The specification search provides **STRONG support** for G1 (Experiment I bid factors: 100% sign consistency, 90% significant at 1%), **MODERATE support** for G3 (decision weights: 100% sign consistency, 83% significant at 5%), and **WEAK support** for G2 (Experiment II price factors: 67% sign consistency, sign reversal in stage 1 subsample). The lab experiment structure limits the specification space to sample and estimator variations, but the results are broadly consistent with the paper's claims for Experiments I and III.
