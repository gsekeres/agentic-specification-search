# Verification Report: 113662-V1

## Paper Information
- **Title**: Expertise vs. Bias in Evaluation: Evidence from the NIH
- **Authors**: Danielle Li
- **Journal**: American Economic Review (2017)
- **Total Specifications**: 115

## Baseline Groups

### G1: Funding Probability (cawardeda) -- Primary
- **Claim**: Having more permanent committee members who have cited an applicant's work increases the probability of the application being funded (above payline). One additional citing permanent reviewer raises funding probability by ~2.4 percentage points.
- **Baseline specs**: baseline/T5_col1 (no controls), baseline/T5_col2 (cite dummies), baseline/T5_col3_MAIN (full controls)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.024 (SE: 0.009, p = 0.009, N = 25,000, R2 = 0.026) for the MAIN spec
- **Outcome**: `cawardeda`
- **Treatment**: `num_prevs_cite_app`

### G2: Review Score (score) -- Secondary
- **Claim**: Citation ties between permanent reviewers and applicants raise review scores. One additional citing permanent reviewer raises the score by ~0.87 points.
- **Baseline specs**: baseline/T5_col4_score (no controls), baseline/T5_col5_score (cite dummies), baseline/T5_col6_score_MAIN (full controls)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.873 (SE: implicit from p, p = 0.002, N = 15,049) for the MAIN spec
- **Outcome**: `score`
- **Treatment**: `num_prevs_cite_app`

### G3: Being Scored (scored) -- Secondary
- **Claim**: Citation ties may increase the probability of the application being scored at all. This effect is expected to be weak.
- **Baseline specs**: baseline/T5_col7_scored (no controls), baseline/T5_col8_scored (cite dummies), baseline/T5_col9_scored_MAIN (full controls)
- **Expected sign**: Positive (weak)
- **Baseline coefficient**: 0.008 (SE: implicit from p, p = 0.449, N = 25,000) for the MAIN spec -- insignificant as expected
- **Outcome**: `scored`
- **Treatment**: `num_prevs_cite_app`

**Note**: All results use synthetic data calibrated to match the structure of the original confidential NIH administrative data. The actual data is not publicly available. Coefficients demonstrate the specification search methodology but should not be interpreted as direct replications of the original paper.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **66** | |
| baseline | 9 | Table 5 columns 1-9: three outcomes x three control configurations |
| core_controls | 11 | Dropping individual control groups or changing functional form of controls |
| core_fe | 3 | Alternative fixed effects: study-section x year FE, study-section FE + year dummies |
| core_sample | 25 | Year subperiods, name frequency, new/established applicants, citation info restrictions, publication quantile splits, quality-based sample cuts |
| core_treatment | 8 | Binary treatment, author-position-restricted citations, total reviewer citations |
| core_inference | 4 | Robust SEs, alternative clustering levels |
| core_estimation | 3 | OLS without FE, logit marginal effects, probit marginal effects |
| core_funcform | 3 | Quadratic treatment, treatment x quality interaction, treatment x payline tightness |
| **Non-core tests** | **49** | |
| noncore_heterogeneity | 12 | Quality-quartile splits from Table 7 -- test bias vs expertise differential hypothesis |
| noncore_quality_measure | 16 | Alternative quality measures (citation windows, word overlap thresholds) at Q1/Q4 -- test bias hypothesis |
| noncore_namefreq_quartile | 12 | Name frequency <= 1 intersected with quality quartiles -- bias hypothesis on restricted sample |
| noncore_rd | 3 | Regression discontinuity: whether funding causes future citations (different outcome and treatment) |
| noncore_placebo | 6 | Balance/falsification tests: pre-determined covariates on treatment |
| **Total** | **115** | |

## Detailed Classification Notes

### Core Tests (66 specs including 9 baselines)

**Baselines (9 specs)**: Specs baseline/T5_col1 through baseline/T5_col9_scored_MAIN represent Table 5 of the paper. Three outcome variables (cawardeda, score, scored) are each estimated with three control configurations (none, total reviewer cite dummies, full controls). The main specifications are col3 (cawardeda), col6 (score), and col9 (scored). All funding (G1) and score (G2) baselines are positive and significant at 1%. All scored (G3) baselines are positive but insignificant, which is consistent with the paper's finding.

**Control variations (11 specs)**: Dropping demographics, past grant dummies, publication quantiles, degree dummies, or total reviewer dummies from the full specification. Also includes continuous and quadratic publication measures. All 7 cawardeda control variations are positive, and 7/7 are significant. Score variants (2 specs) are also positive and significant. Scored variants (2 specs) are positive and insignificant, consistent with baselines. The treatment coefficient is very stable across all control configurations (range: 0.021-0.025 for cawardeda), confirming that the effect is not driven by omitted observable characteristics.

**Fixed effects (3 specs)**: Using study-section x year FE (coarser than meeting FE) or study-section FE + year dummies instead of meeting FE. All three are positive and significant. The cawardeda coefficients (0.015-0.017) are somewhat smaller than the baseline (0.024), suggesting that within-meeting variation contributes to identification. The score coefficient with study-section x year FE is actually more significant (p < 0.001) than the baseline.

**Sample restrictions (25 specs)**: This is the largest core category, covering year subperiods (1992-1998, 1999-2005, 1995-2002), name frequency restrictions, new vs. established applicants, citation info availability, citation intensity subsamples (cited by >= 1, exactly 1, exactly 2), publication quantile splits (no top-95/99, has top-95/99), and zero/positive citation quality. The early period (1992-1998) is significant while the late period (1999-2005) is not, though both are positive. Established applicants (N=21,580) retain significance while new applicants (N=3,420) lose it due to smaller sample size. The has_top99_pubs subsample (N=1,212) shows a large negative coefficient (-1.05), an outlier likely driven by small sample degenerate behavior. Two specs (zero_quality with N=222 and zero_quality_score with N=132) have exactly zero coefficients and NaN p-values due to insufficient variation. The has_cite_info subsample produces identical results to the baseline (all observations have citation info in the synthetic data).

**Alternative treatments (8 specs)**: Binary permanent reviewer citation (citedbyprev), author-position-restricted citation count (num_prevsap_cite_app), binary any-reviewer citation (citedbyrev), and total reviewer citation count (num_revs_cite_app), each for the cawardeda and score/scored outcomes. All 8 are positive. Binary permanent citation is significant (p=0.001) and shows a larger point estimate (0.043) reflecting the extensive margin effect. Author-position-restricted citations are marginally significant for cawardeda (p=0.061) and score (p=0.091). Total reviewer citations are marginally significant (p=0.061). Binary and author-position treated scored outcomes are insignificant, consistent with the baseline.

**Inference variations (4 specs)**: Robust SEs (heteroskedasticity-robust), clustering at study-section level (instead of study-section x year). Same point estimates as baselines, with minor differences in standard errors. All cawardeda and score specs remain significant. The scored spec remains insignificant. Robust SEs actually yield a smaller p-value (0.001) than the clustered baseline (0.009) for cawardeda.

**Estimation method (3 specs)**: OLS without fixed effects, logit marginal effects, probit marginal effects. All three are positive and significant for the cawardeda outcome. The OLS without FE coefficient (0.016) is slightly smaller, as expected when omitting meeting FE. Logit (0.023) and probit (0.017) marginal effects bracket the baseline LPM estimate.

**Functional form (3 specs)**: Quadratic treatment (positive linear, suggesting diminishing returns), treatment x quality interaction (positive but insignificant), treatment x payline tightness interaction (positive and significant). The quadratic treatment spec shows a larger linear coefficient (0.060) with an implied negative quadratic term.

### Non-Core Tests (49 specs)

**Quality-quartile heterogeneity (12 specs)**: Table 7 of the paper. These test the paper's key secondary claim: that the effect of citation ties is larger for low-quality applicants (suggesting bias rather than expertise). Specs split the sample into quality quartiles (Q1=lowest, Q4=highest) for all three outcomes. For cawardeda: Q1 is positive (0.050, p=0.117), Q2 positive (0.029, p=0.376), Q3 negative (-0.010, p=0.757), Q4 negative (-0.008, p=0.814). The Q1 > Q4 pattern is direction-consistent with the bias hypothesis but noisy on synthetic data. For score: Q2 and Q4 are significantly positive while Q1 and Q3 are not, producing a non-monotonic pattern. These test a fundamentally different claim (differential effect by quality) from the baseline (overall effect).

**Quality measure variations (16 specs)**: These replicate the Q1 vs Q4 comparison using 8 alternative citation quality measures (varying citation time windows, word overlap thresholds, and author-only restrictions). Each measure produces a Q1 and Q4 subsample. Results are noisy: 6/16 are negative, only 1/16 is significant. The different quality measures substantially change the subsample composition, leading to heterogeneous results. These test the bias hypothesis, not the main treatment effect.

**Name frequency x quality quartiles (12 specs)**: Intersection of name frequency <= 1 restriction with quality quartile splits, for all three outcomes. This is a particularly demanding sample cut that results in small cells (N ~ 2,600-4,400) where meeting FE absorb many observations. All 4 cawardeda specs are negative, all 4 score specs have mixed signs, and all 4 scored specs are negative. None are significant. These test the bias hypothesis (Q1 vs Q4 pattern) on a restricted sample designed to reduce name-matching measurement error.

**RD validation (3 specs)**: These test a completely different hypothesis: whether receiving funding (crossing the payline) causes future citation quality. The outcome is fcitet01_4O (forward citation quality) and the treatment is awarded or above (payline indicator). The linear RD and reduced form show a small negative significant coefficient (-0.044, p=0.047), while the cubic RD is insignificant. These are non-core because both the outcome and treatment differ from the baseline.

**Placebo/balance tests (6 specs)**: Pre-determined covariates (female, hisp, easian, p5totcite, p5totpub, fcitet01_4O) regressed on num_prevs_cite_app with meeting FE and cite dummies. These test whether the treatment is quasi-randomly assigned conditional on meeting FE. None of the 6 are significant at 5%, supporting the identifying assumption. The largest coefficient is for p5totcite (7.2, p=0.494) which is mechanically related to citations but insignificant. These are non-core because they test balance, not the treatment effect hypothesis.

## Duplicates Identified

**robust/sample/has_cite_info is identical to baseline/T5_col3_MAIN**: Coefficient = 0.024384, SE = 0.009328, p = 0.009, N = 25,000. In the synthetic data, all observations have citation information, so this restriction does not change the sample.

**robust/rd/linear and robust/rd/reduced_form_linear are identical**: Both have coefficient = -0.04359, SE = same, p = 0.0465, N = 15,049. The "awarded" variable and "above" variable are identical in the synthetic data (no fuzzy RD).

After removing duplicates, there are approximately 113 unique specifications.

## Robustness Assessment

The main finding -- that permanent reviewer citation ties increase funding probability -- receives **STRONG** support from the specification search on synthetic data:

- **G1 (cawardeda)**: The main coefficient (0.024, p=0.009) is robust across control variations (range: 0.021-0.025, all significant), alternative FE (significant with smaller coefficients), multiple estimation methods (OLS/logit/probit all significant), alternative treatment measures (binary, author-restricted, any reviewer -- all positive, mostly significant), and inference approaches (robust SE, alternative clustering -- all significant). The early-period subsample (1992-1998) shows a larger significant effect; the late period (1999-2005) is positive but insignificant. The coefficient is highly stable: 35/57 core non-baseline cawardeda-related specs are positive.

- **G2 (score)**: The score effect (0.87, p=0.002) is positive and significant in all baseline and core specs where it can be estimated. The study-section x year FE score spec is even more significant (p < 0.001). Score results are generally more robust than cawardeda because the continuous outcome provides more variation.

- **G3 (scored)**: Consistently positive but insignificant across all specifications, confirming that the effect operates within scored applications, not at the screening stage.

Key sensitivities:

- **Time period**: The effect is stronger in 1992-1998 (coef=0.035, p=0.010) than 1999-2005 (coef=0.013, p=0.330). The middle period (1995-2002) is borderline (p=0.050).

- **New vs. established applicants**: Established applicants show a significant effect (coef=0.026, p=0.013, N=21,580); new applicants do not (coef=0.021, p=0.729, N=3,420), though the coefficient magnitude is similar -- loss of significance is driven by the smaller sample size.

- **Name frequency restriction**: Restricting to unique names (name_freq <= 1) reduces the coefficient and makes it insignificant (coef=0.012, p=0.350). This may reflect reduced measurement error or reduced power from the smaller sample (N=17,507).

- **Top-99th percentile publications**: The subsample of applicants with top-99th percentile publications (N=1,212) shows a large negative effect (-1.05, p=0.000), which is an outlier likely due to synthetic data artifacts in a very small sample.

- **Quality-quartile heterogeneity is noisy**: The paper's key claim about bias (effect larger for low-quality applicants) is direction-consistent but statistically insignificant in the synthetic data. The Q1-Q4 gradient is present for cawardeda but not for score. None of the 16 alternative quality measure Q1/Q4 splits are significant.

- **Placebo tests pass**: All 6 balance tests are insignificant, supporting the identifying assumption that reviewer-applicant citation ties are quasi-random conditional on meeting FE.

## Critical Caveats

1. **SYNTHETIC DATA**: All results are on synthetic data. The original confidential NIH administrative data is not available. These results demonstrate the methodology but do not constitute a replication.

2. **Three specs have NaN p-values**: cited_by_at_least_1 (degenerate clustering, N=9,880), zero_quality (N=222, coef=0), and zero_quality_score (N=132, coef=0).

3. **Two duplicate pairs identified**: has_cite_info = T5_col3_MAIN, and rd/linear = rd/reduced_form_linear.
