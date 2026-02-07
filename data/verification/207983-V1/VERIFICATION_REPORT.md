# Verification Report: 207983-V1

## Paper Information
- **Paper**: Kolesar and Walters, "Contamination Bias in Multiple-Treatment Regressions"
- **Journal**: American Economic Review
- **Type**: Methodology paper using Project STAR as illustrative example

## Baseline Groups

### G1: Small Class Effect (Primary)
- **Spec IDs**: baseline
- **Claim**: Small class size (vs. regular) improves kindergarten standardized test scores
- **Expected sign**: Positive
- **Baseline coefficient**: 5.267 (SE=0.778, p<0.001)
- **Model**: OLS with school FE, no additional controls, full STAR sample (N=5,902)

### G2: Aide Treatment Effect (Secondary)
- **Spec IDs**: baseline_aide
- **Claim**: Teaching aide (vs. regular class) has no significant effect on test scores
- **Expected sign**: Zero/null
- **Baseline coefficient**: 0.242 (SE=0.721, p=0.737)
- **Model**: Same regression as G1 (both treatments included)

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **75** |
| Baselines | 2 |
| Core tests (incl. baselines) | 55 |
| Non-core tests | 20 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 22 | Controls variations (add/drop/leave-one-out), baselines |
| core_sample | 20 | Sample restrictions (gender, race, SES, trimming, winsorizing, teacher subsample, performance quartiles) |
| core_funcform | 7 | Functional form (log, asinh, rank, standardized, quantile regressions) |
| core_inference | 5 | Inference variations (classical, HC1, HC2, HC3, school-clustered SE) |
| core_fe | 1 | FE variation (no FE) |
| noncore_heterogeneity | 11 | Interaction models and subsample splits for heterogeneity analysis |
| noncore_placebo | 5 | Balance/placebo tests on predetermined outcomes |
| noncore_alt_treatment | 2 | Different treatment definitions (any_treatment, small_vs_aide) |
| noncore_alt_outcome | 2 | Benhassine dataset (different study entirely) |

## Top 5 Most Suspicious Rows

### 1. robust/het/by_gender_male and robust/sample/male_only (exact duplicates)
- These two spec_ids produce identical coefficients (6.964) and SE (1.075)
- robust/het/by_gender_male is the same regression as robust/sample/male_only but filed under heterogeneity
- Similarly: by_gender_female = female_only, by_race_whiteasian = whiteasian_only, by_race_minority = minority_only, by_ses_low = freelunch_only, by_ses_high = nolunch_only
- **Issue**: 6 specs are exact duplicates of other specs with different spec_ids
- **Recommendation**: Deduplicate or flag these pairs. I classified the sample/ versions as core_sample and the het/ versions as noncore_heterogeneity based on their stated purpose.

### 2. robust/placebo/white_teacher (significant placebo)
- Coefficient: 0.049, SE: 0.009, p<0.001
- This placebo test is significant, suggesting small class assignment predicts having a white teacher
- **Issue**: This could indicate a problem with randomization or school-level sorting. However, with school FE this is within-school variation, so this may reflect a genuine correlation of class assignment with teacher assignment within schools.
- **Recommendation**: Flag this in paper analysis; may not actually be a balance failure if teacher assignment is post-treatment.

### 3. robust/placebo/masters (significant placebo)
- Coefficient: -0.051, SE: 0.014, p<0.001
- Small class assignment significantly predicts teacher having a masters degree (negative)
- **Issue**: Same concern as above -- teacher characteristics may be post-treatment if class assignment affects which teacher is assigned.
- **Recommendation**: Investigate whether teacher assignment is pre- or post-treatment randomization.

### 4. robust/sample/low_performers and robust/sample/high_performers
- These condition on the outcome variable (bottom/top quartile of test scores)
- Coefficients are tiny (0.69 and 0.33) and insignificant
- **Issue**: Conditioning on the outcome is methodologically problematic. These subsample definitions are endogenous to treatment.
- **Recommendation**: These should perhaps be flagged as invalid rather than core_sample, but since the spec search explicitly generated them, they are kept as core_sample with low confidence (0.7).

### 5. robust/sample/girls_only and robust/sample/boys_only (Benhassine dataset)
- These use a completely different dataset (Morocco CCT), different outcome (enroll_attend_May2010), different treatment (lct_father), different FE (stratum), and different clustering (schoolid)
- **Issue**: These are from a separate study and are not comparable to either STAR baseline
- **Recommendation**: These should be removed from the STAR specification search or given their own baseline group for the Benhassine study. Currently classified as noncore_alt_outcome.

## Recommendations for Spec Search Script

1. **Deduplicate heterogeneity subsamples**: The robust/het/by_* subsample specs are exact copies of robust/sample/* specs. Either remove duplicates or clearly differentiate the purpose.

2. **Separate Benhassine analysis**: The two Benhassine specs should either have their own baseline or be excluded from this paper's specification search entirely, since they test a completely different claim in a different dataset.

3. **Reconsider outcome-conditioned subsamples**: The low/middle/high performer splits condition on the dependent variable, which creates selection bias. Consider removing these or clearly flagging them.

4. **Investigate significant placebo results**: The white_teacher and masters placebo tests are highly significant. The script should note whether these are genuine balance failures or reflect post-randomization teacher assignment.

5. **Clarify interaction model coefficients**: The heterogeneity interaction specs report the base treatment effect (for the omitted category), not the average effect. The script should either report the average marginal effect or clearly label these as heterogeneity analyses distinct from the main effect.
