# Verification Report: 131981-V1

**Paper**: Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey
**Journal**: American Economic Journal: Applied Economics
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Mental distress effect of age-specific curfews
- **Claim**: Age-specific curfews imposed on individuals over 65 during COVID-19 increased mental distress (z_depression) at the RD cutoff of December 1955 birth month.
- **Expected sign**: Positive (curfews increase mental distress)
- **Baseline spec_ids**: baseline
- **Outcome**: z_depression (standardized SRQ-20 mental health index)
- **Treatment**: before1955 (born before December 1955, subject to over-65 curfew)
- **Method**: Sharp RD, bandwidth=45 months, linear polynomial, controls for ethnicity/education/female, clustered on birth month (modate)
- **Baseline coefficient**: 0.140 (SE=0.103, p=0.180) -- NOT significant at conventional levels

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **87** |
| Baseline | 1 |
| Core tests | 57 |
| Non-core tests | 30 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 13 | Bandwidth variations, linear polynomial variants, baseline duplicate |
| core_funcform | 21 | Quadratic polynomial, mental health sub-indices, individual symptoms, winsorized outcomes |
| core_controls | 5 | No controls, female-only control, leave-one-out (drop ethnicity/education/female) |
| core_inference | 3 | Robust SEs, cluster on modate, cluster on province |
| core_sample | 15 | Donut holes, married/unmarried, education splits, chronic disease splits, psych support splits, trimming, symmetric sample |
| noncore_alt_outcome | 11 | Mobility outcomes (outside_week, never_out) and channel outcomes (employment, money, household, conflict, physical/social limitations) |
| noncore_diagnostic | 9 | First-stage (under_curfew) and 8 covariate balance tests |
| noncore_heterogeneity | 6 | Male/female-only subsamples and 4 interaction specifications (gender, marital, chronic, psych support) |
| noncore_placebo | 4 | Placebo cutoff tests at -24, -12, +12, +24 months |

---

## Classification Notes

### Core test rationale
- **Bandwidth variations** (8 specs): These vary the RD bandwidth while keeping the same outcome (z_depression), treatment (before1955), and control set. They directly test the sensitivity of the main claim to bandwidth choice. Classified as core_method.
- **Polynomial order** (6 specs): Linear and quadratic polynomial specifications at different bandwidths. Linear variants duplicate the method variation; quadratic variants change the functional form. All preserve the estimand.
- **Control variations** (5 specs): Adding/dropping individual controls from the baseline set. These preserve the estimand and test sensitivity to covariate adjustment.
- **Clustering variations** (3 specs): Same point estimate, different standard errors. Pure inference robustness.
- **Donut hole** (4 specs): Exclude observations closest to the cutoff. These test the same estimand with sample restrictions around the discontinuity, a standard RD robustness check.
- **Mental health sub-indices** (z_somatic, z_nonsomatic, sum_srq): These are alternative measurements of the same underlying mental health concept using the same SRQ-20 instrument. Classified as core_funcform.
- **Individual SRQ symptoms** (12 specs): Each symptom is a binary component of the z_depression index. While they measure narrower constructs, they are disaggregated views of the same estimand. Classified as core_funcform with lower confidence (0.75).
- **Sample restrictions** (married/unmarried, education, chronic disease, psych support): These subset the sample. While they could be viewed as heterogeneity analyses, they retain the z_depression outcome and before1955 treatment and test whether the main claim holds across subpopulations. Classified as core_sample with moderate confidence (0.65-0.70).
- **Winsorized/trimmed outcomes** (5 specs): Outlier robustness checks on the same outcome. Classified as core_funcform (winsorized) or core_sample (trimmed).
- **Symmetric sample** (1 spec): Standard RD sample restriction. Classified as core_sample.

### Non-core rationale
- **Male/female-only subsamples**: Given the strong documented gender heterogeneity (significant interaction p=0.001), these are best understood as heterogeneity analyses rather than overall effect robustness. The male-only and female-only results tell different stories. Classified as noncore_heterogeneity.
- **Interaction specifications** (4 specs): These add interaction terms between the treatment and a moderating variable. The reported coefficient is the main effect conditional on the interaction, which changes the interpretation. Classified as noncore_heterogeneity.
- **Mobility/first-stage outcomes** (outside_week, under_curfew, never_out): These measure whether the curfew affected mobility, not mental health. They are a different estimand (mechanism/first-stage). Classified as noncore_alt_outcome or noncore_diagnostic.
- **Channel outcomes** (9 specs): Employment, financial, household, and activity limitation outcomes are different causal objects from mental health. They test downstream channels, not the baseline mental health claim.
- **Placebo cutoff tests** (4 specs): Fake cutoffs at alternative birth months. These are validity diagnostics, not tests of the main claim. The treatment variable changes to before_placebo.
- **Covariate balance tests** (8 specs): These regress predetermined covariates (highschool, illiterate, female, married, non_turk, pre_covid_hhsize, psych_support, chronic_disease) on the treatment. They are RD validity diagnostics with entirely different outcomes.

---

## Top 5 Most Suspicious Rows

1. **robust/outcome/z_depression (row 16)**: This is an exact duplicate of the baseline specification (same coefficient, SE, p-value, all fields identical). It appears the spec-search script re-ran the baseline under a different spec_id path. Not harmful but redundant.

2. **rd/bandwidth/bw_45 (row 6)**: Also an exact duplicate of the baseline (BW=45 is the baseline bandwidth). Same coefficient 0.1398, SE 0.1034, p=0.180. Redundant entry.

3. **rd/poly/linear_bw45 (row 11)**: Another exact duplicate of the baseline (linear polynomial at BW=45 is the baseline). Same numbers. Three rows are identical to baseline.

4. **robust/het/interaction_gender (row 45)**: The reported treatment coefficient here (0.264, p=0.018) is the main effect of before1955 conditional on the gender interaction. This is NOT the overall treatment effect -- it is the effect for males specifically (since female=0 is the omitted category). The spec_id labels it as the treatment coefficient, but it should be understood as a conditional effect. The interaction itself is -0.345 (p=0.001), meaning the female effect is approximately 0.264 - 0.345 = -0.081 (null).

5. **rd/poly/quadratic_bw30 (row 13)**: This is the only specification where the coefficient flips sign to a substantial negative value (-0.246). The quadratic polynomial with the narrowest bandwidth (30 months, only 858 obs) is likely overfitting. This is not an error per se, but it highlights the sensitivity of the RD estimate to specification choices.

---

## Recommendations for the Spec-Search Script

1. **Deduplicate baseline**: Three specs (baseline, rd/bandwidth/bw_45, rd/poly/linear_bw45, and robust/outcome/z_depression) are exact duplicates. The script should detect and mark duplicates or avoid generating them.

2. **Separate interaction coefficients**: For heterogeneity/interaction specs, the script reports the main treatment coefficient conditional on the interaction term. It would be more informative to also report (or separately flag) the interaction coefficient and the implied effect for each group. Currently, the coefficient_vector_json contains the interaction details for the gender spec but not consistently for others.

3. **Flag first-stage / mechanism outcomes**: The script lumps mobility outcomes, first-stage checks, channel outcomes, and mental health outcomes under the same robustness/measurement.md tree path. A clearer taxonomy in spec_tree_path (e.g., first_stage/, channels/, symptoms/) would improve automated classification.

4. **Gender-split specs should be marked as heterogeneity**: The male_only and female_only specs are under robustness/sample_restrictions.md, but given the paper's strong documented gender heterogeneity, these are better classified as heterogeneity analyses. The script could check whether a sample split corresponds to a significant interaction and flag accordingly.

5. **Consider optimal bandwidth**: The paper likely uses a data-driven bandwidth (e.g., Imbens-Kalyanaraman or Calonico-Cattaneo-Titiunik). The baseline bandwidth of 45 months appears to be the paper's chosen bandwidth. The script should document which bandwidth selection method was used, as this matters for interpreting bandwidth sensitivity.
