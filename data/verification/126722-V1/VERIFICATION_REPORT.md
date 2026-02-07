# Verification Report: 126722-V1

## Paper
- **Title**: Allocating Health Care Resources Efficiently: The Simple Economics of Vouchers vs. In-Kind Provision
- **Authors**: Lopez, Sautmann, Schaner (2020)
- **Journal**: AEJ-Applied
- **Method**: Cross-sectional OLS from a randomized controlled trial in Mali

## Baseline Groups

### G1: Patient voucher effect on malaria treatment purchase
- **Baseline spec_id**: baseline
- **Claim**: Patient vouchers for antimalarial drugs increase the probability that patients purchase any malaria treatment (simple or severe).
- **Outcome**: treat_sev_simple_mal (binary: purchased any malaria treatment)
- **Treatment**: patient_voucher (random assignment to patient voucher arm)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.153 (SE=0.038, p<0.001)
- **Interpretation**: 15.3 percentage point increase in treatment purchase from patient voucher

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **92** |
| Baseline | 1 |
| Core tests | 58 |
| Non-core tests | 34 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 29 | Control set variations (leave-one-out, incremental, no controls, full controls, include doctor info, patient_voucher_only model) |
| core_inference | 3 | Clustering variations (robust HC1, clinic-day, date) |
| core_sample | 26 | Sample restrictions (age, gender, risk, illness duration, language, literacy, ethnicity, education, symptoms) |
| noncore_alt_outcome | 15 | Alternative outcomes (prescribed treatment, severe malaria, voucher usage, match quality, predicted malaria) |
| noncore_alt_treatment | 2 | Different treatment variables (doctor_voucher, any_voucher) |
| noncore_heterogeneity | 13 | Interaction models (malaria risk, age, gender, symptoms, info, education, illness, ethnicity, French, literacy, triple interaction) |
| noncore_placebo | 4 | Placebo tests (predicted malaria, age, days ill, symptoms) |

## Top 5 Most Suspicious Rows

1. **robust/funcform/expected_match** (spec_id): This is labeled as a functional form variation but has outcome=expected_mal_match_any, which is a completely different outcome from the baseline (treat_sev_simple_mal). It is also an exact duplicate of robust/outcome/expected_mal_match_any (same coefficient -0.092, same p-value). Classified as noncore_alt_outcome.

2. **robust/funcform/pred_mal_continuous** (spec_id): Labeled as functional form but has outcome=pred_mal_pos, which is a predicted malaria probability -- entirely different from the baseline outcome. This is more akin to a placebo/diagnostic test. Classified as noncore_alt_outcome.

3. **robust/het/malaria_risk** and **robust/het/risk_x_info**: These two heterogeneity specs report treatment_var=patient_voucher_high rather than patient_voucher, meaning they extract the interaction coefficient rather than the main effect. This makes them non-comparable to the baseline even if one wanted to include heterogeneity results.

4. **robust/control/date_fe_only**: This has an identical coefficient and p-value to the baseline (0.153, p=5.76e-05), suggesting it IS the baseline specification -- the baseline already only includes date dummies as controls. This is not problematic but worth noting the duplication.

5. **robust/cluster/clinic_day**: Also identical to baseline (same coefficient and p-value), since the baseline already clusters at clinic-day level. This is a true duplicate, not suspicious, but inflates the count.

## Classification Notes

### Core tests (58 specs)
The 58 core specifications all share the same outcome (treat_sev_simple_mal) and treatment (patient_voucher) as the baseline. They vary along three dimensions:
- **Controls** (29 specs): No controls, date FE only, basic patient controls, full controls, leave-one-out drops of individual controls, incremental additions, including doctor info treatment, and omitting doctor_voucher from the model.
- **Inference** (3 specs): Robust HC1, clinic-day clustering (baseline), date clustering.
- **Sample** (26 specs): Restrictions by age, gender, malaria risk, illness duration, symptoms, language, literacy, ethnicity, education, patient respondent status, and non-pregnant.

### Non-core: Alternative outcomes (15 specs)
These test the effect of patient_voucher on different outcome variables: prescribed treatment (RXtreat_sev_simple_mal), severe malaria treatment (treat_severe_mal, RXtreat_severe_mal), voucher usage (used_vouchers_admin), match quality (expected_mal_match_any and variants), and predicted malaria (pred_mal_pos). While informative for the paper, these test different claims than the baseline.

### Non-core: Alternative treatments (2 specs)
doctor_voucher_only and any_voucher change the causal object from patient_voucher to a different treatment variable.

### Non-core: Heterogeneity (13 specs)
All heterogeneity specs add interaction terms to the model, changing the interpretation of the main effect coefficient. Even when the reported coefficient is the main effect of patient_voucher, the inclusion of an interaction term means the estimate represents the effect for the omitted category rather than the average effect. Two specs (malaria_risk and risk_x_info) report the interaction coefficient itself.

### Non-core: Placebo (4 specs)
These test whether the voucher affects outcomes it should not (patient age, days ill, symptoms, predicted malaria), serving as balance/randomization checks.

## Recommendations

1. **Remove duplicate specs**: robust/control/date_fe_only and robust/cluster/clinic_day are identical to the baseline. Consider dropping or flagging them.

2. **Fix functional form category**: robust/funcform/pred_mal_continuous and robust/funcform/expected_match are not functional form variations -- they use entirely different outcome variables. They should be recategorized as outcome variations or diagnostic tests in the spec search script.

3. **Clarify heterogeneity extraction**: For robust/het/malaria_risk and robust/het/risk_x_info, the treatment_var is listed as patient_voucher_high, indicating the interaction coefficient was extracted rather than the main effect. The spec search script should either consistently extract the main effect or clearly flag these as interaction estimates.

4. **Consider separating baseline claim for match quality**: The paper discusses treatment-illness match quality as a secondary finding. If this is considered a distinct claim, it could warrant its own baseline group (G2). Currently all match-quality outcomes are classified as noncore_alt_outcome.
