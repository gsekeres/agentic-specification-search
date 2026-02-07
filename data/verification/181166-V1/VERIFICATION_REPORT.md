# Verification Report: 181166-V1

**Paper**: Technological Change and the Consequences of Job Loss
**Authors**: J. Carter Braxton and Bledi Taska
**Journal**: AER: Insights (2023)
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Main effect of technological change on earnings losses

**Claim**: Workers displaced from occupations experiencing greater technological change (increased computer/software requirements between 2007-2017) suffer larger earnings losses.

**Baseline spec_id**: `baseline`

- **Outcome**: `d_ln_real_earn_win1` (change in log real weekly earnings, winsorized 2.5%)
- **Treatment**: `d_computer_2017_2007_n` (normalized change in computer/software requirements, 2007-2017)
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0234 (p = 0.007)
- **FE**: Year, year of job loss
- **Clustering**: Displaced occupation (dwocc1990)
- **Model**: OLS with survey weights

There is only one baseline group corresponding to the paper's single core finding.

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **67** |
| Baseline | 1 |
| Core tests (including baseline) | 60 |
| Non-core tests | 6 |
| Invalid | 1 |

### Core test breakdown

| Category | Count |
|----------|-------|
| core_controls | 21 |
| core_sample | 27 |
| core_fe | 4 |
| core_inference | 3 |
| core_funcform | 4 |
| core_method | 1 |

### Non-core breakdown

| Category | Count |
|----------|-------|
| noncore_heterogeneity | 4 |
| noncore_placebo | 1 |
| noncore_alt_treatment | 1 |
| invalid | 1 |

---

## Top 5 Most Suspicious Rows

### 1. `robust/placebo/shuffled_treatment` (INVALID)

**Critical issue**: This spec has coefficient = -0.02344, SE = 0.00871, p = 0.00714, R-squared = 0.2327 -- all **exactly identical** to the baseline. The treatment variable is labeled `treat_shuffled` but the results are numerically indistinguishable from the original treatment. This means the shuffle was not actually applied, or the shuffled variable was overwritten by the original. This spec provides no diagnostic value and is marked invalid.

### 2. `robust/heterogeneity/age_interaction` (treatment_var label mismatch)

The `treatment_var` column says `treat_x_old` (the interaction term), but the coefficient reported (-0.02271) corresponds to the **main effect** of `d_computer_2017_2007_n` in the coefficient_vector_json, not the interaction term (which has coefficient -0.00241, p=0.84). The reported coefficient is actually the main treatment effect in a model that includes the interaction. This is a labeling inconsistency. The same pattern applies to all four heterogeneity specs.

### 3. `robust/heterogeneity/gender_interaction` (treatment_var label mismatch)

Same issue as above. The `treatment_var` says `treat_x_male` but the coefficient (-0.03839) is for the main effect `d_computer_2017_2007_n`. The actual interaction coefficient is 0.02483 (p=0.08), which is the substantively interesting quantity for a heterogeneity test.

### 4. `robust/treatment/quintile_dummies` (non-core)

This spec uses quintile dummies for the treatment and reports the coefficient on `d_computer_q5 (highest)` = 0.019 (p=0.52). This is the only specification with a positive coefficient. The change from continuous to categorical treatment fundamentally changes the estimand (from marginal effect to group difference), making it not directly comparable to the baseline. Additionally, the positive sign is inconsistent with the paper's main claim, though it is not statistically significant.

### 5. `robust/sample/winsorize_1pct`

This spec reports coefficient, SE, p-value, N, and R-squared all **identical** to the baseline. While this is plausible if no additional observations are affected by the narrower 1% winsorization threshold compared to the 2.5% baseline, it is worth flagging. The identical results suggest the winsorization at 1% vs 2.5% made no difference to the data, which could indicate the implementation did not correctly apply a different threshold, or that the data simply has no observations between the 1% and 2.5% tails.

---

## Detailed Category Notes

### Controls variations (21 specs)

These include two types: (1) leave-one-out drops of individual controls (9 specs), and (2) progressive addition of controls from none to full (10 specs), plus the no-controls specification. All use the same outcome, treatment, FE, and clustering. These are well-constructed robustness checks. Note that `robust/control/add_age` is identical to baseline since age is the last control added in the progression.

### Sample restrictions (27 specs)

These include: (1) dropping individual job-loss years (11 specs), (2) dropping individual survey years (5 specs), (3) demographic subsamples by age, gender, and education (7 specs), (4) full-time workers only (1 spec), and (5) alternative winsorization thresholds (3 specs). The demographic subsamples (age_young, age_old, male, female, edu_hs_only, edu_some_college, edu_college_plus) are classified as core_sample rather than heterogeneity because they run the same regression on subsamples rather than using interaction terms. They test whether the main effect holds across different populations.

### Fixed effects variations (4 specs)

These test different FE structures: no FE, year only, year-of-job-loss only, and baseline FE plus state FE. All preserve the estimand.

### Inference variations (3 specs)

These change clustering: by year of job loss, by state, and heteroskedasticity-robust SEs (no clustering). All produce the same point estimate (as expected), only SEs differ.

### Functional form (4 specs)

These include non-winsorized log earnings, earnings in levels (dollars), non-normalized treatment, and IHS transformation of the outcome. All test the same claim with different measurement/scaling.

### Heterogeneity (4 specs, non-core)

These add interaction terms (treatment x age, education, gender, unemployment duration). They are classified as non-core because the primary purpose is testing whether the effect varies by subgroup, which is a different claim from the baseline. Additionally, all four have a labeling issue where treatment_var references the interaction term but the coefficient is actually the main effect.

---

## Recommendations for the Spec-Search Script

1. **Fix the shuffled_treatment placebo**: The shuffle was not applied correctly. The script should verify that the shuffled treatment variable has a different distribution from the original before running the regression. Consider setting a random seed and confirming the shuffled variable is not identical to the original.

2. **Fix heterogeneity spec treatment_var labels**: For interaction models, the `treatment_var` field should either (a) report the main treatment variable name (since that is what the coefficient corresponds to) or (b) if the intent is to report the interaction coefficient, extract the correct coefficient from the interaction term rather than the main effect.

3. **Verify winsorization implementation**: Confirm that winsorize_1pct actually applies 1% winsorization (not 2.5%). The identical results are plausible but should be verified.

4. **Consider reporting both main effect and interaction in heterogeneity models**: For interaction specifications, it would be more informative to report the interaction coefficient (which tests heterogeneity directly) rather than the main effect, or to report both.

5. **Quintile treatment spec**: The treatment variable label includes parenthetical text `d_computer_q5 (highest)` which may cause parsing issues. Consider standardizing variable names to avoid spaces and parentheses.
