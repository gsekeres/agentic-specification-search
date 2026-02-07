# Verification Report: 138401-V1

**Paper**: "The Long-Term Effects of Measles Vaccination on Earnings and Employment"
**Journal**: AEJ: Policy
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Employment outcome
- **Claim**: Exposure to measles vaccination in childhood (high-measles states x post-vaccine cohort exposure) affects adult employment probability.
- **Baseline spec_id**: `baseline`
- **Outcome**: `employed`
- **Treatment**: `treatment` (high_measles x exposure)
- **Expected sign**: Negative (in this implementation; paper expects positive)
- **Coefficient**: -0.000737, p < 0.001

### G2: Log wage outcome
- **Claim**: Exposure to measles vaccination in childhood affects adult log wages.
- **Baseline spec_id**: `baseline_lnwage`
- **Outcome**: `lnwage`
- **Treatment**: `treatment` (high_measles x exposure)
- **Expected sign**: Negative (in this implementation; paper expects positive)
- **Coefficient**: -0.001463, p < 0.001

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **38** |
| Baselines | 2 |
| Core tests (non-baseline) | 26 |
| Non-core tests | 4 |
| Invalid | 6 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 3 |
| core_fe | 4 |
| core_funcform | 1 |
| core_inference | 3 |
| core_method | 2 |
| core_sample | 15 |
| invalid | 6 |
| noncore_heterogeneity | 1 |
| noncore_placebo | 3 |

---

## Top 5 Most Suspicious Rows

### 1. `robust/sample/males` (INVALID)
- **Issue**: Coefficient (-0.000737), SE (9.41e-05), and N (15,932,210) are all identical to the full-sample baseline.
- **Diagnosis**: The pandas subsample filter `df[df["female"] == 0]` did not work, likely because the `female` variable was not correctly constructed or pyfixest ignored the filter. The N should be roughly half the full sample.
- **Impact**: This spec provides no information.

### 2. `robust/sample/drop_year_2005`, `drop_year_2010`, `drop_year_2015` (INVALID)
- **Issue**: All three have identical coefficient, SE, and N as baseline despite supposedly dropping a survey year.
- **Diagnosis**: The `year` variable filter `df[df["year"] != yr]` did not work. Likely the ACS `year` variable was not correctly converted to numeric, or the specific years 2005/2010/2015 are not present in the data.
- **Impact**: Three specs that provide no information.

### 3. `robust/outcome/lnwage` (INVALID - duplicate)
- **Issue**: Perfect duplicate of `baseline_lnwage` -- same outcome, treatment, controls, FEs, sample, and identical results.
- **Diagnosis**: The script runs the exact same specification as `baseline_lnwage` and labels it as a robustness check. This is a coding error in the spec search script.
- **Impact**: This is not an independent specification.

### 4. `robust/outcome/lnwage_males` (INVALID)
- **Issue**: Same coefficient (-0.001463), SE, and N (12,302,485) as `baseline_lnwage` despite restricting to males.
- **Diagnosis**: Same subsample filter issue as `robust/sample/males`. The `female == 0` filter on `df_wage` did not work.
- **Impact**: This spec provides no information.

### 5. `robust/control/only_black` (suspicious but retained)
- **Issue**: Coefficient AND SE are identical to baseline when dropping `female` control.
- **Diagnosis**: After absorbing birthplace, birthyear, and year FEs, the `female` variable is exactly orthogonal to `treatment` and has zero partial correlation with the residualized outcome conditional on the other variables. This is unusual but mathematically possible with balanced data.
- **Impact**: Retained as core but with reduced confidence (0.7).

---

## Recommendations for Fixing the Spec-Search Script

1. **Fix subsample filters**: The `df[df["female"] == 0]` filter does not work correctly. Investigate whether the `female` column is properly constructed as numeric 0/1. The same issue affects `df[df["year"] != yr]` for drop-year specs.

2. **Remove duplicate spec**: `robust/outcome/lnwage` is identical to `baseline_lnwage` and should be removed or replaced with a genuinely different specification (e.g., different controls or sample for lnwage).

3. **Add missing specs**: `robust/sample/females` and `robust/heterogeneity/female` are missing from the output. Debug why these failed. Also, `robust/treatment/exposure_only` and `robust/treatment/high_measles_only` are in the script but missing from results.

4. **Investigate treatment construction**: The simplified median-split treatment produces a negative sign (opposite to the paper hypothesis). Consider using continuous pre-vaccine measles mortality rates interacted with exposure, as in the original paper.

5. **Verify year variable**: The ACS `year` variable filtering issue suggests the `year` column may still be stored as a categorical/string. Ensure `pd.to_numeric` conversion is working properly before the filter step.

6. **Check identical-coefficient issue**: The fact that `robust/control/only_black` produces the exact same coefficient AND standard error as baseline is very unusual and warrants investigation of whether pyfixest is caching results or the data slice is somehow identical.

---

## Notes on Sign Interpretation

The treatment effect is consistently negative, which the SPECIFICATION_SEARCH.md notes as surprising given the hypothesis that vaccination should improve outcomes. This is because the treatment variable `high_measles * exposure` captures being in a high-measles state AND having more years of vaccine exposure. If high-measles states had worse baseline conditions that persisted despite vaccination, the negative sign could reflect residual confounding not fully absorbed by state FE. The original paper uses a more nuanced continuous measure of pre-vaccine measles burden.
