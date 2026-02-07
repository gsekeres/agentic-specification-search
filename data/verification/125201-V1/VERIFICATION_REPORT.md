# Verification Report: 125201-V1

## Paper: Temperature and Mortality in Mexico
**Journal**: AEJ-Applied
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Hot days increase overall mortality in Mexican municipalities
- **Baseline spec_id**: baseline
- **Outcome**: death_rate_scaled
- **Treatment**: hot_days (days with max temp >= 32C)
- **Fixed effects**: muni_id + year_month
- **Controls**: cold_days, precipitation
- **Clustering**: muni_id
- **Coefficient**: 0.0058, p < 0.001
- **Expected sign**: Positive

---

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **64** |
| Core specifications | 43 |
| Non-core specifications | 21 |
| Invalid | 0 |
| Unclear | 0 |

### Core Test Breakdown

| Category | Count |
|----------|-------|
| core_controls | 9 |
| core_sample | 20 |
| core_inference | 4 |
| core_fe | 5 |
| core_funcform | 3 |
| core_method | 2 |
| **Core total** | **43** |

Note: The baseline itself is counted in core_controls (is_baseline=1). So 42 non-baseline core tests plus 1 baseline = 43 total core rows.

### Non-Core Breakdown

| Category | Count |
|----------|-------|
| noncore_heterogeneity | 14 |
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 3 |
| noncore_placebo | 2 |
| noncore_diagnostic | 0 |
| **Non-core total** | **21** |

Note: Circulatory and respiratory death rates are classified as noncore_alt_outcome (mechanism tests). 14 heterogeneity specs cover demographic subgroups (male, female, elderly, children, working-age, four age brackets), geographic subgroups (hot states, cold states, large/small municipalities), and one interaction specification.

---

## Top 5 Most Suspicious Rows

1. **robust/het/interaction_population** (spec_id: robust/het/interaction_population)
   - This spec adds an interaction term (hot_days * pop_scaled) but still reports the main hot_days coefficient. It could be argued this is a controls variation. However, the purpose is to test heterogeneity by population size, and the main effect is no longer cleanly interpretable as the overall average effect (it represents the effect at pop_scaled=0). Classified as noncore_heterogeneity with confidence 0.80.

2. **robust/control/all_temp_bins** (spec_id: robust/control/all_temp_bins)
   - The treatment variable changes from hot_days to scale_32_p_m. This appears to be the same heat concept (days above 32C) re-parameterized with full temperature bin controls. Classified as core_controls with confidence 0.80 because the treatment variable name differs but the conceptual estimand is preserved.

3. **robust/funcform/quadratic_temp** (spec_id: robust/funcform/quadratic_temp)
   - Treatment is MEAN_TEMP_m (mean temperature) rather than hot_days. Even though it is listed under functional form, using mean temperature as treatment changes the causal object from "extreme heat days" to "average temperature level." Classified as noncore_alt_treatment with confidence 0.85.

4. **robust/placebo/lead_temp** (spec_id: robust/placebo/lead_temp)
   - Uses hot_days_lead1 (future temperature) as treatment. The coefficient is positive and marginally significant (p=0.026), which is somewhat surprising for a placebo test and could indicate serial correlation or displacement effects. Properly classified as noncore_placebo.

5. **panel/method/first_diff** (spec_id: panel/method/first_diff)
   - Both outcome (d_death_rate) and treatment (d_hot_days) are first-differenced. This is a legitimate alternative estimation method for the same claim, but the variable names differ from baseline. Classified as core_method with confidence 0.85.

---

## Classification Notes

### Heterogeneity vs. Sample Restrictions
Several specifications could be argued to be either heterogeneity or sample restrictions. The key distinction used here:
- **Sample restrictions** (core): Specs that drop observations to test sensitivity of the *overall* claim (e.g., drop a year, drop a state, winsorize). The outcome variable remains death_rate_scaled and treatment remains hot_days.
- **Heterogeneity** (non-core): Specs that restrict to a subgroup OR use a subgroup-specific outcome to test whether the effect differs across groups (e.g., male vs female death rates, elderly death rates, hot vs cold states). These answer a different question than the baseline.

### Geographic Subsamples
hot_states and cold_states are classified as heterogeneity because they split the sample to compare effect sizes across climate zones, not to test sensitivity of the overall finding. In contrast, drop_state_1 through drop_state_5 are classified as sample restrictions because they test whether any single state is driving the result.

### Cause-of-Death Outcomes
Circulatory and respiratory death rates are classified as noncore_alt_outcome rather than noncore_heterogeneity because they represent different outcome concepts (cause-specific mortality) rather than demographic subgroups of the overall death rate.

### Alternative Treatment Measurements
- extreme_heat_binary is classified as core_method because it is an alternative measurement of the same heat exposure concept (binary indicator for extreme heat vs continuous hot days count).
- cold_days and mean_temp are classified as noncore_alt_treatment because they represent fundamentally different temperature exposures.
- quadratic_temp uses MEAN_TEMP_m as treatment, changing the causal object despite being filed under functional form.

---

## Recommendations

1. **No major issues with baseline definition.** The baseline is clearly identified and well-specified.

2. **Consider separating het/interaction_population.** The interaction specification could be re-run without the interaction to produce a cleaner comparison, or the total effect could be computed at the sample mean of population.

3. **The lead_temp placebo is marginally significant (p=0.026).** This may warrant investigation in the original paper context -- it could indicate serial correlation or displacement effects. Not a data quality issue but worth noting.

4. **The all_temp_bins treatment variable rename (scale_32_p_m vs hot_days) should be documented.** If this is the same variable under a different name, the spec search code should standardize it.

5. **Overall quality is high.** All 64 specifications have valid coefficients and standard errors. No missing or non-finite values detected. The specification tree is well-organized and categories are clearly delineated.
