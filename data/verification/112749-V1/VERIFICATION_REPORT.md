# Verification Report: 112749-V1

## Paper: "When the Levee Breaks: Black Migration and Economic Development in the American South"

**Journal**: AER
**Method**: Panel Fixed Effects / Difference-in-Differences with continuous treatment intensity
**Core Hypothesis**: Flooded counties experienced larger declines in Black population share post-1927

---

## Baseline Groups

### G1: Effect of flood intensity on log Black population share
- **Baseline spec_ids**: `baseline`
- **Outcome**: `lnfrac_black` (log fraction Black)
- **Treatment**: `flood_intensity` (continuous flood intensity interacted with post-period)
- **Fixed Effects**: County FE, State-Year FE
- **Clustering**: County (fips)
- **Baseline coefficient**: 0.026 (SE = 0.054, p = 0.63)
- **Expected sign**: Positive (per coefficient convention in the reconstructed data)

**Note**: The baseline coefficient is small, positive, and statistically insignificant. This raises concerns that the reconstructed analysis may not faithfully replicate the paper's main table result. The original paper likely has a more precisely estimated effect, possibly using additional controls (geography x year interactions, New Deal controls) or different sample restrictions.

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **57** |
| Baseline | 1 |
| Core tests (non-baseline) | 30 |
| Non-core tests | 17 |
| Invalid | 8 |
| Unclear | 1 |

### Core Tests by Category

| Category | Count |
|----------|-------|
| core_sample | 21 |
| core_inference | 2 |
| core_fe | 1 |
| core_funcform | 3 |
| core_method | 4 |
| core_controls | 0 |

### Non-Core Tests by Category

| Category | Count |
|----------|-------|
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 1 |
| noncore_heterogeneity | 5 |
| noncore_placebo | 1 |
| noncore_diagnostic | 8 |

---

## Classification Details

### Core Specifications (31 total including baseline)

**Sample restrictions (21)**: The largest group of core specs. These include leave-one-out state drops (states 1, 5, 12, 13, 20, 22, 28, 37, 45, 47), leave-one-out year drops (1930, 1940, 1950, 1960, 1970), early and late period restrictions, flooded-only subsample, and winsorization at 1%, 5%, and 10%. All maintain the same outcome, treatment, FE structure, and clustering as the baseline.

**Inference variations (2)**: HC1 robust SEs and state-level clustering. Both keep the identical point estimate (0.026) as the baseline but change the standard errors.

**FE structure (1)**: The County + Year FE (two-way) specification is classified as core because it preserves within-county identification while using a slightly different time-varying control structure than State-Year FE.

**Functional form (3)**: (a) IHS transform of Black share (`ihs_frac_black`), (b) frac_black in levels (classified under robust/outcome/frac_black), and (c) adding squared flood intensity. These preserve the same conceptual claim while varying the dependent variable transformation or treatment nonlinearity.

**Method (4)**: The baseline itself, population-weighted regression, Black-population-weighted regression, and long differences (1930-1970).

### Non-Core Specifications (17 total)

**Alternative outcomes (2)**: Log Black population and log total population measure different concepts than Black population share. Population outflows vs. share changes have different interpretations.

**Alternative treatment (1)**: Binary flood indicator changes the causal object from intensity-response to extensive margin.

**Heterogeneity (5)**: Interactions with plantation status, high intensity, Mississippi state, and subsamples by Black share (high/low). These test heterogeneity, not the main effect.

**Placebo (1)**: Random permutation of flood treatment assignment.

**Diagnostics (8)**: County-FE-only, Year-FE-only, No-FE, State+Year-FE specifications that omit critical identification structure. Four cross-sectional OLS regressions at individual years (1930, 1950, 1960, 1970) that estimate level associations rather than changes.

### Invalid Specifications (8)

All 8 invalid specs are control variation models (lagged DV variants) where the treatment coefficient was not extracted (null values). This includes `baseline_lagged_dv`, `robust/control/lagged_dv_only`, `robust/control/add_lag2`, `robust/control/add_lag3`, `robust/control/add_lag4`, `robust/control/drop_lag2`, `robust/control/drop_lag3`, and `robust/control/drop_lag4`.

### Unclear (1)

`robust/funcform/levels` appears to be identical to `robust/outcome/frac_black` (same coefficient 0.0801, same SE, same p-value). It is likely a duplicate entry with a different categorization label.

---

## Top 5 Most Suspicious Rows

1. **baseline_lagged_dv and all lagged-DV control specs (8 rows)**: Treatment coefficient is null for all 8 specifications. The extraction script failed to capture the flood_intensity coefficient when lagged dependent variables were added as controls. This is a systematic extraction error that eliminates all control-variation robustness checks from the analysis.

2. **panel/fe/none, panel/fe/time, panel/fe/state_year**: These specifications lack county FE, which means they do not estimate within-county variation. The coefficients are very large (0.73-0.96 for year-only and no-FE; 0.73 for state-year) because they capture cross-county level differences, not the causal effect of flooding. These should not be interpreted as robustness of the DiD result.

3. **custom/cross_section_* (4 rows)**: Cross-sectional OLS at individual years (1930, 1950, 1960, 1970) with only state FE. The large positive coefficients (~0.74-0.78) reflect that counties in the flood zone had higher Black population shares in levels, which is a pre-existing geographic pattern, not a causal effect of the flood.

4. **robust/outcome/frac_black vs robust/funcform/levels**: These produce identical results (coef=0.0801, SE=0.0198, p=5.94e-05) but are recorded as different specification types. This is a data integrity issue -- either one should be removed or they should be flagged as duplicates.

5. **robust/heterogeneity/high_intensity**: The coefficient (0.0257) and SE (0.0537) are identical to the baseline. This suggests the high-intensity interaction term was not properly implemented, or the "high" indicator has no within-sample variation.

---

## Recommendations for Fixing the Specification Search Script

1. **Fix treatment coefficient extraction for lagged-DV models**: The 8 null-coefficient specs represent a significant loss of robustness information. The extraction function should specifically look for the `f_int_1930` (or equivalent treatment) variable coefficient in the regression output, even when additional controls are added.

2. **Validate the baseline against the published paper**: The baseline coefficient (0.026, p=0.63) appears weak relative to what a published AER paper would report. The script should ensure it matches the paper's main table. The original paper likely uses more granular controls (geography x year interactions), different sample restrictions (e.g., the ">10% Black 1920" sample), or a different specification of the treatment x post interaction.

3. **Remove or merge duplicate specifications**: `robust/outcome/frac_black` and `robust/funcform/levels` are the same regression. Keep one.

4. **Reclassify FE-only and cross-section specs**: The FE structure variants that omit county FE (year-only, no-FE, state-year) and the cross-sectional OLS specs do not test the paper's within-county DiD claim. They should be explicitly labeled as diagnostics in the spec_tree_path rather than appearing alongside valid robustness checks.

5. **Add genuine control-variation specs**: The paper likely varies geography x year controls, New Deal controls, and other county-level characteristics. These are missing from the current specification set because all such attempts produced null treatment coefficients.
