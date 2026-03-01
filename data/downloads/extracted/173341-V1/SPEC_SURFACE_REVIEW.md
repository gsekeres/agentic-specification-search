# Specification Surface Review: 173341-V1

**Paper**: "Vulnerability and Clientelism" (Bobonis, Gertler, Gonzalez-Navarro, and Nichter, AER 2022)

**Reviewer**: Verifier agent (prompt 04)

**Date**: 2026-02-24

---

## Summary of Baseline Groups

### G1: Cisterns treatment and rainfall on clientelist requests

- **Outcome**: `ask_private_stacked` (binary: requested any private good from politician)
- **Treatment**: `treatment` (cistern receipt, randomized) + `rainfall_std_stacked` (standardized rainfall shock, natural variation)
- **Estimand**: Joint ITT effects on private good requests
- **Population**: Individuals in semi-arid NE Brazil municipalities, stacked 2012+2013 data
- **Baseline specs**: Table 3 Col 3 (treatment + rainfall) and Col 4 (adds interaction)

### G2: Electoral outcomes (voting section level)

- **Outcome**: `incumbent_votes_section` (votes for incumbent mayor)
- **Treatment**: `tot_treat_by_section_2` (section-level treatment share, rescaled from household randomization)
- **Estimand**: ITT effect of treatment exposure on incumbent vote share
- **Population**: Voting sections in 21 municipalities where incumbent mayor ran for re-election
- **Baseline spec**: Table 4 Col 1

**Why two groups**: G1 and G2 differ in outcome concept (clientelist behavior vs electoral behavior), unit of analysis (individual vs voting section), treatment definition (household-level binary vs section-level share), and data source. These are fundamentally different claim objects requiring separate baseline groups.

---

## Changes Made

### 1. Fixed inconsistent baseline_spec_ids for G1

The original surface listed `baseline__table3_col4` as the sole baseline_spec_id but only had Table 3 Col 3 in the `baseline_specs` array. Since the paper presents both Col 3 (treatment + rainfall) and Col 4 (adds interaction) as key specifications, both are now listed as baselines. **Added Col 4 to `baseline_specs` and both Col 3 and Col 4 to `baseline_spec_ids`.**

### 2. Fixed inconsistent baseline_spec_ids for G2

The original surface listed `baseline__table4_col2` as the baseline_spec_id, but Col 2 uses the broader 39-municipality sample (a robustness variant). The actual preferred specification is Col 1 (21 municipalities where incumbent mayor ran). **Changed `baseline_spec_ids` to `baseline__table4_col1`.** The 39-municipality sample remains as `rc/sample/subvariant/broad_incumbency_sample`.

### 3. Moved `ask_public` from RC to diagnostics/exploration

The original surface listed `rc/form/outcome/ask_public` as a core RC spec. However, the paper explicitly frames public goods requests (Table 3 Col 8) as a placebo test: if the mechanism is clientelist-specific, cisterns should not affect requests for public goods. This is a diagnostic/falsification exercise, not an estimand-preserving robustness check. **Moved to `explore/outcome/ask_public` and added corresponding diagnostic entry.**

### 4. Moved `askrec_private` from RC to exploration

`askrec_private_stacked` (request AND receive private good) changes the outcome concept from "requesting" to "requesting and receiving." It only appears in the heterogeneity Table 5 (Col 5), not in the main Table 3 results. **Moved to `explore/outcome/askrec_private`.**

### 5. Moved G2 alternative electoral outcomes to exploration

The original surface listed `challenger_votes`, `turnout`, and `blank_null_votes` as `rc/form/outcome/*` for G2. These change the outcome concept from incumbent votes to different electoral measures. **Moved to `explore/outcome/*`.** Also added the Table A8 outcomes (PT votes, coalition votes, right-leaning votes) to the exploration universe.

### 6. Moved household-level aggregation to exploration

`rc/data/level/household_level` changes the unit of analysis from individual to household, altering the population concept. **Moved to `explore/data_level/household_level`.**

### 7. Removed redundant `rc/controls/progression/*` entries

The original surface listed `rc/controls/progression/bivariate`, `rc/controls/progression/treatment_only`, `rc/controls/progression/treatment_rainfall`, and `rc/controls/progression/treatment_rainfall_interaction`. These are not control progressions -- they change which treatment variables are included (Cols 1-4 of Table 3). Cols 1-2 (cistern only, rainfall only) are separate treatment model specifications, while Cols 3-4 are the baselines. **Replaced with `rc/form/treatment/cisterns_only` and `rc/form/treatment/rainfall_only` to capture the treatment-specification dimension accurately.**

### 8. Added explicit explore_universe blocks

Created `explore_universe` sections for both baseline groups documenting what is excluded from core and why.

### 9. Made controls bundling policy explicit

Table A7 shows that engagement controls are added in bundles of (level + treatment_interaction + rainfall_interaction). The constraints note now explicitly states that these must be added as bundles, not componentwise.

### 10. Expanded diagnostics_plan

Added Table A6 (clientelism marker balance with treatment), Table 3 Col 8 (public goods placebo), and Table A3 (electoral balance) to the diagnostics plans for the appropriate baseline groups.

### 11. Reduced G2 budget

The original surface set `max_specs_core_total: 20` for G2, but after moving alternative outcomes to exploration, only ~6 core specs remain. **Reduced to 15** to remain realistic while leaving headroom.

---

## Key Constraints and Linkage Rules

### G1 constraints
- **No linked adjustment**: Simple OLS with absorbed municipality FE.
- **Treatment specification is the main variation dimension**: The paper reveals 6 different treatment parameterizations across Table 3 columns (cisterns only, rainfall only, joint, joint+interaction, cisterns by year, rainfall by year). All use the same outcome, FE, and clustering.
- **Engagement controls bundling**: Table A7 controls (mem_assoc, pres_assoc, b_voted) must be added with their corresponding treatment and rainfall interactions as a bundle. The surface now records this explicitly.
- **Municipality FE is the stratum FE**: Since randomization was stratified by municipality, municipality FE is the natural baseline adjustment.

### G2 constraints
- **Rescaled regressors**: Treatment is rescaled from household-level assignment to voting-section-level share (`tot_treat_by_section_2`). This rescaling is design-inherent and should not be varied.
- **Location FE absorbs voting location heterogeneity**: The absorbed `location_id` FE is the natural adjustment for this ecological design.
- **Small number of clusters**: Only ~21 municipalities with incumbent re-election. Wild-cluster bootstrap is necessary for valid inference at the municipality level.

---

## Budget and Sampling Assessment

### G1
- 2 baseline specs (Table 3 Cols 3-4)
- 1 design variant (diff-in-means)
- 7 control set variants (none, mun FE only, mun FE + year, 4 engagement bundles)
- 5 treatment form variants (cisterns only, rainfall only, by-year cisterns, by-year rainfall, interaction)
- 2 year-split sample variants (2012 only, 2013 only)
- 1 outcome form variant (private excluding water)
- 2 FE variants (no mun FE, mun FE only)
- 1 outlier trim
- **Total: ~21 core specs, well within 50-spec budget.**

### G2
- 1 baseline spec (Table 4 Col 1)
- 1 design variant (diff-in-means)
- 1 sample variant (39-municipality broad incumbency)
- 2 control drops
- 1 FE variant
- **Total: ~6 core specs, well within 15-spec budget.**

Full enumeration is feasible for both groups. No random sampling needed.

---

## What's Missing

1. **Table 2 vulnerability outcomes**: Correctly excluded as mechanism/intermediate. These show cisterns reduce CES-D depression, improve health, food security -- they are the mediating pathway, not the main claim about clientelism.

2. **Table 5 heterogeneity by clientelist relationship**: Correctly classified as exploration. Interacting with `frequent_interactor` changes the estimand from ATE to CHET (conditional heterogeneous treatment effect).

3. **Table A4 wealth and treatment**: Balance/mechanism check showing cisterns do not differentially affect wealth composition. Diagnostic, not core.

4. **Table A9 politician responses and citizen beliefs**: Different outcome concepts (politician behavior, honesty/competence perceptions). Correctly excluded.

5. **Table A10 citizen preferences**: Risk/altruism/reciprocity/discount rate tasks. Different outcome concept. Correctly excluded.

6. **Compliance-adjusted (IV) estimates**: The paper does not report TOT/LATE estimates using compliance as an instrument. The ITT is the only estimand presented.

---

## Data Note

All four final data files exist in `data/final_data/`:
- `clientelism_individual_data.dta` (individual-level, single cross-section)
- `clientelism_individual_data_stacked.dta` (individual-level, stacked 2012+2013)
- `clientelism_household_data.dta` (household-level)
- `voting_data.dta` (voting-section-level)

The runner script should use the stacked individual data for G1 and the voting data for G2.

---

## Approved to Run

**Status: APPROVED**

The surface is well-specified with two clearly separated baseline groups reflecting the paper's dual empirical settings (survey clientelism + electoral outcomes). The corrections above address misclassified outcome variants (public goods placebo, alternative electoral outcomes), inconsistent baseline ID references, and the need for explicit treatment-specification tracking. No blocking issues remain.
