# Specification Surface Review: 157781-V1

**Paper**: "Rebel on the Canal: Disrupted Trade Access and Social Conflict in China, 1650-1911"

**Reviewer**: Verifier agent (prompt 04)

**Date**: 2026-02-24

---

## Summary of Baseline Groups

**G1: Canal closure and rebellion incidence** -- Single baseline group, correctly scoped.

- **Outcome**: asinh(rebellion onset per 1600 population) = `ashonset_cntypop1600`
- **Treatment**: `interaction1` = `alongcanal * reform` (Along Canal x Post)
- **Estimand**: DiD ATT of canal closure on rebellion incidence
- **Population**: 575 counties in 6 provinces, 1650-1911
- **Preferred spec**: Table 3, Col 4 (full FE, no time-varying controls)

The single baseline group is appropriate. The paper has one main claim tested across all tables.

---

## Changes Made

### 1. Removed redundant `rc/controls/progression/*` entries

The original surface listed 5 `rc/controls/progression/col*` spec IDs that exactly duplicate the 5 baseline specs (Table 3 Cols 1-5). Cols 1-4 are already listed as `baseline_specs`, and Col 5 is equivalent to `rc/controls/sets/full_controls`. Running these as separate RC specs would double-count the baseline FE progression. **Removed all 5 progression entries.**

### 2. Moved treatment alternatives from `rc/form/treatment/*` to `explore/treatment/*`

The original surface listed alternative treatment definitions (canal density, canal town share, distance to canal) as `rc/form/treatment/*`. However, changing the treatment variable from a binary indicator (`alongcanal * reform`) to a continuous intensity measure (`canal_den * reform`, `canaltown_after`, `distance_canal * reform`) changes the treatment concept and thus the claim object. Per the CLAIM_GROUPING rules, these belong in exploration. **Moved to `explore/treatment/*`.**

### 3. Added missing outcome transformations from Table A3

The code in `tableA3.do` reveals 5 alternative outcome constructions:
- `ashonset_cntypop1820` (1820 population normalization) -- was missing
- `ashonset_cntypop` (time-varying yearly population) -- was present
- `ashonset_km2` (area normalization) -- was present
- `ashonset_num` (asinh of raw count, no normalization) -- was missing
- `onset_cntypop1600` (raw per-capita, no asinh) -- was present but mislabeled as `log_onset_pop1600`

**Added `ashonset_cntypop1820` and `ashonset_num`. Corrected label for `onset_cntypop1600_raw`.**

### 4. Added missing sample restrictions from Table A2

Table A2 reveals a systematic 5x6 grid of sample restrictions:
- **Spatial**: within prefecture, within 100/150/200km of canal, all counties
- **Temporal**: 50/100/150/200-year windows centered around canal closure, full period

The original surface had only `period_1700_1911` and `period_1750_1911` (invented, not matching Table A2 windows). **Replaced with the actual Table A2 temporal windows and added all spatial restrictions.**

### 5. Added explicit explore_universe section

Created an `explore_universe` block to house:
- Treatment alternatives (Table 4)
- North/South heterogeneity triple-diff (Table 5)
- Opium/Taiping interaction heterogeneity (Table 7 Panel B)
- State capacity mechanism channels (Table A4)

### 6. Updated design_audit

- Separated `fe_structure_baseline` (preferred Col 4 full FE) from `fe_structure_minimal` (minimal Col 1 FE) for clarity.
- Removed the ambiguous `additional_fe_revealed` field, since the full FE structure is now recorded directly.

### 7. Added Table 6 placebo tests to diagnostics_plan

Table 6 tests placebo treatments (along Yangtze, Yellow River, Coast, courier routes). These are falsification diagnostics, not estimates of the baseline claim. Added to `diagnostics_plan`.

### 8. Made controls policy explicit

Added `mandatory_controls` (empty) and `optional_pool` (all 15 controls) to `constraints` to make the control-set policy fully explicit. The preferred spec uses zero controls.

---

## Key Constraints and Linkage Rules

- **No linked adjustment**: OLS with absorbed FE; no bundled estimator components.
- **FE structure is the primary robustness dimension**: The paper's main variation is adding progressively richer FE (county, year, pre-rebellion trends, province-year, prefecture trends). This is captured via the 5 baseline specs and the `rc/fe/*` entries.
- **Sharp single-date treatment**: No staggered adoption, so TWFE is the correct and only needed DiD estimator. No need for Callaway-Sant'Anna, Sun-Abraham, or other heterogeneity-robust estimators.
- **Controls are all post-reform interactions**: Every control in `$ctrls` is a time-invariant characteristic interacted with `reform`. They enter as level shifters in the post period only.

---

## Budget and Sampling Assessment

**Updated count**:
- 5 baseline specs (Table 3 Cols 1-5)
- 15 LOO control drops (from Col 5 baseline)
- 4 control subset groupings (none, climate, geography, agriculture)
- 4 FE variations (drop/add individual FE layers from preferred spec)
- 6 sample restrictions (2 from Table 7A + 4 spatial/temporal from Table A2)
- 6 outcome form variations (5 from Table A3 + binary onset)
- 1 outlier trim

**Total core RC specs**: ~41, well within the 100-spec budget. Full enumeration is feasible; no sampling needed.

---

## What's Missing

1. **Prefecture-level aggregation** (Table A2 Col 6): The paper also runs prefecture-level regressions, aggregating county data to the prefecture level. This changes the unit of analysis (and hence the population concept) and is more naturally `explore/aggregation/prefecture_level`. Not added to core because it changes the claim object.

2. **Grain price mechanism** (additional appendix tables): Some appendix analyses examine grain price effects as a mechanism. These are mechanism/exploration analyses and correctly excluded from core.

3. **Pre-reform rebellion measure sensitivity**: The `ashprerebels` variable is constructed from pre-reform rebellion counts and used in FE interactions. Its construction could be varied (e.g., alternative pre-reform windows), but the paper does not reveal this variation, so we do not add it.

---

## Data Note

The analysis dataset `Data/Final/rebellion.dta` does not exist in the extracted package. It must be generated by running `Program/Clean/clean.do`. Only raw data files are present in `Data/Raw/`. The runner script will need to either (a) run the cleaning code first, or (b) have the cleaned dataset pre-generated.

---

## Approved to Run

**Status: APPROVED**

The surface is well-specified with a clear single baseline group, appropriate DiD design classification, and a manageable core universe. The corrections above address redundancies, mislabeled treatment alternatives, and missing revealed variations. No blocking issues remain.
