# Specification Surface: 138922-V1

**Paper**: Marcus, Siedler & Ziebarth, "The Long-Run Effects of Sports Club Vouchers for Primary School Children," *AEJ: Economic Policy*

**Created**: 2026-02-24

---

## 1. Baseline Groups

### G1: Effect of C2SC voucher on sports club membership

**Claim object**:
- **Outcome**: Sports club membership (binary)
- **Treatment**: C2SC (Come to the Sports Club) voucher program in Saxony, available from school year 2008/09
- **Estimand**: ATT of voucher availability on sports club membership
- **Population**: Primary school children in Saxony (treatment) vs Brandenburg and Thuringia (control states), examination cohorts 2006/07 through 2010/11

**Design**: Difference-in-differences using repeated cross-sections of school health examination cohorts. The treatment state is Saxony (bula_3rd=13), with Brandenburg (4) and Thuringia (16) as controls. The voucher program was introduced for the 2008/09 cohort onward, so pre-treatment cohorts are 2006/07 and 2007/08, post-treatment are 2008/09 through 2010/11.

**Baseline specification (Table 2, Column 3)**:
```
reg sportsclub treat i.year_3rd i.bula_3rd i.cityno if sample & yrs, vce(cluster cityno)
```
- `treat = tbula_3rd * tcoh` (Saxony x post-2008 interaction)
- Fixed effects: year, state, city
- Clustering: city (cityno), approximately 93 clusters
- Sample: `inlist(bula_3rd, 4, 13, 16) & target == 1 & nonmiss == 1` with `inrange(year_3rd, 2006, 2010)`
- N ~ 13,333

---

## 2. What is Included

### Baseline variants (3 specs)
- **baseline**: Table 2 Col 3 (year + state + city FE, cluster cityno) -- the paper's preferred specification
- **baseline__table2_col1**: Table 2 Col 1 (group dummies only: tbula_3rd + tcoh)
- **baseline__table2_col2**: Table 2 Col 2 (year + state FE only, no city FE)

### RC axes

**Controls (10 specs)**:
- `rc/controls/add/individual_controls`: Add all 9 individual controls (Table 5, s12): female, siblings, born_germany, parent_nongermany, newspaper, art_at_home, academictrack, sportsclub_4_7, music_4_7
- `rc/controls/loo/drop_*`: 9 LOO specs dropping each control one at a time from the full control set

**Sample window (4 specs)**:
- `rc/sample/window/drop_first_cohort`: 2007-2010 (Table 4, r1)
- `rc/sample/window/extended_window_2000_2010`: 2000-2010 (Table 4, r2)
- `rc/sample/window/extended_window_2006_2011`: 2006-2011 (Table 4, r3)
- `rc/sample/window/shortened_window_2006_2009`: 2006-2009 (Table 4, r4)

**Sample composition (5 specs)**:
- `rc/sample/composition/drop_brandenburg`: Drop Brandenburg from controls (Table 4, r5 uses Sachsen+Thuringia only)
- `rc/sample/composition/drop_thuringia`: Drop Thuringia from controls (Table 4, r6 uses Sachsen+Brandenburg only)
- `rc/sample/composition/drop_nonmiss_restriction`: Use full target population without nonmiss==1 restriction (Table 4, r7)
- `rc/sample/composition/data_quality_filter`: Additional data quality filters (Table 5, s8)
- `rc/sample/composition/no_sibling_contamination`: Exclude sibling-contaminated cohorts (Table 5, s9)
- `rc/sample/composition/no_older_siblings`: Restrict to children without older siblings (Table 5, s9b)

**Fixed effects (3 specs)**:
- `rc/fe/add_controls_as_covariates`: Add individual controls as covariates (keeping FE structure)
- `rc/fe/drop_cityno_fe`: Drop city FE (year + state FE only)
- `rc/fe/drop_bula_fe`: Drop state FE (year + city FE only, state absorbed by city)

**Data construction / treatment definition (3 specs)**:
- `rc/data/treatment_definition_v2`: Use alternative treatment coding treat_v2 (Table 5, s14)
- `rc/data/treatment_first_wave`: Use first-wave treatment (t_tcoh_1st) with first-wave FE (Table 5, s10)
- `rc/data/treatment_current_state`: Use current-state treatment (t_tcoh_bula) with current-state FE (Table 5, s11)

**Weights (1 spec)**:
- `rc/weights/survey_weights`: Use inverse probability weights (Table 5, s13)

**Alternative outcomes (5 specs)**:
- `rc/form/outcome/kommheard`: Knowledge of program
- `rc/form/outcome/kommgotten`: Received voucher
- `rc/form/outcome/kommused`: Redeemed voucher
- `rc/form/outcome/sport_hrs`: Weekly hours of sport
- `rc/form/outcome/oweight`: Overweight indicator

**Joint robustness (14 specs)**:
Combinations of controls with other RC axes to assess stability of findings with individual-level covariates.

### Total planned core specs: ~52

---

## 3. What is Excluded

- **Heterogeneity analyses** (Table B1, Figure 2): These are `explore/*` objects, not core robustness
- **Age-based DiD** (Table B2): Different design (panel of ages within individuals); not preserving the same estimand
- **Synthetic control** (Table 7): Different estimator family; would be explore/design variant
- **Power calculations** (Table B6-B7): Diagnostic
- **Parent-reported outcomes** (Table 6): Different data source; could be explore
- **Pre-trend tests**: Not included as specs (would be `diag/*`)

---

## 4. Inference Plan

**Canonical**: CRV1 clustered at cityno (93 clusters) -- matches the paper's baseline
**Variants**:
1. HC1 (robust, no clustering) -- Table B5 column 1
2. Cluster at state level (bula_3rd, only 3 clusters -- very aggressive) -- Table B5 column 4
3. Cluster at cohort level -- Table B5 column 5

---

## 5. Constraints

- **Controls count**: min=0 (baseline), max=9 (full individual controls from Table 5, s12)
- **Linked adjustment**: false (no bundled estimator)
- **Mandatory FE**: Year FE present in all specs except Table 2 Col 1

---

## 6. Budget and Sampling

Full enumeration is feasible. The control pool is small (9 binary controls), so LOO from the full set is tractable. No random sampling needed. Target ~52 specifications plus inference variants.
