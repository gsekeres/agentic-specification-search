# Specification Surface: 114701-V1

## Paper Overview
- **Title**: Changing How Literacy is Taught: Evidence on Synthetic Phonics (Machin, McNally, and Viarengo, 2018)
- **Design**: Difference-in-differences with school fixed effects
- **Data**: National Pupil Database (confidential; not included in package, only do-files provided)
- **Key context**: Studies two literacy interventions (EDRp pilot and CLLD Phase 1) in English primary schools using treatment-by-cohort DiD with school FE. Outcomes are standardized test scores at ages 5 (FSP), 7 (KS1), and 11 (KS2).

## Baseline Groups

### G1: EDRp Program Effects (Table 3 Panel A)

**Claim object**: The EDRp synthetic phonics program improves student literacy scores at multiple assessment stages.

**Baseline specification**:
- Formula: `stdcll ~ bc_99 + bc_00 + bc_01 + yy2-yy4 + 30_controls | school_FE`
- Outcome: `stdcll` (standardized communication/language/literacy score, FSP)
- Treatment: Treatment-by-year interactions (bc_99, bc_00, bc_01 for 3 cohorts)
- Controls: 15 individual-level (fsm, male, ethnicity dummies, language, SEN) + 15 school-mean counterparts
- FE: School (urn) via areg
- Clustering: School (urn)

**Additional baseline-like rows**:
- KS1 reading (ks1_readps) -- age 7
- KS2 reading (stdread) -- age 11 (long-run persistence)

### G2: CLLD Program Effects (Table 3 Panel B)

**Claim object**: The CLLD program improves student literacy scores.

**Baseline specification**:
- Same structure as G1 but with 4 treatment-year indicators (bc_99-bc_02) and CLLD treatment schools.

## RC Axes Included

### Controls
- **Leave-one-out**: Drop key individual controls (fsm, male, sen_statement, emt_r, absent)
- **Block removal**: Drop all ethnicity (individual), drop all school means, drop language controls
- **Standard sets**: Individual controls only (drop school means), school means only, minimal (fsm + male + emt_r)
- **Random subsets**: 15-20 draws from the 30-variable control pool

### Sample restrictions
- Outlier trimming on standardized scores
- EMT only (native English speakers)
- EAL only (English as additional language)
- FSM only (free school meals eligible)
- Non-FSM only

### Design variants
- Pilot-only vs Phase 1-only analysis
- Pooled pilot and phase 1

### Fixed effects / clustering
- School + year dummies (baseline)
- Two-way clustering at school-year

## What Is Excluded and Why

- **Heterogeneity by subgroup (Table 4)**: Treatment effects by speech nativity and FSM status. These are subgroup analyses, better as explore/* rather than core RC.
- **Tables 5-8**: Additional program comparisons and long-run analyses using similar methods but different populations/programs.
- **Figure 1 / Table A2**: Descriptive statistics and trends, not regression estimates.

## Budgets and Sampling

- **G1 max core specs**: 70
- **G2 max core specs**: 35
- **Max control subsets**: 20 (G1), 10 (G2)
- **Seed**: 114701
- **Sampling**: Block-based (4 blocks: core demographics, ethnicity, language, school means)

## Inference Plan

- **Canonical**: Cluster at school (urn)
- **Variants**: HC1 robust, two-way school-year clustering
