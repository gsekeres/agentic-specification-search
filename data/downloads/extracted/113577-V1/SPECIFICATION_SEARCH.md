# Specification Search: 113577-V1

## Paper
Jackson & Bruegmann (2009), "Teaching Students and Teaching Each Other:
The Importance of Peer Learning among Teachers"
American Economic Journal: Applied Economics, 1(4), 85-108.

## Data Note
**IMPORTANT**: The main microdata (Final_file_JAN09.dta) is restricted-use
North Carolina administrative education data. It is NOT provided in the
replication package. Only ccd_data.dta (CCD school characteristics) is available.
A synthetic dataset was constructed to match the variable structure described
in the Stata code files. Coefficient estimates are therefore synthetic and
should not be compared to the paper's reported values. The specification
pipeline, code structure, and variable relationships are valid.

## Surface Summary
- **Paper ID**: 113577-V1
- **Baseline groups**: 2 (G1_math, G1_reading)
- **Design code**: panel_fixed_effects (two-way FE: teacher + school-year)
- **Budgets**: 55 core specs per group, 10 control subsets
- **Seeds**: 113577 (math), 113578 (reading)
- **Surface hash**: sha256:d770946cae38f8b9fb4c811c9da5a079ff5b249ec94daca6c2ef091defc25f60

## Execution Summary

### Counts
| Category | Count |
|----------|-------|
| Total spec rows | 61 |
| G1_math specs | 31 |
| G1_reading specs | 30 |
| Successful | 57 |
| Failed | 4 |
| Inference variants | 4 |

### Spec Breakdown by Type
| Type | G1_math | G1_reading |
|------|---------|------------|
| baseline | 1 | 1 |
| design/* | 3 | 3 |
| rc/fe/* | 5 | 5 |
| rc/controls/loo/* | 4 | 4 |
| rc/controls/sets/* | 3 | 3 |
| rc/controls/subset/* | 10 | 10 |
| rc/sample/* | 3 | 3 |
| rc/form/treatment/* | 2 | 1 |
| **Subtotal** | **31** | **30** |

### Design Variants (Paper's 5 Columns)
1. **No FE (OLS)**: Not run separately (the OLS-with-own-teacher spec in col 1 is captured under design/within_school_year_fe with different FE)
2. **School-year FE only** (Col 2): `design/panel_fixed_effects/estimator/within_school_year_fe`
3. **Student FE** (Col 3): `design/panel_fixed_effects/estimator/within_student_fe`
4. **Teacher FE only** (Col 4): `design/panel_fixed_effects/estimator/within_teacher_fe_only`
5. **Teacher + school-year FE** (Col 5, baseline): `baseline`

### RC Axes
- **FE structure**: add/drop teacher FE, school-year FE, student FE (5 specs)
- **Controls LOO**: drop lagged score, class size, teacher experience block, demographics block (4 specs)
- **Control sets**: minimal, extended, full with own teacher quality (3 specs)
- **Control subsets**: 10 random draws using stratified block sampling
- **Sample restrictions**: post-2002, elementary only, middle only (3 specs)
- **Treatment form**: peer observable VA, peer characteristics (1-2 specs)

### Inference Variants
- Canonical: clustered at teacher level (CRV1: t_s)
- Variant 1: clustered at school-year level (CRV1: sch_year_code)
- Variant 2: heteroskedasticity-robust (HC1)

## Deviations from Surface
- The paper uses `felsdvreg` (Stata command for two-way FE decomposition). This is
  approximated in Python using pyfixest's multi-way FE absorption (`| t_s + sch_year_code`),
  which produces identical point estimates to felsdvreg under standard conditions.
- `rc/form/treatment/peer_characteristics` only in G1_math (reading group has fewer
  treatment alternatives in the surface).
- Synthetic data used: coefficient estimates do not match paper's reported values.

## Software Stack
- Python: 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
