# Specification Search: 113566-V1

## Paper
Jacob & Lefgren (2009), "The Effect of Grade Retention on High School Completion",
American Economic Journal: Applied Economics, 1(3), 33-58.

## DATA NOTE
The underlying data are **confidential Chicago Public Schools student records** and
are not available in the replication package. This specification search uses **synthetic
data** that replicates the variable structure and DGP described in the Stata do files.
Estimates should NOT be interpreted as replication of the original results. The purpose
is to exercise the specification search framework and verify the code can run the full
set of specifications.

## Surface Summary
- **Paper ID**: 113566-V1
- **Baseline groups**: 1 (G1)
- **Design**: Fuzzy regression discontinuity
- **Running variable**: `index` (normalized test score relative to cutoff)
- **Treatment**: `dret2` (retained or transition center)
- **Outcome**: `dropF2005` (dropout by Fall 2005)
- **Instruments**: Experiment-specific splines (`gpmarg_*`, `gpind4_index_above_*`)
- **Canonical inference**: Clustered at `gp` (index x experiment cell)
- **Budget**: 70 max specs core total
- **Seed**: 113566

## Counts
| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 3 | 3 | 0 |
| Design   | 30 | 30 | 0 |
| RC       | 30 | 30 | 0 |
| **Total** | **63** | **63** | **0** |
| Inference variants | 6 | 6 | 0 |

## Specification Variants Executed

### Baselines (3)
- `baseline` (Grade 6, preferred spec with full controls, samp_m10p5)
- `baseline__table2_grade8` (Grade 8)
- `baseline__table2_older_grade8` (Older Grade 8 / newgrade=9)

### Design variants (30)
**Bandwidth** (3 x 3 grades = 9):
- `samp_m15p10`: index in [-1.5, 1.0]
- `samp_m20p15`: index in [-2.0, 1.5]
- `samp_m8p3`: index in [-0.8, 0.3]

**Polynomial order** (3 x 3 grades = 9):
- Quadratic index (+ gpind2_*)
- Cubic index (+ gpind2_*)
- Cubic split above/below cutoff (+ gpind3_*)

**Instrument set / procedure** (4 x 3 grades = 12):
- Fixed knots (agg_index_marg + index_above)
- Single pass dummy IV
- Experiment x pass dummies IV (gppass_* + pass)
- Marginal area only IV (gpmarg_* only, gpind4_* as controls)

### RC variants (30)
**Control sets** (3 x 3 grades = 9):
- Group only (gp_*)
- Group + index (gp_* gpind_*)
- Full covariates (all interactions)

**Sample restrictions** (3 grade restrictions + 3 cohort x 3 grades + 3 fail type x 3 grades = 27):
- Grade 6 only, Grade 8 only, Older Grade 8 only
- Cohort 1997, 1998, 1999 (each x 3 grades)
- Failed reading only, Failed math only, Failed both (each x 3 grades)

### Inference variants (6)
- HC1 (heteroskedasticity-robust, no clustering) x 3 baselines
- Classical / IID (no robust SE) x 3 baselines

## Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3

## Deviations and Notes
- **Synthetic data**: Original confidential CPS data not available. Synthetic data
  constructed to match variable structure from `ret_cleaned_8_19_2008.do` and
  `datanew_31oct2005.ado`. N ~ 46,500 synthetic students.
- **Covariate interactions**: Simplified gpcov2_* (used squared test scores rather than
  full 3rd-order polynomial interactions) due to synthetic data. Original paper uses
  p2j*, p3j*, qq* polynomial interactions.
- The paper's `ivreg2` Stata command maps to pyfixest `feols()` with IV syntax.
- All specifications use the linked IV adjustment (instruments and controls varied jointly).
