# Specification Surface: 113577-V1

## Paper Overview
- **Title**: Teacher Peer Effects (study of how peer teacher quality affects student achievement)
- **Design**: Panel fixed effects with two-way FE decomposition (teacher + school-year)
- **Key claim**: Being assigned to a school with higher-quality peer teachers improves student test score growth, even after controlling for own teacher quality and school-year fixed effects. Uses felsdvreg for the two-way FE decomposition.

## Baseline Groups

### G1_math: Peer Teacher Effects on Math Achievement

**Claim object**: Effect of peer teacher value-added (average quality of other teachers in same school-year) on student math test score growth.

**Baseline specification**:
- Formula: `m_growth ~ peer_tfx_m + peer_tfx_miss_m + l_math + [demographics/class controls] | t_s + sch_year`
- Two-way FE: teacher (t_s) and school-year (sch_year) via felsdvreg
- Clustering: teacher level (t_s)
- Focal coefficient: `peer_tfx_m`

### G1_reading: Peer Teacher Effects on Reading Achievement

Same structure as math but with reading outcome (r_growth) and reading peer VA (peer_tfx_r).

## RC Axes Included

### FE structure (the paper's main axis of variation)
The paper's 5 columns progressively add FE:
1. No FE (plain OLS)
2. School-year FE (areg, a(s_s))
3. Student FE (areg, a(mastid)) -- drops demographics
4. Teacher FE (areg, a(t_s))
5. Teacher + school-year FE (felsdvreg, preferred)

Additional: student FE robustness check using felsdvreg with i(mastid) j(t_s).

### Controls
- **Leave-one-out**: Lagged score, class size, teacher experience block, demographics block
- **Standard sets**: Minimal (year*grade only), extended (+ classroom), full (+ own teacher quality)
- **Control subsets**: 10 random draws stratified by block

### Sample restrictions
- Post-2002 (drop early years with missing data)
- Elementary only, middle only

### Treatment definition
- **Peer observable VA**: Use VA based on observable teacher characteristics instead of estimated teacher FE
- **Peer characteristics**: Use raw peer teacher characteristics (from Table in Part 1) instead of VA summary

## What Is Excluded and Why

- **Part 1 (teacher characteristics regressions)**: The `2_Part_1.do` regressions examine own and peer teacher observable characteristics rather than estimated VA. These use a different treatment concept (raw characteristics vs estimated VA) and are partially included as `rc/form/treatment/*`.
- **Part 4 (future teachers and lagged peers)**: Uses leads/lags of peer composition as falsification/mechanism tests. These are `diag/*` or `explore/*`.
- **Part 5 (interactions)**: Appendix interaction specifications are `explore/*` heterogeneity analyses.

## Budgets and Sampling

- **Max core specs per group**: 55
- **Max control subsets**: 10
- **Seeds**: 113577 (math), 113578 (reading)

## Inference Plan

- **Canonical**: Clustered at teacher level (matches paper)
- **Variants**: Cluster at school-year level, HC1 robust
