# Specification Surface: 163241-V1

## Paper: Pay Transparency and the Gender Gap (Baker, Halberstam, Kroft, Mas, & Messacar)

## Baseline Groups

### G1: Effect of pay transparency on gender wage gap
- **Outcome**: ln_salary_annual_rate (log annual salary of university faculty)
- **Treatment**: treated indicator (province adopted transparency law AND peer salary was revealed)
- **Focal coefficient**: 1.female#1.treated (interaction of female indicator with treatment)
- **Estimand**: ATT of pay transparency on gender wage gap reduction
- **Population**: Canadian university faculty (UCASS data, full-time tenured/tenure-track, 1989-2018)
- **Baseline spec**: Table 4 Col 2 (Inst-Dept peers) -- reghdfe ln_salary_annual_rate i.female##i.treated i.has_responsibilities, absorb(id3 prov_year_sex) cl(inst)
- **Key finding**: female#treated = 0.0197 (SE=0.0059, p=0.002) -- transparency RAISES female salaries relative to males

### Additional baselines
- Inst-Dept-Rank peer group definition (Table 4 Col 2, second run): female#treated = 0.0120 (SE=0.0043)

## Core Universe

### Controls axes
- **LOO**: Drop has_responsibilities (only time-varying control in individual FE spec)
- **Add**: appoint_inst_numyears, degree_high_numyears (from cross-sectional spec, Table 4 Col 1)
- **Cross-sectional spec**: Table 4 Col 1 with institution + subject + prov_year_sex FE instead of individual FE

### Sample restrictions (from paper Tables 5-7)
- Balanced institutions only (Table 5 Col 2)
- Minimum 10 observations per individual (Table 5 Col 3)
- By academic rank: assistant, associate, full professors (Table 6)
- NFDP sample only (nfdp==1)
- Trim salary at 1st/99th percentile (already done at 0.5/99.5 in data cleaning)

### Fixed effects axes
- Swap individual FE (id3) for institution + subject FE (cross-sectional approach)
- Add department-gender-specific time trends (i.inst_subj_sex#c.year, Table 5 Col 4)

### Data construction
- Peer group definition: Inst-Dept vs Inst-Dept-Rank (two main variants)
- Provincial-only treatment (treated = province has law, without peer revelation requirement)
- Ontario salary adjustment (fiscal vs calendar year timing adjustment)

### Functional form
- Level salary (not log) -- changes interpretation substantially

### Weights
- Unweighted (baseline has no explicit weights)

## Inference Plan
- **Canonical**: Cluster SEs at institution level (~49 clusters)
- **Variant 1**: HC1 robust SEs (used in Table 5 Col 4 with dept-gender trends where clustering is absorbed)
- **Variant 2**: Cluster at province level (~10 clusters, very few, but natural treatment variation level)

## Constraints
- Control-count envelope: [1, 3] (has_responsibilities is always included; cross-sectional adds 2 experience measures)
- The two peer group definitions (Inst-Dept, Inst-Dept-Rank) change how treatment is constructed but not the claim object
- The treatment definition has two components: (1) province adopted law, AND (2) a peer's salary was revealed -- the second component introduces individual-level variation
- Staggered adoption across provinces means TWFE may have heterogeneous treatment effects issues

## Budget
- Max core specs: 80
- No control subset sampling needed (very few controls available)
- Total planned: approximately 30-40 specs
- Seed: 163241

## What is excluded and why
- Event study figures (Fig 4, 5, A4-A9): diagnostic (pre-trends), not alternative estimates
- Table 6 bottom (union status heterogeneity): exploration, not baseline claim
- Table 7 top (initial gender pay gap heterogeneity): exploration, not baseline claim
- Table 7 bottom (continuous treatment): changes estimand (dose-response rather than binary ATT)
- Table 8 (early vs late adopters): exploration of treatment effect heterogeneity
- Appendix tables with wider treatment intervals: changes treatment definition substantially

## Data Availability Note
The replication package includes log files with full regression output and do-files. The main analysis data (ucass_all_regs_matched.dta) is confidential Statistics Canada data. However, auxiliary data files (union_data.dta, CPI.dta, gapdata.dta, drop_data.dta) are included, and the log files provide the exact coefficients, standard errors, and sample sizes needed to verify the baseline specification.
