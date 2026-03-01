# Specification Surface: 112791-V1

## Paper Overview
- **Title**: The Impact of Medicaid on Labor Market Activity and Program Participation: Evidence from the Oregon Health Insurance Experiment (Baicker et al., 2014, AER P&P)
- **Design**: Randomized experiment (lottery-based random assignment)
- **Key limitation**: Administrative data (SSA records) are not publicly available. Replication code is provided but data must be obtained through restricted-access channels. The specification surface is built from the code and the paper's description of its analysis.

## Baseline Groups

### G1: Employment and Earnings (Table 1)

**Claim object**: The ITT effect of Medicaid lottery selection on labor force participation and earnings, measured using SSA administrative records (W-2 wages, self-employment income, earnings above FPL).

**Why a separate group**: Table 1 is the paper's primary result table on labor supply. The three earnings outcomes (any earnings, total earnings, earnings above FPL) are the central claim about Medicaid's labor market effects.

**Baseline specifications**:
- `any_earn2009 ~ treatment + nnn* + any_earn2007 [pw=weight_ssa_admin], cluster(reservation_id)` (binary: any positive earnings)
- `earn2009 ~ treatment + nnn* + earn2007 [pw=weight_ssa_admin], cluster(reservation_id)` (continuous: total earnings)
- `earn_ab_fpl_adj_2009 ~ treatment + nnn* + earn_ab_fpl_adj_2007 [pw=weight_ssa_admin], cluster(reservation_id)` (binary: earnings above FPL)

**Controls**:
- `nnn*`: Lottery draw fixed effects (~8 dummies). These are mandatory for RCT design integrity (randomization was stratified by lottery draw).
- Lagged outcome (2007 version of the dependent variable): used in baseline specification for precision.
- `lottery_list`: Demographic/geographic variables from the lottery signup list (birthyear_list, female_list, english_list, self_list, first_day_list, have_phone_list, pobox_list, zip_msa, zip_hh_inc_list). Used only in the "all controls" robustness check.

### G2: Government Benefit Receipt (Table 2)

**Claim object**: The ITT effect of Medicaid lottery selection on government benefit receipt (SNAP, TANF, SSI, SSDI), measured using SSA and state administrative records.

**Why a separate group**: Table 2 is the paper's second main result table. The eight benefit outcomes (4 binary indicators + 4 dollar amounts) represent a distinct claim about program participation effects vs. the labor supply effects in G1.

**Baseline specifications**: Eight ITT regressions (one per outcome), each of the form:
- `{outcome}2009 ~ treatment + nnn* + {outcome}2007 [pw=weight_ssa_admin], cluster(reservation_id)`

## RC Axes Included

### Controls
- **Drop lagged outcome**: Run without the 2007 lagged dependent variable (matches paper's "no controls" robustness check in Tables A10-A11)
- **Add lottery_list demographics**: Add the 9 lottery signup characteristics (matches paper's "all controls" robustness check in Tables A10-A11)

### Sample / Time Period
- **Year 2008**: Replace 2009 outcomes with 2008 outcomes (with appropriate first-stage variable ohp_all_ever_ad2008)
- **Years 2008-2009 pooled**: Use 0809 combined outcomes
- These time-period variants are explicitly in the paper's robustness tables (Tables A10-A11)

### Weights
- **Unweighted**: Drop probability weights (weight_ssa_admin). The paper always uses weights; unweighted is a standard robustness check.

### Outcome alternatives (G1 only)
- **Wage income only** (wage2009): W-2 income only, dropping self-employment
- **Self-employment only** (se2009): Self-employment income only
- **Any wage** (any_wage2009): Binary indicator for any W-2 income
- **Any self-employment** (any_se2009): Binary indicator for any SE income
- These are reported in the paper's detailed earnings table (Table A6)

### Design estimator variants
- **Difference-in-means**: No lottery-draw FE, no lagged outcome. Pure raw comparison.
- **Strata FE**: Lottery-draw FE only (nnn*), no lagged outcome. This is the design-minimal specification.

## What Is Excluded and Why

- **IV/LATE estimates**: The paper also reports 2SLS estimates (instrumenting Medicaid enrollment with lottery selection). These target a different estimand (LATE for compliers) and would require a separate design code (instrumental_variables). They are excluded from the core surface because the ITT is the primary claim; IV is reported as supplementary.
- **Disability application outcomes** (Table A8): These are additional outcomes not emphasized as main results.
- **Income summary indices** (econ_sufficient, econ_suff_ns_any): These are composite summary indices reported in Table A9, not the paper's focal claims.
- **SNAP timing analysis** (Table A7): Month-by-month SNAP onset analysis is an event-timing exercise, not the main claim.
- **Exploration/heterogeneity**: No subgroup analysis in the paper.
- **Sensitivity**: No sensitivity analysis needed (RCT with random assignment).

## Budgets and Sampling

- **G1 max core specs**: 60 (3 baseline outcomes x ~18 spec variants + design variants + 5 alternative outcomes)
- **G2 max core specs**: 70 (8 baseline outcomes x ~8 spec variants + design variants)
- **Control subsets**: Exhaustive enumeration (small control pool: lottery FE always mandatory, optional lagged outcome and demographics)
- **Seed**: 112791

## Inference Plan

- **Canonical**: Cluster at household/reservation_id (the randomization unit). This matches the paper's inference throughout.
- **Variants**: HC1 robust (no clustering), HC3 jackknife. These stress-test the clustering assumption.

## Key Linkage Constraints

- Lottery draw FE (nnn*) are always included -- they are part of the randomization design and removing them would violate exchangeability.
- Weights (weight_ssa_admin) correct for the sampling design in post-lottery periods; the unweighted variant is a robustness check, not the preferred specification.
- Lagged outcomes improve precision but are optional from a design perspective (randomization ensures balance without them).
