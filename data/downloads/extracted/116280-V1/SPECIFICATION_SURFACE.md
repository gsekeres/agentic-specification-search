# Specification Surface: 116280-V1

## Paper Overview

This paper studies the determinants of insurance company organizational form (mutual vs. stock) during 1900-1949 in the United States, focusing on how state-level regulation affects the choice to form as a mutual company. The main analysis uses logit regressions of a binary outcome (mutual=1, stock=0) on state regulatory variables, with standard errors clustered at the state level.

## Baseline Groups

### G1: Effect of state regulation on mutual organizational form

**Claim object**: State regulatory environment (financial requirements for mutuals and stocks, regulatory dummies) predicts whether an insurance company is formed as a mutual.

**Baseline specification** (Table 2, Column 1): Logit regression of `mutual` on `mlaw` (mutual financial requirement), `slaw` (stock financial requirement), `regulate`, `nfc` (fraternal), `reform` (reorganization dummy), and reform interactions (`refmlaw`, `refslaw`, `refregulate`, `refnfc`), plus decade dummies (`ten2`-`ten5`), with SEs clustered by state.

- N = 881 observations across 47 states
- Key coefficient on `mlaw`: -2.12 (z = -5.48, p < 0.001)
- Pseudo R2 = 0.265

**Why one baseline group**: The paper's central claim is the effect of the regulatory environment on organizational form. The different table columns (Tables 2-5) represent alternative specifications of the same claim, using different measures of regulation (e.g., `favor` instead of `mlaw`/`slaw`) and different estimators (logit vs clogit). These are treated as RC variants rather than separate claims.

## Included Robustness Checks

### Controls

1. **Leave-one-out (LOO)**: Drop each non-treatment control one at a time (8 LOO specs: drop each of slaw, regulate, nfc, reform, refmlaw, refslaw, refregulate, refnfc)
2. **Control sets**: The paper itself uses several control configurations:
   - No decade dummies (rcorp instead)
   - Add `rcorp` (real interest rate) to baseline
   - Replace `mlaw`/`slaw` with `favor` (as in the paper's alternative specification)
   - `favor` spec with `rcorp` instead of decade dummies
   - Full specification with `rcorp` added
   - Financial requirements only (mlaw, slaw)
   - Regulatory dummies only (regulate, nfc)
3. **Control progression**: Build up from bivariate to full specification
4. **Random control subsets**: 10 randomly sampled subsets from the control pool

### Sample

- **Trimming**: Winsorize `mlaw` at 1/99 percentiles (it has a skewed distribution with max 11.89)
- **Trimming**: Winsorize `slaw` at 1/99 percentiles
- **Exclude states with no mutual formations**: The clogit analysis drops 13 states (131 obs) with no mutual companies; this restriction is also informative for logit
- **Pre-1930 only**: Restrict to companies formed before 1930 (earlier regulatory environment)
- **Post-1920 only**: Restrict to companies formed after 1920 (more modern regulatory environment)

### Fixed Effects

- **Add state FE**: Conditional fixed-effects logit (clogit) with state grouping, as in Table 5 of the paper

### Functional Form / Estimator

- **LPM (OLS)**: Linear probability model as the design-code estimator alternative
- **Probit**: As a parametric alternative to logit
- **Conditional logit (clogit)**: With state FE, as in the paper's Table 5
- **LPM with decade FE**: Absorb decade effects as factor variables
- **Binary treatment coding**: Dichotomize mlaw/slaw/favor at median as alternative treatment definitions

## Excluded from Core

- **Heterogeneity analyses** (e.g., by decade, by region): These would be `explore/*` specs
- **Alternative outcome definitions**: No obvious alternatives since `mutual` is well-defined binary
- **DML/AIPW**: Not applicable given the logit framework and lack of clear binary treatment

## Inference Plan

- **Canonical**: Cluster-robust SEs at state level (47 clusters) -- matches the paper
- **Variant 1**: HC1 robust (no clustering)
- **Variant 2**: Conventional (non-robust) SEs

## Budget and Sampling

- **Total core budget**: 60 specifications
- **Controls subset budget**: 10 random draws
- **Seed**: 116280
- **Sampler**: stratified_size (stratify by number of controls included)

Full enumeration of LOO and control-set specs is feasible. The random subset specs provide additional coverage of the combinatorial control space.

## Key Linkage Constraints

- Reform interaction terms (`refmlaw`, `refslaw`, `refregulate`, `refnfc`) are mechanically linked to `reform` -- dropping `reform` while keeping interactions is incoherent
- `mlaw`/`slaw` specifications and `favor` specifications represent alternative regulatory measures -- they are substitutes, not additive
- Decade dummies and `rcorp` serve similar time-control roles -- the paper uses one or the other

## Estimated Spec Count

| Category | Count |
|---|---|
| Baseline | 1 |
| Additional baselines (favor, rcorp specs) | 2 |
| Design (LPM) | 1 |
| RC/controls/loo | 12 |
| RC/controls/sets | 8 |
| RC/controls/progression | 5 |
| RC/controls/subset | 10 |
| RC/sample | 4 |
| RC/fe | 1 |
| RC/form (estimator/treatment) | 8 |
| **Total** | **52** |
