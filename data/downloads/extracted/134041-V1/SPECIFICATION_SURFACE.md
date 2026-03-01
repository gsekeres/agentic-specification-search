# Specification Surface: 134041-V1

## Paper Overview

"How Do Beliefs About the Gender Wage Gap Affect the Demand for Public Policy?" by Sonja Settele (AEJ: Economic Policy). This paper uses a survey experiment in which US adults are randomly assigned to receive information about the true gender wage gap (women earn 74 cents per male dollar). The paper estimates the ITT effect of this information treatment on (1) perceptions of the gender wage gap as a problem and (2) demand for labor market policies to address the gap.

## Baseline Groups

### G1: Effect of information treatment on perceptions (perception index)

**Claim object**: The information treatment causes respondents to update their perceptions about the gender wage gap being a problem, as measured by a z-scored perception index (`z_mani_index`), which summarizes three components: whether gender differences in wages are large, are a problem, and whether government should mitigate them.

**Baseline specification** (Table 5, Panel A, Column 6): OLS regression of `z_mani_index` on `T1` with 21 pre-treatment controls, probability weights (`pweight`), and HC1 robust SEs.

**Why a separate group**: The perception outcome family is the paper's "first stage" -- whether the treatment shifts beliefs. This is conceptually distinct from policy demand.

**Additional baseline specs**: Individual perception components (`posterior`, `zposterior`, `large`, `problem`, `govmore`) are listed as additional baseline spec_ids since the paper reports all of these as headline results in Table 5 Panel A.

### G2: Effect of information treatment on policy demand (policy demand index)

**Claim object**: The information treatment increases demand for gender-equity labor market policies, as measured by the policy demand index (`z_lmpolicy_index`), which summarizes support for quotas, affirmative action, equal pay legislation, wage transparency, reporting websites, and childcare subsidies.

**Baseline specification** (Table 5, Panel B, Column 7): OLS regression of `z_lmpolicy_index` on `T1` with 21 pre-treatment controls, probability weights, HC1 robust SEs.

**Why a separate group**: Policy demand is the paper's ultimate outcome of interest -- the causal chain from treatment to beliefs to policy preferences. The paper treats this as a distinct set of headline results.

**Additional baseline specs**: Individual policy components (`quotaanchor`, `AAanchor`, `legislationanchor`, `transparencyanchor`, `UKtool`, `childcare`) are listed as additional baseline spec_ids.

## Included Robustness Checks

### Controls (both groups)

1. **Leave-one-out (LOO)**: Drop each of the 9 most substantive controls one at a time (wave, gender, prior, democrat, indep, otherpol, anychildren, loghhinc, associatemore). Age dummies and employment categories are treated as blocks.
2. **Control sets**:
   - No controls (pure treatment-control comparison)
   - Demographics only (gender, age dummies, region)
   - Demographics + politics (add democrat, indep, otherpol)
   - Demographics + economics (add loghhinc, employment dummies, education)
   - Full baseline (all 21 controls)
3. **Control progression**: Sequential buildup from bivariate to full
4. **Random control subsets**: 10 (G1) or 5 (G2) randomly sampled subsets

### Sample

- **Wave A only**: Test whether results hold in each survey wave separately
- **Wave B only**: Same
- **Incentivized prior only**: Restrict to respondents with incentivized prior beliefs (prior1==1), as used in some tables

### Weights

- **Unweighted**: Run without probability weights to test sensitivity

### Functional Form / Outcome

- **Raw posterior belief**: Use `posterior` (0-200 scale) instead of z-scored version
- **Z-scored posterior**: Use `zposterior` as alternative

### Design Estimator

- **Difference-in-means**: Simple mean comparison without controls (the purest experimental estimator)
- **With covariates**: OLS with pre-treatment covariates (the paper's baseline approach)

## Excluded from Core

- **2SLS / IV estimates** (Table 5, Panel C): These instrument z-scored posterior beliefs with T1 to estimate the effect of beliefs on policy demand. This changes the estimand from ITT to a causal mediation estimate and belongs in `explore/*`.
- **Follow-up survey persistence** (Table 6): These use different outcome measures from the obfuscated follow-up survey (Stage II). They test persistence rather than the primary effect and belong in `explore/*`.
- **Heterogeneity by gender/politics** (Table 7): These add interaction terms (T1*female, T1*democrat) and change the estimand to conditional effects. They belong in `explore/*`.
- **Beliefs about underlying factors** (Table 8): Different outcome family (personal vs impersonal factors, fairness). Exploratory.
- **Policy effectiveness heterogeneity** (Table 9): Wave B only, interacts with perceived policy effectiveness. Exploratory.
- **Willingness to pay** (Table 10): Control group only, correlational analysis. Not experimental.

## Inference Plan

- **Canonical**: HC1 robust SEs (matches Stata `vce(r)`)
- **Variant 1**: Conventional (non-robust) SEs
- **Variant 2**: HC3 robust SEs (finite-sample correction)

Note: The paper uses sharpened q-values (FDR correction) for multiple testing across outcome variables within families. This is a post-processing step (`post/*`) rather than an inference variant.

## Budget and Sampling

- **G1 total core budget**: 80 specifications
- **G2 total core budget**: 60 specifications
- **Controls subset budget**: 10 (G1), 5 (G2)
- **Seeds**: 134041 (G1), 134042 (G2)
- **Sampler**: stratified_size

## Key Linkage Constraints

- Controls are pre-treatment covariates for precision, not identification. Adding/dropping controls should not change the estimand in a well-executed experiment. This means the control sensitivity analysis tests precision rather than bias.
- The `$controls` global used in all specifications is identical across tables, ensuring consistency.
- Probability weights (`pweight`) are used throughout. The unweighted specification is a legitimate robustness check since the experiment is individually randomized.
- Wave is included as both a control and a sample restriction -- when restricting to a single wave, the wave dummy should be dropped from controls.

## Estimated Spec Count

### G1 (Perceptions)
| Category | Count |
|---|---|
| Baseline (z_mani_index) | 1 |
| Additional baselines (posterior, zposterior, large, problem, govmore) | 5 |
| Design (diff_in_means, with_covariates) | 2 |
| RC/controls/loo | 9 |
| RC/controls/sets | 5 |
| RC/controls/progression | 5 |
| RC/controls/subset | 10 |
| RC/sample | 3 |
| RC/weights | 1 |
| RC/form/outcome | 2 |
| **Total** | **~43 per baseline + baselines = ~51** |

### G2 (Policy demand)
| Category | Count |
|---|---|
| Baseline (z_lmpolicy_index) | 1 |
| Additional baselines (6 individual policies) | 6 |
| Design (diff_in_means, with_covariates) | 2 |
| RC/controls/loo | 9 |
| RC/controls/sets | 5 |
| RC/controls/progression | 5 |
| RC/controls/subset | 5 |
| RC/sample | 3 |
| RC/weights | 1 |
| **Total** | **~37 per baseline + baselines = ~44** |

**Combined total: ~95 core specifications across both groups**
