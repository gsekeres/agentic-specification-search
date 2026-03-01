# Specification Search Log: 116280-V1

## Paper
Mayers & Smith (2005), insurance organizational form choice (mutual vs stock).

## Surface Summary
- **Paper ID**: 116280-V1
- **Surface hash**: sha256:63733aa94a57227727262cc2d15fe56f592634ec16d5138f089570fbde8c1cbd
- **Baseline groups**: 1
  - G1: Effect of state regulation on mutual organizational form (Table 2)
- **Design**: cross_sectional_ols (logit estimator)
- **Canonical inference**: Cluster at state (47 clusters)
- **Budget**: max 60 core specs
- **Seed**: 116280
- **Control subset sampler**: stratified_size

## Execution Summary

### Counts
| Category | Planned | Executed | Successful | Failed |
|----------|---------|----------|------------|--------|
| Baseline | 3 | 3 | 3 | 0 |
| Design variants | 1 | 1 | 1 | 0 |
| RC variants | 46 | 46 | 46 | 0 |
| **Total estimate rows** | **50** | **50** | **50** | **0** |
| Inference variants | 2 | 2 | 2 | 0 |

### Specifications Executed

#### Baselines
- Table 2 col 1: logit mutual mlaw controls, cluster(state) -- N=881, pseudo R2=0.265
- Favor spec: logit mutual favor controls, cluster(state)
- Rcorp spec: logit mutual mlaw controls+rcorp (no decade dummies), cluster(state)

#### Design Variant
- LPM (OLS): Linear probability model with same controls and clustering

#### RC: Controls (LOO)
- Drop each of: slaw, regulate, nfc, reform (with interactions), refmlaw, refslaw, refregulate, refnfc, ten2, ten3, ten4, ten5

#### RC: Controls (Sets)
- No decade dummies, add rcorp, add favor, favor instead of mlaw/slaw, favor+rcorp, full+rcorp, minimal financial, minimal regulatory

#### RC: Controls (Progression)
- Bivariate, financial reqs only, regulatory only, financial+reform, full no interactions

#### RC: Controls (Random Subsets)
- 10 random subsets (seed=116280), 4-10 controls each

#### RC: Sample
- Trim mlaw 1/99, trim slaw 1/99, exclude states with no mutuals, pre-1930 only, post-1920 only

#### RC: Fixed Effects
- Conditional logit with state FE (Table 5 approach)

#### RC: Functional Form
- LPM, probit, conditional logit (state), LPM with decade FE absorbed, favor as treatment

#### Inference Variants
- HC1 (heteroskedasticity-robust), IID (conventional)

### Skipped / Deviations
- **Binary treatment coding** (mlaw_binary, slaw_binary, favor_binary) excluded per surface (changes coefficient interpretation -- in excluded_from_core).
- **Conditional logit uses statsmodels ConditionalLogit** rather than Stata's clogit; SEs may differ slightly from Stata's clustered clogit.
- No diagnostics executed (diagnostics_plan is empty).

## Software Stack
- Python 3.12.7
- statsmodels (logit, probit)
- pyfixest (LPM)
- pandas, numpy
