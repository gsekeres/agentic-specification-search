# Verification Report: 116359-V1

## Paper
**Title**: High Trade Costs and Their Consequences: An Estimated Dynamic Model of African Agricultural Storage and Trade
**Author**: Obie Porteous
**Journal**: American Economic Review, 2019
**Method**: Cross-sectional OLS with heteroskedasticity-robust and clustered standard errors; IV/2SLS for demand parameter estimation

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Overland trade costs increase with distance (~12 pp per 1000 km) and crossing borders (~6.8 pp) | + | baseline_t5c1, baseline_t5c2, baseline_t5c3, baseline_t5c4 |
| G2 | International link trade costs are not significantly explained by distance; FTA reduces costs ~14 pp | + (distance) | baseline_t6c1, baseline_t6c2, baseline_t6c3, baseline_t6c4, baseline_t6c5 |
| G3 | Port-to-world trade costs negatively associated with corruption (insignificant, N=47) | - | baseline_tA13c1, baseline_tA13c2 |
| G4 | Elasticity of substitution sigma ~ 1.5 (OLS with both FE sets) | + | demand/sigma_ols |
| G5 | Elasticity of demand epsilon ~ -0.26 (OLS with region FE, clustered) | - | demand/epsilon_ols_clustered |

### G1: Overland Trade Costs (Table 5)
- **Baseline estimate (T5C1)**: distance coef = 0.000119 per km (SE = 3.95e-05), p = 0.003, N = 413
- **International border penalty**: 0.068 (SE = 0.017), p < 0.001
- **R-squared**: 0.054
- **Parameters**: OLS with no clustering, taubar ~ distance + internatl
- Cols 2/4 decompose distance by road type (paved/unpaved/water); Cols 3/4 add population controls

### G2: International Link Trade Costs (Table 6)
- **Baseline estimate (T6C5)**: distance coef = 4.95e-05 (SE = 1.30e-04), p = 0.70, N = 135
- **FTA coefficient (T6C5)**: -0.138 (SE = 0.066), p = 0.035
- **R-squared**: 0.188
- **Parameters**: OLS clustered by country pair, taubar ~ distance + language + FTA + corruption + LPI

### G3: Port-to-World Trade Costs (Table A13)
- **Baseline estimate (A13C1)**: corrupt2013 coef = -0.0109 (SE = 0.0102), p = 0.28, N = 47
- **R-squared**: 0.178
- **Parameters**: OLS clustered by country, taubar ~ corrupt2013 + sub500 + hivol + gdppercap + lpicustoms

### G4: Elasticity of Substitution (Table A7)
- **Baseline estimate**: lnp coef = 0.493 (SE = 0.219), p = 0.024, N = 463
- **Implied sigma**: 1 + 0.493 = 1.493
- **R-squared**: 0.930
- **Parameters**: OLS with region-grain + country-year FE, clustered by country-grain

### G5: Elasticity of Demand (Table A7)
- **Baseline estimate**: lnp coef = -0.256 (SE = 0.071), p < 0.001, N = 289
- **Implied epsilon**: -0.256
- **R-squared**: 0.683
- **Parameters**: OLS with region FE, clustered by region code

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **70** |
| Baselines | 13 |
| Core tests (non-baseline) | 57 |
| Non-core tests | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 13 | Paper table columns: 4 from Table 5, 5 from Table 6, 2 from Table A13, 1 sigma OLS, 1 epsilon OLS |
| core_se | 9 | SE/clustering variations: HC1, HC3, alternative cluster variables |
| core_funcform | 12 | Functional form: log-log, semi-log, quadratic, asinh, quantile regression, per-km outcome |
| core_controls | 14 | Control set variations: drop/add controls, kitchen sink, distance-only |
| core_sample | 13 | Sample restrictions: domestic-only, drop outliers, paved-only, winsorize, distance cutoffs |
| core_estimation | 4 | IV vs OLS, fixed effects variations for demand parameters |

## Classification Decisions

### All 70 Specifications Are Baseline or Core

Every specification in this package is either a direct replication of a paper table column (baseline) or a robustness test of one of the paper's five main empirical claims (core). There are no specifications testing tangential claims, auxiliary statistics, or unrelated outcomes. This is consistent with the paper's structure: the specification search was designed around the paper's three trade cost regression tables (Tables 5, 6, A13) and the demand estimation table (Table A7).

### Baseline Classifications (13 specs)

**Table 5 (G1, 4 baselines)**: Columns 1-4 of Table 5 are all presented as primary results in the paper. Col 1 (distance + internatl) and Col 2 (road-type decomposition) are the two main specifications; Cols 3-4 add population controls. All four are classified as baselines because they appear as distinct columns in the paper's main results table.

**Table 6 (G2, 5 baselines)**: Columns 1-5 of Table 6 are a build-up specification where each column adds one control. Col 5 (the kitchen-sink) is the focal specification for the primary coefficient (distance). All five are baselines because they appear as distinct table columns.

**Table A13 (G3, 2 baselines)**: The two appendix table columns (with and without the trains variable) are both paper results.

**Demand (G4-G5, 2 baselines)**: The paper's preferred sigma estimate (OLS with both FE sets, G4) and epsilon estimate (OLS clustered, G5) are identified as the primary demand parameter estimates from Table A7.

### Core Classifications (57 specs)

**G1 robustness (33 specs)**: All 33 non-baseline Table 5 robustness specifications test whether the distance-trade cost relationship holds under alternative standard errors (5 specs), functional forms (7 specs), control sets (8 specs), and sample restrictions (11 specs). These are all legitimate robustness tests of the same core claim. Two specs change the treatment variable: per-km outcome (robust/form/perkm_outcome tests internatl rather than distance, but the claim is the same), and per-tkm outcome (robust/outcome/pertkm reverses the sign because per-km cost falls with distance).

**G2 robustness (13 specs)**: All 13 non-baseline Table 6 robustness specifications test the distance coefficient for international links under varied SE/clustering (2), functional form (2), control sets (6), and sample restrictions (3). All are directly relevant to the same claim.

**G3 robustness (4 specs)**: The 4 non-baseline port regressions vary clustering, controls, and functional form for the corruption-trade cost relationship. All are core.

**G4 robustness (4 specs)**: IV, alternative FE sets, and HC1 SE for the sigma estimate.

**G5 robustness (3 specs)**: No-cluster, IV, and no-FE variations for the epsilon estimate.

### Why No Non-Core Classifications

Unlike some papers where the specification search includes auxiliary outcomes or tangential analyses, this package is tightly focused on the paper's stated empirical claims. Every specification tests either the distance-trade cost relationship (G1/G2), the corruption-port cost relationship (G3), or a demand parameter (G4/G5). There are no placebo tests, falsification exercises, or supplementary descriptive statistics that would warrant a non-core classification.

## Robustness Assessment

### G1: Overland Distance-Trade Cost Relationship -- STRONG

This is the paper's core finding and it is highly robust:
- **33 of 37 total specs (89%)** show a positive distance coefficient
- **30 of 37 (81%)** are significant at the 5% level
- **26 of 37 (70%)** are significant at the 1% level
- The baseline coefficient of ~0.000119 per km is stable across specifications (range: 0.000046 to 0.000230 excluding functional form changes)
- The international border dummy (~0.068) is consistently significant at p < 0.001 across all Table 5 specifications

Key sensitivity findings:
- **SE/clustering**: All 5 SE variations maintain significance at 5% except clustering on destination country (p=0.092, marginally insignificant)
- **Functional form**: Log-log, semi-log, asinh, and all three quantile regressions confirm the positive relationship. The quadratic specification makes the linear term insignificant (p=0.59), suggesting the relationship may not be strictly linear
- **Controls**: Adding road type dummies, population controls, or Michelin indicators attenuates the distance coefficient by ~10-35% but preserves significance
- **Sample**: Results hold for domestic-only, drop-outlier, and market-size subsamples. Two notable exceptions: paved-roads-only shows a negative insignificant coefficient (the distance-cost relationship is driven by unpaved roads), and international-links-only is insignificant (consistent with G2)

### G2: International Distance-Trade Cost Relationship -- CONSISTENTLY NULL

The null finding for international links is itself robust:
- **0 of 18 total specs** show distance significant at the 5% level
- p-values range from 0.30 to 0.999
- The distance coefficient is consistently near zero or slightly positive
- This null finding is robust to SE/clustering variations, functional form, control set variations, sample restrictions, and winsorizing
- The FTA coefficient is the most robust finding for international links: significant at 5% in 7 of 10 specs that include it

### G3: Port Corruption-Trade Cost Relationship -- WEAK (underpowered)

- **0 of 6 total specs** show corruption significant at the 5% level
- Corruption coefficient is consistently negative (as expected) across all 6 specs
- With N=47, the small sample severely limits power
- No-clustering variant (p=0.107) comes closest to significance

### G4: Elasticity of Substitution -- FRAGILE

- Baseline OLS with both FE sets: sigma = 1.49, p = 0.024 (significant)
- HC1 robust SE: p = 0.009 (more significant due to smaller SE)
- **But**: removing either FE set makes the estimate insignificant (p = 0.66 for country-year only, p = 0.74 for region-grain only)
- IV estimate: sigma = 1.10, insignificant (p = 0.73)
- The sigma estimate depends critically on having both fixed effects sets -- it is not robust to specification changes

### G5: Elasticity of Demand -- MODERATE

- Baseline OLS clustered: epsilon = -0.26, p < 0.001 (highly significant)
- OLS no cluster: p = 0.008 (still significant)
- No region FE: epsilon = -0.40, p < 0.001 (larger in magnitude, more significant)
- **But**: IV estimate gives epsilon = -0.14, insignificant (p = 0.29)
- The OLS estimate is robust to SE variations and FE choices, but the IV specification raises endogeneity concerns

## Notable Issues

### 1. The structurally-estimated dependent variable
The outcome variable (taubar) is itself the output of a structural dynamic model, not raw data. The reduced-form regressions in Tables 5-6 and A13 explain the determinants of a model-generated object. This means the standard errors may understate uncertainty because they do not account for estimation error in the first-stage structural model.

### 2. Paved vs unpaved road heterogeneity
The distance-trade cost relationship is entirely driven by unpaved roads. For paved-only links (N=217), the distance coefficient is negative and insignificant. For unpaved-only links (N=183), it is nearly twice the baseline magnitude and highly significant. This heterogeneity is well-documented in the paper (Table 5, Cols 2/4) but is an important qualification to the headline finding.

### 3. International links: consistent null for distance
The paper transparently reports that distance does not significantly predict trade costs for international links (G2). The specification search confirms this is not an artifact of any particular specification choice -- the null holds across all 18 robustness variations.

### 4. Demand parameter sensitivity to identification
The sigma estimate (G4) is fragile: it requires both region-grain and country-year FE to be significant, and the IV estimate is insignificant. The epsilon estimate (G5) is more robust to OLS variations but the IV estimate is also insignificant. This suggests potential endogeneity concerns that the instruments do not fully resolve.

### 5. Small sample sizes for G2 and G3
The international link regressions (N=135) and port regressions (N=47) have limited statistical power. The consistent null results for G2 and G3 may reflect underpoweredness rather than true zero effects.

## Recommendations

1. **Focus the specification curve on G1 (overland distance-trade cost) for the primary analysis**: These 37 specs cover the paper's central claim with a well-structured set of robustness variations, and the result is strongly supported.

2. **G2 is valuable as a contrast case**: The 18 international link specs show a uniformly null distance coefficient, which could serve as a useful "null result" benchmark in a specification curve comparison.

3. **G4 (sigma) demonstrates fragility**: The 5 sigma specs show how the same coefficient goes from significant (p=0.024) to completely insignificant depending on FE choice and estimation method. This could illustrate specification sensitivity in a compelling way.

4. **All 70 specs are core**: No specs need to be excluded as non-core. The specification search is tightly focused on the paper's stated claims.

5. **The per-km outcome spec (robust/outcome/pertkm) has a negative coefficient by construction**: This is expected because longer links have lower per-km trade costs (the total cost rises sublinearly with distance). It should be flagged as testing the same relationship with reversed sign interpretation.
