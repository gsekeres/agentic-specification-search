# Verification Report: 116243-V1

## Paper
**Title**: Accounting for the Growth of MNC-based Trade using a Structural Model of U.S. MNCs
**Author**: Michael Keane
**Journal**: American Economic Review
**Year**: 2006
**Method**: Structural Maximum Likelihood Estimation (simulated MLE)
**Software**: Fortran 77 (custom code: `jpe43_aer.f`)

## Critical Context: Structural Model with Confidential Data

This paper is fundamentally different from reduced-form empirical papers. It develops and estimates a large structural model of U.S. multinational corporations via simulated maximum likelihood. The key facts that shape this verification:

1. **Confidential data**: The underlying BEA microdata on 551 U.S. MNCs (1983-1996) cannot be accessed. No re-estimation or alternative specifications are possible.
2. **Single estimation run**: All 151 "specifications" in the CSV are individual parameter estimates extracted from one Fortran output file (`outjpe810`), not independent robustness tests.
3. **Joint estimation**: All 401 parameters (168 iterated) are estimated simultaneously. Individual t-statistics do not have the same interpretation as in reduced-form regressions because parameters are interdependent.
4. **No specification variation**: There is no variation in sample, controls, estimation method, or functional form across the rows. Every row reports a coefficient from the same model applied to the same data.

As a result, this specification search does **not** provide the kind of robustness evidence that specification searches of reduced-form papers provide. It is better understood as a structured parameter inventory.

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Structural model of U.S. MNCs: core technology parameters (returns to scale, substitution elasticity, input share, technical progress) characterize multinational production | + | baseline |

There is a single model and a single estimation. The "baseline" row reports the four core technology parameters (RET, MU, THETA, ATRU) as a coefficient vector.

- **Baseline coefficient (RET)**: 0.1652 (t = 49.71)
- **Full coefficient vector**: RET = 0.165, MU = 0.716, THETA = 0.222, ATRU = 0.045
- **Log-likelihood**: -272,235.41
- **N**: 7,714 firm-year observations (551 firms, 14 years; 446 "good" firms used)

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **152** |
| Baselines | 1 |
| Core tests (non-baseline) | 11 |
| Non-core tests | 140 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_technology | 5 | 1 baseline + 4 core technology parameters (RET, MU, THETA, ATRU) |
| core_demand | 6 | Demand shift intercepts, wage loadings, and time trends (domestic and foreign) |
| core_trade | 4 | Trade cost parameters (TAU0-D, TAU0-F, TAU-L, CORR) |
| noncore_structural_param | 46 | Labor, materials, capital, growth, and transition probability parameters |
| noncore_discrete_choice | 36 | Logit coefficients for ND, NF, E, I decisions (including some duplicates) |
| noncore_funcform | 12 | Box-Cox transformation parameters |
| noncore_nuisance | 34 | Scale, VPSI variance, SPHI diagonal, SETA diagonal parameters |
| noncore_model_fit | 6 | Wage equation fit and predicted participation frequencies |

## Classification Decisions

### Core Test Classifications

**Core technology parameters (4 specs)**: RET (returns to scale), MU (elasticity of substitution), THETA (input share), and ATRU (technical progress rate). These are the paper's central structural parameters governing the CES/Cobb-Douglas production function of MNCs. All four are extremely precisely estimated (t-statistics ranging from 5.67 to 494.40). These parameters are the foundation of the paper's claims about MNC production technology and trade.

**Core demand parameters (6 specs)**: DELTD-0, DELTD-W, DELTD-T (domestic) and DELTF-0, DELTF-W, DELTF-T (foreign). These intercepts, wage loadings, and time trends in the demand shift equations are core to the paper's explanation of how demand for domestic vs. foreign MNC output evolves over time. The time trends in particular are central to explaining the growth of MNC-based trade. The plant-count loadings (DELTD-ND, DELTD-NF) are classified as non-core because they are ancillary parameters governing how the number of plants enters demand, rather than the primary demand evolution mechanism.

**Core trade parameters (4 specs)**: TAU0-D, TAU0-F, TAU-L, and CORR. The trade cost structure is directly central to the paper's title claim about accounting for MNC-based trade growth. TAU-L (trade cost loading on wages, t = 79.43) is the most precisely estimated trade parameter and is highlighted in the paper as a key finding.

### Non-Core Classifications

**Labor, materials, and capital input demand parameters (18 specs)**: These are ancillary structural parameters governing how input demands (labor, materials, capital) depend on time trends, plant counts, and state variables. While necessary for the model to function, they are not the primary objects of interest for the paper's claims about trade growth. Many have mixed significance.

**Growth/accumulation parameters (8 specs)**: Capital accumulation parameters (GAM1-*, GAM2-*) governing the dynamic investment process. These are structural primitives of the dynamic model but not the focus of the paper's main claims. Several are not individually significant.

**Transition probability parameters (12 specs)**: Parameters governing the probability of state transitions (opening/closing plants). These are part of the dynamic discrete choice model infrastructure but are ancillary to the main trade claims.

**Box-Cox transformation parameters (12 specs)**: Functional form parameters (BC-1 through BC-12) governing the transformation of dependent variables. These are nuisance parameters that ensure proper functional form but are not of substantive interest. The SPECIFICATION_SEARCH.md notes that "the data strongly rejects linear specifications," which is useful but does not test the paper's main claims.

**Discrete choice logit parameters (36 specs)**: Coefficients for the four binary decisions (new domestic plant, new foreign plant, export, import). While these decisions are part of the model, the individual logit coefficients are not the paper's primary focus. Many are not individually significant (especially macro variables like GDP, exchange rates, and wages). The SPECIFICATION_SEARCH.md notes that "discrete choice equations show mixed significance."

**Variance-covariance parameters (34 specs)**: VPSI (discrete choice variance, 10 specs), SPHI diagonal (12 specs), and SETA diagonal (12 specs). These are nuisance parameters governing error distributions. All are significant, but they are not of substantive economic interest.

**Scale parameters (2 specs)**: Error scale parameters (SCALE1, SCALE2). Nuisance parameters for the simulation-based estimation procedure.

**Model fit diagnostics (6 specs)**: Wage equation fit (US and Canada mean error/RMSE) and predicted participation frequencies. These are descriptive diagnostics, not testable parameter estimates. The predicted frequencies (with SE = 0 and p = 1) are point predictions without uncertainty measures.

## Duplicate Specifications

Four parameter estimates appear to be duplicated across categories, where the same structural parameter appears both as an input demand parameter and as a discrete choice logit coefficient:

1. **AKD-E** appears as both `structural/params/capital/akd_e` (capital demand) and `structural/discrete/e/akd_e` (export logit) -- identical coefficient 4.019905, SE 5.21156
2. **AKF-I** appears as both `structural/params/capital/akf_i` (capital demand) and `structural/discrete/i/akf_i` (import logit) -- identical coefficient 2.630641, SE 1.97467
3. **GAM1-E** appears as both `structural/params/growth/gam1_e` (growth) and `structural/discrete/e/gam1_e` (export logit) -- identical coefficient -33.278549, SE 25.14163
4. **GAM2-I** appears as both `structural/params/growth/gam2_i` (growth) and `structural/discrete/i/gam2_i` (import logit) -- identical coefficient -30.323467, SE 10.89766

Additionally, two spec_ids appear twice in the CSV with different values, likely representing different aspects of the same parameter:
- `structural/params/growth/gam1_nd` appears twice (coef 0.005529 and coef 14.883907)
- `structural/params/growth/gam2_nf` appears twice (coef -0.000156 and coef 62.100152)

These duplicate IDs are problematic for downstream analysis and should be disambiguated.

## Notable Issues

### 1. Not a specification search in the traditional sense
The 151 rows are individual parameters extracted from a single estimation. There are no alternative samples, control sets, estimation methods, or functional forms being compared. This means the specification search provides no information about robustness to modeling choices.

### 2. Confidential data prevents replication
Because the BEA data is confidential, the specification search agent could not run any code. All results were extracted from the existing output file. No verification of computational accuracy is possible.

### 3. Convergence concerns
The SPECIFICATION_SEARCH.md notes the estimation "hit the iteration limit (24 iterations) but appears to have converged based on small parameter changes." Without access to gradient norms or convergence diagnostics, it is unclear whether the MLE actually converged.

### 4. Joint estimation makes individual t-tests misleading
In a jointly estimated structural model, individual t-statistics reflect the precision of each parameter conditional on all others being at their estimated values. They do not isolate the importance of individual parameters for model fit or for the paper's claims. A parameter could be individually insignificant but still be identified through its structural restrictions.

### 5. Model fit statistics are not specifications
The last 6 rows (wage equation fit and predicted frequencies) are descriptive statistics about model performance, not parameter estimates. The predicted frequencies have SE = 0 and p-value = 1, which are placeholder values rather than meaningful test statistics.

## Recommendations

1. **Flag this as a degenerate specification search**: All "specifications" come from one model, one estimation, one sample. Standard robustness metrics (e.g., share of significant specifications) are not meaningful here.

2. **Remove or flag duplicates**: The 4 cross-category duplicates (AKD-E, AKF-I, GAM1-E, GAM2-I) and 2 same-ID duplicates (GAM1-ND, GAM2-NF) inflate the specification count. After removing 6 duplicates, the true count is approximately 145 unique parameters.

3. **Distinguish parameter types in downstream analysis**: If this paper enters a meta-analysis, the core technology (4) and trade (4) parameters are the most relevant to the paper's claims. The other ~140 parameters are structural infrastructure, nuisance parameters, or fit diagnostics.

4. **Consider excluding from specification-search-based robustness analysis**: Because there is no variation in modeling choices, this paper cannot contribute to analyses of how results change across specifications. It may be more appropriate to treat it as a single-specification paper.
