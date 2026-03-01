# Specification Surface: 112474-V1

**Paper**: Dinkelman (2011), "The Effects of Rural Electrification on Employment: New Evidence from South Africa"
**Design**: Instrumental Variables (2SLS)
**Created**: 2026-02-24

---

## 1. Paper Summary

This paper estimates the causal effect of rural electrification on employment in South Africa using an instrumental variables strategy. The treatment variable (T) is a binary indicator for whether a community received an Eskom electrification project between 1996 and 2001. The instrument is land gradient (mean_grad_new), which affects the cost of extending the electricity grid. Outcomes are first-differenced employment rates (2001 minus 1996) for women and men separately.

The main analysis uses community-level census data from rural tribal areas in KwaZulu-Natal (KZN), restricted to communities with at least 100 adults in both census years (N=1816).

---

## 2. Baseline Groups

### G1: Female Employment (Main Claim)
- **Outcome**: d_prop_emp_f (change in female employment proportion)
- **Treatment**: T (electrification indicator), instrumented by mean_grad_new
- **Estimand**: LATE of electrification on female employment
- **Population**: Rural tribal communities in KZN with >= 100 adults (largeareas==1)
- **Baseline spec**: Table 4, Column 9 (IV with full controls + district FE + service changes)
  - Controls: kms_to_subs0, baseline_hhdens0, base_hhpovrate0, prop_head_f_a0, sexratio0_a, prop_indianwhite0, kms_to_road0, kms_to_town0, prop_matric_m0, prop_matric_f0, d_prop_waterclose, d_prop_flush
  - FE: district (dccode0)
  - Clustering: community (placecode0)

### G2: Male Employment (Main Claim)
- **Outcome**: d_prop_emp_m (change in male employment proportion)
- Same treatment, instrument, population, controls, FE, and clustering as G1
- The paper presents both outcomes as co-equal headline results (Table 4 has parallel columns for women and men)

**Rationale for two baseline groups**: The paper's main Table 4 reports results for both female and male employment as parallel headline findings. The female employment result is statistically significant and economically large; the male result is smaller and not significant. Both are interpreted as main claims.

---

## 3. Design Details

- **Estimator**: 2SLS (ivreg2/ivreg in Stata)
- **Instrument**: mean_grad_new (community land gradient from SRTM satellite data)
- **Identification**: Land gradient affects electrification cost but is argued to be excludable from employment changes conditional on controls
- **Bundled estimator**: IV is a multi-component bundle (first stage + second stage). Controls and district FE are shared across stages (linked_adjustment = true).

---

## 4. Core Universe Design

### Additional Baseline Specs
For each group, the paper's Table 4 progression includes:
- **Col 7**: IV, no controls (bivariate)
- **Col 8**: IV with controls, district FE, no service changes
- **Col 9** (baseline): IV with controls, district FE, plus service changes (d_prop_waterclose, d_prop_flush)

### Design Variants
- **LIML**: Limited Information Maximum Likelihood as robustness to weak instruments (standard IV diagnostic alternative)

### RC Axes

#### Controls (LOO)
Leave-one-out dropping each of the 12 controls (preserving district FE). This is the main RC axis.

#### Controls (Progression)
- Bivariate: IV with no controls, no FE
- Geographic: kms_to_subs0, kms_to_road0, kms_to_town0
- Demographic: baseline_hhdens0, base_hhpovrate0, prop_head_f_a0, sexratio0_a, prop_indianwhite0, prop_matric_m0, prop_matric_f0
- Full without services: all 10 baseline controls + district FE, excluding d_prop_waterclose and d_prop_flush

#### Controls (Random Subsets)
10 random control subsets sampled from the 10 baseline controls (excluding service changes), stratified by size. Seed: 112474.

#### Sample Restrictions
- Full sample: all 1992 observations (no largeareas restriction)
- No roads: exclude communities with count_roads==1 (paper's Appendix 3 Table 2)
- No nearby treated 1km: exclude controls near treated areas (1km buffer)
- No nearby treated 5km: exclude controls near treated areas (5km buffer)

#### Trimming
- Trim outcome at 1st/99th percentiles
- Trim outcome at 5th/95th percentiles

#### Fixed Effects
- Drop district FE entirely
- Add political heterogeneity control (hetindex) per Appendix 3 Table 2

#### Functional Form
- Inverse hyperbolic sine transform of outcome

#### Estimation
- OLS reduced form: regress outcome directly on instrument (mean_grad_new) with controls and FE, to verify reduced-form relationship

---

## 5. Inference Plan

**Canonical**: Cluster-robust SEs at the community level (placecode0), matching the paper's baseline.

**Variants**:
- HC1 robust (no clustering)
- Cluster at district level (dccode0) -- only 10 clusters, so this is a stress test

---

## 6. Constraints

- Controls count envelope: [0, 12]
- Linked adjustment: true (controls shared across IV stages)
- District FE: dccode0 (10 districts)
- Clustering: placecode0 (274 unique communities in sample)

---

## 7. Budgets and Sampling

- Max core specs per group: ~45 (2 additional baselines + 1 LIML + 12 LOO + 4 progression + 10 random subsets + 6 sample + 2 trim + 2 FE + 1 form + 1 OLS reduced form + baseline = ~42)
- Control subset budget: 10
- Seed: 112474
- Sampler: stratified_size (stratify by number of controls included)

---

## 8. What is Excluded and Why

- **Heterogeneity analyses** (Appendix 2 age groups, poverty quintiles): These explore who drives the LATE, not the main estimand. Would be `explore/*`.
- **Placebo test** (early-treated areas): This is a falsification check, not an estimate of the main causal effect. Would be `diag/*`.
- **Services outcomes** (Table 7: electricity, wood, cooking): Different outcome concepts, not the employment claim. Would be `explore/*`.
- **Composition changes** (Table 9: population, education): Different outcome concepts. Would be `explore/*`.
- **Household survey data** (Table 6): Different data source with different unit of analysis.
- **Spatial SE corrections**: These would require custom spatial HAC routines not readily available in Python. Noted as potentially relevant inference variant but excluded from feasible set.
