# Specification Surface: 130784-V1

**Paper**: "Child Marriage Bans and Female Schooling and Labor Market Outcomes: Evidence from Natural Experiments in 17 Low- and Middle-Income Countries"

**Design**: Generalized Difference-in-Differences

---

## 1. Baseline Groups

### G1: Effect of child marriage bans on child marriage

**Claim object**: The paper's headline claim is that child marriage ban legislation reduces child marriage rates, with the effect identified through cohort-by-region intensity variation across 17 low- and middle-income countries.

- **Outcome**: `childmarriage` (binary: married before age 18)
- **Treatment**: `bancohort_pcdist` = `bancohort_pc` x `distance`, where `bancohort_pc` indicates whether a woman's birth cohort was affected by the ban (age < 18 + interviewyear - banyear_pc), and `distance` captures pre-ban regional child marriage intensity (mean years married before 18 among pre-ban cohorts in the same country-region-urban cell)
- **Fixed effects**: country x age (`countryage`) and country x region x urban (`countryregionurban`)
- **Clustering**: `countryregionurban` (282 clusters)
- **Sample**: Women age 15-49, `regsample_pc==1` (17 countries with at least one post-ban DHS round), complete education and employment data

The paper presents multiple outcomes in Table 4 (childmarriage, marriage_age, age_firstbirth, educ, employed). We focus on `childmarriage` as the primary claim object (G1 baseline) and include `marriage_age`, `educ`, and `employed` as additional baseline specs.

---

## 2. Why one baseline group

The paper uses the same identification strategy, same treatment variable, same fixed effects, and same sample for all five outcomes. The outcomes measure different aspects of the same policy question: what are the effects of child marriage bans? However, `childmarriage` is the most directly targeted outcome and appears first in Table 4.

The additional outcomes (marriage_age, educ, employed) are listed as `baseline_spec_ids` within G1 since the paper treats them as co-equal headline results.

---

## 3. Core Universe Design

### Baseline specs
- `baseline`: childmarriage ~ bancohort_pcdist | countryage + countryregionurban (Table 4, Panel A, Col 1)
- `baseline__marriage_age`: marriage_age outcome (Table 4, Panel A, Col 2)
- `baseline__educ`: educ outcome (Table 4, Panel A, Col 4)
- `baseline__employed`: employed outcome (Table 4, Panel A, Col 5)

### RC axes included

**Controls (add)**: The baseline specification has NO controls beyond FE. The paper's robustness checks add:
- Linear survey year control (`interviewyear`)
- Quadratic survey year (`interviewyear` + `interviewyear_sqd`)
- Compulsory schooling law control (`cslcohortdist`)
- Survey year + ban year FE (`i.interviewyear i.banyear_pc`)
- Region-specific linear trends (`regtrend*`)
- Demographics: country x religion, country x ethnicity, country x visitor status FE

**Sample restrictions**:
- Urban-only, Rural-only subsamples (Table 4 Panels B, C)
- Countries grouped by baseline legal minimum age (Table 4 Panels D, E, F)
- Age restrictions: drop individuals > 40 years from ban (Table A13), > 30 years
- Leave-one-country-out jackknife (Table A7): drop each of 17 countries

**Treatment construction (data)**:
- Alternative intensity measures: distance2 (proportion), distance40, distance50, distance25, distance_reg
- Binary intensity: above-mean, above-75th-percentile
- Alternative ban-cohort cutoffs: age 17, 16, 15 (Table A14)

**Outcome thresholds**:
- childmarriage16, childmarriage15, childmarriage14, childmarriage13, childmarriage12 (Table 5)

**Additional FE**:
- Interview year FE, ban year FE (Table A6 Panel C)

---

## 4. Constraints

- **Controls count**: min=0, max=3 (baseline has 0 controls; robustness adds at most a few continuous controls or FE sets)
- **Linked adjustment**: false (no bundled estimator)
- No control-subset sampling needed (controls are added one-at-a-time, not sampled from a pool)

---

## 5. Budget

- Target: ~55-60 core specifications (well within budget of 100)
- No controls-subset sampling needed
- Full enumeration feasible

---

## 6. Inference Plan

- **Canonical**: CRV1 clustered at countryregionurban (matches paper)
- **Variants**: (1) Cluster at country level (17 clusters); (2) HC1 robust (no clustering)

---

## 7. What is excluded

- **Event study (Figure 5)**: This is a dynamic analysis with age-at-ban dummies x intensity; treated as exploration/diagnostic, not core
- **IV regressions (Table 6)**: Different estimand (LATE); excluded from core
- **Occupation outcomes (Table A2)**: Heterogeneity, not headline claim
- **Employment type outcomes (Tables A8-A10)**: Sub-outcomes of employment
- **Heterogeneity by income level (Table A12)**: Exploration
- **Years married before 18 (Table A11)**: Alternative outcome coding, not headline
