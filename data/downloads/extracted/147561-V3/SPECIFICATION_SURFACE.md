# Specification Surface: 147561-V3

## Paper: Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo (Balan, Bergeron, Tourek, and Weigel)

## Baseline Groups

### G1: Effect of local (chief-led) collection on property tax compliance

- **Outcome**: taxes_paid (binary: whether property tax was paid during the 2018 campaign)
- **Treatment**: t_l (indicator for local/chief-led collection assignment, tmt==2 vs tmt==1 central)
- **Estimand**: ITT effect of assigning a neighborhood to local chief-led tax collection (vs central/state collection) on individual-level property tax compliance
- **Population**: Property owners in Kananga, Kasai Central, D.R. Congo, excluding villa properties and properties with missing rate information, during the 2018 property tax campaign
- **Baseline spec**: Table 4, Compliance Panel, Col 4 (reg taxes_paid t_l i.stratum i.house i.time_FE_tdm_2mo_CvL if inlist(tmt,1,2), cl(a7))
- **Additional baseline**: Table 4, Revenues Panel, Col 4 uses taxes_paid_amt as outcome (continuous revenue amount)

### Why one baseline group

The paper's central question is whether delegating tax collection to local chiefs (who have local information) increases compliance relative to central/state-led collection. Table 4 presents the main results for both tax compliance (extensive margin, binary) and revenues (intensive margin, continuous amount). These are two facets of the same claim object. The revenues outcome is included as an additional baseline spec (baseline__revenues) rather than a separate group because the treatment concept and population are identical.

Tables 5-9 explore mechanisms (assessment accuracy, bribes, attitudes, collector visits, property values) rather than independent main claims. The CLI (central with local information) and CxL (central x local) arms are explored in Appendix tables, not the main headline comparison.

## Experimental Design

- **Randomization**: Neighborhoods (polygons, identified by a7) were randomly assigned to treatment arms within strata.
- **Treatment arms**: (1) Central collection, (2) Local/chief collection, (3) Central with local information (CLI), (4) Central x Local (CxL), (5) Control (no campaign).
- **Focal comparison**: Local (tmt==2) vs Central (tmt==1). The paper restricts to `inlist(tmt,1,2)` for Table 4.
- **Clustering**: At the neighborhood (a7) level, matching randomization.
- **Fixed effects**: Stratum FE (randomization block), house-type FE, month/time FE.

## Core Universe

### Design estimator implementations

- **Difference in means**: No FE, simple regression of taxes_paid on t_l with clustering
- **Strata FE only**: Includes stratum FE but no month or house FE (Table 4 Col 1 equivalent)

### Fixed effects axes (the main revealed variation in Table 4)

Table 4 progressively adds fixed effects across 5 columns:
- Col 1: Stratum FE only
- Col 2: Stratum + Month FE
- Col 3: Stratum + Month FE, collapsed to polygon means (robust SE)
- Col 4: Stratum + Month + House FE
- Col 5: Stratum + Month + House FE, excluding exempt properties

This progression (rather than control variation) is the main revealed search dimension. The core RC spec IDs enumerate these combinations:
- stratum_only
- stratum_month
- stratum_month_house (baseline)
- drop house FE from baseline
- drop time FE from baseline

### Sample axes

- Exclude exempt properties (Table 4 Col 5: `exempt!=1`)
- Collapse to polygon means (Table 4 Col 3 equivalent)
- Trim taxes_paid_amt at 1st/99th percentile (for revenue outcome)

### Functional form axes (for revenue outcome only)

- log(1 + taxes_paid_amt)
- asinh(taxes_paid_amt)

### Treatment definition axes

- Include CLI arm (tmt==3) alongside central and local
- Include CxL arm (tmt==4) alongside central and local
- Pooled local-type treatment: combine tmt==2 and tmt==4 vs tmt==1

## Inference Plan

- **Canonical**: Cluster SEs at neighborhood (a7) level, matching the unit of randomization. Used in all Table 4 columns except Col 3.
- **Variant**: HC1 robust SEs (used in the polygon-mean specification, Table 4 Col 3)

## Constraints

- Control-count envelope: [0, 0]. The paper does not include individual-level covariates in Table 4. The only controls are fixed effects (stratum, month, house type).
- No linkage constraints: single-equation RCT design.
- Sample restriction: always restricted to Central vs Local arms (inlist(tmt,1,2)) unless the treatment definition RC explicitly broadens this.
- Data construction: villas (house==3) dropped, missing rate observations dropped, pilot polygons dropped. These are not varied.

## Budget

- Max core specs: 60
- No control subset sampling needed (no individual controls to vary)
- Estimated total core specs: ~18 (1 baseline compliance + 1 baseline revenues + 2 design + 7 FE combos + 3 sample + 2 functional form + 3 treatment definition = ~19)
- Seed: 147561

## What is excluded and why

- **Table 5 (mechanisms: assessment accuracy, bribes, attitudes)**: These are mechanism outcomes, not the main compliance claim. Belong in explore/variable_definitions.
- **Table 6 (collector visits)**: Mechanism analysis of how local chiefs contact property owners.
- **Table 7 (chief knowledge and consultations)**: Mechanism analysis.
- **Table 8 (heterogeneity by property value)**: Heterogeneity analysis, not a separate main claim. Belongs in explore/heterogeneity.
- **Table 9 (property value and willingness to pay)**: Mechanism/secondary analysis.
- **CLI and CxL arm comparisons (Tables A8-A15)**: Alternative treatment arm comparisons explored in appendix. The focal comparison is Local vs Central.
- **Table 1 (summary statistics)**: Descriptive, not an estimate.
- **Table 3 (balance and attrition)**: Diagnostics, recorded in diagnostics_plan.
