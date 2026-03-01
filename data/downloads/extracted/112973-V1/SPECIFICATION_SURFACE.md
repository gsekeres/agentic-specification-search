# Specification Surface: 112973-V1

## Paper Overview
- **Title**: The Housing Market Impacts of Shale Gas Development (Muehlenbachs, Spiller, & Timmins, 2015, AER)
- **Design**: Difference-in-differences with property FE and county-by-year FE
- **Data**: Repeat-sale residential properties in the Marcellus Shale region of Pennsylvania
- **Key tables**: Table 2 (main DiD, Panel A = well pads, multiple distance radii), Table 3 (PWSA/mini-PWSA), Tables A1-A7 (appendix robustness)

## Baseline Groups

### G1: Shale Gas Drilling Impact on House Prices (Table 2, Panel A)

**Claim object**: Shale gas well pad drilling within a given distance radius reduces property values, with a differential (more negative) effect for properties dependent on groundwater (GW) vs. those on piped water. Identification exploits within-property variation over time in the number of nearby well pads, with county-by-year FE absorbing local trends.

**Baseline specifications**:
- Table 2, Panel A, K=2km, Full sample: `xtreg logprice2012 post postGW npads npadsGW _IyeaXcou_* dq_*, i(property) fe robust cluster(censustractstatecounty)`
- Same for Boundary sample (restricts to narrow band around public water service area border)
- Distance radii: K = 1km, 1.5km, 2km, 2.5km, 3km
- Focal parameter: `postGW` (differential GW effect), or `post + postGW` (total effect for GW properties)

## RC Axes Included

### Data construction (distance radius)
- **K = 1km, 1.5km, 2km (baseline), 2.5km, 3km**: The paper's main variation across columns. Each radius redefines the treatment variable `post` and its GW interaction.

### Sample restrictions
- **Full vs. boundary sample**: Both in the paper. Boundary sample restricts to properties near the PWSA border.
- **Single-well-only**: The robustness do-file restricts to properties with at most one well within 2km (cleaner treatment).
- **Outlier trimming**: Trim extreme sale prices
- **Period restrictions**: Sensitivity to time window

### Controls
- **Add house characteristics**: The repeat-sales design does not need time-invariant controls, but time-varying characteristics (renovations, etc.) could be added if available
- **LOO on npads/npadsGW**: Drop the 20km well count control

### Fixed effects
- **Decompose county-year FE**: Use separate county + year FE instead of interacted
- **Add finer FE**: Census tract FE or school district FE

### Treatment definition
- **Bore count instead of pad count**: Different granularity of well measurement
- **Production intensity**: Use actual gas production instead of well presence
- **Permitted but undrilled**: Wells that are permitted but not yet drilled (placebo-like treatment)
- **Time decomposition**: Separate wells drilled <1 year ago vs. >1 year ago

### Preprocessing
- **Nominal prices**: Use nominal sale prices instead of inflation-adjusted

### Joint variations
- Distance radius x sample (full/boundary) combinations
- Distance radius x treatment definition combinations

## What Is Excluded and Why

- **Table 3 (PWSA/mini-PWSA analysis)**: Uses a different sample restriction and treatment definition (distance to PWSA boundary). This is a robustness approach, not a separate claim object, but the required data construction is significantly different.
- **Table 4 (matching estimates)**: Matching-based estimator changes the identification strategy; treated as exploration.
- **Tables A4-A5 (probability of sale, new construction)**: Different outcomes, not the main house-price claim.
- **Figures 4-5 (Linden-Rockoff plots)**: Spatial discontinuity analysis, different design.

## Budgets and Sampling

- **Max core specs**: 80
- **Max control subsets**: 15
- **Seed**: 112973
- Most variation from 5 distance radii x 2 samples x treatment definitions
- Control variation is limited (property FE absorbs most cross-sectional controls)

## Inference Plan

- **Canonical**: Cluster-robust SE at census tract level (matching the paper)
- **Variants**: Cluster by county (coarser), robust without clustering, cluster by property (for repeat sales)

## Key Constraints

- Property FE are always included (repeat-sales design)
- County-by-year FE are always included in baseline (though decomposition is an RC)
- GW interaction structure (post, postGW, npads, npadsGW) should be maintained for comparability
- The distance radius (K) defines the treatment variable and must be specified as a data construction parameter
