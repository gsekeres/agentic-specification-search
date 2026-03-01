# Specification Surface: 113779-V1

## Paper Overview
- **Title**: Traffic Congestion and Infant Health: Evidence from E-ZPass (Currie and Walker, 2011)
- **Design**: Difference-in-differences
- **Data**: Birth certificate data from NJ and PA linked to proximity to toll plazas. Compressed .dta.gz files included.
- **Key context**: Classic DiD exploiting the staggered adoption of E-ZPass electronic toll collection to reduce traffic congestion and pollution near toll plazas.

## Baseline Groups

### G1: Birth Outcomes Near Toll Plazas (Table 3)

**Claim object**: E-ZPass adoption improves infant health outcomes (birth weight, prematurity, low birth weight, gestation) for mothers living near toll plazas, through reduced traffic congestion and air pollution.

**Baseline specification**:
- Formula: `birth_weight ~ postXnearEZpass + postEZpass + nearEZpass + controls | mother_FE + year + month`
- Outcome: Birth weight (primary), with prematurity, LBW, gestation as additional outcomes
- Treatment: Post-E-ZPass x Near-toll-plaza interaction (postXnearEZpass)
- Near distance threshold: 2km (baseline), 1.5km (robustness)
- Control distance: 10km (baseline), 5km (robustness)
- Trim window: 3 years pre/post
- FE: Mother FE (preferred), or zip/area FE
- Clustering: Date level

**Additional baseline-like rows**:
- Prematurity indicator
- Low birth weight indicator
- Gestation weeks

## Design Variants

The paper explores extensive robustness along design parameters:
1. **Near distance thresholds**: 1.5km vs 2km
2. **Control distance**: 5km vs 10km
3. **FE structure**: Mother FE, zip FE, no geographic FE
4. **Trim window**: 3-year vs 5-year windows around E-ZPass adoption
5. **Geographic aggregation**: Individual births vs zip-level vs county-level

## RC Axes Included

### Controls
- **Leave-one-out**: Drop each maternal demographic control individually
- **Standard sets**: Minimal (post/near/interaction only), demographics, full with smoking indicators
- **Additions**: Smoking indicator, race indicator

### Sample restrictions
- Outlier trimming on birth weight
- Race-specific subsamples (blacks only)
- Smoker exclusion
- Geographic aggregation levels

### Fixed effects
- Mother FE + year-month (preferred)
- Zip + year-month
- Year-month only

### Clustering alternatives
- Date (baseline), zip, toll plaza

### Functional form
- Log birth weight

### Preprocessing
- Winsorization of outcome

## What Is Excluded and Why

- **First-stage air quality analysis (Table in separate do-file)**: Different dataset (air quality monitors, not birth data). Different claim object.
- **Housing sales analysis (separate do-file)**: Different outcome and dataset.
- **Event study graphs**: These are visual displays of the DiD dynamics, not separate claim objects. Covered by the pre-trends diagnostic.
- **Propensity score trimming**: This is a design parameter variant included in the design specs.

## Budgets and Sampling

- **Max core specs**: 70
- **Max control subsets**: 20
- **Seed**: 113779
- **Sampling**: Stratified for control subsets. Full enumeration for design variants.

## Inference Plan

- **Canonical**: Cluster at date level (matching paper)
- **Variants**: Cluster at zip, toll plaza, HC1 robust
