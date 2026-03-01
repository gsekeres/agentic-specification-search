# Specification Surface Review: 112973-V1

## Summary

The surface was reviewed against the paper's Stata do-files (06_DID_mainGW_tables-2-panel-A-and-A1.do, 06_DID_mainGWrobustness.do, and related files) and the classification. This is Muehlenbachs, Spiller, & Timmins (2015), studying shale gas drilling impacts on house prices in Pennsylvania.

## Baseline Groups

### G1: Shale Gas Drilling Impact on House Prices
- **Status**: Correctly defined. Single claim object (drilling -> house prices, with GW heterogeneity).
- **Design code**: `difference_in_differences` is correct. The design exploits within-property variation over time in nearby well drilling, with county-by-year FE.
- **Design audit**: Comprehensive. Notes the property FE, county-by-year FE, census tract clustering, distance radius parameter, and the GW interaction structure.
- **Both full and boundary samples captured** as baseline specs (correct -- the paper presents both as headline results).

### No additional baseline groups needed
- The GW interaction (postGW) is part of the same specification, not a separate claim.
- Tables on probability of sale, new construction, and matching are correctly excluded from the core surface.

## Checklist Results

### A) Baseline groups
- Single baseline group with two baseline specs (full and boundary sample) -- correct.
- The paper's headline result is the GW interaction effect. The distance radius variation is correctly treated as a data construction RC, not separate baseline groups.

### B) Design selection
- `difference_in_differences` is correct for this repeat-sales panel DiD.
- The `design/difference_in_differences/estimator/twfe` variant is included (already the baseline estimator).
- No heterogeneity-robust DiD estimators (Callaway-Sant'Anna, etc.) are included. This is acceptable because the treatment is continuous (well pad count), and the staggered-DiD literature primarily applies to binary treatments with staggered rollout. However, could be considered as exploration.

### C) RC axes
- **Data construction (distance radius)**: Critical axis. 5 radii (1km through 3km) matching the paper. Correctly captured.
- **Sample**: Full vs. boundary, single-well-only, outlier trimming -- good coverage.
- **Controls**: Limited by the repeat-sales design (property FE absorbs cross-sectional controls). LOO of npads/npadsGW is appropriate.
- **Treatment definition**: Bore count, production intensity, time decomposition are all present in the paper's appendix tables. Good coverage.
- **FE variations**: Decomposing county-year FE and adding finer FE are appropriate.

### D) Controls multiverse policy
- `controls_count_min=4`, `controls_count_max=6` -- correct given the small set of regressors.
- Mandatory controls (post, npads, npadsGW) correctly identified.
- `linked_adjustment=false` -- correct.
- The GW interaction structure should be maintained for comparability -- correctly noted.

### E) Inference plan
- Canonical census tract clustering matches the paper -- correct.
- County-level clustering (coarser) is an important variant.
- Property-level clustering is appropriate for repeat sales.

### F) Budgets + sampling
- 80 specs total is reasonable for 5 distance radii x 2 samples x treatment definitions + joint combinations.
- Full enumeration of the small control pool is appropriate.
- Seed specified (112973).

### G) Diagnostics plan
- Permitted-but-undrilled wells as a falsification test is correctly included. This is a natural placebo in this design.

## Key Constraints and Linkage Rules
- Property FE always included (repeat-sales design foundation).
- County-by-year FE are part of the baseline identification.
- GW interaction structure (four variables: post, postGW, npads, npadsGW) should be maintained together.
- Distance radius K parameterizes the treatment variable construction.

## What's Missing
- Could consider adding house characteristics that vary over time (e.g., renovations) if available in the data, but the repeat-sales design is designed to avoid this need.
- Spatial clustering (e.g., Conley SE) could be a useful inference variant given the geographic nature of the treatment, but is not blocking.

## Final Assessment
**Approved to run.** The surface correctly identifies the single baseline claim object, captures the key distance-radius variation as a data construction RC, and includes the paper's full/boundary sample comparison. The diagnostic (permitted-undrilled wells) is appropriate. No blocking issues.
