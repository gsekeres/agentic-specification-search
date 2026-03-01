# Specification Surface Review: 113893-V1

## Summary

The surface was reviewed against the paper's main and appendix do-files and the .dta dataset. This paper uses geographic variation in Serbian radio signal strength to identify the effect of cross-border media on nationalism in Croatia.

## Baseline Groups

### G1: Nationalist Vote Share (Nazi_share)
- **Status**: Correctly defined. The headline claim is about radio exposure increasing nationalist voting.
- **Design code**: `instrumental_variables` is correct. Signal strength instruments for radio availability.
- **Design audit**: Comprehensive. Includes instrument details, weighting, clustering, and control set structure. The `linked_adjustment=true` flag correctly reflects that the same controls appear in both the first stage and reduced form/2SLS.

### G2: Ethnic Graffiti
- **Status**: Correctly defined as a secondary baseline group. The graffiti outcome is a separate behavioral measure of ethnic tensions.
- **Design code**: `instrumental_variables` is correct (same identification strategy).
- Budget appropriately smaller for secondary outcome.

## Checklist Results

### A) Baseline groups
- Two baseline groups (vote share, graffiti) -- appropriate. These are the paper's two main claim types (voting behavior and ethnic conflict).
- Other party vote shares (hdz_share, sdp_share) and turnout are correctly excluded as supplementary.
- Survey and lab experiment results are correctly excluded as different datasets/settings.

### B) Design selection
- `instrumental_variables` is appropriate. The paper uses both OLS (intent-to-treat / availability) and reduced form (signal strength) approaches.
- Design variants correctly include 2SLS, continuous signal strength, signal dummies, and multiple station dummies.

### C) RC axes
- **Controls**: Excellent coverage. The paper itself uses two control sets (short/long) and the surface captures LOO, standard sets, and the robustness additions from Table 6 (Hungarian radio, Croatian radio, free-space loss).
- **Sample**: Distance restriction (75km) from Table 7 is appropriately included.
- **Weights**: Unweighted variant is a useful robustness check.
- **Missing**: No data construction axes, which is appropriate for a cross-sectional analysis with standardized variables.

### D) Controls multiverse policy
- `controls_count_min=6` (geography + region only) and `controls_count_max=21` (full long set) -- correct range reflecting the paper's revealed search space.
- `linked_adjustment=true` correctly enforces same controls in first stage and reduced form.
- Region dummies (r1-r5) treated as mandatory -- appropriate given the geographic identification strategy.

### E) Inference plan
- Canonical clustering at municipality matches all paper specifications.
- Spatial Conley SE is a valuable addition given the geographic identification.

### F) Budgets + sampling
- Budget of 75 (G1) + 25 (G2) = 100 total specs is reasonable.
- Seed specified (113893).
- Block-based control organization is well-structured.

### G) Diagnostics plan
- First-stage F-statistic is essential for IV.
- AET balance test captures the paper's own exclusion restriction checks (Table 2B).

## Key Constraints and Linkage Rules
- **Linked adjustment**: Same controls in first stage and reduced form. This is correctly flagged.
- Region dummies (r1-r5) are mandatory across all specifications.
- Population weighting is the default; unweighted is an explicit RC variant.

## What's Missing
- Election-year comparisons (2003 vs 2011) could be useful but are correctly categorized as explore/*.
- Nothing blocking from the core specification search.

## Final Assessment
**Approved to run.** The surface correctly identifies the IV identification strategy, handles the control set structure well, and appropriately separates the two main claim objects. The linked adjustment constraint is correctly flagged. Data is available (.dta format).
