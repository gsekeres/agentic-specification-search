# Verification Report: 113710-V1

## Paper
**"Does Electoral Competition Curb Party Favoritism?"** by Curto-Grau, Sole-Olle, and Sorribas-Navarro (2018), AEJ: Applied Economics.

## Baseline Groups

### G1: Alignment causes higher transfers
- **Claim**: Political alignment (same party controls municipal and regional government) causes higher intergovernmental transfers per capita, identified via fuzzy RD at the 50% vote share threshold.
- **Expected sign**: Positive
- **Baseline spec_ids**: baseline
- **Baseline coefficient**: 98.06 (IV-2SLS, local quadratic, region FE, municipality clusters, N=6050)

This is the only baseline group. The paper secondary hypothesis about electoral competition moderating favoritism is tested via heterogeneity splits and interaction terms, which are classified as non-core.

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **95** |
| Baseline | 1 |
| Core tests (including baseline) | 60 |
| Non-core tests | 35 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 12 | Control progression and leave-one-out variations |
| core_sample | 30 | Bandwidth, period, region, outlier, donut, and panel restrictions |
| core_inference | 4 | Clustering variations (municipality, region, province, robust) |
| core_funcform | 6 | Polynomial order, log/asinh transforms, time interaction |
| core_fe | 1 | Dropping region fixed effects |
| core_method | 7 | LIML, OLS, reduced form, DiD, coalition/bloc alignment |
| noncore_heterogeneity | 19 | Competition, size, density, debt, and region-specific splits |
| noncore_placebo | 4 | Placebo cutoff tests at non-zero thresholds |
| noncore_diagnostic | 11 | First-stage regressions and covariate balance tests |
| noncore_alt_outcome | 1 | Total transfers (tc) instead of targeted transfers (tk) |

## Top 5 Most Suspicious Rows

1. **rd/poly/local_quadratic** (row 8): This is an exact duplicate of the baseline specification (same coefficient=98.057, same N=6050, same everything). It appears under the polynomial-order category but is numerically identical to baseline. Not invalid, but inflates the spec count.

2. **robust/cluster/municipality** (row 10): Exact duplicate of baseline -- same coefficient, same clustering. The baseline already clusters at municipality level (codiine), so this adds no new information.

3. **rd/controls/full** (row 20): Exact duplicate of robust/control/add_tipo (both have coefficient=84.151, N=6050, same controls). Listed under two different categories.

4. **robust/sample/min_obs_3** (row 86): Exact duplicate of robust/sample/balanced_panel (both have coefficient=66.842, N=4062). Requiring 3 observations in a 3-period panel is equivalent to balanced panel.

5. **rd/placebo/cutoff_neg0_15** (row 57): This placebo cutoff test at -0.15 is significant (p=5.85e-05, coef=88.9), which is noted in the SPECIFICATION_SEARCH.md as potentially concerning because -0.15 is close to the true cutoff at 0. However, this is correctly classified as a placebo test and the significance may reflect spillover. Methodologically appropriate to flag.

## Notes on Classification Decisions

### Alternative treatment definitions (core_method, not noncore_alt_treatment)
The robust/treatment/coalition_alignment (abcd) and robust/treatment/bloc_alignment (bloc) specs use alternative definitions of political alignment. Because alignment is the core concept and these are alternative measurements of the same concept (not fundamentally different causal objects), they are classified as core_method. However, both specs have identical coefficients to rd/bandwidth/bw_full (82.284), suggesting these may be re-labeled versions of the same regression with different variable names rather than genuinely different treatment definitions.

### Heterogeneity splits (noncore)
All heterogeneity specifications (high/low competition, large/small municipality, high/low density, high/low debt, and region-specific) are classified as noncore_heterogeneity. While they test the alignment effect within subgroups, they are not robustness tests of the main claim but rather explorations of effect heterogeneity. The competition splits are particularly relevant to the paper secondary hypothesis, but that hypothesis is about the interaction (moderation), not the main alignment effect.

### First-stage and reduced form
First-stage regressions (outcome=ab, treatment=dab) test instrument strength, not the main claim, so they are noncore_diagnostic. The reduced-form regression (outcome=tk, treatment=dab) is classified as core_method because it tests the same directional claim (alignment causes transfers) using the ITT estimand.

### OLS specifications
Several specs (rows 66, 74-86, 92-95) use OLS instead of IV-2SLS. For the region-specific heterogeneity specs (region_1 through region_10), this is likely because IV cannot be run within single regions (too few observations), so these are classified as noncore_heterogeneity rather than core. The panel-restriction OLS specs (balanced_panel, min_obs_2, min_obs_3) and symmetric bandwidth OLS specs test the same directional claim with a different estimator and are classified as core_sample.

### Total transfers (noncore_alt_outcome)
The robust/outcome/total_transfers spec uses tc (total transfers) instead of tk (targeted transfers). This is a different outcome variable that captures a broader transfer concept, making it a different claim rather than a robustness check of the original one.

## Recommendations for Spec-Search Script

1. **Eliminate exact duplicates**: Three pairs of duplicates inflate the spec count (rd/poly/local_quadratic = baseline; robust/cluster/municipality = baseline; rd/controls/full = robust/control/add_tipo; robust/sample/min_obs_3 = robust/sample/balanced_panel). The script should detect when a specification produces identical point estimates and flag or deduplicate.

2. **Alternative treatment variables may be mislabeled**: The coalition_alignment and bloc_alignment specs produce identical coefficients to bw_full, suggesting these may not be genuine alternative treatment definitions. The script should verify that the actual regression inputs differ.

3. **Distinguish IV from OLS specs more clearly**: Several specs switch from IV-2SLS to OLS without explicit labeling. The model_type field captures this, but the spec_tree_path does not always reflect the estimator change. Consider adding an estimator dimension to the spec tree.

4. **Consider separating the competition-moderation hypothesis**: The paper secondary hypothesis (competition moderates favoritism) could be treated as a separate baseline group if the goal is to also audit that claim. Currently all competition-related heterogeneity specs are noncore relative to the alignment-level claim.
