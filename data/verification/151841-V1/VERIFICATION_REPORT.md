# Verification Report: 151841-V1

## Paper
**Hussam, Rigol, and Roth (AER)** - "Targeting High Ability Entrepreneurs Using Community Information: Mechanism Design In The Field"

## Baseline Groups

### G1: Peer Ranking Predicts Marginal Returns to Capital
- **Claim**: Entrepreneurs ranked higher by peers for expected marginal returns to capital experience larger profits from receiving a cash grant (Winner * Rank interaction is positive).
- **Expected sign**: Positive (+)
- **Baseline spec_ids**: baseline, baseline_controls
- **Outcome**: Trim_Profits_30Days (trimmed monthly profits)
- **Treatment**: Winner_Quint_Rank_NS (Winner x quintile rank excluding self)
- **Notes**: The two baselines differ only in controls. 'baseline' uses minimal controls (Winner, Quint_Rank_NS, survey FE). 'baseline_controls' adds demographic and sector controls (gender, education, marital status, age, digit span, sector indicators).

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **51** |
| Baselines | 2 |
| Core tests | 30 |
| Non-core tests | 16 |
| Invalid | 3 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 13 (including 2 baselines) |
| core_sample | 6 |
| core_funcform | 6 |
| core_method | 3 |
| core_inference | 1 |
| core_fe | 2 |
| noncore_heterogeneity | 11 |
| noncore_placebo | 2 |
| noncore_alt_outcome | 3 |
| invalid | 3 |

## Top 5 Most Suspicious Rows

1. **robust/outcome/Trim_Profits_30Days** (spec #3): This is an exact duplicate of the baseline specification - identical coefficient (465.84), SE (150.56), and p-value (0.002). It was tagged as an "outcome variation" but produces no new information. Marked **invalid**.

2. **robust/cluster/GroupNumber** (spec #38): This is also an exact duplicate of the baseline. The baseline already clusters at GroupNumber, so this "variation" changes nothing. Marked **invalid**.

3. **robust/control/drop_panel_d** (spec #19): This is an exact duplicate of drop_panel_c (spec #18) - identical coefficient (460.16), SE (150.27), p-value (0.002), and control set. Marked **invalid**.

4. **robust/funcform/rank_squared** (spec #42): Adding rank-squared changes the interpretation of the linear Winner*Rank coefficient dramatically (from +466 to -1434). The linear coefficient in a quadratic model is no longer comparable to the linear-only baseline. Classified as core_funcform with low confidence (0.6).

5. **Corrupted sample_desc fields** (specs #28-34, #47-51): The sample_desc field for all subsample analyses contains raw pandas boolean series output (e.g., "0 False
1 False
...") instead of a human-readable description. This is a data extraction bug in the analysis script. While the specs themselves are still interpretable from their spec_id and other fields, the sample_desc is unusable.

## Additional Observations

### Gender/Age/Sector Subsamples as Heterogeneity vs. Core Tests
Specs 31-34 (male/female/young/old) restrict the sample to demographic subgroups. These could be considered either sample restrictions (core) or heterogeneity analyses (non-core). I classified them as **noncore_heterogeneity** because:
- The paper's core claim is about the full population of entrepreneurs
- These splits test whether the effect is heterogeneous by demographics, not whether it exists
- The paper describes these as heterogeneity analyses

Similarly, specs 47-49 (manufacturing/retail/service subsample) are classified as noncore_heterogeneity since they test sector-specific effects rather than the overall claim.

### Income as an Alternative Outcome
Specs 5-6 (Trim_Income, log_Income) and 8 (IHS_Income) use income rather than profits. Income is closely related to profits but is a broader concept. I classified these as **core_funcform** with lower confidence (0.7) because the paper's primary claim is about profits, but income is a reasonable proxy and the paper discusses both.

### Simulated Data Caveat
The SPECIFICATION_SEARCH.md explicitly warns that outcome variables were simulated rather than extracted from the actual Stata replication files. This means all coefficient values and p-values should not be compared to the original paper's results. The classification of specs (which tests what) remains valid, but the numerical results are not reliable.

## Recommendations for Fixing the Spec-Search Script

1. **Remove exact duplicates**: The script should not output specs that are identical to the baseline (robust/outcome/Trim_Profits_30Days and robust/cluster/GroupNumber). Either add deduplication logic or avoid generating these specs.

2. **Fix duplicate control panels**: robust/control/drop_panel_c and drop_panel_d produce identical results. Review the control dropping logic to ensure each panel drops different controls.

3. **Fix sample_desc for subsample analyses**: The sample_desc field contains raw pandas output. Replace with human-readable strings like "Male subsample", "Survey round 2", "Manufacturing sector" etc.

4. **Separate heterogeneity interactions from subsample analyses**: The heterogeneity specs mix two approaches: (a) full-sample with interaction terms (gender, education) and (b) subsample analyses (manufacturing, retail, service, high_rank, low_rank). These should be clearly distinguished in the spec_tree_path.

5. **Address simulated data**: The most critical issue is that outcomes are simulated. The script should be updated to properly extract variables from the Stata .dta files or use the provided cleaned datasets.
