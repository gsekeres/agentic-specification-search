# Verification Report: 207766-V1

**Paper**: "Organized Voters: Evidence from Governmental Transfers" by Camille Urvoy (AER 2025)
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## 1. Baseline Groups

### G1: Congruent Organization Transfers (PRIMARY)
- **Claim**: Municipalities where the ruling party narrowly won receive significantly more government transfers per capita to politically congruent nonprofit organizations (~1.39 EUR/capita, p < 0.001).
- **Baseline spec_ids**: `baseline` (outcome = `amount_congr1_3_pcap`)
- **Expected sign**: Positive (+)
- **Notes**: This is the paper's central finding. The congruent classification uses 3-group text-based LASSO classification.

### G2: Total Organization Transfers (SECONDARY)
- **Claim**: Total government transfers per capita to nonprofit organizations are higher in aligned municipalities (~1.08 EUR/capita, p = 0.023).
- **Baseline spec_ids**: `baseline` (outcome = `amount_pcap`)
- **Expected sign**: Positive (+)
- **Notes**: Weaker and less robust than the congruent-specific effect. The paper emphasizes the congruent channel.

---

## 2. Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **149** |
| Core tests (is_core_test=1) | 62 |
| Non-core tests (is_core_test=0) | 87 |
| Baselines (is_baseline=1) | 4 |
| Invalid | 0 |
| Unclear | 0 |

### Core Tests by Category
| Category | Count |
|----------|-------|
| core_method | 26 |
| core_sample | 22 |
| core_funcform | 12 |
| core_inference | 2 |
| core_fe | 0 |

### Non-Core Tests by Category
| Category | Count |
|----------|-------|
| noncore_alt_outcome | 51 |
| noncore_heterogeneity | 18 |
| noncore_diagnostic | 12 |
| noncore_placebo | 6 |

### Core Tests by Baseline Group
| Group | Core (incl baseline) |
|-------|---------------------|
| G1 (congruent) | 51 |
| G2 (total) | 11 |

---

## 3. Classification Rationale

### Why many specs are non-core (51 alt_outcome)
The specification search runs each specification across multiple outcome variables simultaneously. For each kernel, polynomial, or inference variation, 4 outcomes are estimated: `amount_congr1_3_pcap` (congruent), `amount_congr2_3_pcap` (moderate), `amount_congr3_3_pcap` (non-congruent), and `amount_pcap` (total). The moderate and non-congruent outcomes serve as **falsification/contrast tests** -- the paper's theory predicts null effects for these, so they are not robustness checks of the main claim. Additionally, outcome decompositions by organization age (old/young), funding source (multi-ministry/single-ministry), headquarters location (local/non-local), and margin type (extensive/intensive) are mechanism explorations rather than robustness checks.

### Heterogeneity (18 specs)
The paper reports substantial heterogeneity analysis: campaign spending level, government party local popularity, incumbency status, turnout level, municipality size, and election timing. These test effect moderation, not the main claim, so they are classified as non-core. Some were coded under `robustness/sample_restrictions.md` by the spec search (e.g., large/small municipalities, pre/non-pre-election years) but are substantively heterogeneity splits.

### Diagnostics (12 specs)
Individual year regressions (year_2005 through year_2016) run the RD on single cross-sections. These are diagnostic checks of within-year effects and lack panel power, so they are classified as diagnostics rather than robustness checks. The leave-one-year-out variants (drop_year_*), by contrast, are genuine robustness checks and are classified as core.

### Placebos (6 specs)
Six placebo cutoff tests (at margins -15, -10, -5, +5, +10, +15) test design validity rather than the core claim.

---

## 4. Top 5 Most Suspicious / Noteworthy Rows

1. **rd/sample/campaign_spending_capped (row 78)**: This is classified under `robustness/sample_restrictions.md` but restricts to Pop >= 9000. This is an institutional threshold for campaign spending regulation, making it genuinely a sample restriction (not pure heterogeneity). However, the complementary spec (`campaign_spending_uncapped`, Pop < 9000) creates a de facto heterogeneity split. Classified as `core_sample` with confidence 0.80 -- borderline.

2. **rd/sample/two_candidates + amount_congr2_3_pcap (row 81)**: Shows a significant negative effect (coef = -0.649, p = 0.003) for moderate organizations in two-candidate races. This is unexpected given the null expectation for non-congruent outcomes. However, this row is correctly classified as non-core (alt_outcome). The significant result here may reflect that two-candidate races have cleaner political alignment measurement, revealing a substitution effect.

3. **rd/placebo/cutoff_-5 (row 141)**: The placebo cutoff at -5pp shows a significant result (coef = -2.449, p = 0.027). The SPECIFICATION_SEARCH.md notes this as a concern. While correctly classified as non-core placebo, this is worth flagging as it raises design validity concerns.

4. **rd/poly/order_1 specs (rows 22-25)**: These are exact duplicates of the baseline specs (same polynomial, kernel, and sample). The spec search generated redundant rows. The coefficients and p-values match exactly. Not a classification error, but inflates the core count by 2 (the congr1 and pcap rows).

5. **rd/kernel/triangular specs (rows 34-37)**: Similarly, the triangular kernel is the baseline kernel, so these 4 rows are exact duplicates of the baseline. The congr1 and pcap rows are classified as core but are duplicates, inflating the core count by 2.

---

## 5. Recommendations

1. **Deduplicate baseline-replicating specs**: The `rd/poly/order_1` and `rd/kernel/triangular` specs are exact copies of the baseline. The spec search script should detect when a robustness variation produces the same specification as the baseline and skip or flag it. This affects 8 rows (4 outcomes x 2 duplicated specs).

2. **Separate heterogeneity from sample restrictions**: The spec search script places some heterogeneity splits (large/small municipalities, pre/non-pre-election years) under `robustness/sample_restrictions.md`. These should be tagged under `robustness/heterogeneity.md` to facilitate correct classification downstream. A complementary pair of subsamples that partition the full sample is heterogeneity, not a sample restriction.

3. **Tag individual year regressions as diagnostics**: The single-year regressions could be tagged with a dedicated `diagnostic/year_specific` path rather than `sample_restrictions` to distinguish them from leave-one-year-out robustness checks.

4. **Consider omitting non-congruent outcome rows from robustness counts**: Since 51 of 149 specs are contrast tests with null-expected outcomes (congr2, congr3), they substantially inflate the total spec count without testing the main claim. The spec search summary statistics (e.g., "50.3% significant at 5%") mix core and contrast tests, which is misleading. Consider reporting core-only statistics separately.

5. **Baseline claim is correctly identified**: The spec search correctly identifies `amount_congr1_3_pcap` as the primary outcome and the RD at `alignm = 0` as the main specification. No corrections needed to the baseline definition.

---

## 6. File Listing

- `verification_baselines.json` -- Baseline group definitions (G1, G2)
- `verification_spec_map.csv` -- Row-level classification of all 149 specifications
- `VERIFICATION_REPORT.md` -- This report
