# Verification Report: 114854-V1

**Paper**: "Paging Inspector Sands: The Costs of Public Information"
**Authors**: Sacha Kapoor and Arvind Magesan
**Journal**: American Economic Journal: Economic Policy
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## 1. Baseline Groups

### G1: Pedestrian Flow (Table 8 Col 7)
- **Spec ID**: baseline
- **Outcome**: ped_vol8hr (8-hour pedestrian volume)
- **Treatment**: countdown (pedestrian countdown signal installation)
- **Coefficient**: -228.44, SE=669.40, p=0.733, N=1,912
- **Fixed Effects**: day + month + year + main street + side route
- **Controls**: ns_ind, ew_ind (location indicators)
- **Match to paper**: Good. Values match documented Table 8 Col 7 specification.

### G2: Automobile Flow (Table 9 Col 7)
- **Spec ID**: baseline_auto
- **Outcome**: tot_count (total automobile count)
- **Treatment**: countdown
- **Coefficient**: -348.67, SE=225.52, p=0.122, N=28,996
- **Fixed Effects**: day + month + year + street1 + street2
- **Controls**: ew_ind, ns_ind, nvar1
- **Match to paper**: **POOR**. The SPECIFICATION_SEARCH.md documents Table 9 Col 7 as coef=166.46, SE=352.17, p=0.637. The CSV baseline_auto shows a coefficient of the opposite sign (-348.67 vs +166.46), with a different SE and p-value. The clustering variable (id_pcs) and FE implementation may differ from the original Stata code.

---

## 2. Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **90** |
| Baselines | 2 |
| Core tests | 80 |
| Non-core tests | 10 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 14 |
| core_fe | 11 |
| core_funcform | 6 |
| core_inference | 2 |
| core_sample | 47 |
| noncore_alt_outcome | 3 |
| noncore_heterogeneity | 5 |
| noncore_placebo | 2 |

---

## 3. Classification Notes

### Core tests (80 specs)
The large majority of specifications are legitimate robustness checks:

- **Control variations (14)**: Progressive addition of controls/FE from no controls up to the full specification, plus leave-one-out control variations. These directly test sensitivity of the main estimate to model specification.

- **FE variations (11)**: Leave-one-out fixed effects (drop year FE, drop day FE, drop main street FE, etc.). These test whether any single FE dimension drives the result.

- **Sample restrictions (47)**: Leave-one-year-out (dropping each year individually), geographic subsample restrictions (east/west/north/south), volume-based splits (above/below median), weekday-only, seasonal (summer/winter), winsorization (1%/5%/10%), and trimming (1%). These are standard robustness checks.

- **Functional form (6)**: Log, inverse hyperbolic sine, and square root transformations of the outcome. These preserve the direction of the estimand while changing the functional form.

- **Inference (2)**: Robust vs. IID standard errors for the automobile flow specification. Same coefficient, different standard errors.

### Non-core tests (10 specs)

- **Placebo tests (2)**: fake_timing_ped and fake_timing_auto use a fake treatment variable (countdown_fake) shifted one year early. These test pre-trends / falsification, not the main claim.

- **Heterogeneity (5)**: Specifications with interaction terms (countdown * ew_ind, countdown * ns_ind, countdown * high_freq) that test whether the treatment effect varies by location or frequency. The main effect in these regressions has a different interpretation due to the interaction.

- **Alternative outcomes (3)**: am_pk_vol, pm_pk_vol, and off_pk_vol are different time-of-day automobile volume measures (AM peak, PM peak, off-peak). These are distinct outcomes from the baseline tot_count and were not part of the paper main Table 9 claim.

---

## 4. Top 5 Most Suspicious Rows

1. **baseline_auto**: The documented baseline for Table 9 Col 7 shows coef=166.46 (positive) but the CSV records coef=-348.67 (negative). This is the most critical discrepancy. All 41 automobile flow robustness specifications inherit whatever error caused this mismatch. **Severity: HIGH**.

2. **robust/control/drop_sideroute**: This spec has the exact same coefficient (137.48), SE (260.47), p-value (0.598), and R-squared (0.522) as robust/control/time_location_main. These are duplicates. **Severity: LOW**.

3. **robust/control/drop_nvar_auto**: This spec has the exact same coefficient (-346.34), SE (224.89), and R-squared (0.8278) as robust/control/time_loc_streets_auto. These are duplicates. **Severity: LOW**.

4. **robust/placebo/fake_timing_ped**: With only N=555 observations and an R-squared of 0.998, this specification appears overfit. **Severity: LOW**.

5. **robust/sample/winter_auto**: Shows a very large negative coefficient (-5562.36) that is significant at p=0.008, while the baseline is only -348.67 (p=0.122). This dramatic difference in the winter subsample could indicate seasonal confounding. **Severity: INFORMATIONAL**.

---

## 5. Recommendations for the Spec-Search Script

1. **Fix automobile baseline replication**: The most important issue is the discrepancy between the documented Table 9 Col 7 values and the baseline_auto specification in the CSV. The estimation script should be debugged to determine why the coefficient sign flips. Possible causes: different clustering implementation, different FE absorption, or data filtering differences.

2. **Deduplicate identical specifications**: drop_sideroute duplicates time_location_main, and drop_nvar_auto duplicates time_loc_streets_auto. The script should check for and remove duplicate results.

3. **Missing collision data analysis**: The paper primary analysis uses collision data (Tables 2-7, 10) which requires a separate data license. The current specification search only covers the secondary flow analysis.

4. **Consider adding the paper original Table 8 and Table 9 column progressions**: The paper builds up specifications column by column. A more systematic replication of each table column would help validate the script.

---

## 6. Data Quality Notes

- All 90 specifications have non-missing coefficients, standard errors, and p-values.
- No infinite or NaN values detected.
- The treatment variable is consistently countdown for all core specs and countdown_fake for placebo specs.
- The pedestrian flow baseline (G1) replicates well; the automobile flow baseline (G2) does not match the paper.
