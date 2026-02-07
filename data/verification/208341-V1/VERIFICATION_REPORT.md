# Verification Report: 208341-V1

## Paper: Land Rental Subsidies and Agricultural Productivity
**Authors**: Acampora, Casaburi, Willis
**Journal**: AER
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

| Group | Claim | Baseline spec_id | Outcome | Expected Sign |
|-------|-------|-------------------|---------|---------------|
| G1 | Rental subsidy increases agricultural value added | baseline/ETwadj_ag_va1_r6_qaB_1 | ETwadj_ag_va1_r6_qaB_1 | + |
| G2 | Rental subsidy increases plot cultivation | baseline/ETd2_1_plot_use_cltvtd_1 | ETd2_1_plot_use_cltvtd_1 | + |
| G3 | Rental subsidy increases input expenditure | baseline/ETd34_ag_inputs1_B_1 | ETd34_ag_inputs1_B_1 | + |
| G4 | Rental subsidy affects land rental expenditure | baseline/ETL_val_1 | ETL_val_1 | unknown |
| G5 | Rental subsidy increases harvest value | baseline/ETe1_3_h_value1_qa_1 | ETe1_3_h_value1_qa_1 | + |

The primary claim is G1 (value added). The paper is an RCT evaluating land rental subsidies vs. cash transfers on agricultural outcomes in Kenya, with 509 farmers across 4 endline rounds (1957 observations). All 5 baseline specs use rental_subsidy as treatment, full baseline controls, stratum + endline_round FE, and farmer-level clustering.

---

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 69 |
| Baseline specifications | 5 |
| Core test specifications (non-baseline) | 41 |
| Non-core specifications | 23 |
| Invalid specifications | 0 |
| Unclear specifications | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 22 (incl. 5 baselines) |
| core_sample | 12 |
| core_funcform | 7 |
| core_fe | 3 |
| core_inference | 2 |
| noncore_alt_treatment | 7 |
| noncore_heterogeneity | 6 |
| noncore_diagnostic | 3 |
| noncore_placebo | 3 |
| noncore_alt_outcome | 4 |

### Core tests by baseline group

| Group | Baselines | Core tests | Total |
|-------|-----------|------------|-------|
| G1 (value added) | 1 | 32 | 33 |
| G2 (plot cultivation) | 1 | 2 | 3 |
| G3 (input expenditure) | 1 | 2 | 3 |
| G4 (land value) | 1 | 2 | 3 |
| G5 (harvest value) | 1 | 2 | 3 |

The overwhelming majority of robustness checks target G1 (value added), the primary outcome.

---

## Classification Logic

### Baselines (5 specs)
The 5 specs with spec_id starting with "baseline/" each represent a different outcome variable from the paper main results table, all using rental_subsidy as treatment with full baseline controls, stratum + round FE, and farmer-level clustering.

### Core tests of G1 (value added) -- 32 specs
- **Control variations** (17 specs): No controls, plot-size only, leave-one-out drops (7 controls dropped one at a time), progressive additions (1-7 controls).
- **Sample restrictions** (12 specs): Individual rounds (1-4), stratum splits (C/NC), early/late rounds, winsorization at 1%/5%/10%, and 1% trimming.
- **FE variations** (3 specs): Strata only, round only, no FE.
- **Inference variations** (2 specs): Robust SE, stratum-level clustering.
- **Functional form** (1 spec): IHS transformation of value added.

### Core tests of G2-G5
Each secondary outcome has a no-controls variant and/or IHS/log transformations as core tests.

### Non-core: Alternative treatment (7 specs)
5 cash_drop ITT specs (different treatment arm) + 2 rental_minus_cash comparison specs (different estimand).

### Non-core: Heterogeneity (6 specs)
Large/small plot and high/low baseline cultivation subsample splits.

### Non-core: Diagnostics (3 specs)
First stage (subsidy and cash compliance) + mechanism (plot rented).

### Non-core: Placebo (3 specs)
Effects on pre-treatment baseline outcomes.

### Non-core: Additional outcomes (4 specs)
Maize cultivation, commercial crops, seed value, improved seeds.

---

## Top 5 Most Suspicious Rows

1. **robust/control/add_7_controls**: Coefficient = 19.8411, p = 0.1532 -- identical to baseline/ETwadj_ag_va1_r6_qaB_1. This is an exact duplicate of the baseline (all 7 controls = full controls).

2. **robust/control/add_6_controls**: Coefficient = 19.8059, p = 0.1528 -- nearly identical to baseline. The 7th control adds almost nothing.

3. **itt/cash_drop specs with tree_path = methods/cross_sectional_ols.md#baseline**: The tree_path tag "#baseline" could misleadingly suggest these are baselines. They are alternative treatment ITT estimates.

4. **robust/sample/winsorize_* outcome variable**: Uses "ETwadj_ag_va1_r6_qaB_1_wins" with _wins suffix, changing the outcome_var string. Automated matching would miss the connection to baseline.

5. **robust/fe/none**: Coefficient drops to 4.78 (p = 0.69) from 19.84 baseline, highlighting the importance of strata FE for precision in this stratified RCT.

---

## Recommendations

1. **Remove or flag the duplicate**: robust/control/add_7_controls is identical to the baseline.

2. **Clarify cash_drop tree paths**: Should use a distinct spec_tree_path to avoid confusion.

3. **Consider adding IV/2SLS specs**: The paper uses IV/2SLS as its primary specification, but only ITT was run.

4. **Standardize winsorized outcome naming**: Use same outcome_var name with a transformation note.

5. **Consider reclassifying heterogeneity specs**: If the paper presents heterogeneous effects as main findings, some could be reclassified as core_sample.
