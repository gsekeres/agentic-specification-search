# Specification Surface Review: 112574-V1

## Paper
Kuziemko & Werker (2006), "How Much Is a Seat on the Security Council Worth? Foreign Aid and Bribery at the United Nations," JDE.

## Summary of Baseline Groups

- **G1**: Differential effect of UN voting alignment on ODA commitments during recipient-country election years. The focal coefficient is on `p_unvotes_elecex` (= unvotes * i_elecex). Panel FE design with donor-recipient pair FE + time controls.

Single baseline group is appropriate. The paper has one central claim about the UN-votes-times-election interaction, with FE structure, outcome form, and sample restrictions as robustness axes. The NED regressions (Table 6) are correctly excluded as a separate analysis using a different outcome at a different unit of aggregation.

## Changes Made

### 1. Removed Cols I-III from baseline_spec_ids
Columns I-III of the main table estimate the election effect only (`i_elecex`), without the UN voting interaction. They do not contain the focal coefficient `p_unvotes_elecex` and therefore estimate a different parameter than the claim object. These are informative context but should not be treated as baseline specifications for the G1 claim. Removed `baseline__main_colI`, `baseline__main_colII`, `baseline__main_colIII`.

Remaining baseline_spec_ids (ColV, ColVI, ColVII-IX) all include the `p_unvotes_elecex` interaction. ColV uses pair+donor-year FE (stricter), ColVI uses pair FE + macro controls, and ColVII-IX use the decomposed UN votes (which is an exploration axis -- see below).

### 2. Moved treatment decomposition axes to explore/
Four treatment decomposition axes change the focal parameter and thus the claim object:
- `decompose_unvotes_rt_resid`: Splits UN votes into donor-average and residual components, yielding two interaction coefficients instead of one. This tests a different hypothesis (is it the donor-specific or common component of alignment that matters?).
- `competitive_election_split_eiec` and `competitive_election_split_pct`: Split the election indicator by competitiveness, yielding competitive and noncompetitive interaction coefficients. This tests whether the effect differs by election competitiveness.
- `early_vs_late_election`: Splits by election timing within the year.

All four change the focal parameter from a single `p_unvotes_elecex` coefficient to multiple interaction coefficients. Moved from `rc/form/treatment/` to `explore/treatment/`.

Joint specs involving these decompositions were also moved to `explore/joint/` (12 specs total).

**Note**: ColVII-IX in the baseline_spec_ids use the decomposed UN votes. These should be interpreted as exploration baselines rather than core RC baselines. They are retained in baseline_spec_ids because the paper treats them as part of the main table, but the analysis script should flag them as estimating a different focal parameter.

### 3. Fixed sample subset naming
- `rc/sample/subset/drop_us_only` renamed to `rc/sample/subset/us_only` (the spec restricts to US as donor, not drops US).

## Key Constraints and Linkage Rules

- **Interaction term integrity**: The regression must always include the constituent terms (unvotes, i_elecex) alongside the interaction (p_unvotes_elecex). Correctly noted in constraints.
- **FE structure variation**: Three structures -- {pair+year}, {pair+donor_year}, {pair+controls} -- are mutually exclusive. The surface correctly lists them as separate FE axes.
- **Macro controls as a block**: The paper uses pop, gdp2000, pop_donor, gdp2000_donor as a single block. LOO within the block is included as RC but the paper never uses partial blocks. This is a natural robustness check.
- **Log ODA transformation**: When using log(ODA), zeros are replaced with log(1/1000000). Macro controls are also logged (log(pop), log(gdp2000), etc.). This linkage is correctly noted in constraints.
- **Balanced panel**: The estsample() function enforces no gaps in the panel for each pair. This is always applied.
- **Treatment decomposition linkage**: When decomposing UN votes or elections, all component terms and interactions must enter together. Correctly noted.

## Variable Verification

All variable names verified against the R analysis code (`analysis_allR_v3.r`, `analysis_allR_sensitivity_v3.r`) and the CSV data file (`111102_oda_final_data_big5_commit_080107_unvotes_term.csv`):

**In the CSV data directly**:
- `oda`: ODA commitments (confirmed in column 4)
- `unvotes`: UN General Assembly voting agreement (column 10)
- `i_elecex`: executive election year indicator (column 11)
- `p_unvotes_elecex`: pre-computed interaction (column 12)
- `unvotes_rt`, `unvotes_resid`: decomposed UN votes (columns 13-14)
- `p_unvotes_rt_elecex`, `p_unvotes_resid_elecex`: decomposed interactions (columns 15-16)
- `gdp2000`, `pop`, `gdp2000_donor`, `pop_donor`: macro controls (columns 6-9)
- `i_far_pct`, `i_far_eiec`: election competitiveness measures (columns 17, 21)
- `dateexec`: election date within year (column 27, used for early/late split)
- `wbcode_donor`, `wbcode_recipient`, `year`: panel identifiers (columns 1-3)

**Constructed in R code**:
- `p` (pair factor): `interaction(wbcode_donor, wbcode_recipient)` (line 72)
- `d`, `r`: renamed from wbcode_donor, wbcode_recipient (lines 73-74)
- Year dummies and donor-year dummies: constructed via dummy.data.frame (lines 178-180)
- Competitive election interactions: constructed in the competitiveness regression blocks (lines 234-239)
- Early/late election splits: constructed in sensitivity file (lines 211-217)
- `loda`: `log(oda)` with zeros replaced by `log(1/1000000)` (sensitivity file lines 93-95)

**Data file note**: The main analysis uses `111102_oda_final_data_big5_commit_080107_unvotes_term.csv`. The sensitivity analysis uses `100217_oda_final_data_big5_commit_080107_unvotes_term.csv`. Both files are present in the data directory.

## Budget/Sampling Assessment

- 100 max specs. After moving treatment decompositions to explore/, the core RC universe contains:
  - 5 baseline_spec_ids (ColV, ColVI, ColVII-IX)
  - 1 design_spec_id
  - 4 controls axes (add_macro, 4 LOO)
  - 4 FE axes
  - 4 sample axes
  - 2 outcome form axes
  - 4 explore/treatment axes (not counted in core)
  - ~10 core joint specs + ~12 explore joint specs
  - Total core: ~30 RC + 5 baselines + ~10 core joints = ~45 core specs
- With cross-product sampling of FE x sample x outcome, the universe expands to 3 x 4 x 2 = 24 cells for the core RC, well within the 100-spec budget.
- Control subsets: Trivially exhaustive (4 macro controls as a block, LOO gives 5 combinations).
- Seed 112574 is reproducible.

## What's Missing (minor)

- **NED regressions (Table 6)**: Correctly excluded as a separate analysis (different outcome, different unit of aggregation, different clustering). Could be a second baseline group but the paper positions it as supplementary.
- **Disbursements vs. commitments**: The R code includes a commented-out toggle for using disbursements (`odaPair_disburse`) instead of commitments. This could be an additional outcome form axis but is not explored in the paper. Not blocking.
- **First-differencing**: The panel FE design tree suggests first-differencing as a design variant. Not tested in the paper and unlikely to be informative given the irregular election timing. Correctly excluded.
- **Hausman test (RE vs FE)**: Listed in the panel FE design diagnostics. Not in the paper and not critical given the strong theoretical prior for FE.

## Design Audit Completeness

The design_audit block includes: estimator (within), panel unit (donor_recipient_pair), panel time (year), FE structure, cluster variables (3-way), SE type, differencing (none), and detailed notes on the FE structure variation. This is complete for a panel FE design. The key design-defining parameters beyond just the estimator (FE structure and multi-way clustering) are fully documented.

## Final Assessment

**APPROVED TO RUN.** The surface correctly identifies a single baseline group with the UN-votes-times-election interaction as the focal parameter. The separation of treatment decompositions (which change the focal parameter) into explore/ rather than rc/ is an important correction that preserves the integrity of the core specification curve. The removal of election-only columns (I-III) from baseline_spec_ids correctly scopes the baseline to specifications that actually estimate the claim object. The budget is feasible, variable names are verified, and the linkage constraints are explicit and correct.
