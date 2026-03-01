# Verification Report: 145141-V1

## Paper
Bursztyn, Ferman, Fiorin, Kanz, Rao (2018), "Status Goods: Experimental Evidence from Platinum Credit Cards", QJE

## Baseline Groups Found

### G1: YMCA Field Experiment -- Attendance
- **Baseline spec_run_ids**: 145141-V1_run_001, 145141-V1_run_002
- **Baseline spec_ids**: baseline__table2_col2_coh, baseline__table2_col3_coh
- **Claim**: Effect of random assignment to public recognition (attendance posted on public board) on YMCA gym attendance during a 30-day experiment period
- **Baseline coefficient (image)**: 1.190 (SE=0.459, p=0.010, N=370) [Col 2]; 1.273 (SE=0.447, p=0.005, N=370) [Col 3]
- **Expected sign**: Positive (public recognition increases attendance)

### G2: Charity Real-Effort Experiment -- Points Scored
- **Baseline spec_run_ids**: 145141-V1_run_017, 145141-V1_run_018, 145141-V1_run_019
- **Baseline spec_ids**: baseline__table5_col1_prolific, baseline__table5_col2_berkeley, baseline__table5_col3_bu
- **Claim**: Effect of public recognition (SR round vs anonymous round) on points scored in within-subject charitable giving task
- **Baseline coefficient (SR)**: 105.0 (SE=12.3, p<0.001, N=2904) [Prolific]; 134.4 (SE=22.6, p<0.001, N=1152) [Berkeley]; 103.6 (SE=45.3, p=0.024, N=354) [BU]
- **Expected sign**: Positive (public recognition increases effort)

### G3: WTP Elicitation -- Reputational Return Function
- **Baseline spec_run_ids**: 145141-V1_run_060, 145141-V1_run_061, 145141-V1_run_062, 145141-V1_run_063
- **Baseline spec_ids**: baseline__table3_col2_coh_ymca, baseline__table6_col2_prolific, baseline__table6_col4_berkeley, baseline__table6_col6_bu
- **Claim**: WTP for public recognition increases with hypothetical performance level (positive slope of the reputational return function)
- **Baseline coefficient (visits/interval)**: 0.361 (SE=0.036, p<0.001, N=4070) [YMCA]; 0.155 (SE=0.018, p<0.001, N=16456) [Prolific]; 0.379 (SE=0.070, p<0.001, N=6528) [Berkeley]; 0.309 (SE=0.116, p=0.009, N=2006) [BU]
- **Expected sign**: Positive (WTP increases with performance)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 96 |
| Valid (is_valid=1) | 96 |
| Invalid (is_valid=0) | 0 |
| Core tests (is_core_test=1) | 96 |
| Non-core | 0 |
| Baseline rows | 9 |
| Inference variants (inference_results.csv) | 7 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 13 |
| core_controls | 9 |
| core_sample | 48 |
| core_funcform | 18 |
| core_preprocess | 5 |
| core_fe | 3 |

## Robustness Assessment

### G1: YMCA Attendance (16 specs)

**Sign consistency**: 16/16 (100%) positive coefficients.

**Statistical significance**: 14/16 (87.5%) significant at 5%. The two non-significant specs are the diff-in-means estimator (run_003, p=0.113) and the no-controls OLS (run_004, p=0.113), which are identical because OLS on treatment alone with no controls equals diff-in-means. These show that the significance depends on controlling for pre-treatment attendance (past), which explains a large share of outcome variance (R2 jumps from 0.007 to 0.561).

**Coefficient range** (attendance outcome): [0.823, 1.273]. The topcoded-at-15 spec (run_016) produces the smallest effect (0.823), consistent with compressing the high end of attendance.

**Inference sensitivity**: HC3 SEs on baseline (infer_001) give p=0.011, essentially unchanged from HC1 (p=0.010). No inference fragility.

### G2: Charity Real-Effort (43 specs)

**Sign consistency**: 41/43 (95.3%) positive. Two negative coefficients are both BU subsample specs with very high p-values:
  - run_031 (first_round_only, BU): coef=-27.7, p=0.832, N=118
  - run_050 (log1p_pts, BU): coef=-0.130, p=0.563, N=354

These sign flips are noise from the small BU sample, not evidence against the claim.

**Statistical significance**: 39/43 (90.7%) significant at 5%. The 4 non-significant specs are all BU sample variants (N=118-354), reflecting low power in the smallest sample.

**Coefficient range** (pts outcome, excluding transformed outcomes): [-27.7, 168.5]. The wide range is driven by BU subsample noise. Restricting to Prolific (N>2000): [95.8, 112.9].

**Inference sensitivity**: HC1 (no clustering) on baselines:
  - Prolific: p<0.001 (baseline clustered p<0.001) -- unchanged
  - Berkeley: p<0.001 (baseline clustered p<0.001) -- unchanged
  - BU: p=0.172 (baseline clustered p=0.024) -- loses significance without clustering

The BU result is fragile: it depends on clustering at the individual level, and the small sample (118 individuals, 354 obs) means clustering makes a substantial difference.

### G3: WTP Elicitation (37 specs)

**Sign consistency**: 37/37 (100%) positive. No sign flips.

**Statistical significance**: 34/37 (91.9%) significant at 5%. The 3 non-significant specs are all BU charity subsample variants:
  - run_084 (close_to_score_charity, BU): p=0.209, N=977
  - run_093 (trim_wtp_5_95, BU): p=0.564, N=1817
  - run_083 (close_to_score_charity, BU): p=0.079, N=3305

**Inference sensitivity**: HC1 (no clustering) on charity baselines:
  - Prolific: p<0.001 (baseline clustered p<0.001) -- unchanged
  - Berkeley: p<0.001 (baseline clustered p<0.001) -- unchanged
  - BU: p=0.051 (baseline clustered p=0.009) -- borderline

### Overall Assessment

**94 of 96** specifications (97.9%) produce positive coefficients consistent with the claim.

**87 of 96** specifications (90.6%) are statistically significant at the 5% level.

The result is **STRONG** across all three experimental settings. The main source of fragility is the BU sample, which is the smallest and noisiest of the three online experiment samples. The YMCA field experiment and Prolific/Berkeley online experiments are highly robust.

## Structural Audit

### Step 0: Sanity Checks (all passed)
- `spec_run_id`: unique, 96 distinct values
- `baseline_group_id`: present for all rows, 3 groups (G1: 16, G2: 43, G3: 37)
- `spec_tree_path`: all reference valid spec-tree paths with anchors where appropriate
- `run_success`: all 96 rows have run_success=1
- `coefficient_vector_json`: all rows have required keys (coefficients, inference, software, surface_hash)
- No `infer/*` rows in specification_results.csv
- All numeric fields finite for run_success=1 rows (r_squared NaN only for 2 Tobit specs, which is correct)
- rc/form/* rows all contain `functional_form` block
- rc/controls/* rows all contain `controls` block
- rc/sample/* rows all contain `sample` block
- rc/fe/* rows all contain `fixed_effects` block
- Inference spec_ids match surface canonical: G1 uses infer/se/hc/hc1, G2 and G3 use infer/se/cluster/id

### Step 1: Surface Alignment (perfect match)
- All 2 G1 baseline spec_ids present in results
- All 3 G2 baseline spec_ids present in results
- All 4 G3 baseline spec_ids present in results
- All design spec_ids present
- All rc spec_ids present
- No missing or spurious baseline groups

### Step 2: Baseline Identification
- G1: 2 baselines (Table 2 Cols 2-3, coherent sample)
- G2: 3 baselines (Table 5 Cols 1-3, Prolific/Berkeley/BU)
- G3: 4 baselines (Table 3 Col 2 YMCA, Table 6 Cols 2/4/6 charity)

### Step 3: Classification
- All 96 rows classified as core tests (is_core_test=1)
- No outcome or treatment concept drift: all G1 outcome transforms (log1p, asinh, standardized) are legitimate functional form variations of attendance; all G2 transforms (log1p, standardized, pts_hundreds) are legitimate rescalings of pts; all G3 treatment codings (visits, interval_raw, ln_visits, interval_idx) are alternative performance-level parametrizations within the same BDM elicitation
- No population changes that warrant non-core classification: multi-sample specs within G2 and G3 correspond to the paper's own separate-sample analysis

## Top Issues

1. **No-op WTP trimming for YMCA**: Runs 095 and 096 (trim_wtp_1_99 and trim_wtp_5_95 for YMCA in G3) produce results identical to baseline (run_060). This means the YMCA WTP variable has no values beyond the 1st/99th percentile trim thresholds, making these specs trivially identical. They are valid but uninformative. Marked is_valid=1 with confidence=0.90.

2. **Duplicated spec_ids across samples**: G2 and G3 have many spec_ids that appear 3 times (once per experimental sample). This is by design -- the paper runs separate regressions per sample -- but it means spec_id alone is not a unique row identifier. The sample_desc field disambiguates.

3. **Individual FE coefficient match**: G2 individual FE specs (runs 057-059) produce coefficients identical to ownpay-only specs (runs 023, 025, 027). This is correct: in a within-subject design with 3 obs per person, absorbing individual FE drops order dummies and makes ownpay collinear with individual effects, leaving only the SR within-person contrast. The much higher R2 (0.75-0.80 vs 0.01-0.02) confirms FE is being absorbed.

4. **BU sample fragility**: The BU sample (N=118-354) is consistently the noisiest across G2 and G3, producing sign flips and loss of significance. This is a power issue, not an extraction error.

## Recommendations

1. Consider flagging no-op trim specs (identical to baseline) in the runner to avoid inflating the spec count with uninformative duplicates.
2. For multi-sample papers, appending a sample suffix to spec_id (e.g., `rc/controls/sets/ownpay_only__prolific`) would make spec_id unique per row and improve traceability.
3. The BU first-round-only spec (N=118, 1 round per person in a within-subject design) collapses the within-subject variation entirely, making it a cross-sectional comparison on a tiny sample. Consider noting this more prominently.

## Conclusion

The specification search confirms **STRONG support** for the paper's claims across all three experimental settings. The effects of public recognition on attendance (G1), charitable effort (G2), and WTP for recognition (G3) are robust to control variation, sample restrictions, functional form changes, outlier treatment, and alternative estimators. The main vulnerability is the small BU sample, where results are noisy and sometimes lose significance, but this is a power limitation rather than evidence against the claim. Inference sensitivity is minimal: HC3 SEs in G1 and HC1 SEs in G2/G3 produce only minor changes in significance, with the BU subsample being the sole exception.
