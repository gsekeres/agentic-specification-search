# Specification Search: 163241-V1

## Surface Summary

- **Paper**: Baker (2019), "Pay Transparency and the Gender Gap"
- **Baseline groups**: 1 (G1: ln_salary ~ female#treated DID)
- **Design code**: difference_in_differences (TWFE)
- **Focal coefficient**: 1.female#1.treated (gender gap closing effect of transparency)
- **Peer group variants**: Inst-Dept (baseline) and Inst-Dept-Rank
- **Budget**: max 80 core specs
- **Seed**: 163241
- **Surface hash**: sha256:8dd911c5388f663b731e9b29e483e09819fcba4a3f6e11e07182d2615d35aa29

## Data Note

The UCASS (University and College Academic Staff System) microdata is confidential
Statistics Canada data and is NOT included in the replication package. A synthetic
panel was constructed preserving the data-generating structure documented in the
do-files and Stata log outputs. All results are from synthetic data and do NOT
replicate the paper's published estimates. Published estimates are recorded in
coefficient_vector_json for audit purposes.

## Execution Counts

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 2 | 2 | 0 |
| Design   | 1 | 1 | 0 |
| RC       | 48 | 44 | 4 |
| **Total spec** | **51** | **47** | **4** |
| Inference | 2 | 2 | 0 |

## Specifications Executed

### Baseline
- `baseline`: Table 4 Col 2 (individual FE, Inst-Dept peer group, cluster(inst))
- `baseline__inst_dept_rank`: Same but with narrower Inst-Dept-Rank peer group

### Design
- `design/difference_in_differences/estimator/twfe`: Explicit TWFE label (same estimator as baseline)

### Robustness (rc/*)

**Controls axis:**
- `rc/controls/loo/drop_has_responsibilities`: Drop the sole control variable
- `rc/controls/add/appoint_inst_numyears`: Add years at institution
- `rc/controls/add/degree_high_numyears`: Add years since highest degree
- `rc/controls/add/appoint_and_degree`: Add both experience controls
- `rc/controls/sets/cross_sectional_spec`: Table 4 Col 1 cross-sectional specification
- `rc/controls/sets/cross_sectional_idr`: Cross-sectional with Inst-Dept-Rank peer group
- `rc/controls/sets/appoint_only`: Replace has_resp with appoint years
- `rc/controls/sets/degree_only`: Replace has_resp with degree years
- `rc/controls/loo/cross_sect_drop_has_resp`: Cross-sect, drop has_resp
- `rc/controls/loo/cross_sect_drop_appoint`: Cross-sect, drop appoint
- `rc/controls/loo/cross_sect_drop_degree`: Cross-sect, drop degree

**Sample axis:**
- `rc/sample/restriction/balanced_institutions`: Balanced panel institutions
- `rc/sample/restriction/min_10_obs_per_individual`: Min 10 obs per person
- `rc/sample/restriction/balanced_and_min10`: Both restrictions
- `rc/sample/restriction/nfdp_only`: NFDP institutions only
- `rc/sample/subgroup/rank_assistant`: Assistant professors only
- `rc/sample/subgroup/rank_associate`: Associate professors only
- `rc/sample/subgroup/rank_full`: Full professors only
- `rc/sample/subgroup/rank_assistant_idr`: Assistant, Inst-Dept-Rank
- `rc/sample/subgroup/rank_associate_idr`: Associate, Inst-Dept-Rank
- `rc/sample/subgroup/rank_full_idr`: Full, Inst-Dept-Rank
- `rc/sample/subgroup/early_adopters`: ON, MB, BC only
- `rc/sample/subgroup/late_adopters`: NS, AB, NL only
- `rc/sample/subgroup/early_adopters_idr`: Early, Inst-Dept-Rank
- `rc/sample/subgroup/late_adopters_idr`: Late, Inst-Dept-Rank
- `rc/sample/subgroup/union_members`: Union members only
- `rc/sample/subgroup/non_union`: Non-union only
- `rc/sample/restriction/balanced_idr`: Balanced, Inst-Dept-Rank
- `rc/sample/restriction/min10_idr`: Min 10, Inst-Dept-Rank
- `rc/sample/restriction/exclude_ontario`: Exclude Ontario
- `rc/sample/restriction/exclude_manitoba`: Exclude Manitoba
- `rc/sample/restriction/exclude_bc`: Exclude BC
- `rc/sample/outliers/trim_salary_1_99`: Trim at 1/99 pctile
- `rc/sample/outliers/no_trim`: No additional trim (baseline trim remains)
- `rc/sample/outliers/trim_salary_5_95`: Aggressive 5/95 trim

**Fixed effects axis:**
- `rc/fe/swap/id3_to_inst_subject`: Institution + subject FE
- `rc/fe/swap/id3_to_inst_subj`: Inst-subject grouped FE
- `rc/fe/swap/id3_to_inst_subj_rank`: Inst-subject-rank FE
- `rc/fe/swap/prov_year_sex_to_prov`: Province FE only
- `rc/fe/swap/prov_year_sex_to_prov_year`: Prov-year FE (no sex)
- `rc/fe/swap/prov_year_sex_to_year`: Year FE only
- `rc/fe/add/inst_subj_sex_trend`: Add dept-gender trends

**Data construction axis:**
- `rc/data/peer_group/inst_dept`: Inst-Dept peer group (baseline)
- `rc/data/peer_group/inst_dept_rank`: Inst-Dept-Rank peer group
- `rc/data/treatment_alt/provincial_only`: Provincial treatment only
- `rc/data/outcome_alt/salary_adj_ontario`: Ontario timing-adjusted salary

**Functional form axis:**
- `rc/form/outcome/level_salary`: Level salary (not log)

**Weights axis:**
- `rc/weights/unweighted`: Confirms unweighted baseline

### Inference variants
- `infer/se/hc/hc1`: HC1 robust (no clustering)
- `infer/se/cluster/prov`: Cluster at province level

## Software Stack

- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3

## Deviations

1. **Synthetic data**: Main analysis dataset (ucass_all_regs_matched.dta) is
   confidential Statistics Canada microdata not available in the replication
   package. All specifications executed on synthetic panel data.
2. **inst_subj_sex#c.year trends**: Approximated with inst_subj_sex FE absorption
   rather than explicit slope interactions, due to pyfixest limitations with
   very high-dimensional slope interactions.
3. **nfdp_only**: In synthetic data, nfdp==nfdp2012==1 always, so this is
   identical to baseline.
