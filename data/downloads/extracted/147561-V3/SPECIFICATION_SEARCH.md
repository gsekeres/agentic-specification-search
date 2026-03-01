# Specification Search Report: 147561-V3

## Paper
"Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo" (Balan, Bergeron, Tourek, Weigel)

## Surface Summary
- **Paper ID**: 147561-V3
- **Baseline groups**: 1 (G1)
- **Design code**: randomized_experiment
- **Baseline outcome**: taxes_paid (binary compliance) and taxes_paid_amt (revenue)
- **Treatment**: t_l (local/chief collection assignment)
- **Focal comparison**: Local (tmt==2) vs Central (tmt==1)
- **Canonical inference**: cluster SEs at a7 (neighborhood/polygon level)
- **Budget**: max 60 core specs
- **Seed**: 147561

## Data Construction
The analysis dataset was constructed from raw data files following `2_Data_Construction.do`:
1. Merged flier assignment data with stratum, treatment assignment, registration (cartography), taxroll, and tax payment data
2. Constructed `taxes_paid` (binary compliance) and `taxes_paid_amt` (=taxes_paid * rate)
3. Created time FE bins matching Stata `egen cut` with breakpoints at Stata dates 21355, 21415, 21475, 21532
4. Dropped villas (house==3), pilot polygons, and observations with missing rate
5. Restricted to Central (tmt==1) vs Local (tmt==2) for focal comparison

**Analysis sample**: 28872 observations in Central vs Local comparison

## Execution Summary
- **Total specification rows**: 51
- **Successful**: 51
- **Failed**: 0
- **Inference variant rows**: 1
- **Inference successful**: 1
- **Inference failed**: 0

## Specs Executed

### Baselines (2)
- `baseline`: Table 4 Compliance Col 4 (taxes_paid ~ t_l | stratum + house + time_FE, cl(a7))
- `baseline__revenues`: Table 4 Revenues Col 4 (taxes_paid_amt outcome)

### Design variants (2)
- `design/randomized_experiment/estimator/diff_in_means`: No FE, compliance
- `design/randomized_experiment/estimator/diff_in_means__revenues`: No FE, revenues

### RC: FE sets (6 compliance + 6 revenues = 12 total, but some are identical to baseline)
- stratum_only, stratum_month, stratum_month_house (baseline-equivalent)
- Each for both compliance and revenue outcomes

### RC: FE drop (2)
- drop house FE
- drop time FE

### RC: Sample restrictions (7)
- exclude_exempt (compliance + revenues)
- polygon_means (compliance + revenues)
- trim revenues at 1st/99th percentile
- Cross-combinations with various FE sets

### RC: Functional form (5)
- log(1+amt) (full FE, stratum only, excl exempt)
- asinh(amt) (full FE, stratum only)

### RC: Treatment definition (10)
- include CLI arm (compliance + revenues)
- include CxL arm (compliance + revenues)
- pooled local-type vs central (compliance + revenues)
- include all arms (compliance + revenues)
- Various FE combinations

### Inference variants (1)
- HC1 robust SEs on baseline

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3

## Deviations from Surface
- Data was constructed from raw files (no pre-built analysis_data.dta available)
- The midline (monitoring) survey data was not merged because it was not needed for Table 4 outcomes
- The exact Stata date imputation for missing `today_carto` was approximated (polygon-level min TDM date or max carto date)
- Sample sizes may differ slightly from the paper due to differences in merge order or missing value handling
