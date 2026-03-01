# Specification Search: 113513-V1

## Surface Summary

- **Paper**: Trends in Economics Undergraduate Majors by Demographics (IPEDS, 2001-2014)
- **Design**: Cross-sectional OLS (descriptive correlations)
- **Baseline groups**: 1 (G1)
- **Baseline formula**: `econshare ~ level_d{j} + year, robust`
- **Budget**: max 55 core specs
- **Seed**: 113513 (unused -- full enumeration)
- **Surface hash**: `sha256:28ad57bbb8596dbd76f51f9306a450c6a1b5d6167476e3e4a22b98ea9947ac58`

## Data Construction

The raw IPEDS CSV files (`IPEDS_degrees_by_sex.csv`, `IPEDS_degrees_by_race.csv`) were processed
following the Stata do files (`majortrendssex_datain.do`, `majortrendsrace_datain.do`, `majortrends.do`):

1. Detailed disciplines mapped to 12 groups (economics, business/mgmt, poli sci, psych, etc.)
2. For sex data: aggregated by year x discipline x gender, computed shares
3. For race data: white = White + Asian/Pacific Islander; nonwhite = Black + Hispanic + Am. Indian;
   dropped Temporary Resident and Other/unknown
4. Collapsed to year-level: econshare = econ BA / total BA, level_d{j} = discipline j BA / total BA
5. Year range: 2001-2014 (14 observations per regression)

## Execution Summary

### Specification Results

| Category | Planned | Executed | Success | Failed |
|----------|---------|----------|---------|--------|
| Baseline (d2) | 1 | 1 | 1 | 0 |
| Additional baselines (d3-d12) | 10 | 10 | 10 | 0 |
| rc/sample/restriction/females_only | 11 | 11 | 11 | 0 |
| rc/sample/restriction/nonwhites_only | 11 | 11 | 11 | 0 |
| rc/form/outcome/second_major_share | 11 | 11 | 11 | 0 |
| rc/sample/time/drop_first_year | 11 | 11 | 11 | 0 |
| rc/sample/time/drop_last_year | 11 | 11 | 11 | 0 |
| rc/sample/time/pre_2008 | 11 | 11 | 11 | 0 |
| rc/sample/time/post_2008 | 11 | 11 | 11 | 0 |
| rc/controls/single/add_year_squared | 11 | 11 | 11 | 0 |
| rc/form/model/no_year_control | 11 | 11 | 11 | 0 |
| **Total** | **110** | **110** | **110** | **0** |

### Inference Results

| Variant | Specs | Success | Failed |
|---------|-------|---------|--------|
| infer/se/hac/nw_auto (Newey-West, lags=2) | 11 | 11 | 0 |
| infer/se/classical/ols | 11 | 11 | 0 |
| **Total** | **22** | **22** | **0** |

## Deviations and Notes

- **Year range**: Data covers 2001-2014 (14 years), not 2000-2015 as stated in the surface notes.
  The surface estimates of ~16 observations were approximate; actual N=14 per regression.
- **No controls multiverse**: Each regression is bivariate (one discipline share + year).
  The specification space is naturally small.
- **Small sample caveat**: With N=14 per regression, statistical power is very limited.
  Newey-West HAC SEs use 2 lag(s) (floor of N^(1/3)).
- **Second major share (econshare2d)**: Available for all students. This matches the paper's
  Table 1 which reports both first and second major regressions.
- **Full enumeration**: All planned specs were executed (no random sampling needed).

## Software Stack

- Python 3.12.7
- statsmodels 0.14.6
- pandas 2.2.3
- numpy 2.1.3
