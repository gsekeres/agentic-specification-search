# Specification Search Report: 125321-V1

## Paper
- **Title**: Can Technology Solve the Principal-Agent Problem? Evidence from China's War on Air Pollution
- **Paper ID**: 125321-V1

## Surface Summary
- **Baseline groups**: 1 (G1)
- **Design code**: regression_discontinuity (sharp)
- **Running variable**: T = date - auto_date (days from automation)
- **Cutoff**: 0
- **Baseline spec**: Residualized PM10, rdrobust p(1) q(2) kernel(tri) cluster(code_city)
- **Budget**: max 55 core specs
- **Seed**: 125321

## Canonical Inference
- City-level clustering (cluster code_city) with robust bias-corrected inference from rdrobust

## Data Preparation
- Generated station_day_1116 by merging pollution_1116 + weather_1116 + station_list (replicating Prepare_Data.do)
- Generated station_month by monthly aggregation + AOD merge
- Residualized PM10 using iterative demeaning (station FE + month FE) then OLS on weather
- Also residualized log(PM10) for functional form variant

## Execution Summary
- **Total core specifications**: 23
- **Successful**: 23
- **Failed**: 0
- **Inference variants**: 2 (2 successful)
- **Diagnostics**: 6 (6 successful)

## Specifications Executed

### Baselines (2 specs)
| spec_id | description |
|---------|-------------|
| `baseline` | Residualized PM10, all stations (Table 1A Row 1 Col 2) |
| `baseline__raw_pm10` | Raw PM10, all stations (Table 1A Row 1 Col 1) |

### Design Variants (7 specs)
| spec_id | description |
|---------|-------------|
| `design/regression_discontinuity/bandwidth/half_baseline` | Half MSE-optimal bandwidth |
| `design/regression_discontinuity/bandwidth/double_baseline` | Double MSE-optimal bandwidth |
| `design/regression_discontinuity/poly/local_quadratic` | Local quadratic (p=2, q=3) |
| `design/regression_discontinuity/kernel/uniform` | Uniform kernel |
| `design/regression_discontinuity/kernel/epanechnikov` | Epanechnikov kernel |
| `design/regression_discontinuity/procedure/conventional` | Conventional (non-bias-corrected) |
| `design/regression_discontinuity/procedure/robust_bias_corrected` | Robust bias-corrected |

### RC Variants (14 specs)
| spec_id | description |
|---------|-------------|
| `rc/controls/sets/none_no_residualization` | No controls/FE (raw PM10) |
| `rc/controls/sets/weather_only_no_fe` | Weather only, no station/month FE |
| `rc/sample/restrict/wave1_only` | Wave 1 stations |
| `rc/sample/restrict/wave2_only` | Wave 2 stations |
| `rc/sample/restrict/deadline_only` | Deadline stations |
| `rc/sample/restrict/76_cities` | 76 cities with fewer missing obs |
| `rc/sample/restrict/no_missing_pm10` | Non-missing PM10 only |
| `rc/sample/outliers/trim_y_1_99` | Trim 1/99 pct |
| `rc/sample/outliers/trim_y_5_95` | Trim 5/95 pct |
| `rc/sample/donut/exclude_1day` | Donut +/- 1 day |
| `rc/sample/donut/exclude_3days` | Donut +/- 3 days |
| `rc/form/outcome/log_pm10` | Log(PM10) outcome |
| `rc/fe/alt/station_yearmonth_fe` | Year-month FE |
| `rc/data/time_aggregation/monthly` | Monthly aggregation |

### Inference Results (2 variants)
| spec_id | description |
|---------|-------------|
| `infer/se/cluster/station` | Station-level clustering |
| `infer/se/hc/hc1` | Heteroskedasticity-robust (NN VCE, no clustering) |

### Diagnostics (6 checks)
| diag_spec_id | description |
|-------------|-------------|
| `diag/regression_discontinuity/manipulation/rddensity` | McCrary density test |
| `diag/regression_discontinuity/balance/weather_continuity` | Wind speed continuity |
| `diag/regression_discontinuity/balance/weather_continuity` | Rain continuity |
| `diag/regression_discontinuity/balance/weather_continuity` | Temperature continuity |
| `diag/regression_discontinuity/balance/weather_continuity` | Relative humidity continuity |
| `diag/regression_discontinuity/placebo_outcome/aod_satellite` | AOD satellite placebo |

## Deviations from Surface
- None. All planned specifications executed.

## Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- rdrobust: 1.3.0
- rddensity: unknown
- pandas: 2.2.3
- numpy: 2.1.3
