# Local Projection Methods

Local projections (Jordà 2005) estimate impulse response functions directly by regressing future outcomes on current shocks at each horizon.

## Method Classification
- **Code**: `local_projection`
- **Primary Use**: Dynamic effects, impulse responses
- **Data Structure**: Time series or panel

## Baseline Specification

```
spec_id: lp/baseline
spec_tree_path: methods/local_projection.md#baseline
```

Standard local projection at horizon h:
```
y_{t+h} = α + β_h * shock_t + controls + ε_{t+h}
```

Run for h = 0, 1, 2, ..., H (typically H = 12-24 for quarterly, 3-5 for annual)

## Specification Variations

### Horizon Selection
```
spec_id: lp/horizon/{h}
spec_tree_path: methods/local_projection.md#horizon-selection
```
- Short horizon (h = 1-4)
- Medium horizon (h = 5-12)
- Long horizon (h = 12+)

### Control Sets
```
spec_id: lp/controls/{variant}
spec_tree_path: methods/local_projection.md#control-sets
```
- Lags of dependent variable only
- Lags of shock only
- Both lags
- Additional macro controls

### Fixed Effects (Panel LP)
```
spec_id: lp/fe/{type}
spec_tree_path: methods/local_projection.md#fixed-effects
```
- Unit FE
- Time FE
- Two-way FE

### Standard Errors
```
spec_id: lp/se/{type}
spec_tree_path: methods/local_projection.md#standard-errors
```
- Newey-West HAC (specify bandwidth)
- Driscoll-Kraay (panel)
- Clustered by unit
- Wild bootstrap

### Shock Identification
```
spec_id: lp/shock/{method}
spec_tree_path: methods/local_projection.md#shock-identification
```
- Recursive ordering (Cholesky)
- External instrument (LP-IV)
- Narrative identification
- High-frequency identification

## Robustness Checks

### Lag Length
```
spec_id: lp/robust/lags/{p}
spec_tree_path: methods/local_projection.md#lag-length
```
Test sensitivity to number of control lags (p = 2, 4, 8)

### Sample Period
```
spec_id: lp/robust/sample/{period}
spec_tree_path: methods/local_projection.md#sample-period
```
- Pre-Great Moderation
- Post-Great Moderation
- Excluding recessions

### Transformation
```
spec_id: lp/robust/transform/{type}
spec_tree_path: methods/local_projection.md#transformation
```
- Level specification
- Log specification
- Growth rates
- Cumulative responses

## Output Requirements

For each horizon h, record:
- `coefficient`: β_h estimate
- `std_error`: Standard error (specify HAC/cluster method)
- `p_value`: Two-sided test
- `ci_lower`, `ci_upper`: Confidence interval
- `n_obs`: Effective sample size at horizon h
- `horizon`: The forecast horizon h

## References

- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections." American Economic Review.
- Ramey, V. A. (2016). "Macroeconomic Shocks and Their Propagation." Handbook of Macroeconomics.
