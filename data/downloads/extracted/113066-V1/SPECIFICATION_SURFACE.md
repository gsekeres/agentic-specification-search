# Specification Surface: 113066-V1

## Paper Overview
- **Title**: Real Effects of Information Frictions: When the States and the Kingdom became United (Steinwender, 2018, AER)
- **Design**: Difference-in-differences (before/after telegraph, daily time series)
- **Data**: Daily cotton prices in New York and Liverpool, ~1860-1868, plus trade/export data
- **Key tables**: Table 2 (price difference level and variance), Tables 3-9 (information lag IV, price transmission, trade)
- **Secondary design**: IV (Table 3) where telegraph instruments for information lag

## Baseline Groups

### G1: Telegraph Effect on Cotton Price Difference (Table 2, Cols 1-3)

**Claim object**: The transatlantic telegraph reduced the average cotton price difference between Liverpool and New York. This is the paper's core finding on price convergence / market integration.

**Baseline specifications**:
- Table 2, Col 1: `newey diff tele, lag(2)` -- raw price difference on telegraph dummy
- Table 2, Col 2: `newey difffrctotal tele l1nyrec, lag(2)` -- freight-adjusted price difference, controlling for lagged NY receipts, excluding no-trade days
- Table 2, Col 3: Forward-looking price difference net of freight
- Focal coefficient: `tele` (telegraph effect on average price difference)

### G2: Telegraph Effect on Price Difference Variance (Table 2, Cols 4-6)

**Claim object**: The telegraph reduced the variance of the cotton price gap. This captures the quality-of-information dimension (not just the average gap, but the dispersion of the gap).

**Baseline specifications**:
- Table 2, Col 4: `newey dev2 tele, lag(2)` -- variance of raw price difference
- Table 2, Col 5: `newey dev2 tele l1nyrec, lag(2)` -- variance of freight-adjusted difference
- dev2 is constructed as the sample-corrected squared deviation from the pre/post period mean

## RC Axes Included

### Controls (both groups)
- **Add/drop l1nyrec**: Lagged NY cotton receipts (the only substantive control)
- **Add freight cost / total transport cost**: As a control rather than outcome adjustment

### Sample restrictions (both groups)
- **Exclude/include no-trade days**: Different time-series constructions (ct_t vs ct_notrade)
- **Symmetric window**: Equal number of pre and post-telegraph observations
- **Narrow/wide window**: Sensitivity to how far from the event the sample extends
- **Outlier trimming**: Extreme price differences

### Outcome definitions (G1)
- **Raw vs. freight-adjusted**: diff vs difffrctotal
- **Forward vs. contemporaneous**: Current vs forward-looking Liverpool price
- **Log absolute difference**: Transform to logs
- **Detrending**: Remove linear time trend before regression

### Outcome definitions (G2)
- **Raw vs. freight-adjusted variance**: Different underlying price difference
- **Forward freight variance**: Based on forward-looking price difference
- **Log variance**: Log transformation of dev2
- **Absolute deviation**: Use |diff - mean| instead of squared deviation

### Joint variations
- Outcome definition x sample restriction combinations
- Outcome definition x control inclusion combinations

## What Is Excluded and Why

- **Table 3 (information lag IV)**: Uses tele as instrument for infolag, which is a different estimand (effect of information lag vs. effect of telegraph per se). Treated as exploration/secondary design.
- **Tables 4-5 (ARIMA price transmission)**: Different econometric framework (ARIMA with AR corrections). These are structural time-series models, not regression specifications amenable to our standard RC framework.
- **Tables 6-7 (lagged price effects, export response)**: Different outcomes (lagged price gap, export levels). Could be separate baseline groups but are framed as supporting evidence.
- **Tables 8-9 (trade effects)**: Different outcome (cotton exports). The paper treats price convergence as the main result and trade effects as secondary.
- **Tables 10-13 (welfare analysis, demand/supply estimation)**: Structural estimation with different identification. Out of scope for standard specification search.

## Budgets and Sampling

- **G1**: Max 55 core specs, 10 control subsets
- **G2**: Max 30 core specs, 5 control subsets
- **Seed**: 113066
- **Full enumeration**: Very small control pool, full enumeration trivial

## Inference Plan

- **Canonical**: Newey-West HAC SE with 2 lags (matching the paper)
- **Variants**: Newey-West with 4 and 8 lags (bandwidth sensitivity), HC1 (robust only)
- Bandwidth sensitivity is particularly important here since the autocorrelation structure of daily cotton prices is the key concern

## Key Constraints

- Single treatment variable (tele = 0/1), no variation in treatment intensity
- Time-series design with no cross-sectional variation -- limits control possibilities
- Outcome construction (raw diff vs freight-adjusted vs forward) is the main source of variation
- Variance outcome (G2) requires constructing dev2 using within-period means, which can be sensitive to the sample definition
