# Verification Report: 112496-V1

## Paper Information
- **Title**: Understanding International Prices: Customers as Capital
- **Authors**: Lukasz A. Drozd, Jaromir B. Nosal
- **Journal**: American Economic Review (2012)
- **Total Specifications**: 3,064

## Baseline Groups

### G1: Correlation of Real Export and Import Prices (corr(px, pm))
- **Claim**: Real export and import prices are highly positively correlated across OECD countries.
- **Baseline spec**: spec_id=33
- **Expected sign**: Positive (strongly positive, ~0.75-0.85)
- **Baseline estimate**: 0.7953
- **Outcome**: `corr_px_pm`
- **Method**: Cross-country average of pairwise correlations of HP-filtered log real export and import price deflators
- **Table 1**

### G2: Volatility Ratio of Terms of Trade to Real Exchange Rate (vol(TOT)/vol(RER))
- **Claim**: Terms of trade are much less volatile than the real exchange rate.
- **Baseline spec**: spec_id=34
- **Expected sign**: Positive but well below 1 (~0.5-0.7)
- **Baseline estimate**: 0.5688
- **Outcome**: `vol_tot_over_rer`
- **Method**: Cross-country average of SD(TOT)/SD(RER) computed on HP-filtered log series
- **Table 2**

### G3: Volatility Ratio of Quantities to Prices (vol(Q)/vol(P))
- **Claim**: The quantity (product mix) ratio has comparable or slightly lower volatility than the price ratio.
- **Baseline spec**: spec_id=2393
- **Expected sign**: Positive, near 1 (~0.8-0.9)
- **Baseline estimate**: 0.7939
- **Outcome**: `volatility_ratio_Q_over_P`
- **Method**: Cross-country average of SD(quantity ratio)/SD(price ratio) from national accounts deflators; baseline uses shorter sample 1980-2000
- **Table 3**

### G4: Relative Volatility of Export Prices to Real Exchange Rate (vol(px)/vol(RER))
- **Claim**: Real export prices are consistently less volatile than the real exchange rate.
- **Baseline spec**: spec_id=2913
- **Expected sign**: Positive but below 1 (~0.5-0.7)
- **Baseline estimate**: 0.6140
- **Outcome**: `rel_vol_px_to_rer`
- **Method**: Cross-country average of SD(px)/SD(RER) on HP-filtered log series
- **Supporting statistic complementing Table 2**

**Note**: This is a descriptive statistics / calibration targets paper with no regression analysis. All specifications compute cross-country summary statistics (correlations, volatility ratios) on detrended macroeconomic time series from OECD national accounts (12 countries, 1980Q1-2004Q2). The specification search varies analytical choices combinatorially. An additional 16 specs compute nominal (undeflated) export-import price correlations (`corr_EPI_IPI_nominal`); these are classified as non-core alternative outcomes assigned to G1.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Baseline** | **4** | Paper's primary reported specifications for each fact |
| **Core** | **2,700** | Standard analytical variations across all dimensions |
| **Non-core** | **360** | Extreme HP parameter (lambda=100000) or alternative outcome (nominal correlations) |
| **Total** | **3,064** | |

### Core Tests by Baseline Group

| Group | Baseline | Core | Non-core | Total |
|-------|----------|------|----------|-------|
| G1 (corr_px_pm) | 1 | 1,023 | 144 | 1,168 |
| G2 (vol_tot/vol_rer) | 1 | 1,023 | 128 | 1,152 |
| G3 (vol_Q/vol_P) | 1 | 511 | 64 | 576 |
| G4 (vol_px/vol_rer) | 1 | 143 | 24 | 168 |

Note: G1 total (1,168) includes 16 `corr_EPI_IPI_nominal` specs classified as non-core.

### Core Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_multi_3d | 1,393 | Vary 3 dimensions simultaneously |
| core_multi_4d | 686 | Vary 4 dimensions simultaneously |
| core_countries_sample | 161 | Vary country set and sample period |
| core_countries_hp_lambda | 140 | Vary country set and HP parameter |
| core_hp_lambda_sample | 115 | Vary HP parameter and sample period |
| core_countries_detrend | 42 | Vary country set and detrending method |
| core_detrend_sample | 42 | Vary detrending method and sample period |
| core_countries | 28 | Vary only country set |
| core_sample | 23 | Vary only sample period |
| core_hp_lambda | 20 | Vary only HP filter parameter |
| core_countries_deflator | 14 | Vary country set and deflator |
| core_deflator_sample | 14 | Vary deflator and sample period |
| core_deflator_hp_lambda | 10 | Vary deflator and HP parameter |
| core_detrend | 6 | Vary only detrending method |
| core_deflator_detrend | 4 | Vary deflator and detrending method |
| core_deflator | 2 | Vary only deflator (CPI to PPI) |

### Non-core Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| noncore_extreme_multi_3d | 189 | HP=100000 combined with 2+ other dimension changes |
| noncore_extreme_multi_4d | 98 | HP=100000 combined with 3+ other dimension changes |
| noncore_extreme_param | 57 | HP=100000 with at most 1 other dimension change |
| noncore_alt_outcome | 16 | Nominal (undeflated) export-import price correlations |

## Classification Rationale

### Why most specs are classified as core

This is a descriptive statistics paper where specifications represent combinatorial variations of standard analytical choices in international macroeconomics. Unlike a regression-based paper where changing the outcome variable or adding/removing controls can represent qualitatively different analyses, here every specification computes the same type of statistic (correlation or volatility ratio) with the same conceptual interpretation. The dimensions varied are:

1. **Detrending method** (HP filter, linear detrend, first-differencing): All three are standard and widely used in the RBC/international macro literature. First-differencing emphasizes higher-frequency variation but remains a legitimate approach.

2. **HP filter parameter** (lambda = 400 through 10000): Values from 400 to 10000 are all used in the literature (400 for annual data mapping, 1600 is the Hodrick-Prescott standard, 6400 for quarterly-to-annual frequency conversion). These are classified as core.

3. **HP filter parameter = 100000**: This extreme value essentially removes the HP filter's smoothing, approximating linear detrending. Since linear detrending is already included as a separate method, HP(100000) is redundant and classified as non-core.

4. **Sample period** (8 windows): All are economically motivated subsamples (pre/post-Plaza Accord, pre/post-euro, decade-specific). All are core.

5. **Price deflator** (CPI vs PPI for G1/G2; national accounts for G3): Both CPI and PPI are standard deflators. The choice matters substantively (PPI removes more common price variation) but both are defensible.

6. **Country subset** (8 groups): All represent standard groupings in international macro (OECD, G7, euro area, Anglo-Saxon, etc.). All are core.

### Why HP=100000 is non-core

HP filtering with lambda=100000 produces a trend that closely tracks the raw series, leaving essentially no cyclical component beyond a near-linear residual. This is practically identical to linear detrending, which is already covered as a separate detrend_method. Including it would double-count one analytical choice. The 344 specs with HP=100000 (minus 16 nominal specs) yield 328 non-core specs; however, 16 of these are also nominal correlations, producing a total of 344 non-core from HP=100000 and 16 from the alternative outcome.

## Robustness Assessment

### G1: corr(px, pm) -- Extremely Robust

- **1,024 core specs** (including baseline)
- Core mean: 0.671, median: 0.654, SD: 0.134, range: [0.378, 0.936]
- **100% of core specs show positive correlation**
- **90% of core specs show correlation above 0.5**
- The finding that real export and import prices are strongly positively correlated is extremely robust across all standard analytical variations.

**Key sensitivities**:
- **Deflator**: CPI-based mean = 0.795, PPI-based mean = 0.558. Using PPI substantially reduces the average correlation, but it remains strongly positive in all specifications. This is the single largest source of variation.
- **Detrending method**: HP mean = 0.684, linear mean = 0.718, first-diff mean = 0.581. First-differencing produces lower correlations by focusing on quarter-to-quarter changes.
- **Country subset**: Euro-core countries show higher correlations (~0.86) than Anglo countries (~0.70), reflecting different trade structures and exchange rate regimes.

### G2: vol(TOT)/vol(RER) -- Highly Robust

- **1,024 core specs** (including baseline)
- Core mean: 0.587, median: 0.585, SD: 0.121, range: [0.367, 1.067]
- **99.8% of core specs show ratio below 1**
- **93.8% of core specs show ratio below 0.8**
- The finding that terms of trade are much less volatile than the real exchange rate is highly robust.

**Key sensitivity**: Only 2 out of 1,024 core specs produce a ratio above 1, and these occur in narrow subsamples with atypical exchange rate behavior.

### G3: vol(Q)/vol(P) -- Moderately Robust

- **512 core specs** (including baseline)
- Core mean: 0.915, median: 0.873, SD: 0.213, range: [0.390, 1.624]
- **68.6% of core specs show ratio below 1**
- **98.2% of core specs show ratio above 0.5**
- The finding that quantity adjustment is not dramatically larger than price adjustment is moderately robust. The ratio exceeds 1 in about 31% of core specifications, particularly in the 1990s-2000s subsample.

**Key sensitivity**: In the later sample periods (1990-2004), the ratio rises above 1, suggesting quantity adjustment became relatively larger. The paper acknowledges this sensitivity.

### G4: vol(px)/vol(RER) -- Robust

- **144 core specs** (including baseline)
- Core mean: 0.639, median: 0.629, SD: 0.110, range: [0.499, 0.855]
- **100% of core specs show ratio below 1**
- Real export prices are consistently less volatile than the real exchange rate across all core specifications.

## Duplicates and Mechanical Issues

No exact duplicates were identified in the specification search. Each of the 3,064 specifications represents a unique combination of the 5 analytical dimensions.

There are no mechanical identities or tautological specifications in this search. All statistics are computed from data and represent genuine empirical measurements.

## Data Notes

- Small differences from published values arise because the specification search uses the OECD REER index directly, rather than constructing a trade-weighted real exchange rate from bilateral exchange rates with time-varying weights as in the original MATLAB code. The qualitative patterns match the paper's reported values.
- The `corr_EPI_IPI_nominal` outcome (16 specs) measures nominal export-import price correlations. These are classified as non-core because they measure a fundamentally different object (nominal vs. real prices). Their mean of 0.838 is somewhat higher than the CPI-deflated baseline (0.795), consistent with common inflation components boosting nominal correlations.
