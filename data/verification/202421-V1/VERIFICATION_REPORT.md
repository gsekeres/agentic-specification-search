# Verification Report: 202421-V1

## Paper Information
- **Title**: Who Watches the Watchmen? Local News and Police Behavior in the United States
- **Authors**: Nicola Mastrorocco, Arianna Ornaghi
- **Journal**: AEJ: Economic Policy
- **Total Specifications**: 63

## Baseline Groups

### G1: Crime Search Volume
- **Claim**: Sinclair Broadcasting acquisition of local TV stations reduces Google Trends search volume for "crime" in affected DMAs.
- **Baseline spec**: `baseline/google_trends/crime`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0150 (SE: 0.0154, p = 0.332)
- **Outcome**: `log_hits_crime`
- **Treatment**: `ownedby_sinclair`

### G2: Police Search Volume
- **Claim**: Sinclair Broadcasting acquisition reduces Google Trends search volume for "police" in affected DMAs.
- **Baseline spec**: `baseline/google_trends/police`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0247 (SE: 0.0179, p = 0.170)
- **Outcome**: `log_hits_police`
- **Treatment**: `ownedby_sinclair`

**Note**: Neither baseline is statistically significant. The Google Trends analysis (Table 4) is secondary evidence; the paper's primary results are in Tables 1-3 (content analysis and clearance rates) which require proprietary data.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **48** | |
| core_controls | 15 | Baselines (2) + control set variations (none, minimal, partial, full, leave-one-out) |
| core_fe | 10 | Fixed effects variations (unit only, time only, year vs month, none) |
| core_sample | 14 | Sample restrictions (early/late period, drop always/never treated, outliers, trimming, WLS) |
| core_funcform | 6 | Functional form (IHS, levels, standardized for crime and police) |
| core_inference | 1 | Robust SEs instead of clustered |
| core_method | 2 | First differences estimation |
| **Non-core tests** | **15** | |
| noncore_placebo | 11 | Placebo outcomes (weather, YouTube in various forms) and fake treatment timing |
| noncore_heterogeneity | 4 | Heterogeneity by crime-search level and adoption timing |
| **Total** | **63** | |

Baselines (is_baseline=1): 2 specs (`baseline/google_trends/crime`, `baseline/google_trends/police`), both also classified as core_controls.

## Detailed Classification Notes

### Core Tests (48 specs including 2 baselines)

**Control variations (15 specs, including 2 baselines)**: The specification search systematically varies the control set from no controls through minimal (crime trend only), partial (crime + police trends), to full controls (all four DMA 2010 characteristics x time trends). Leave-one-out robustness drops each individual control. This is done for both crime and police outcomes. Two specs are exact duplicates of the baselines: `did/controls/full/crime` = `baseline/google_trends/crime` and `did/controls/full/police` = `baseline/google_trends/police`. One additional duplicate: `did/controls/none_dma_ym/police` = `did/controls/none/police`.

**Fixed effects variations (10 specs)**: Tests include DMA FE only (no time FE), time FE only (no DMA FE), coarser time FE (year instead of year-month), and no FE at all. Applied to both crime and police outcomes. Two duplicates: `did/fe/year_instead_month/police` appears twice and `did/fe/unit_only/police_ctrl` duplicates `did/fe/unit_only/police`.

**Sample restrictions (14 specs)**: Early period (2010-2013) vs late period (2014-2017); dropping always-treated or never-treated DMAs; dropping top/bottom 10% by search volume; trimming 1% outcome tails; excluding first/last year; removing the min>0 filter; WLS weighted by search volume. Most are for crime, with some for police.

**Functional form (6 specs)**: Inverse hyperbolic sine (IHS), levels, and standardized outcome transformations for both crime and police outcomes.

**Inference (1 spec)**: Robust (heteroskedasticity-consistent) standard errors instead of DMA-clustered SEs. Same point estimate as baseline.

**Method (2 specs)**: First differences instead of fixed effects for crime and police.

### Non-Core Tests (15 specs)

**Placebo tests (11 specs)**: These test the research design validity rather than the main claim:
- Placebo outcomes at baseline: weather and YouTube search volumes (2 baseline placebo specs + 2 explicit placebo specs = 4, noting that baseline/weather = placebo/weather and baseline/youtube = placebo/youtube are duplicates)
- Alternative functional forms for placebo outcomes: IHS and levels for weather and YouTube -- 4 specs
- Fake treatment timing: shifted 1 and 2 years earlier -- 2 specs
- Random permutation of treatment -- 1 spec

**Heterogeneity (4 specs)**: Split by above/below median crime search intensity and by early vs late Sinclair adopter timing. The early/late adopter specs use different treatment variables (`early_treat`, `late_treat`), making them heterogeneity analyses rather than direct robustness tests.

## Duplicates Identified

The following specs produce identical coefficient/SE values:
1. `baseline/google_trends/crime` = `did/controls/full/crime` (both coef = -0.01502, SE = 0.01544)
2. `baseline/google_trends/police` = `did/controls/full/police` (both coef = -0.02468, SE = 0.01790)
3. `baseline/google_trends/weather` = `placebo/weather_outcome` (both coef = 0.01497, SE = 0.01578)
4. `baseline/google_trends/youtube` = `placebo/youtube_outcome` (both coef = -0.02164, SE = 0.01201)
5. `did/controls/none/police` = `did/controls/none_dma_ym/police` (both coef = -0.02046, SE = 0.02049)
6. `did/fe/unit_only/police` = `did/fe/unit_only/police_ctrl` (both coef = -0.02392, SE = 0.01847)
7. `did/fe/year_instead_month/police` (row 23) = `did/fe/year_instead_month/police` (row 63) (both coef = -0.02814, SE = 0.01773)

After accounting for duplicates, there are approximately 56 unique specifications.

## Robustness Assessment

The crime search coefficient (G1) ranges from -0.145 to -0.006 across core specifications (all negative), with the unfiltered sample producing the only significant result at 5%. The police search coefficient (G2) ranges from -0.058 to +0.063, with the sign flipping only in the no-FE specification. The results are directionally consistent but statistically weak, which aligns with the paper's framing of Google Trends as secondary evidence.

Key sensitivity:
- Removing the min>0 sample filter increases magnitude dramatically (from -0.015 to -0.145, p=0.011)
- Time FE only or no FE specifications produce very different results, confirming the importance of DMA fixed effects
- Early period shows stronger effects than late period
- First differences yields significant results where levels does not
