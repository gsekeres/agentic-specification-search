# Verification Report: 113561-V1

## Paper Information
- **Title**: What Determines Giving to Hurricane Katrina Victims?
- **Authors**: Christina M. Fong, Erzo F.P. Luttmer
- **Journal**: American Economic Journal: Applied Economics (2009)
- **Total Specifications**: 157

## Baseline Groups

### G1: Main Effect on Giving -- All Respondents
- **Claim**: Showing pictures of black Katrina victims (picshowblack) affects charitable giving. The paper expects the main effect to be small/insignificant because positive (black respondent) and negative (white respondent) effects offset.
- **Baseline spec**: `baseline_tab3_col2`
- **Expected sign**: Negative (weakly)
- **Baseline coefficient**: -2.30 (SE: 3.85, p = 0.55)
- **Outcome**: `giving` (real allocation, $0-$100)
- **Treatment**: `picshowblack`
- **Table 3, Column 2**

### G2: Main Effect on Giving -- White Respondents
- **Claim**: White respondents reduce giving when shown black victims, reflecting weaker cross-racial solidarity.
- **Baseline spec**: `baseline_tab3_col5_white`
- **Expected sign**: Negative
- **Baseline coefficient**: -4.33 (SE: 4.68, p = 0.35)
- **Outcome**: `giving`
- **Treatment**: `picshowblack`
- **Table 3, Column 5**

### G3: Main Effect on Giving -- Black Respondents
- **Claim**: Black respondents increase giving when shown black victims, reflecting in-group solidarity.
- **Baseline spec**: `baseline_tab3_col6_black`
- **Expected sign**: Positive
- **Baseline coefficient**: +5.63 (SE: 8.54, p = 0.51)
- **Outcome**: `giving`
- **Treatment**: `picshowblack`
- **Table 3, Column 6**

### G4: Manipulation Check
- **Claim**: Showing black victims significantly shifts perceived racial composition of recipients.
- **Baseline spec**: `manip_check_per_hfhdif_all`
- **Expected sign**: Positive
- **Baseline coefficient**: +16.35 (SE: 3.99, p < 0.001)
- **Outcome**: `per_hfhdif` (perceived % black recipients, difference)
- **Treatment**: `picshowblack`

**Note**: The paper's actual empirical contribution is about heterogeneous (INTERACTION) effects -- e.g., white respondents who feel close to their ethnic group reduce giving more when shown black victims (Table 3, cols 3-6; Table 6). The specification search tests only the main effect of picshowblack, which the paper itself reports as insignificant. Therefore the weak/null main effects are expected and do not constitute a replication failure.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Baselines** | **4** | Table 3 cols 2/5/6 (giving by all/white/black) + manipulation check |
| **Core tests (incl. baselines)** | **136** | |
| core_controls | 48 | 3 baselines + nraudworthy variant, no-charity, minimal, pic+audio only, extra region, extra hhsize, LOO (20 controls), full manip (3 duplicates of baselines) |
| core_sample | 54 | White/black/all x city (main survey, Slidell, Biloxi) x 4 outcomes, race-shown treatment restriction x 3 samples x 4 outcomes |
| core_outcome | 12 | Alternative outcomes (hypgiv_tc500, subjsupchar, subjsupgov) x (all/white/black), plus 3 Table 5 s1 duplicates |
| core_inference | 8 | Unweighted OLS versions for giving/hypgiv_tc500/subjsupchar/subjsupgov x all/white |
| **Non-core tests** | **21** | |
| noncore_alt_outcome | 18 | Perception outcomes (9 variables x all/white): per_votebush, per_income, per_govtcash, per_prepared, per_attrelig, per_crimrec, per_helpoth, per_workhard, per_windspeed |
| noncore_diagnostic | 3 | Manipulation check (per_hfhdif) for all/white/black -- validates design, not main effect |
| **Total** | **157** | |

## Detailed Classification Notes

### Core Tests (136 specs including 4 baselines)

**Baselines (4 specs)**: Three correspond to Table 3 columns 2, 5, and 6 (all/white/black respondents, giving outcome, WLS with tweight, robust SE, full demographic controls). The fourth is the manipulation check (per_hfhdif, all respondents), which establishes that the experimental treatment works.

**Control variations (48 specs)**: The largest category. Includes:
- Audio control variants: nraudworthy composite vs full audio manipulation dummies (baseline_tab4_r1_all, outcome_giving_white/black, ctrl_full_manip_all/white/black)
- Reduced controls: no demographic controls (s5 variants, 12 specs across outcomes x samples), minimal (picture manips only, 4 specs), picture + audio only (2 specs), no prior charity controls (3 specs)
- Extra controls: region dummies (4 specs), household size (2 specs), extra controls including HfH effectiveness + life priorities (s6 variants, 12 specs)
- Leave-one-out (20 specs): Each drops one of 20 demographic controls from the white-respondent giving specification

**Sample restrictions (54 specs)**: Systematic exploration across:
- City subsamples: main survey (s2), Slidell (s3), Biloxi (s4) -- tested for all 4 outcomes x 3 race groups
- Treatment arm: race-shown only (s8, excluding obscured-race condition) -- tested for all 4 outcomes x 3 race groups

**Alternative outcomes (12 specs)**: Three outcomes beyond the primary giving measure:
- `hypgiv_tc500`: hypothetical giving (topcoded at $500)
- `subjsupchar`: support for charitable spending
- `subjsupgov`: support for government spending
Each tested across all/white/black samples. These are core because the paper reports results for all four outcomes in Tables 3-5.

**Inference variations (8 specs)**: Unweighted OLS (dropping survey weights) for 4 outcomes x 2 samples (all and white respondents).

### Non-Core Tests (21 specs)

**Perception outcomes (18 specs)**: Nine perception variables tested for all and white respondents:
- `per_votebush`, `per_income`, `per_govtcash`, `per_prepared`, `per_attrelig`, `per_crimrec`, `per_helpoth`, `per_workhard`, `per_windspeed`
These measure how showing black victims changes respondents' perceptions of the victim population along various dimensions. They test a distinct mechanism (stereotype activation) rather than the giving/support outcomes, so they are classified as non-core alternative outcomes. Note: `per_windspeed` serves as a placebo -- showing black victims should not change perceived windspeed.

**Manipulation check (3 specs)**: The per_hfhdif outcome (perceived % black recipients) for all/white/black subsamples. These validate the experimental design rather than test the main effect, so they are classified as non-core diagnostics. G4 baseline is included here.

## Duplicates Identified

The following specs produce identical coefficients and SEs:
1. `ctrl_full_manip_all` = `baseline_tab3_col2` (coef = -2.30, SE = 3.85, N = 1343)
2. `ctrl_full_manip_white` = `baseline_tab3_col5_white` (coef = -4.33, SE = 4.68, N = 915)
3. `ctrl_full_manip_black` = `baseline_tab3_col6_black` (coef = 5.63, SE = 8.54, N = 247)
4. `tab5_hypgiv_tc500_s1_baseline_white` = `outcome_hypgiv_tc500_white` (coef = -2.18, SE = 4.06, N = 913)
5. `tab5_subjsupchar_s1_baseline_white` = `outcome_subjsupchar_white` (coef = -0.22, SE = 0.16, N = 907)
6. `tab5_subjsupgov_s1_baseline_white` = `outcome_subjsupgov_white` (coef = -0.44, SE = 0.20, N = 913)
7. `loo_drop_black_white` = `loo_drop_other_white` (coef = -4.20, SE = 4.68, N = 915) -- race dummies have no variation in white-only sample

After removing duplicates, there are approximately 150 unique specifications.

## Robustness Assessment

The main effect of `picshowblack` on giving is **WEAK/NULL** -- consistent with the paper's own framing and results:

- **G1 (all respondents, giving)**: The coefficient ranges from -3.60 (Biloxi subsample) to -0.38 (Slidell) across core specs, and is never significant at 5%. Median coefficient around -2.0. This is expected: the paper argues positive and negative subgroup effects cancel.

- **G2 (white respondents, giving)**: Coefficient ranges from -8.87 (Biloxi) to +1.46 (Slidell). Never significant at 5% across 57 giving specifications. LOO analysis shows remarkable stability: coefficient stays between -3.6 and -4.3 regardless of which control is dropped. The most negative estimate comes from the extra-controls specification (-4.92).

- **G3 (black respondents, giving)**: Coefficient ranges from -3.07 (Slidell) to +19.89 (Biloxi, p = 0.08). Small N (~125-247) limits power. Generally positive but never significant at 5%.

- **G4 (manipulation check)**: Strong and robust. Coefficient is +16.35 (all, p < 0.001), +12.07 (white, p = 0.008), +26.02 (black, p = 0.026). The experimental manipulation clearly works.

Key findings across outcomes:
- **Giving (real)**: 0% significant at 5% across all 57 specifications. Consistent null.
- **Hypothetical giving**: 0% significant. Even weaker than real giving.
- **Charity support (subjsupchar)**: 0% significant. Consistently negative for white respondents.
- **Government support (subjsupgov)**: 24% significant (6/25 specs), primarily for white respondents. The coefficient for white respondents is around -0.44 (p ~ 0.02-0.04), the most robust negative effect found.
- **Perception outcomes**: Only per_votebush_all shows significance (p = 0.014). Most perceptions are unaffected.

The specification search confirms that the paper's main finding is NOT about the main effect of picshowblack but about heterogeneous effects (interactions with racial attitudes), which are not captured in these main-effect specifications. The weak main effects are internally consistent and expected.
