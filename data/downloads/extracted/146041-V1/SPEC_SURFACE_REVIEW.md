# Specification Surface Review: 146041-V1

## Summary of Baseline Groups

- **G1**: Cross-country elasticity of aggregate skill quality (AQ) with respect to income per worker
  - Bivariate OLS: log(AQ measure) ~ log(income per worker)
  - Baseline: Table 2, Row 1, Column 3 (irAQ53_dum_skti_hrs_secall, sigma=1.5)
  - Unit: country (~11 in micro sample, ~90 in Barro-Lee sample)
  - No controls by design (all regressions are bivariate)

## Changes Made

1. **Issue: Table 2 do-file does NOT use `robust` SEs.** The surface states canonical inference is HC1 (`robust`), but `tab_2.do` runs `qui reg l_outcome l_y` with no `, robust` option. This means OLS standard errors are used. However, other do-files in the code (e.g., `devacc_main.do`) may use robust. The Table 2 regressions are simple bivariate with N~11, where robust and classical SEs are nearly identical. **Updated the canonical inference description** to note that the Table 2 code uses default (homoskedastic) SEs, and HC1 is a reasonable choice for our pipeline but is not strictly the paper's baseline.

2. **Verified `rc/form/outcome/*` axis is dominant.** The code confirms Table 2 varies along: (a) wage premium method (education dummies vs experience+gender adjusted), (b) self-employment inclusion, (c) sector subsamples (agriculture, manufacturing, low-skill services, high-skill services). Table 3 varies along: sample restriction (US immigrants, pooled countries, bilateral controls, selection-adjusted, 10+ years, good English, no downgrading, sorting sectors, sorting regions). These match the `rc/sample/subgroup/*` and `rc/sample/restriction/*` specs in the surface.

3. **Verified no controls axis.** Confirmed: all Table 2 and Table 3 regressions are `reg l_outcome l_y` with zero controls. The surface correctly sets `controls_count_min/max` to 0.

4. **Added `rc/form/outcome/aq_sigma_4p0`**: The devacc_main.do code includes `gen sigma4 = 4` for an alternative sigma=4 calculation. This was missing from the surface. Added it.

5. **Noted overlap between baseline_spec_ids and rc_spec_ids.** Some specs like `baseline__aq_sigma_high` and `rc/form/outcome/aq_sigma_2p0` may overlap (both correspond to sigma=2.0 AQ). This is not problematic as long as the runner deduplicates. No change needed but flagged for awareness.

## Key Constraints and Linkage Rules

- **No controls**: All regressions are bivariate. The control-count envelope is correctly [0, 0].
- **Outcome construction is the main axis**: The spec universe is defined by how AQ is constructed (sigma, wage measure, labor supply, skill threshold, sample) rather than what controls are included.
- **Sample and outcome are confounded**: Changing the immigrant sample (e.g., 10+ years in US) changes the Q estimate, which changes the AQ variable, which is the outcome. The surface correctly treats these as `rc/sample/restriction/*`.
- **No fixed effects**: Cross-country bivariate regression with no FE.

## Budget/Sampling Assessment

- Target 50-80 specs is feasible with full enumeration.
- No control-subset sampling needed (no controls).
- The combinatorial space is: sigma (3) x wage_measure (2) x labor_supply (3) x sample (9 Table 3 rows) x sector (5 from Table 2), but not all combinations exist in the data. The surface correctly lists specific specs rather than trying to enumerate a full cross-product.
- Budget of 80 is adequate.

## What's Missing

- **Common Mincerian return assumption** (`rc/form/outcome/aq_mincerian_common`): Listed in surface, verified in code. OK.
- **Body count vs hours vs population**: Listed in surface as `rc/data_construction/labor_supply/*`. Verified these correspond to code variants. OK.
- **Alternative treatment (GDP PPP vs PWT)**: `rc/form/treatment/log_gdp_ppp` and `rc/form/treatment/log_gdp_pwt` are listed. The code uses PWT income data. These are appropriate.

## Verification Against Code

- `tab_2.do`: Bivariate regressions of log(outcome) on l_y with sample_micro==1 filter. Rows correspond to: (1) baseline dummies, (2) experience+gender adjusted, (3) baseline on self-employment sample, (4) self-employment, (5-8) sector subsamples (agriculture, manufacturing, low-skill services, high-skill services). All match surface.
- `tab_3.do`: Nine rows varying the Q computation: US immigrants only, pooled countries, bilateral controls, selection adjusted, 10+ years in US, good English, no downgrading, sorting (sectors), sorting (geographic). All match surface `rc/sample/*` specs.
- `devacc_main.do`: Development accounting with alternative sigma values. Matches surface.
- Data: `temp/Q_origins.dta` contains pre-computed country-level AQ measures with various construction choices.

## Final Assessment

**APPROVED TO RUN.** The surface is well-structured and faithful to the paper's revealed search space. The main axis of variation (outcome construction) is correctly identified as the dominant specification dimension. The corrections are minor: (1) noting the default vs robust SE discrepancy, and (2) adding sigma=4 variant. The budget and sampling plan are adequate for full enumeration.
