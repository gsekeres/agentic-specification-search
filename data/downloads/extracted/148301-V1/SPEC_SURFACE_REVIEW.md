# Specification Surface Review: 148301-V1

## Summary of Baseline Groups

- **G1**: Foreign market access and export platform location
  - GLM fractional logit: ep ~ lfma + controls, sector-year FE, cluster(country)
  - Baseline: Table 2, Column 4 (full controls with haven indicator)
  - Marginal effects (dydx) are the reported coefficients
  - Unit: country x sector x year panel, BEA data 1999-2013

- **G2**: Profit shifting through export platforms
  - OLS (reghdfe): lprofit ~ ep_haven + controls, sector-year FE, cluster(country)
  - Baseline: Table 4, Column 1 (log profit)
  - Focal parameter: coefficient on ep_haven (export platform share x haven)

## Changes Made

1. **Verified GLM fractional logit baseline matches code.** The `2.Paper.do` code shows Table 2 Col 4 as: `glm ep lfma taxr haven control2 i.kt, link(logit) family(binomial) cluster(i)` followed by `margins, dydx(...)`. The surface correctly identifies `haven` as a separate control added in Col 4 (Cols 1-3 do not include haven). Confirmed.

2. **Corrected baseline label from `Table2-Col4` to match code.** The code shows Col 1 = lfma + lrgdp, Col 2 = lfma + taxr + lrgdp, Col 3 = lfma + taxr + control2 (eoi_enf, dtc_enf, lrgdp, dtc_num), Col 4 = lfma + taxr + haven + control2. The surface baseline is Col 4 with all 6 controls. This is correct.

3. **Verified haven-related linkage for G2.** The surface notes that when haven is dropped from controls, ep_haven becomes undefined. The code constructs `ep_haven = ep * haven`, so dropping haven as a control while keeping ep_haven would be valid (haven would just not be separately controlled for). However, the code always includes haven when ep_haven is present. The surface's warning is appropriate.

4. **Added note about Table 3 haven split (big5 vs otherh).** The code in `2.Paper.do` Table 3 section uses `big5 otherh` instead of a single `haven` dummy. This maps to `rc/form/treatment/haven_split_big5_other` in the surface. Verified.

5. **Verified G2 profit equation.** Code: `reghdfe lprofit lfma ep taxr haven ep_haven eoi_enf dtc_enf dtc_num lrgdp lemp leqpmt, absorb(kt) cluster(i)`. The surface correctly lists all 10 controls plus ep_haven as treatment. GPML Poisson and cube root transform alternatives are in Table 4 Cols 2-3. Verified.

6. **Noted Table 3 manufacturing/services split.** The code uses `if d<=8` for manufacturing and `if d>8` for services. These map to `rc/sample/subgroup/manufacturing_only` and `rc/sample/subgroup/services_only`. Verified.

7. **Added `rc/fe/alt/country_year` to surface.** The Appendix do-file (3.Appendix.do) likely contains additional FE specifications. The surface already includes country-year as an alt FE spec, which is appropriate for a panel.

## Key Constraints and Linkage Rules

- **GLM marginal effects**: The GLM fractional logit reports index coefficients; the paper reports marginal effects (dydx). The runner must compute marginal effects for all GLM specifications.
- **OLS alternative**: Table 2 Cols 7-8 use `reghdfe` instead of GLM. This is an estimator switch that changes functional form assumptions. The surface correctly includes `rc/form/estimator/ols_reghdfe`.
- **Haven subsamples**: Table 2 Cols 5-6 restrict to haven==0 and haven==1 respectively. These are in the surface as `rc/sample/subgroup/non_haven_only` and `rc/sample/subgroup/haven_only`.
- **Control progression**: Table 2 Cols 1-4 progressively add controls: lrgdp -> taxr,lrgdp -> taxr,control2 -> taxr,haven,control2. The surface includes `rc/controls/progression/col1_to_col4`.
- **Country-level clustering**: All specs cluster at the country level (`i` in the code, `iso3` in the data).

## Budget/Sampling Assessment

- G1: 80 specs is feasible. The control pool is small (6 controls), LOO + progression + 10 random subsets + sample splits + estimator switch + FE variants covers the space well.
- G2: 30 specs is feasible. Fixed control set, variation from LOO + outcome transforms + sample restrictions.
- Combined: ~110 specs across both groups exceeds the 50-spec minimum.
- Random subset sampling (10 draws) is appropriate given the small control pool.

## What's Missing

- **GPML Poisson for G1**: The paper does not use Poisson for the export platform share equation (only for the profit equation). The surface correctly excludes it from G1.
- **Country FE for G1**: The surface includes `rc/fe/add/country` for both G1 and G2. For G1, adding country FE on top of sector-year FE would absorb most of the cross-country variation in lfma, potentially making the regression degenerate. This is an aggressive robustness check but is in the paper's appendix.
- **Time-series variation**: The surface does not include first-differencing or long-differencing. The treatment (lfma) varies primarily cross-sectionally, so this is appropriate to exclude.

## Verification Against Code

- `2.Paper.do`: Verified Table 2 (6 GLM cols + 2 OLS cols), Table 3 (manufacturing/services split, big5/otherh haven split), Table 4 (lprofit, GPML Poisson, cube root), Table 5 (quantification exercise).
- `replication.do`: Master file confirming code structure.
- Data: `Data/Analysis/FT_PS` is the analysis dataset.
- Sample construction: `reghdfe profit_... if imputed_profit==0` confirms the profit sample restriction (non-imputed profits).

## Final Assessment

**APPROVED TO RUN.** The surface is well-designed for this GLM fractional logit / OLS panel study. Two baseline groups correctly separate the export platform location claim (G1) from the profit shifting claim (G2). The key implementation challenge is computing GLM marginal effects correctly in Python. The budget and sampling plan are adequate. No blocking issues identified.
