# Specification Surface Review: 116139-V1

## Paper: Kosfeld and Rustagi (2015 AER) "Leader Punishment and Cooperation in Groups: Experimental Field Evidence from Ethiopia"

**Reviewer**: Pre-run audit (verifier agent)
**Date**: 2026-02-25

---

## Summary of Baseline Groups

### G1: Leader Type and Group Cooperation (Table 6, Col 3)
- **Outcome**: `pct` (forest condition percentage, measure of group cooperation)
- **Treatment**: `leq`, `leqef`, `las` (three leader type dummies entered jointly; omitted category is non-punishment type)
- **Controls**: `ccs`, `ed`, `pp`, `gs`, `wmk`, `time`, `fem`, `ginic` (8 group-level characteristics)
- **FE**: Village FE (`i.vcode`, 5 villages)
- **SE**: Heteroskedasticity-robust
- **Dataset**: `Leader_Group_AER_2014.dta` (N = 53 total rows, 51 with non-missing `lcode`)
- **Verified against do-file**: MATCH. Line 147: `xi: reg pct $leader $control i.vcode, robust` where `$leader = "leq leqef las"` and `$control = "ccs ed pp gs wmk time fem ginic"`.

### G2: Leader Punishment Behavior (Table 3)
- **Outcome**: `pi` (number of punishment points assigned)
- **Treatment**: `i.cd1` (9 experimental condition dummies, cd1 = 2..10, base = cd1 = 1)
- **Controls**: None
- **Clustering**: Session/group level (`fcode`)
- **Model**: Poisson
- **Dataset**: `Leader_Pun_Poisson_AER_2014.dta` (N = 510)
- **Verified against do-file**: MATCH. Line 62: `poisson pi i.cd1, vce(cluster fcode)`.

**Claim grouping is correct**: G1 is the primary claim (leader type predicts group cooperation). G2 is a secondary/complementary claim (leaders punish differently across experimental conditions).

---

## Variable Name Verification

### G1 variables (Leader_Group_AER_2014.dta, 53 obs x 51 vars)

**All verified present**:
- Outcome: `pct` CONFIRMED, `pct2` CONFIRMED
- Treatment: `leq`, `leqef`, `las` CONFIRMED
- Controls: `ccs`, `ed`, `pp`, `gs`, `wmk`, `time`, `fem`, `ginic` CONFIRMED
- Extended controls: `chet`, `shet`, `ginil`, `lage`, `ledu` CONFIRMED
- FE: `vcode` CONFIRMED
- Sample restrictions: `vlcode`, `lcode` CONFIRMED
- Appendix controls: `turnover`, `leaderskill`, `clan1`, `clan2`, `clan3`, `peren`, `seas`, `slope`, `lpun` CONFIRMED

**ISSUE FOUND AND FIXED -- `lclan` vs `lclanp`**:
- The surface originally referenced `lclan` (matching the do-file text), but `Leader_Group_AER_2014.dta` contains `lclanp`, NOT `lclan`.
- The do-file works because Stata allows variable name abbreviation: `lclan` uniquely matches `lclanp` in the Group dataset.
- In Python code, we must use the exact name `lclanp`.
- The `Leader_Pun_Poisson_AER_2014.dta` dataset does contain `lclan` (not `lclanp`), so G2 references to `lclan` are correct.
- **Fixed**: Updated `rc/controls/single/add_lclan` to `rc/controls/single/add_lclanp` in G1, and updated the constraints and sampling notes accordingly.

### G2 variables (Leader_Pun_Poisson_AER_2014.dta, 510 obs x 19 vars)

**All verified present**:
- Outcome: `pi` CONFIRMED, `pj` CONFIRMED
- Treatment: `cd1` CONFIRMED, pre-generated dummies `_Icd1_2` through `_Icd1_10` CONFIRMED
- Controls: `lage`, `ledu`, `lclan` CONFIRMED
- Clustering: `fcode` CONFIRMED
- Subsampling: `lcode`, `cases3` CONFIRMED

---

## Key Constraints and Linkage Rules

- **Leader type dummies always enter jointly**: `leq`, `leqef`, `las` must always be included together in G1 (they define the treatment of interest; dropping one changes the estimand). Correctly documented.
- **No bundled estimator**: Simple OLS/Poisson, no linkage across equations.
- **Village FE treatment**: The surface correctly treats village FE as part of the preferred specification (Col 3), with `rc/fe/drop_village_fe` as an explicit robustness variant.
- **Small sample caution**: N ~ 47-51 for G1 means some specifications with many controls may have very few degrees of freedom. The surface notes this.
- **Influential observations**: The do-file hardcodes specific row indices for influential observations (lines 163-168): rows 7, 13 (2 influential), and additionally rows 5, 32 (4 influential). The runner should use DFITS diagnostics to identify these or use the hardcoded indices.

---

## Changes Made

1. **Fixed `lclan` to `lclanp` in G1**: Changed `rc/controls/single/add_lclan` to `rc/controls/single/add_lclanp`. Updated `constraints.notes` and `sampling.notes` to reference `lclanp`.

2. **Updated sample size note in `design_audit.notes`**: Clarified that the dataset has 53 groups total with 51 having non-missing `lcode`.

---

## Budget/Sampling Assessment

- **G1 budget (60 core specs, 20 controls subset)**:
  - LOO from 8 baseline controls = 8 specs
  - Single additions of 6 extended controls (chet, shet, ginil, lage, ledu, lclanp) = 6 specs
  - Single additions of 8 appendix controls (turnover, leaderskill, clan1, clan2, clan3, peren, seas, slope) = 8 specs
  - 4 predefined control sets (no controls, group only, group+leader, full+heterogeneity) = 4 specs
  - Progressive build-up = ~4 specs
  - 20 random subsets = 20 specs
  - 2 FE variants (drop village FE, village FE only) = 2 specs
  - 2 treatment variants (lcode dummy, lpun continuous) = 2 specs
  - 1 outcome variant (pct2) = 1 spec
  - 3 sample restrictions (drop influential 2, drop influential 4, drop vice leaders, drop LNP) = 4 specs
  - Joint controls x sample = ~4 specs
  - Total ~ 63 specs. Budget of 60 is tight but feasible with minor pruning.

- **G2 budget (20 core specs)**:
  - 2 model variants (nbreg, OLS/LPM) = 2 specs
  - 4 control additions (lage, ledu, lclan individually + all 3 together) = 4 specs
  - 1 outcome variant (pj) = 1 spec
  - 3 subsample by leader type = 3 specs
  - 1 design alternative (diff in means) = 1 spec
  - Baseline = 1 spec
  - Total ~ 12 specs. Well within budget.

- **Combined target (~80 specs)**: Achievable.

---

## What's Missing

1. **Table A5 additional controls**: The do-file (Table A5) tests 12 additional leader characteristics as controls: `leaderskill`, `firstleader`, `lsiblings`, `lbirthorderall`, `lbirthordersons`, `lwealthall`, `lwealthsons`, `lwives`, `lwivessiblings`, `lchildren`, `lheight`, `lchest`. The surface includes `leaderskill` but not the others. Some of these (e.g., `firstleader`, `lsiblings`) could be added as RC variants if they are in the data. Not critical -- Table A5 is supplementary.

2. **Table A6 additional specifications**: The do-file tests `cc_maj` (CC majority), `ccs2` (CCS squared), and clan dummies. The surface includes `clan1`, `clan2`, `clan3` but not `cc_maj` or `ccs2`. Consider adding if desired for completeness.

3. **`turnreason` control**: Table A10 Col 3 adds `turnreason` as a control. The surface includes `turnover` but not `turnreason`. Consider adding.

4. **Table 7 panel analysis**: Currently excluded. This uses a different dataset (`Leq_Panel_AER_2014.dta`) and panel FE design. Correctly excluded from the core surface since it targets a different estimand. Could be added as an `explore/*` spec.

5. **Influential observation identification**: The do-file hardcodes row indices for DFITS-influential observations. The runner should either use the same hardcoded indices or dynamically compute DFITS. Consider adding a note to the sample restriction RC specs.

---

## Final Assessment

**APPROVED TO RUN** -- no blocking issues.

The surface is well-constructed. The baseline specifications match the do-file exactly. The RC axes are comprehensive and appropriate for a cross-sectional OLS design with small sample size. The variable name issue (`lclan` vs `lclanp`) has been corrected. The budget is feasible. The control pool variation (LOO, additions, random subsets) is particularly well-suited for this paper given the paper's own extensive control robustness checks across Table 6 columns.

One minor caution: the very small sample size (N ~ 47-51) means that specifications with many controls (13-14) will have very few degrees of freedom, and some may not converge or may produce unreliable estimates. The runner should track and flag such cases.
