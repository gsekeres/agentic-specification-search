# Specification Surface Review: 116063-V1

## Paper: Dafny (2005 AER) "How Do Hospitals Respond to Price Changes?"

**Reviewer**: Pre-run audit (verifier agent)
**Date**: 2026-02-25

---

## Summary of Baseline Groups

### G1: Intensity Analysis (Table 5, IV for lnprocs)
- **Outcome**: `lnprocs` (log total procedures per DRG-pair-year)
- **Endogenous**: `lnwt` (log DRG weight)
- **Instrument**: `lnlas88instr` (log Laspeyres price index based on 1987 coding and 1988 weights)
- **FE**: DRG dummies (`drg_*`) + DRG-specific linear trends (`drgXyea_*`) + year dummies (`year86-year91`)
- **Weights**: Frequency weights (`totprocs`)
- **SE**: Heteroskedasticity-robust
- **Verified against do-file**: MATCH. Line 300 of `int.do`: `ivreg lnprocs (lnwt=lnlas88instr) year86-year91 drg_* drgXyea_* [w=totprocs] , robust`.

### G2: Upcoding Analysis (Table 3, IV for fracy)
- **Outcome**: `fracy` (fraction of young patients coded with complications)
- **Endogenous**: `spread` (DRG weight spread)
- **Instrument**: `sp8788pt` (1987-88 spread change interacted with post-87 dummy)
- **FE**: DRG dummies (`drg_*`) + year dummies (`year86-year91`)
- **Weights**: Frequency weights (`totyoung`)
- **SE**: Heteroskedasticity-robust
- **Verified against do-file**: MATCH. Line 254 of `up.do`: `ivreg fracy (spread=sp8788pt) year86-year91 drg_* [w=totyoung], robust`.

**Claim grouping is correct**: G1 (intensity/volume response to prices) is the primary claim; G2 (upcoding response to price incentives) is a complementary secondary claim. Both are IV designs with distinct instruments, outcomes, and claim objects.

---

## Variable Name Verification

All variables are constructed within the do-files from raw data. Verified variable construction chain:

**G1 (from int.do)**:
- `lnprocs = ln(totprocs)` -- line 218
- `lnwt = ln(weight)` -- line 220
- `lnchga = ln(totchga/chgprocs)` -- line 214 (charges deflated by hospital CPI)
- `lnlos = ln(totlos/totprocs)` -- line 217
- `lnsurg = ln(totsurg/sgprocs)` -- line 215
- `lnicu = ln(toticu/totprocs)` -- line 216
- `lndeathr = ln(dead/totprocs)` -- line 219
- `lnlas88instr = ln(laspeyres88) - lnwt87` -- line 248 (zeroed pre-reform: line 258)
- `lnlascl88instr` -- residualized on pre-reform charge growth (lines 249-253)
- `drgtype` -- constructed from 1987 admission patterns (elective/urgent/emergent): lines 170-187

**G2 (from up.do)**:
- `fracy = totyoung/syprocs` -- line 156
- `fraco = totold/soprocs` -- line 168
- `spread` = DRG weight (positive for CC code, negative for non-CC code, summed within pair) -- lines 145-147
- `sp8788pt = sp8788 * post` -- line 223 (where sp8788 = spread88 - spread87)
- `fracy87post = fracy87 * post` -- line 222

All variable names in the surface are consistent with the do-file construction.

---

## CRITICAL DATA AVAILABILITY ISSUE

**The raw analysis data is NOT included in the replication package.** The do-files (`int.do`, `up.do`) require intermediate Stata files (`drgcg`, `up`, `uphosp`) that are created by SAS programs (`intdata.sas`, `updata.sas`) from raw MedPAR 20% sample data. The MedPAR data is confidential and not provided.

What IS included:
- `drg85_96.txt` -- DRG weight tables (used to merge on DRG weights)
- `pps85.dta` through `pps91.dta` -- Hospital-level PPS data (used for cost:charge ratios)
- `int.do`, `up.do` -- Analysis do-files
- `intdata.sas`, `updata.sas`, `extract84_96.sas` -- SAS data construction programs
- `inthosp.do`, `nonpps.do` -- Additional analysis files (hospital-level, non-PPS)

**Implication for the runner**: The runner must either:
1. Obtain the MedPAR data and run the SAS programs to construct intermediate files, OR
2. Reconstruct the analysis dataset from the do-file logic (not feasible without raw data), OR
3. Treat this paper as "data-incomplete" and skip the specification search.

**Added a note to the top-level `notes` field in the JSON.**

---

## Key Constraints and Linkage Rules

- **IV bundled estimator**: First and second stage share the same FE structure, weights, and sample. Changes to FE or weights apply to both stages. Correctly flagged with `linked_adjustment: true` for G1.
- **Outcome-weight linkage (G1)**: Different outcomes use different frequency weights:
  - `lnprocs` -> `totprocs`
  - `lnchga` -> `chgprocs`
  - `lnsurg` -> `sgprocs`
  - `lnlos`, `lnicu`, `lndeathr` -> `totprocs`
  This linkage is correctly documented in the `rc/joint/outcome_weight/*` specs.
- **No controls pool**: Both G1 and G2 use only FE structure as controls. No covariates to vary. Correctly reflected.

---

## Changes Made

1. **Added DATA NOTE to top-level `notes`**: Documented that the analysis data requires SAS construction from raw MedPAR data not included in the replication package. Specified what IS and IS NOT available.

---

## Budget/Sampling Assessment

- **G1 budget (60 specs)**: Feasible if data is available. Axis enumeration:
  - 6 outcomes (lnprocs, lnchga, lnlos, lnsurg, lnicu, lndeathr)
  - 2 instrument variants (baseline lnlas88instr, clean lnlascl88instr)
  - ~3 FE variants (full, drop DRG trends, year only)
  - ~3 weight variants (baseline, unweighted, outcome-matched)
  - ~4 sample restrictions (drop 1985, pre-post only, DRG-type subsamples, outlier trimming)
  - 2 design alternatives (OLS, LIML)
  - Plus joint combinations
  - Full enumeration feasible.

- **G2 budget (20 specs)**: Feasible. Fewer axes.

- **Combined target (~80 specs)**: Reasonable.

---

## What's Missing

1. **Reduced-form estimates (G1)**: The do-file runs reduced-form regressions (`areg lnprocs lnlas88instr ...`) as well as IV. The surface includes this implicitly through the instrument variant axis, but a dedicated `rc/form/treatment/reduced_form` spec would be clearer. Currently it is listed for G2 but not explicitly for G1 as a named RC, though the areg reduced-form specs in the do-file (lines 286-296) provide this. Consider adding `rc/form/treatment/reduced_form` to G1's rc_spec_ids.

2. **OLS comparison (G1)**: The surface includes `rc/form/treatment/ols_no_iv` which is correct. The do-file confirms OLS regressions at lines 303, 338-342.

3. **Hausman test diagnostic**: The do-file performs Hausman tests comparing IV and OLS (lines 301-304, 307-308, etc.). The surface includes `diag/instrumental_variables/endogeneity/hausman` which is correct.

4. **Clean instrument reduced form (G1)**: Line 282 runs the first stage with the clean instrument. This is covered by `rc/form/instrument/lnlascl88instr`.

5. **Hospital-level analysis excluded**: Correctly excluded -- the hospital-level data (`uphosp`) is even less available than the DRG-level data, and uses a different IV strategy (share of young patients with complications).

---

## Final Assessment

**CONDITIONALLY APPROVED TO RUN** -- approved if the analysis data can be constructed from the SAS programs and raw MedPAR data. The surface itself is well-structured, with correct baseline specifications, appropriate RC axes, proper IV bundling constraints, and reasonable budgets.

**Blocking issue**: Data availability. The replication package does not include the constructed analysis datasets (`drgcg.dta`, `up.dta`). The runner cannot proceed without these intermediate files or access to the raw MedPAR 20% sample.

If data is available, no structural issues with the specification surface.
