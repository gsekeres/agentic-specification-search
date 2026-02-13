# Replication Agent Prompt

Use this prompt when launching agents to replicate author results from AEA replication packages.

---

## Inputs

- **Package directory**: `{EXTRACTED_PACKAGE_PATH}` (an extracted openICPSR package)
- **Paper ID**: `{PAPER_ID}` (the openICPSR project number, e.g., `112431-V1`)
- **Bibliographic data**: `data/tracking/AEA_universe.jsonl` (one JSON object per paper)
- **Method reference**: `prompts/reference/estimation_methods.md` (Stata→Python translation guide)

Important environment constraint:
- Stata is **not** available. Implement all regressions in **Python** or **R**.
- Translate Stata do-files, SAS scripts, or Matlab code into Python/R equivalents.
- **Consult `prompts/reference/estimation_methods.md`** for translation recipes before implementing.

---

## Task

Open the replication package, read the author's code, and replicate their regression results in Python or R. The goal is to verify that the data and code produce the published estimates.

### Scope

Replicate the paper's **main results** and **key robustness checks** — i.e., regressions that appear in the main text tables and any robustness tables referenced in the body of the paper. Skip appendix-only or supplementary regressions that are not referenced in the main text.

### What counts as a regression

Count only regression-type commands: OLS, 2SLS/IV (including first stages), panel FE, logit/probit, tobit, quantile regression, and similar estimators that produce coefficients and standard errors. Do **not** count summary statistics, t-tests, balance tables, simulations, or figure-generating code.

---

## Outputs (REQUIRED)

Write the following to `{EXTRACTED_PACKAGE_PATH}` (the **top-level** extracted package directory, NOT a subfolder like `Codes-and-data/` or `data/`):

### 1) `replication.csv`

One row per replicated regression. Required columns:

| Column | Description |
|--------|-------------|
| `paper_id` | The paper identifier (e.g., `112431-V1`) |
| `reg_id` | Sequential integer (1, 2, 3, ...) |
| `outcome_var` | Dependent variable name |
| `treatment_var` | Key independent variable / treatment |
| `coefficient` | Point estimate for the focal coefficient |
| `std_error` | Standard error |
| `p_value` | p-value |
| `ci_lower` | Lower bound of 95% CI (blank allowed) |
| `ci_upper` | Upper bound of 95% CI (blank allowed) |
| `n_obs` | Number of observations |
| `r_squared` | R-squared or pseudo-R-squared (blank allowed) |
| `original_coefficient` | The coefficient reported in the paper's code/logs |
| `original_std_error` | The SE reported in the paper's code/logs |
| `match_status` | `exact`, `close`, `discrepant`, or `failed` (see tolerance rules) |
| `coefficient_vector_json` | Full coefficient vector as JSON (all estimated coefficients) |
| `fixed_effects` | Fixed effects included (e.g., `"unit + time"`) |
| `controls_desc` | Brief description of control variables |
| `cluster_var` | Clustering variable(s) (blank if none) |
| `estimator` | Estimator used (e.g., `OLS`, `2SLS`, `logit`, `poisson`) |
| `sample_desc` | Brief sample description or filter applied |
| `notes` | Any issues, warnings, or deviations from original |

### Tolerance rules for `match_status`

- **`exact`**: Coefficient matches the original to 4+ decimal places
- **`close`**: Relative error ≤ 1% (i.e., `|coef - orig| / |orig| ≤ 0.01`)
- **`discrepant`**: Relative error > 1%
- **`failed`**: Regression could not be run (data issues, convergence failure, etc.)

### 2) `replication_report.md`

A structured markdown report containing:

```markdown
# Replication Report: {PAPER_ID}

## Summary
- **Paper**: [title from bibliographic data]
- **Replication status**: [full | small errors | major errors | not possible]
- **Total regressions in original package**: [count]
- **Regressions in scope (main + key robustness)**: [count]
- **Successfully replicated**: [count]
- **Match breakdown**: [N exact, N close, N discrepant, N failed]

## Data Description
- Files used: [list of data files loaded]
- Key variables: [main outcome and treatment variables]
- Sample sizes: [note any discrepancies with published N]

## Replication Details
[For each major table/result, briefly describe:]
- What was replicated
- Whether estimates match
- Any issues encountered (variable naming, missing data, version differences)

## Translation Notes
- Original language: [Stata/R/Python/etc.]
- Translation approach: [brief description of key translation decisions]
- Known limitations: [any Stata-specific features that required approximation]

## Software Stack
- Language: [Python X.Y / R X.Y]
- Key packages: [pyfixest, statsmodels, etc. with versions]
```

### 3) `scripts/paper_replications/{PAPER_ID}.py` (or `.R`)

Save the executable replication script to this repo-relative path. The script should:
- Be self-contained and runnable
- Load data from `{EXTRACTED_PACKAGE_PATH}`
- Print results to console and write `replication.csv`

---

## Tracking

### Append to `data/tracking/replication_tracking.jsonl`

After completing the replication, append one JSON line:

```json
{
  "paper_id": "{PAPER_ID}",
  "doi": "...",
  "title": "...",
  "journal": "...",
  "year": ...,
  "replication": "full|small errors|major errors|not possible",
  "original_specifications": 42,
  "replicated_specifications": 38,
  "exact_matches": 30,
  "close_matches": 6,
  "discrepant": 2,
  "failed": 0,
  "original_language": "stata",
  "replication_language": "python",
  "timestamp": "2026-02-12T..."
}
```

Pull `doi`, `title`, `journal`, `year` from `data/tracking/AEA_universe.jsonl` by matching on `paper_id`.

### Replication status classification

| Status | Criteria |
|--------|----------|
| **full** | All in-scope regressions have `match_status` of `exact` or `close` |
| **small errors** | ≥ 80% of in-scope regressions are `exact` or `close`, remainder are `discrepant` (not `failed`) |
| **major errors** | < 80% match rate, or any `failed` regressions that affect main results |
| **not possible** | Required data is missing/proprietary, or code cannot be translated (e.g., requires proprietary software with no equivalent) |

---

## Missing / Restricted Data

If **any** regressions in the package require proprietary, restricted, or otherwise unavailable data — even if some regressions use included data — classify the entire paper as `"replication": "not possible"`. Do not attempt partial replication.

Specifically:

- Mark the paper as `"replication": "not possible"` in the tracking file.
- Write a brief `replication_report.md` explaining what data is missing and which regressions are affected.
- Set `original_specifications` to the count of regressions found in the code (even if they can't be run).
- Set `replicated_specifications`, `exact_matches`, `close_matches`, `discrepant`, and `failed` all to `0`.
- Leave `replication.csv` empty (header row only).

This applies equally to packages that rely on proprietary datasets (e.g., ACNielsen, CRSP, Compustat), restricted-access government data, or software that cannot be translated (e.g., complex Dynare/Matlab structural models).

---

## Step-by-Step Procedure

### Step 0: Read the package

1. List all files in `{EXTRACTED_PACKAGE_PATH}`.
2. Read the README (usually `README.md`, `README.txt`, or `README.pdf`).
3. Identify the main analysis scripts (`.do`, `.R`, `.py`, `.m`, `.sas`).
4. Identify the data files (`.dta`, `.csv`, `.xlsx`, `.rds`, etc.).

### Step 1: Count original regressions

Scan **all** analysis scripts in the package and count regression commands:

- **Stata**: `reg`, `reghdfe`, `xtreg`, `areg`, `ivreg`, `ivreghdfe`, `ivreg2`, `logit`, `probit`, `tobit`, `qreg`, `poisson`, `nbreg`, `xtpoisson`, `clogit`, `ologit`, `mlogit`, `sureg`, `nl`, `gmm`
- **R**: `lm(`, `glm(`, `felm(`, `feols(`, `ivreg(`, `plm(`, `fixest::`, `lfe::`, `AER::ivreg`
- **Python**: `smf.ols(`, `smf.logit(`, `linearmodels.`, `pyfixest.`, `statsmodels.`
- **SAS**: `PROC REG`, `PROC LOGISTIC`, `PROC GENMOD`

This count is `original_specifications`. Count each invocation, not unique commands (i.e., if `reghdfe` is called 15 times, count 15).

### Step 2: Identify in-scope regressions

From the full set, identify which regressions correspond to:
- Main text tables (Table 1, Table 2, etc.)
- Robustness tables referenced in the main text

This is the subset you will replicate.

### Step 3: Load and prepare data

1. Load the data files into Python (pandas) or R.
2. Check for Stata-specific features: value labels, factor variables, variable labels.
3. Apply any data cleaning/filtering from the original code.
4. Verify sample sizes match the original before running regressions.

### Step 4: Translate and run regressions

For each in-scope regression:

1. Translate the original command to Python/R.
2. Run the regression.
3. Extract the focal coefficient, SE, p-value, and full coefficient vector.
4. Compare against the original output (from log files, or by reading the values the original code produces).
5. Record in `replication.csv`.

**Translation guidelines for Stata → Python**:
- `reghdfe y x, absorb(fe1 fe2) cluster(cl)` → `pyfixest.feols("y ~ x | fe1 + fe2", data=df, vcov={"CRV1": "cl"})`
- `reg y x1 x2, robust` → `pyfixest.feols("y ~ x1 + x2", data=df, vcov="hetero")`
- `ivreg2 y (x = z), robust` → `pyfixest.feols("y ~ 1 | x ~ z", data=df, vcov="hetero")`
- `logit y x1 x2` → `statsmodels.formula.api.logit("y ~ x1 + x2", data=df).fit()`
- `xtreg y x, fe cluster(id)` → `pyfixest.feols("y ~ x | id", data=df, vcov={"CRV1": "id"})`

**Common pitfalls**:
- Stata's `reghdfe` drops singletons by default; pyfixest does too, but verify.
- Stata uses analytic weights with `[aw=w]`; translate to `weights` parameter.
- Stata's `robust` = HC1; Python's default may differ — use HC1 explicitly.
- Stata's `cluster()` uses CR1 (small-sample adjusted); match with `CRV1` in pyfixest.
- Check that categorical/factor variables are handled identically (`i.var` in Stata → `C(var)` in Python).

### Step 5: Write outputs

1. Write `replication.csv` with all results.
2. Write `replication_report.md` with the structured report.
3. Save the script to `scripts/paper_replications/{PAPER_ID}.py` (or `.R`).
4. Append to `data/tracking/replication_tracking.jsonl`.
5. Classify the overall replication status.

---

## Flagging unlisted estimation methods

When you encounter a Stata/Matlab/R command that is **not covered** in `prompts/reference/estimation_methods.md`, add a line to `replication_report.md` in this format:

```
UNLISTED_METHOD: <command> in <paper_id> — <brief description of what it does>
```

This ensures the reference file can be updated for future papers. Do your best to translate the command anyway, but flag it so it can be reviewed.

---

## Quality checks before finishing

- Every `reg_id` in `replication.csv` is unique.
- `coefficient_vector_json` is valid JSON for every row.
- `match_status` is one of: `exact`, `close`, `discrepant`, `failed`.
- `original_specifications` count is documented and plausible.
- The replication script is self-contained and runnable.
- The tracking JSONL line has all required fields.
- Replication status classification follows the criteria table above.
- Any estimation methods not in the reference file are flagged with `UNLISTED_METHOD`.
