"""
Specification Search Script for Rudik (2020)
"Optimal Climate Policy When Damages are Unknown"
American Economic Journal: Economic Policy

Paper ID: 111185-V1

Executes the approved SPECIFICATION_SURFACE.json:
  - G1: log_correct ~ logt (Table 1 baseline, damage exponent estimation)

Outputs:
  - specification_results.csv (core specs: baseline, design/*, rc/*)
  - inference_results.csv (inference variants: infer/*)
  - SPECIFICATION_SEARCH.md (run log)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "111185-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_FILE = os.path.join(PACKAGE_DIR, "estimate_damage_parameters", "10640_2017_166_MOESM10_ESM.dta")
SEED = 111185

# =============================================================================
# Load and prepare data
# =============================================================================
print(f"Loading data from: {DATA_FILE}")
df_raw = pd.read_stata(DATA_FILE)
print(f"Raw data shape: {df_raw.shape}")

# Replicate data transformations from table_1.do
df_raw['correct_d'] = (df_raw['D_new'] / 100) / (1 - df_raw['D_new'] / 100)
df_raw['log_correct'] = np.where(df_raw['correct_d'] > 0, np.log(df_raw['correct_d']), np.nan)
df_raw['logt'] = np.log(df_raw['t'])
df_raw['logt2'] = df_raw['logt'] ** 2
df_raw['t2'] = df_raw['t'] ** 2
df_raw['asinh_correct'] = np.arcsinh(df_raw['correct_d'])

# Baseline regression sample
df_base = df_raw.dropna(subset=['log_correct', 'logt']).copy()
print(f"Baseline regression sample: {len(df_base)} observations (dropped {len(df_raw) - len(df_base)})")

# =============================================================================
# Available control variables
# =============================================================================
CONTROLS_ALL = ['Preindustrial', 'Market', 'Grey', 'Year', 'Repeat_Obs', 'Based_On_Other', 'cat']

CONTROL_BLOCKS = {
    'study_quality': ['Grey', 'Repeat_Obs', 'Based_On_Other'],
    'damage_type': ['Market', 'cat'],
    'study_design': ['Preindustrial', 'Year'],
}
BLOCK_NAMES = list(CONTROL_BLOCKS.keys())

# =============================================================================
# Helper functions
# =============================================================================
spec_results = []
inference_results = []
spec_counter = [0]
inference_counter = [0]

def next_spec_run_id():
    spec_counter[0] += 1
    return f"{PAPER_ID}_spec_{spec_counter[0]:03d}"

def next_inference_run_id():
    inference_counter[0] += 1
    return f"{PAPER_ID}_infer_{inference_counter[0]:03d}"

def run_ols(df, formula, outcome_var, treatment_var, spec_id, spec_tree_path,
            baseline_group_id, controls_desc, sample_desc, fixed_effects="",
            cluster_var="", coef_vector_extra=None):
    """Run an OLS regression and record results."""
    run_id = next_spec_run_id()
    try:
        model = smf.ols(formula, data=df).fit()
        coef = model.params.get(treatment_var, np.nan)
        se = model.bse.get(treatment_var, np.nan)
        pval = model.pvalues.get(treatment_var, np.nan)
        ci = model.conf_int()
        ci_lower = ci.loc[treatment_var, 0] if treatment_var in ci.index else np.nan
        ci_upper = ci.loc[treatment_var, 1] if treatment_var in ci.index else np.nan
        n_obs = int(model.nobs)
        r_sq = model.rsquared

        coef_vector = {k: float(v) for k, v in model.params.items()}
        if coef_vector_extra:
            coef_vector.update(coef_vector_extra)

        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r_sq,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'run_success': 1,
            'run_error': '',
        }
        spec_results.append(row)
        return run_id, model
    except Exception as e:
        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({'error': str(e)}),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'run_success': 0,
            'run_error': str(e)[:200],
        }
        spec_results.append(row)
        return run_id, None

def run_inference_variant(model, treatment_var, spec_run_id, spec_id, spec_tree_path,
                          baseline_group_id, cov_type, cov_kwds=None):
    """Recompute inference for an existing model under a different SE type."""
    infer_id = next_inference_run_id()
    try:
        if cov_kwds:
            refit = model.get_robustcov_results(cov_type=cov_type, **cov_kwds)
        else:
            refit = model.get_robustcov_results(cov_type=cov_type)

        # get_robustcov_results returns arrays, not Series -- use model param names to index
        param_names = list(model.params.index)
        tv_idx = param_names.index(treatment_var)
        coef = float(refit.params[tv_idx])
        se = float(refit.bse[tv_idx])
        pval = float(refit.pvalues[tv_idx])
        ci = refit.conf_int()
        ci_lower = float(ci[tv_idx, 0])
        ci_upper = float(ci[tv_idx, 1])

        coef_vector = {
            'inference': {
                'spec_id': spec_id,
                'method': cov_type,
                'se': float(se),
                'p_value': float(pval),
            }
        }

        row = {
            'paper_id': PAPER_ID,
            'inference_run_id': infer_id,
            'spec_run_id': spec_run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': 'log_correct',
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'cluster_var': '',
            'run_success': 1,
            'run_error': '',
        }
        inference_results.append(row)
    except Exception as e:
        row = {
            'paper_id': PAPER_ID,
            'inference_run_id': infer_id,
            'spec_run_id': spec_run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': 'log_correct',
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({'error': str(e)}),
            'cluster_var': '',
            'run_success': 0,
            'run_error': str(e)[:200],
        }
        inference_results.append(row)

def build_formula(outcome, treatment, controls):
    """Build OLS formula string."""
    rhs = [treatment] + controls
    return f"{outcome} ~ {' + '.join(rhs)}"


def winsorize(s, lower_q=0.01, upper_q=0.99):
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lo, hi)


# =============================================================================
# STEP 1: Baseline specification
# =============================================================================
print("\n=== Running Baseline ===")
baseline_run_id, baseline_model = run_ols(
    df=df_base,
    formula="log_correct ~ logt",
    outcome_var="log_correct",
    treatment_var="logt",
    spec_id="baseline",
    spec_tree_path="designs/cross_sectional_ols.md#baseline",
    baseline_group_id="G1",
    controls_desc="None (bivariate regression)",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)} (6 obs dropped for non-positive damages)",
)
print(f"  Baseline: coef={baseline_model.params['logt']:.4f}, SE={baseline_model.bse['logt']:.4f}, p={baseline_model.pvalues['logt']:.6f}")

# Run inference variants on baseline
for cov_type_label, cov_type_sm, spec_id_infer, tree_path in [
    ("HC1", "HC1", "infer/se/hc/hc1", "modules/inference/standard_errors.md#heteroskedasticity-robust-se-no-clustering"),
    ("HC2", "HC2", "infer/se/hc/hc2", "modules/inference/standard_errors.md#heteroskedasticity-robust-se-no-clustering"),
    ("HC3", "HC3", "infer/se/hc/hc3", "modules/inference/standard_errors.md#heteroskedasticity-robust-se-no-clustering"),
]:
    run_inference_variant(
        model=baseline_model,
        treatment_var="logt",
        spec_run_id=baseline_run_id,
        spec_id=spec_id_infer,
        spec_tree_path=tree_path,
        baseline_group_id="G1",
        cov_type=cov_type_sm,
    )

# =============================================================================
# STEP 2A: Controls -- Single additions
# =============================================================================
print("\n=== Running Single-Control Additions ===")
for ctrl in CONTROLS_ALL:
    formula = build_formula("log_correct", "logt", [ctrl])
    desc = f"logt + {ctrl}"
    run_ols(
        df=df_base,
        formula=formula,
        outcome_var="log_correct",
        treatment_var="logt",
        spec_id=f"rc/controls/single/add_{ctrl}",
        spec_tree_path="modules/robustness/controls.md#c-single-control-treatment--one-control",
        baseline_group_id="G1",
        controls_desc=desc,
        sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
        coef_vector_extra={"controls": {"family": "single", "added": [ctrl], "n_controls": 1}},
    )

# =============================================================================
# STEP 2B: Controls -- Standard sets
# =============================================================================
print("\n=== Running Control Sets ===")

# Minimal set: just study_quality indicators (most directly relevant)
minimal_controls = ['Grey', 'Based_On_Other']
formula = build_formula("log_correct", "logt", minimal_controls)
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/sets/minimal",
    spec_tree_path="modules/robustness/controls.md#a-standard-control-sets",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(minimal_controls)}",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "sets", "set_name": "minimal", "included": minimal_controls, "n_controls": len(minimal_controls)}},
)

# Extended set: study_quality + damage_type
extended_controls = CONTROL_BLOCKS['study_quality'] + CONTROL_BLOCKS['damage_type']
formula = build_formula("log_correct", "logt", extended_controls)
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/sets/extended",
    spec_tree_path="modules/robustness/controls.md#a-standard-control-sets",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(extended_controls)}",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "sets", "set_name": "extended", "included": extended_controls, "n_controls": len(extended_controls)}},
)

# Full set: all controls
full_controls = CONTROLS_ALL
formula = build_formula("log_correct", "logt", full_controls)
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/sets/full",
    spec_tree_path="modules/robustness/controls.md#a-standard-control-sets",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(full_controls)}",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "sets", "set_name": "full", "included": full_controls, "n_controls": len(full_controls)}},
)

# =============================================================================
# STEP 2C: Controls -- Progression
# =============================================================================
print("\n=== Running Control Progressions ===")

# Bivariate (= baseline, but recorded as progression for consistency)
run_ols(
    df=df_base, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/progression/bivariate",
    spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "progression", "step": "bivariate", "n_controls": 0}},
)

# + study_quality
prog_sq = CONTROL_BLOCKS['study_quality']
formula = build_formula("log_correct", "logt", prog_sq)
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/progression/study_quality",
    spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(prog_sq)}",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "progression", "step": "study_quality", "included": prog_sq, "n_controls": len(prog_sq)}},
)

# + damage_type
prog_dt = CONTROL_BLOCKS['study_quality'] + CONTROL_BLOCKS['damage_type']
formula = build_formula("log_correct", "logt", prog_dt)
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/progression/damage_type",
    spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(prog_dt)}",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "progression", "step": "damage_type", "included": prog_dt, "n_controls": len(prog_dt)}},
)

# + study_design (= full)
prog_full = CONTROL_BLOCKS['study_quality'] + CONTROL_BLOCKS['damage_type'] + CONTROL_BLOCKS['study_design']
formula = build_formula("log_correct", "logt", prog_full)
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/controls/progression/full",
    spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(prog_full)}",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
    coef_vector_extra={"controls": {"family": "progression", "step": "full", "included": prog_full, "n_controls": len(prog_full)}},
)

# =============================================================================
# STEP 2D: Controls -- Exhaustive block combinations (2^3 = 8)
# =============================================================================
print("\n=== Running Exhaustive Block Combinations ===")
block_combo_counter = 0
for r in range(len(BLOCK_NAMES) + 1):
    for combo in itertools.combinations(BLOCK_NAMES, r):
        if len(combo) == 0:
            continue  # bivariate already covered
        controls = []
        for block in combo:
            controls.extend(CONTROL_BLOCKS[block])
        if len(controls) > 6:
            continue  # respect max controls constraint
        block_combo_counter += 1
        combo_label = "_".join(combo)
        formula = build_formula("log_correct", "logt", controls)
        run_ols(
            df=df_base, formula=formula,
            outcome_var="log_correct", treatment_var="logt",
            spec_id=f"rc/controls/subset/block_{combo_label}",
            spec_tree_path="modules/robustness/controls.md#e-high-dimensional-control-set-search-combinatorial-budgeted",
            baseline_group_id="G1",
            controls_desc=f"logt + {' + '.join(controls)}",
            sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
            coef_vector_extra={"controls": {"family": "subset", "method": "exhaustive_blocks", "blocks_included": list(combo), "included": controls, "n_controls": len(controls)}},
        )
print(f"  Ran {block_combo_counter} block combinations")

# =============================================================================
# STEP 2E: Controls -- Random variable-level subsets (seeded)
# =============================================================================
print("\n=== Running Random Control Subsets ===")
rng = np.random.default_rng(SEED)
subset_count = 0
target_subsets = 15  # additional variable-level random subsets
for draw_idx in range(target_subsets):
    size = rng.integers(2, min(6, len(CONTROLS_ALL)) + 1)
    chosen = list(rng.choice(CONTROLS_ALL, size=size, replace=False))
    subset_count += 1
    formula = build_formula("log_correct", "logt", chosen)
    run_ols(
        df=df_base, formula=formula,
        outcome_var="log_correct", treatment_var="logt",
        spec_id=f"rc/controls/subset/random_{subset_count:03d}",
        spec_tree_path="modules/robustness/controls.md#e-high-dimensional-control-set-search-combinatorial-budgeted",
        baseline_group_id="G1",
        controls_desc=f"logt + {' + '.join(chosen)}",
        sample_desc=f"Howard & Sterner (2017), N={len(df_base)}",
        coef_vector_extra={"controls": {"family": "subset", "method": "random", "seed": SEED, "draw_index": draw_idx + 1, "included": chosen, "n_controls": len(chosen)}},
    )
print(f"  Ran {subset_count} random subsets")

# =============================================================================
# STEP 3: Sample restrictions
# =============================================================================
print("\n=== Running Sample Restrictions ===")

# 3A: Outlier trimming -- outcome
for pct_label, lo_q, hi_q in [("1_99", 0.01, 0.99), ("5_95", 0.05, 0.95)]:
    lo = df_base['log_correct'].quantile(lo_q)
    hi = df_base['log_correct'].quantile(hi_q)
    df_trim = df_base[(df_base['log_correct'] >= lo) & (df_base['log_correct'] <= hi)].copy()
    run_ols(
        df=df_trim, formula="log_correct ~ logt",
        outcome_var="log_correct", treatment_var="logt",
        spec_id=f"rc/sample/outliers/trim_y_{pct_label}",
        spec_tree_path="modules/robustness/sample.md#b-outliers-and-influential-observations",
        baseline_group_id="G1",
        controls_desc="None (bivariate)",
        sample_desc=f"Trimmed outcome [{pct_label}], N={len(df_trim)}",
        coef_vector_extra={"sample": {"axis": "outliers", "rule": "trim", "var": "log_correct", "lower_q": lo_q, "upper_q": hi_q, "n_obs_before": len(df_base), "n_obs_after": len(df_trim)}},
    )

# 3B: Outlier trimming -- treatment
lo = df_base['logt'].quantile(0.01)
hi = df_base['logt'].quantile(0.99)
df_trim = df_base[(df_base['logt'] >= lo) & (df_base['logt'] <= hi)].copy()
run_ols(
    df=df_trim, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/outliers/trim_x_1_99",
    spec_tree_path="modules/robustness/sample.md#b-outliers-and-influential-observations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Trimmed treatment [1,99], N={len(df_trim)}",
    coef_vector_extra={"sample": {"axis": "outliers", "rule": "trim", "var": "logt", "lower_q": 0.01, "upper_q": 0.99, "n_obs_before": len(df_base), "n_obs_after": len(df_trim)}},
)

# 3C: Cook's D
from statsmodels.stats.outliers_influence import OLSInfluence
infl = OLSInfluence(baseline_model)
cooksd = infl.cooks_distance[0]
threshold = 4.0 / len(df_base)
keep_mask = cooksd <= threshold
df_cooks = df_base[keep_mask].copy()
n_dropped_cooks = (~keep_mask).sum()
run_ols(
    df=df_cooks, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/outliers/cooksd_4_over_n",
    spec_tree_path="modules/robustness/sample.md#b-outliers-and-influential-observations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Dropped {n_dropped_cooks} obs with Cook's D > 4/N, N={len(df_cooks)}",
    coef_vector_extra={"sample": {"axis": "outliers", "rule": "cooksd", "threshold": threshold, "n_dropped": int(n_dropped_cooks), "n_obs_before": len(df_base), "n_obs_after": len(df_cooks)}},
)

# 3D: Drop repeat observations
df_no_repeat = df_base[df_base['Repeat_Obs'] == 0].copy()
run_ols(
    df=df_no_repeat, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/quality/drop_repeat_obs",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Dropped repeat observations, N={len(df_no_repeat)}",
    coef_vector_extra={"sample": {"axis": "quality", "rule": "drop_repeat_obs", "n_obs_before": len(df_base), "n_obs_after": len(df_no_repeat)}},
)

# 3E: Drop based-on-other
df_no_based = df_base[df_base['Based_On_Other'] == 0].copy()
run_ols(
    df=df_no_based, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/quality/drop_based_on_other",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Dropped based-on-other studies, N={len(df_no_based)}",
    coef_vector_extra={"sample": {"axis": "quality", "rule": "drop_based_on_other", "n_obs_before": len(df_base), "n_obs_after": len(df_no_based)}},
)

# 3F: Drop grey literature
df_no_grey = df_base[df_base['Grey'] == 0].copy()
run_ols(
    df=df_no_grey, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/quality/drop_grey_lit",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Dropped grey literature, N={len(df_no_grey)}",
    coef_vector_extra={"sample": {"axis": "quality", "rule": "drop_grey_lit", "n_obs_before": len(df_base), "n_obs_after": len(df_no_grey)}},
)

# 3G: Drop catastrophic damages (cat == 1)
df_no_cat = df_base[df_base['cat'] == 0].copy()
run_ols(
    df=df_no_cat, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/quality/drop_catastrophic",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Dropped catastrophic damage estimates, N={len(df_no_cat)}",
    coef_vector_extra={"sample": {"axis": "quality", "rule": "drop_catastrophic", "n_obs_before": len(df_base), "n_obs_after": len(df_no_cat)}},
)

# 3H: Temporal split -- early studies (Year <= median)
median_year = df_base['Year'].median()
df_early = df_base[df_base['Year'] <= median_year].copy()
run_ols(
    df=df_early, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/time/early_studies",
    spec_tree_path="modules/robustness/sample.md#a-time--period-restrictions",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Early studies (Year <= {median_year:.0f}), N={len(df_early)}",
    coef_vector_extra={"sample": {"axis": "time", "rule": "early_half", "cutoff": float(median_year), "n_obs_before": len(df_base), "n_obs_after": len(df_early)}},
)

# 3I: Late studies
df_late = df_base[df_base['Year'] > median_year].copy()
run_ols(
    df=df_late, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/time/late_studies",
    spec_tree_path="modules/robustness/sample.md#a-time--period-restrictions",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Late studies (Year > {median_year:.0f}), N={len(df_late)}",
    coef_vector_extra={"sample": {"axis": "time", "rule": "late_half", "cutoff": float(median_year), "n_obs_before": len(df_base), "n_obs_after": len(df_late)}},
)

# 3J: Independent estimates only (drop repeat + based_on_other)
df_independent = df_base[(df_base['Based_On_Other'] == 0) & (df_base['Repeat_Obs'] == 0)].copy()
run_ols(
    df=df_independent, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/sample/quality/independent_only",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Independent estimates only (no repeat, no based-on-other), N={len(df_independent)}",
    coef_vector_extra={"sample": {"axis": "quality", "rule": "independent_only", "n_obs_before": len(df_base), "n_obs_after": len(df_independent)}},
)

# =============================================================================
# STEP 4: Functional form variations
# =============================================================================
print("\n=== Running Functional Form Variations ===")

# 4A: Levels outcome: correct_d ~ logt
run_ols(
    df=df_base, formula="correct_d ~ logt",
    outcome_var="correct_d", treatment_var="logt",
    spec_id="rc/form/outcome/level",
    spec_tree_path="modules/robustness/functional_form.md#a-outcome-transformations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}, outcome in levels",
    coef_vector_extra={"functional_form": {"outcome_transform": "level", "treatment_transform": "log", "interpretation": "Semi-elasticity: level damages per unit log-temperature."}},
)

# 4B: Asinh outcome: asinh(correct_d) ~ logt
run_ols(
    df=df_base, formula="asinh_correct ~ logt",
    outcome_var="asinh_correct", treatment_var="logt",
    spec_id="rc/form/outcome/asinh",
    spec_tree_path="modules/robustness/functional_form.md#a-outcome-transformations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}, outcome = asinh(damages)",
    coef_vector_extra={"functional_form": {"outcome_transform": "asinh", "treatment_transform": "log", "interpretation": "Approx log for large y; handles zeros. Preserves damage concept."}},
)

# 4C: Levels treatment: log_correct ~ t (semi-elasticity: damages per degree C)
run_ols(
    df=df_base, formula="log_correct ~ t",
    outcome_var="log_correct", treatment_var="t",
    spec_id="rc/form/treatment/level",
    spec_tree_path="modules/robustness/functional_form.md#b-treatment-transformations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}, treatment in levels (degrees C)",
    coef_vector_extra={"functional_form": {"outcome_transform": "log", "treatment_transform": "level", "interpretation": "Semi-elasticity: pct change in damages per degree C."}},
)

# 4D: Quadratic in log temperature: log_correct ~ logt + logt^2
run_ols(
    df=df_base, formula="log_correct ~ logt + logt2",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/form/model/quadratic_treatment",
    spec_tree_path="modules/robustness/functional_form.md#c-nonlinear-dose-response-within-the-same-concept",
    baseline_group_id="G1",
    controls_desc="logt + logt^2",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}, quadratic in log-temp",
    coef_vector_extra={"functional_form": {"outcome_transform": "log", "treatment_transform": "log + log^2", "interpretation": "Test for nonlinearity in log-log relationship. Focal coef is linear term."}},
)

# 4E: Levels-quadratic: correct_d ~ t + t^2 (polynomial damage function)
run_ols(
    df=df_base, formula="correct_d ~ t + t2",
    outcome_var="correct_d", treatment_var="t",
    spec_id="rc/form/model/levels_quadratic",
    spec_tree_path="modules/robustness/functional_form.md#c-nonlinear-dose-response-within-the-same-concept",
    baseline_group_id="G1",
    controls_desc="t + t^2",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}, polynomial damage function in levels",
    coef_vector_extra={"functional_form": {"outcome_transform": "level", "treatment_transform": "level + level^2", "interpretation": "Quadratic polynomial damage function D = a + b*T + c*T^2. Alternative to power-law specification. Focal coef is linear term."}},
)

# =============================================================================
# STEP 5: Preprocessing variations
# =============================================================================
print("\n=== Running Preprocessing Variations ===")

# 5A: Winsorize outcome
df_winsor_y = df_base.copy()
df_winsor_y['log_correct'] = winsorize(df_winsor_y['log_correct'], 0.01, 0.99)
run_ols(
    df=df_winsor_y, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/preprocess/outcome/winsor_1_99",
    spec_tree_path="modules/robustness/preprocessing.md#outcome-coding-variations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Winsorized outcome [1,99], N={len(df_winsor_y)}",
    coef_vector_extra={"preprocess": {"target": "outcome", "operation": "winsorize", "params": {"lower_q": 0.01, "upper_q": 0.99}}},
)

# 5B: Winsorize treatment
df_winsor_x = df_base.copy()
df_winsor_x['logt'] = winsorize(df_winsor_x['logt'], 0.01, 0.99)
run_ols(
    df=df_winsor_x, formula="log_correct ~ logt",
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/preprocess/treatment/winsor_1_99",
    spec_tree_path="modules/robustness/preprocessing.md#treatment-coding-variations",
    baseline_group_id="G1",
    controls_desc="None (bivariate)",
    sample_desc=f"Winsorized treatment [1,99], N={len(df_winsor_x)}",
    coef_vector_extra={"preprocess": {"target": "treatment", "operation": "winsorize", "params": {"lower_q": 0.01, "upper_q": 0.99}}},
)

# =============================================================================
# STEP 6: Combined specs (joint axis variation)
# =============================================================================
print("\n=== Running Combined Specifications ===")

# 6A: Drop catastrophic + study quality controls
df_no_cat_ctrl = df_base[df_base['cat'] == 0].copy()
ctrl_safe = ['Grey', 'Repeat_Obs', 'Based_On_Other']
formula = build_formula("log_correct", "logt", ctrl_safe)
run_ols(
    df=df_no_cat_ctrl, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/drop_cat_study_quality",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(ctrl_safe)}",
    sample_desc=f"Dropped catastrophic + study quality controls, N={len(df_no_cat_ctrl)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"], "details": {"sample_rule": "drop_cat", "controls": ctrl_safe}}},
)

# 6B: Drop grey + Preindustrial control
df_no_grey2 = df_base[df_base['Grey'] == 0].copy()
formula = build_formula("log_correct", "logt", ['Preindustrial'])
run_ols(
    df=df_no_grey2, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/drop_grey_preindustrial",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="logt + Preindustrial",
    sample_desc=f"Dropped grey lit + Preindustrial control, N={len(df_no_grey2)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"], "details": {"sample_rule": "drop_grey", "controls": ["Preindustrial"]}}},
)

# 6C: Cook's D + minimal controls
formula = build_formula("log_correct", "logt", minimal_controls)
run_ols(
    df=df_cooks, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/cooksd_minimal",
    spec_tree_path="modules/robustness/sample.md#b-outliers-and-influential-observations",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(minimal_controls)}",
    sample_desc=f"Cook's D filtered + minimal controls, N={len(df_cooks)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"], "details": {"sample_rule": "cooksd_4_over_n", "controls": minimal_controls}}},
)

# 6D: Drop repeat obs + Market control
df_no_rep2 = df_base[df_base['Repeat_Obs'] == 0].copy()
formula = build_formula("log_correct", "logt", ['Market'])
run_ols(
    df=df_no_rep2, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/drop_repeat_market",
    spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
    baseline_group_id="G1",
    controls_desc="logt + Market",
    sample_desc=f"Dropped repeat obs + Market control, N={len(df_no_rep2)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"], "details": {"sample_rule": "drop_repeat_obs", "controls": ["Market"]}}},
)

# 6E: Levels treatment + study quality controls
formula = "log_correct ~ t + Grey + Repeat_Obs + Based_On_Other"
run_ols(
    df=df_base, formula=formula,
    outcome_var="log_correct", treatment_var="t",
    spec_id="rc/joint/form_controls/level_t_study_quality",
    spec_tree_path="modules/robustness/functional_form.md#b-treatment-transformations",
    baseline_group_id="G1",
    controls_desc="t + Grey + Repeat_Obs + Based_On_Other",
    sample_desc=f"Howard & Sterner (2017), N={len(df_base)}, treatment in levels + study quality controls",
    coef_vector_extra={"joint": {"axes_changed": ["form", "controls"], "details": {"treatment_transform": "level", "controls": ctrl_safe}}},
)

# 6F: Trim y 5/95 + study quality controls
lo_5 = df_base['log_correct'].quantile(0.05)
hi_95 = df_base['log_correct'].quantile(0.95)
df_trim_5_95 = df_base[(df_base['log_correct'] >= lo_5) & (df_base['log_correct'] <= hi_95)].copy()
formula = build_formula("log_correct", "logt", ctrl_safe)
run_ols(
    df=df_trim_5_95, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/trim_y_5_95_study_quality",
    spec_tree_path="modules/robustness/sample.md#b-outliers-and-influential-observations",
    baseline_group_id="G1",
    controls_desc=f"logt + {' + '.join(ctrl_safe)}",
    sample_desc=f"Trimmed outcome [5,95] + study quality controls, N={len(df_trim_5_95)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"]}},
)

# 6G: Early studies + Market control
formula = build_formula("log_correct", "logt", ['Market'])
run_ols(
    df=df_early, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/early_market",
    spec_tree_path="modules/robustness/sample.md#a-time--period-restrictions",
    baseline_group_id="G1",
    controls_desc="logt + Market",
    sample_desc=f"Early studies + Market control, N={len(df_early)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"]}},
)

# 6H: Late studies + Preindustrial control
formula = build_formula("log_correct", "logt", ['Preindustrial'])
run_ols(
    df=df_late, formula=formula,
    outcome_var="log_correct", treatment_var="logt",
    spec_id="rc/joint/sample_controls/late_preindustrial",
    spec_tree_path="modules/robustness/sample.md#a-time--period-restrictions",
    baseline_group_id="G1",
    controls_desc="logt + Preindustrial",
    sample_desc=f"Late studies + Preindustrial control, N={len(df_late)}",
    coef_vector_extra={"joint": {"axes_changed": ["sample", "controls"]}},
)

# =============================================================================
# STEP 7: Write outputs
# =============================================================================
print("\n=== Writing Outputs ===")

# Specification results
df_spec = pd.DataFrame(spec_results)
spec_out = os.path.join(PACKAGE_DIR, "specification_results.csv")
df_spec.to_csv(spec_out, index=False)
print(f"Wrote {len(df_spec)} specification rows to: {spec_out}")

# Inference results
df_infer = pd.DataFrame(inference_results)
infer_out = os.path.join(PACKAGE_DIR, "inference_results.csv")
df_infer.to_csv(infer_out, index=False)
print(f"Wrote {len(df_infer)} inference rows to: {infer_out}")

# Summary statistics
n_success = df_spec['run_success'].sum()
n_fail = (df_spec['run_success'] == 0).sum()
print(f"\nSummary: {n_success} successful, {n_fail} failed out of {len(df_spec)} total specs")

# Print spec_id distribution
print("\nSpec ID distribution:")
for prefix in ['baseline', 'rc/controls', 'rc/sample', 'rc/form', 'rc/preprocess', 'rc/joint']:
    count = df_spec['spec_id'].str.startswith(prefix).sum()
    if count > 0:
        print(f"  {prefix}: {count}")

# Print baseline coefficient for verification
baseline_row = df_spec[df_spec['spec_id'] == 'baseline'].iloc[0]
print(f"\nBaseline verification: coef={baseline_row['coefficient']:.6f}, SE={baseline_row['std_error']:.6f}, p={baseline_row['p_value']:.6e}")

print("\nDone!")
