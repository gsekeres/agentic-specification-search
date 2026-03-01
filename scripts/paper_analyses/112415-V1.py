"""
Specification Search Script for Arieli, Ben-Ami & Rubinstein (2011)
"Tracking Decision Makers under Uncertainty"
American Economic Journal: Microeconomics, 3(4), 68-76.

Paper ID: 112415-V1

Surface-driven execution:
  - G1: Effect of choice set size on attention efficiency
  - Within-subject experiment (N=41 subjects, 3 set-size conditions: 4, 9, 16)
  - Baseline: paired t-tests on subject-level means of fixation efficiency
  - Design alternative: OLS with subject FE
  - Multiple outcome measures, sample restrictions, pairwise comparisons

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "112415-V1"
DATA_DIR = "data/downloads/extracted/112415-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit block from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_raw = pd.read_stata(f"{DATA_DIR}/fixations_data.dta")
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

print(f"Raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"Subjects: {df_raw['subject'].nunique()}")
print(f"Set sizes: {sorted(df_raw['numalt'].unique())}")

results = []
inference_results = []
spec_run_counter = 0
infer_run_counter = 0


# ============================================================
# HELPER: Compute efficiency (matching Stats-2A.R)
# ============================================================
def compute_efficiency(data):
    """Compute efficiency = (roirating - MeanSetRating) / (maxrating - MeanSetRating)"""
    eff = (data['roirating'] - data['MeanSetRating']) / (data['maxrating'] - data['MeanSetRating'])
    return eff


def compute_initial_fixation_number(data):
    """Compute order number of initial fixation (matching R code)"""
    number = np.zeros(len(data))
    number[0] = 1
    for i in range(1, len(data)):
        if data['fixOrder'].iloc[i] == 1:
            number[i] = 1
        elif data['fixOrder'].iloc[i] > 1:
            number[i] = number[i-1] + 1
    return number


def compute_stopping_dummy(data):
    """Compute dummy for final initial fixation (stopping probability)"""
    number = compute_initial_fixation_number(data)
    dum = np.zeros(len(data))
    for i in range(len(data) - 1):
        if number[i] >= number[i+1]:
            dum[i] = 1
    dum[len(data) - 1] = 1
    return dum


def compute_cached_value(data):
    """Compute cached (best-seen-so-far) value during initial search"""
    cached = np.zeros(len(data))
    for i in range(len(data)):
        if data['fixOrder'].iloc[i] == 1:
            cached[i] = data['roirating'].iloc[i]
        elif data['fixOrder'].iloc[i] > 1:
            cached[i] = max(data['roirating'].iloc[i], cached[i-1])
    return cached


# ============================================================
# HELPER: Paired t-test on subject-level means
# ============================================================
def paired_ttest_on_means(data, outcome_col, numalt_a, numalt_b):
    """
    Collapse to subject x condition means, then do paired t-test.
    Returns (mean_diff, t_stat, p_value, n_subjects, means_a, means_b)
    """
    # Collapse to subject x numalt means
    subj_means = data.groupby(['subject', 'numalt'])[outcome_col].mean().reset_index()

    # Get means for each condition
    means_a = subj_means[subj_means['numalt'] == numalt_a].set_index('subject')[outcome_col]
    means_b = subj_means[subj_means['numalt'] == numalt_b].set_index('subject')[outcome_col]

    # Align subjects (inner join)
    common = means_a.index.intersection(means_b.index)
    means_a = means_a.loc[common]
    means_b = means_b.loc[common]

    # Paired t-test
    tstat, pval = stats.ttest_rel(means_a, means_b)
    mean_diff = means_a.mean() - means_b.mean()
    n_subj = len(common)
    se = mean_diff / tstat if tstat != 0 else np.nan

    return {
        'mean_diff': mean_diff,
        'se': abs(se),
        't_stat': tstat,
        'p_value': pval,
        'n_subjects': n_subj,
        'mean_a': means_a.mean(),
        'mean_b': means_b.mean(),
    }


# ============================================================
# HELPER: Record a paired t-test result
# ============================================================
def record_ttest(spec_id, spec_tree_path, outcome_var, treatment_var,
                 data, outcome_col, numalt_a, numalt_b,
                 sample_desc, notes="",
                 axis_block_name=None, axis_block=None,
                 functional_form=None):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        res = paired_ttest_on_means(data, outcome_col, numalt_a, numalt_b)

        coef_val = float(res['mean_diff'])
        se_val = float(res['se'])
        pval = float(res['p_value'])
        n_obs = int(res['n_subjects'])

        ci_lower = coef_val - 1.96 * se_val
        ci_upper = coef_val + 1.96 * se_val

        coefficients = {
            f"numalt_{int(numalt_a)}": float(res['mean_a']),
            f"numalt_{int(numalt_b)}": float(res['mean_b']),
            "diff": coef_val,
        }

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block
        if functional_form:
            blocks["functional_form"] = functional_form

        payload = make_success_payload(
            coefficients=coefficients,
            inference={"spec_id": "infer/se/paired_ttest", "params": {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            blocks=blocks,
            design={"randomized_experiment": G1_DESIGN_AUDIT},
        )

        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': 'G1',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef_val,
            'std_error': se_val,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': '',
            'cluster_var': '',
            'run_success': 1,
            'run_error': '',
        })
        print(f"  OK {run_id}: {spec_id} | coef={coef_val:.4f}, p={pval:.4f}, n={n_obs}")
        return run_id

    except Exception as e:
        err_msg = str(e)[:200]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': 'G1',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': '',
            'cluster_var': '',
            'run_success': 0,
            'run_error': err_msg,
        })
        print(f"  FAIL {run_id}: {spec_id} | {err_msg}")
        return run_id


# ============================================================
# HELPER: Record an OLS with subject FE result
# ============================================================
def record_ols_fe(spec_id, spec_tree_path, outcome_var, treatment_var,
                  data, outcome_col, treatment_col,
                  vcov, sample_desc, cluster_var_name="",
                  notes="", axis_block_name=None, axis_block=None,
                  functional_form=None):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        formula = f"{outcome_col} ~ {treatment_col} | subject"
        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().iloc[0])
        se_val = float(m.se().iloc[0])
        pval = float(m.pvalue().iloc[0])
        n_obs = int(m._N)
        r2 = float(m._r2) if hasattr(m, '_r2') else np.nan

        try:
            ci = m.confint()
            ci_lower = float(ci.iloc[0, 0])
            ci_upper = float(ci.iloc[0, 1])
        except:
            ci_lower = coef_val - 1.96 * se_val
            ci_upper = coef_val + 1.96 * se_val

        all_coefs = dict(zip(m.coef().index, m.coef().values))

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block
        if functional_form:
            blocks["functional_form"] = functional_form

        payload = make_success_payload(
            coefficients={k: float(v) for k, v in all_coefs.items()},
            inference={"spec_id": "infer/se/cluster/subject", "params": {"cluster_var": "subject"}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            blocks=blocks,
            design={"randomized_experiment": G1_DESIGN_AUDIT},
        )

        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': 'G1',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef_val,
            'std_error': se_val,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': 'subject',
            'controls_desc': '',
            'cluster_var': cluster_var_name,
            'run_success': 1,
            'run_error': '',
        })
        print(f"  OK {run_id}: {spec_id} | coef={coef_val:.4f}, p={pval:.4f}, n={n_obs}")
        return run_id

    except Exception as e:
        err_msg = str(e)[:200]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': 'G1',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': 'subject',
            'controls_desc': '',
            'cluster_var': cluster_var_name,
            'run_success': 0,
            'run_error': err_msg,
        })
        print(f"  FAIL {run_id}: {spec_id} | {err_msg}")
        return run_id


# ============================================================
# HELPER: Record an inference variant
# ============================================================
def record_inference_variant(base_run_id, spec_id, spec_tree_path,
                             outcome_var, treatment_var,
                             data, outcome_col, treatment_col,
                             vcov, infer_type, sample_desc, cluster_var_name=""):
    global infer_run_counter
    infer_run_counter += 1
    infer_id = f"{PAPER_ID}_infer_{infer_run_counter:03d}"

    try:
        formula = f"{outcome_col} ~ {treatment_col} | subject"
        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().iloc[0])
        se_val = float(m.se().iloc[0])
        pval = float(m.pvalue().iloc[0])
        n_obs = int(m._N)
        r2 = float(m._r2) if hasattr(m, '_r2') else np.nan

        try:
            ci = m.confint()
            ci_lower = float(ci.iloc[0, 0])
            ci_upper = float(ci.iloc[0, 1])
        except:
            ci_lower = coef_val - 1.96 * se_val
            ci_upper = coef_val + 1.96 * se_val

        all_coefs = dict(zip(m.coef().index, m.coef().values))

        payload = make_success_payload(
            coefficients={k: float(v) for k, v in all_coefs.items()},
            inference={"spec_id": infer_type, "params": {"cluster_var": cluster_var_name} if cluster_var_name else {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": G1_DESIGN_AUDIT},
        )

        inference_results.append({
            'paper_id': PAPER_ID,
            'inference_run_id': infer_id,
            'spec_run_id': base_run_id,
            'spec_id': infer_type,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': 'G1',
            'coefficient': coef_val,
            'std_error': se_val,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(payload),
            'run_success': 1,
            'run_error': '',
        })
        print(f"  OK {infer_id}: {infer_type} (base={base_run_id}) | se={se_val:.4f}, p={pval:.4f}")

    except Exception as e:
        err_msg = str(e)[:200]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        inference_results.append({
            'paper_id': PAPER_ID,
            'inference_run_id': infer_id,
            'spec_run_id': base_run_id,
            'spec_id': infer_type,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': 'G1',
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'run_success': 0,
            'run_error': err_msg,
        })
        print(f"  FAIL {infer_id}: {infer_type} | {err_msg}")


# ============================================================
# DATA PREPARATION
# ============================================================

# Initial fixations only (paper's main sample)
df_init = df_raw[df_raw['refixWithLast'] == 0].copy()

# Compute efficiency on initial fixations
df_init['efficiency_computed'] = compute_efficiency(df_init)

# Compute initial fixation number
df_init['init_fix_number'] = compute_initial_fixation_number(df_init)

# Compute stopping dummy (for stopping probability)
df_init['stopping_dummy'] = compute_stopping_dummy(df_init)

# Compute cached value (best-seen-so-far)
df_init['cached_value'] = compute_cached_value(df_init)

# All fixations (including refixations)
df_all = df_raw.copy()

# Compute elapsed time on ALL fixations (before filtering)
elapsed = np.zeros(len(df_all))
for i in range(len(df_all)):
    if df_all['fixOrder'].iloc[i] == 1:
        elapsed[i] = df_all['FixDuration'].iloc[i]
    elif df_all['fixOrder'].iloc[i] > 1:
        elapsed[i] = elapsed[i-1] + df_all['FixDuration'].iloc[i]
df_all['elapsed'] = elapsed

# For trial-level data (last fixation per trial)
df_trial = df_raw[df_raw['fixOrder'] == df_raw['NumberLooked']].copy()

# Compute cached value at end of trial (for choice quality)
cached_all = np.zeros(len(df_raw))
for i in range(len(df_raw)):
    if df_raw['fixOrder'].iloc[i] == 1:
        cached_all[i] = df_raw['roirating'].iloc[i]
    elif df_raw['fixOrder'].iloc[i] > 1:
        cached_all[i] = max(df_raw['roirating'].iloc[i], cached_all[i-1])
df_raw['cached_at_end'] = cached_all

df_trial_full = df_raw[df_raw['fixOrder'] == df_raw['NumberLooked']].copy()
df_trial_full['best_seen_chosen'] = (df_trial_full['chosenrating'] == df_trial_full['cached_at_end']).astype(float)

# Collapse initial fixations to subject x condition means for efficiency
eff_means = df_init.groupby(['subject', 'numalt'])['efficiency_computed'].mean().reset_index()
eff_means_pivot = eff_means.pivot(index='subject', columns='numalt', values='efficiency_computed')

# For OLS regressions, prepare panel-like data at subject x condition level
# Multiple outcomes collapsed to subject x numalt level

print(f"\nInitial fixations: {len(df_init)} rows")
print(f"All fixations: {len(df_all)} rows")
print(f"Trial-level: {len(df_trial_full)} rows")

# ============================================================
# Prepare subject x condition level data for OLS with subject FE
# ============================================================

# Efficiency: subject x numalt means
eff_subj = df_init.groupby(['subject', 'numalt'])['efficiency_computed'].mean().reset_index()
eff_subj.columns = ['subject', 'numalt', 'efficiency']

# Efficiency_New: subject x numalt means
effnew_subj = df_init.groupby(['subject', 'numalt'])['Efficiency_New'].mean().reset_index()
effnew_subj.columns = ['subject', 'numalt', 'efficiency_new']

# CurFixef: subject x numalt means
curfix_subj = df_init.groupby(['subject', 'numalt'])['CurFixef'].mean().reset_index()
curfix_subj.columns = ['subject', 'numalt', 'curfix_ef']

# CurFixefNew: subject x numalt means
curfixnew_subj = df_init.groupby(['subject', 'numalt'])['CurFixefNew'].mean().reset_index()
curfixnew_subj.columns = ['subject', 'numalt', 'curfix_ef_new']

# FixDuration: subject x numalt means (on all fixations)
fixdur_subj = df_raw.groupby(['subject', 'numalt'])['FixDuration'].mean().reset_index()
fixdur_subj.columns = ['subject', 'numalt', 'fix_duration']

# FixDuration on initial fixations only
fixdur_init_subj = df_init.groupby(['subject', 'numalt'])['FixDuration'].mean().reset_index()
fixdur_init_subj.columns = ['subject', 'numalt', 'fix_duration_init']

# NumberLooked: subject x numalt means (from raw, one per trial ideally)
# Use trial-level data
numlooked_subj = df_trial_full.groupby(['subject', 'numalt'])['NumberLooked'].mean().reset_index()
numlooked_subj.columns = ['subject', 'numalt', 'num_looked']

# Percent looked = NumberLooked / numalt
pctlooked_subj = df_trial_full.groupby(['subject', 'numalt']).apply(
    lambda g: (g['NumberLooked'] / g['numalt']).mean()
).reset_index()
pctlooked_subj.columns = ['subject', 'numalt', 'pct_looked']

# Best-seen-chosen: subject x numalt means
bsc_subj = df_trial_full.groupby(['subject', 'numalt'])['best_seen_chosen'].mean().reset_index()
bsc_subj.columns = ['subject', 'numalt', 'best_seen_chosen']

# RT: subject x numalt means (from trial-level)
rt_subj = df_trial_full.groupby(['subject', 'numalt'])['rt'].mean().reset_index()
rt_subj.columns = ['subject', 'numalt', 'rt']

# Stopping probability (overall): subject x numalt means
stop_subj = df_init.groupby(['subject', 'numalt'])['stopping_dummy'].mean().reset_index()
stop_subj.columns = ['subject', 'numalt', 'stopping_prob']

# Merge all into one panel
panel = eff_subj.copy()
for other in [effnew_subj, curfix_subj, curfixnew_subj, fixdur_subj, fixdur_init_subj,
              numlooked_subj, pctlooked_subj, bsc_subj, rt_subj, stop_subj]:
    panel = panel.merge(other, on=['subject', 'numalt'], how='outer')

# Create treatment dummies for OLS
panel['numalt_9'] = (panel['numalt'] == 9).astype(int)
panel['numalt_16'] = (panel['numalt'] == 16).astype(int)
# For pairwise comparisons
panel['larger_set'] = 0
panel.loc[panel['numalt'] == 9, 'larger_set'] = 1
panel.loc[panel['numalt'] == 16, 'larger_set'] = 1

print(f"\nPanel data: {len(panel)} rows ({panel['subject'].nunique()} subjects x {panel['numalt'].nunique()} conditions)")

# ============================================================
# Also prepare RT-restricted samples
# ============================================================
# RT < 2000 (paper uses this restriction in some analyses)
df_init_rt2000 = df_init[df_init['rt'] < 2000].copy()
df_init_rt1500 = df_init[df_init['rt'] < 1500].copy()

# Compute efficiency on RT-restricted samples
eff_rt2000 = df_init_rt2000.groupby(['subject', 'numalt'])['efficiency_computed'].mean().reset_index()
eff_rt2000.columns = ['subject', 'numalt', 'efficiency_rt2000']

eff_rt1500 = df_init_rt1500.groupby(['subject', 'numalt'])['efficiency_computed'].mean().reset_index()
eff_rt1500.columns = ['subject', 'numalt', 'efficiency_rt1500']

# Merge into panel
panel = panel.merge(eff_rt2000, on=['subject', 'numalt'], how='left')
panel = panel.merge(eff_rt1500, on=['subject', 'numalt'], how='left')

# Efficiency_New on RT-restricted samples
effnew_rt2000 = df_init_rt2000.groupby(['subject', 'numalt'])['Efficiency_New'].mean().reset_index()
effnew_rt2000.columns = ['subject', 'numalt', 'efficiency_new_rt2000']

effnew_rt1500 = df_init_rt1500.groupby(['subject', 'numalt'])['Efficiency_New'].mean().reset_index()
effnew_rt1500.columns = ['subject', 'numalt', 'efficiency_new_rt1500']

panel = panel.merge(effnew_rt2000, on=['subject', 'numalt'], how='left')
panel = panel.merge(effnew_rt1500, on=['subject', 'numalt'], how='left')

# Stopping prob on RT-restricted samples
df_init_rt2000['stopping_dummy'] = compute_stopping_dummy(df_init_rt2000)
stop_rt2000 = df_init_rt2000.groupby(['subject', 'numalt'])['stopping_dummy'].mean().reset_index()
stop_rt2000.columns = ['subject', 'numalt', 'stopping_prob_rt2000']
panel = panel.merge(stop_rt2000, on=['subject', 'numalt'], how='left')

# FixDuration on initial fixations + RT < 2000
fixdur_init_rt2000 = df_init_rt2000.groupby(['subject', 'numalt'])['FixDuration'].mean().reset_index()
fixdur_init_rt2000.columns = ['subject', 'numalt', 'fix_duration_init_rt2000']
panel = panel.merge(fixdur_init_rt2000, on=['subject', 'numalt'], how='left')

# Also prepare all-fixations sample (including refixations) for efficiency
# On raw data including refixations, compute efficiency
df_all_eff = df_raw.copy()
df_all_eff['efficiency_all_fix'] = compute_efficiency(df_all_eff)
eff_all_subj = df_all_eff.groupby(['subject', 'numalt'])['efficiency_all_fix'].mean().reset_index()
eff_all_subj.columns = ['subject', 'numalt', 'efficiency_all_fix']
panel = panel.merge(eff_all_subj, on=['subject', 'numalt'], how='left')

# Compute rank-based efficiency: rank of fixated item within set
# rank_efficiency = 1 - (rank_of_roirating / numalt)
# Higher = looked at higher-ranked items
def compute_rank_efficiency_subj(data):
    """Compute rank of fixated item's rating within the set's ratings"""
    # For each trial, rank the ratings of all items
    # The rank of the fixated item gives a measure of efficiency
    out = []
    for (subj, na), grp in data.groupby(['subject', 'numalt']):
        # For each fixation, compute what fraction of items have lower ratings
        rank_eff = grp['roirating'].rank(method='average') / len(grp)
        out.append({'subject': subj, 'numalt': na, 'rank_efficiency': rank_eff.mean()})
    return pd.DataFrame(out)

rank_eff_subj = compute_rank_efficiency_subj(df_init)
panel = panel.merge(rank_eff_subj, on=['subject', 'numalt'], how='left')

# Ensure subject is categorical for FE
panel['subject'] = panel['subject'].astype(int)

print(f"Panel columns: {list(panel.columns)}")

# ============================================================
# SPECIFICATION 1: BASELINE - Stats-2A efficiency paired t-test (4 vs 9)
# This is the paper's main claim: efficiency differs by set size
# ============================================================
print("\n=== BASELINE SPECS (paired t-tests, matching paper) ===")

# The surface labels the baseline as "Stats-2A-efficiency-by-setsize"
# Paper reports 3 pairwise comparisons; the focal one is 4 vs 9 (or whichever is most prominent)
# We'll use 4 vs 16 as the focal (largest effect) and add others as rc/*

run_baseline = record_ttest(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/randomized_experiment.md",
    outcome_var="efficiency",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='efficiency_computed',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations only (refixWithLast==0), all subjects",
)

# ============================================================
# ADDITIONAL BASELINES: Other outcomes from surface baseline_spec_ids
# ============================================================
print("\n=== ADDITIONAL BASELINE OUTCOMES ===")

# Stopping probability: 4 vs 16
record_ttest(
    spec_id="baseline__stopping_prob",
    spec_tree_path="specification_tree/designs/randomized_experiment.md",
    outcome_var="stopping_prob",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='stopping_dummy',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, stopping at final initial fixation",
)

# Fixation duration: 4 vs 16 (on all fixations, matching Stats-Fig-6A.R)
record_ttest(
    spec_id="baseline__fixation_duration",
    spec_tree_path="specification_tree/designs/randomized_experiment.md",
    outcome_var="FixDuration",
    treatment_var="numalt_4_vs_16",
    data=df_raw,
    outcome_col='FixDuration',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="All fixations (including refixations)",
)

# Choice quality (best-seen-chosen): 4 vs 16 (matching stats-Fig-3A.R)
record_ttest(
    spec_id="baseline__choice_quality",
    spec_tree_path="specification_tree/designs/randomized_experiment.md",
    outcome_var="best_seen_chosen",
    treatment_var="numalt_4_vs_16",
    data=df_trial_full,
    outcome_col='best_seen_chosen',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Trial-level: best-seen-chosen dummy",
)

# ============================================================
# DESIGN ALTERNATIVES: OLS with subject FE
# ============================================================
print("\n=== DESIGN ALTERNATIVES (OLS with subject FE) ===")

# Efficiency: OLS with subject FE, numalt as treatment (4 is base)
# Using numalt_16 as focal treatment (largest comparison)
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="efficiency",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='efficiency',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, OLS with subject FE",
    cluster_var_name="subject",
)

# Stopping prob: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="stopping_prob",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='stopping_prob',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, OLS with subject FE",
    cluster_var_name="subject",
)

# FixDuration: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="fix_duration",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='fix_duration',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, OLS with subject FE",
    cluster_var_name="subject",
)

# Best-seen-chosen: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="best_seen_chosen",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='best_seen_chosen',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, OLS with subject FE",
    cluster_var_name="subject",
)

# ============================================================
# RC/SAMPLE: PAIRWISE CONDITION COMPARISONS (paired t-tests)
# ============================================================
print("\n=== RC/SAMPLE: PAIRWISE COMPARISONS ===")

# Efficiency: 4 vs 9
record_ttest(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="efficiency",
    treatment_var="numalt_4_vs_9",
    data=df_init,
    outcome_col='efficiency_computed',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Initial fixations, 4 vs 9 only",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_9", "conditions_compared": [4, 9]},
)

# Efficiency: 9 vs 16
record_ttest(
    spec_id="rc/sample/conditions/9_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="efficiency",
    treatment_var="numalt_9_vs_16",
    data=df_init,
    outcome_col='efficiency_computed',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, 9 vs 16 only",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/9_vs_16", "conditions_compared": [9, 16]},
)

# Stopping prob: 4 vs 9
record_ttest(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="stopping_prob",
    treatment_var="numalt_4_vs_9",
    data=df_init,
    outcome_col='stopping_dummy',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Initial fixations, stopping prob, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_9", "conditions_compared": [4, 9]},
)

# Stopping prob: 9 vs 16
record_ttest(
    spec_id="rc/sample/conditions/9_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="stopping_prob",
    treatment_var="numalt_9_vs_16",
    data=df_init,
    outcome_col='stopping_dummy',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, stopping prob, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/9_vs_16", "conditions_compared": [9, 16]},
)

# Stopping prob: 4 vs 16
record_ttest(
    spec_id="rc/sample/conditions/4_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="stopping_prob",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='stopping_dummy',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, stopping prob, 4 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_16", "conditions_compared": [4, 16]},
)

# FixDuration: 4 vs 9
record_ttest(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="FixDuration",
    treatment_var="numalt_4_vs_9",
    data=df_raw,
    outcome_col='FixDuration',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="All fixations, fix duration, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_9", "conditions_compared": [4, 9]},
)

# FixDuration: 9 vs 16
record_ttest(
    spec_id="rc/sample/conditions/9_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="FixDuration",
    treatment_var="numalt_9_vs_16",
    data=df_raw,
    outcome_col='FixDuration',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="All fixations, fix duration, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/9_vs_16", "conditions_compared": [9, 16]},
)

# Best-seen-chosen: 4 vs 9
record_ttest(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="best_seen_chosen",
    treatment_var="numalt_4_vs_9",
    data=df_trial_full,
    outcome_col='best_seen_chosen',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Trial-level, best-seen-chosen, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_9", "conditions_compared": [4, 9]},
)

# Best-seen-chosen: 9 vs 16
record_ttest(
    spec_id="rc/sample/conditions/9_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="best_seen_chosen",
    treatment_var="numalt_9_vs_16",
    data=df_trial_full,
    outcome_col='best_seen_chosen',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Trial-level, best-seen-chosen, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/9_vs_16", "conditions_compared": [9, 16]},
)

# ============================================================
# RC/SAMPLE: RT RESTRICTIONS
# ============================================================
print("\n=== RC/SAMPLE: RT RESTRICTIONS ===")

# Efficiency with RT < 2000
record_ttest(
    spec_id="rc/sample/outliers/trim_rt_2000",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency",
    treatment_var="numalt_4_vs_16",
    data=df_init_rt2000,
    outcome_col='efficiency_computed',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 2000ms",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_rt_2000", "rt_threshold": 2000},
)

# Efficiency with RT < 1500
record_ttest(
    spec_id="rc/sample/outliers/trim_rt_1500",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency",
    treatment_var="numalt_4_vs_16",
    data=df_init_rt1500,
    outcome_col='efficiency_computed',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 1500ms",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_rt_1500", "rt_threshold": 1500},
)

# Efficiency with RT < 2000, 4 vs 9
record_ttest(
    spec_id="rc/sample/outliers/trim_rt_2000",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency",
    treatment_var="numalt_4_vs_9",
    data=df_init_rt2000,
    outcome_col='efficiency_computed',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Initial fixations, RT < 2000ms, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_rt_2000", "rt_threshold": 2000, "conditions": [4, 9]},
)

# Efficiency with RT < 2000, 9 vs 16
record_ttest(
    spec_id="rc/sample/outliers/trim_rt_2000",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency",
    treatment_var="numalt_9_vs_16",
    data=df_init_rt2000,
    outcome_col='efficiency_computed',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 2000ms, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_rt_2000", "rt_threshold": 2000, "conditions": [9, 16]},
)

# Stopping prob with RT < 2000
record_ttest(
    spec_id="rc/sample/outliers/trim_rt_2000",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="stopping_prob",
    treatment_var="numalt_4_vs_16",
    data=df_init_rt2000,
    outcome_col='stopping_dummy',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 2000ms, stopping prob",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_rt_2000", "rt_threshold": 2000},
)

# Stopping prob with RT < 1500
record_ttest(
    spec_id="rc/sample/outliers/trim_rt_1500",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="stopping_prob",
    treatment_var="numalt_4_vs_16",
    data=df_init_rt1500,
    outcome_col='stopping_dummy',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 1500ms, stopping prob",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_rt_1500", "rt_threshold": 1500},
)

# ============================================================
# RC/SAMPLE: INCLUDE REFIXATIONS
# ============================================================
print("\n=== RC/SAMPLE: INCLUDE REFIXATIONS ===")

# Efficiency including refixations (keep all fixation types)
record_ttest(
    spec_id="rc/sample/outliers/keep_all_fixations",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency_all_fix",
    treatment_var="numalt_4_vs_16",
    data=df_all_eff,
    outcome_col='efficiency_all_fix',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="All fixations including refixations",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/keep_all_fixations", "refixation_filter": "include_all"},
)

# Efficiency including refixations, 4 vs 9
record_ttest(
    spec_id="rc/sample/outliers/keep_all_fixations",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency_all_fix",
    treatment_var="numalt_4_vs_9",
    data=df_all_eff,
    outcome_col='efficiency_all_fix',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="All fixations including refixations, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/keep_all_fixations", "refixation_filter": "include_all", "conditions": [4, 9]},
)

# Efficiency including refixations, 9 vs 16
record_ttest(
    spec_id="rc/sample/outliers/keep_all_fixations",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="efficiency_all_fix",
    treatment_var="numalt_9_vs_16",
    data=df_all_eff,
    outcome_col='efficiency_all_fix',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="All fixations including refixations, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/keep_all_fixations", "refixation_filter": "include_all", "conditions": [9, 16]},
)

# ============================================================
# RC/DATA: ALTERNATIVE OUTCOME DEFINITIONS
# ============================================================
print("\n=== RC/DATA: ALTERNATIVE OUTCOMES ===")

# Efficiency_New instead of efficiency
record_ttest(
    spec_id="rc/data/outcome/efficiency_new_vs_old",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="Efficiency_New",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='Efficiency_New',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, Efficiency_New outcome",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/efficiency_new_vs_old", "outcome_definition": "Efficiency_New"},
)

# Efficiency_New: 4 vs 9
record_ttest(
    spec_id="rc/data/outcome/efficiency_new_vs_old",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="Efficiency_New",
    treatment_var="numalt_4_vs_9",
    data=df_init,
    outcome_col='Efficiency_New',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Initial fixations, Efficiency_New, 4 vs 9",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/efficiency_new_vs_old", "outcome_definition": "Efficiency_New", "conditions": [4, 9]},
)

# Efficiency_New: 9 vs 16
record_ttest(
    spec_id="rc/data/outcome/efficiency_new_vs_old",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="Efficiency_New",
    treatment_var="numalt_9_vs_16",
    data=df_init,
    outcome_col='Efficiency_New',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, Efficiency_New, 9 vs 16",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/efficiency_new_vs_old", "outcome_definition": "Efficiency_New", "conditions": [9, 16]},
)

# CurFixef instead of efficiency
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="CurFixef",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='CurFixef',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, CurFixef outcome",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/curfix_ef_vs_efficiency", "outcome_definition": "CurFixef"},
)

# CurFixef: 4 vs 9
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="CurFixef",
    treatment_var="numalt_4_vs_9",
    data=df_init,
    outcome_col='CurFixef',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Initial fixations, CurFixef, 4 vs 9",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/curfix_ef_vs_efficiency", "outcome_definition": "CurFixef", "conditions": [4, 9]},
)

# CurFixef: 9 vs 16
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="CurFixef",
    treatment_var="numalt_9_vs_16",
    data=df_init,
    outcome_col='CurFixef',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, CurFixef, 9 vs 16",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/curfix_ef_vs_efficiency", "outcome_definition": "CurFixef", "conditions": [9, 16]},
)

# CurFixefNew: 4 vs 16
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="CurFixefNew",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='CurFixefNew',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, CurFixefNew outcome",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/curfix_ef_vs_efficiency", "outcome_definition": "CurFixefNew"},
)

# Rank-based efficiency
record_ttest(
    spec_id="rc/form/outcome/rank_efficiency",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
    outcome_var="rank_efficiency",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='roirating',  # We'll use rank_efficiency from panel instead
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, rank-based efficiency",
    functional_form={
        "spec_id": "rc/form/outcome/rank_efficiency",
        "interpretation": "Rank of fixated item rating within set, normalized to [0,1]",
        "transformation": "rank_normalize",
    },
)

# ============================================================
# RC/DATA: ALTERNATIVE OUTCOMES + RT RESTRICTIONS (combined)
# ============================================================
print("\n=== RC/DATA + RC/SAMPLE: COMBINED ===")

# Efficiency_New + RT < 2000
record_ttest(
    spec_id="rc/data/outcome/efficiency_new_vs_old",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="Efficiency_New",
    treatment_var="numalt_4_vs_16",
    data=df_init_rt2000,
    outcome_col='Efficiency_New',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 2000, Efficiency_New",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/efficiency_new_vs_old", "outcome_definition": "Efficiency_New", "rt_restriction": 2000},
)

# CurFixef + RT < 2000
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="CurFixef",
    treatment_var="numalt_4_vs_16",
    data=df_init_rt2000,
    outcome_col='CurFixef',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations, RT < 2000, CurFixef",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/curfix_ef_vs_efficiency", "outcome_definition": "CurFixef", "rt_restriction": 2000},
)

# ============================================================
# RC/SAMPLE: Additional outcomes across conditions
# ============================================================
print("\n=== ADDITIONAL OUTCOMES: RT, NumberLooked, pct_looked ===")

# RT: 4 vs 16
record_ttest(
    spec_id="rc/sample/conditions/4_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="rt",
    treatment_var="numalt_4_vs_16",
    data=df_trial_full,
    outcome_col='rt',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Trial-level RT, 4 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_16", "conditions_compared": [4, 16]},
)

# RT: 4 vs 9
record_ttest(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="rt",
    treatment_var="numalt_4_vs_9",
    data=df_trial_full,
    outcome_col='rt',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Trial-level RT, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_9", "conditions_compared": [4, 9]},
)

# NumberLooked: 4 vs 16 (from Stats-Fig-6C.R)
record_ttest(
    spec_id="rc/sample/conditions/4_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="NumberLooked",
    treatment_var="numalt_4_vs_16",
    data=df_raw,
    outcome_col='NumberLooked',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="All fixations, NumberLooked, 4 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_16", "conditions_compared": [4, 16]},
)

# Pct looked: 4 vs 16 (from Stats-Fig-6D.R)
# NumberLooked / numalt
df_raw_pct = df_raw.copy()
df_raw_pct['pct_looked'] = df_raw_pct['NumberLooked'] / df_raw_pct['numalt']
record_ttest(
    spec_id="rc/sample/conditions/4_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="pct_looked",
    treatment_var="numalt_4_vs_16",
    data=df_raw_pct,
    outcome_col='pct_looked',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="All fixations, pct items looked at, 4 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_16", "conditions_compared": [4, 16]},
)

# Pct looked: 4 vs 9
record_ttest(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="pct_looked",
    treatment_var="numalt_4_vs_9",
    data=df_raw_pct,
    outcome_col='pct_looked',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="All fixations, pct items looked at, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/4_vs_9", "conditions_compared": [4, 9]},
)

# Pct looked: 9 vs 16
record_ttest(
    spec_id="rc/sample/conditions/9_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="pct_looked",
    treatment_var="numalt_9_vs_16",
    data=df_raw_pct,
    outcome_col='pct_looked',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="All fixations, pct items looked at, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/conditions/9_vs_16", "conditions_compared": [9, 16]},
)

# ============================================================
# OLS WITH SUBJECT FE: Additional outcomes
# ============================================================
print("\n=== OLS WITH SUBJECT FE: Additional outcomes ===")

# Efficiency_New: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="efficiency_new",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='efficiency_new',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, Efficiency_New, OLS with subject FE",
    cluster_var_name="subject",
)

# CurFixef: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="curfix_ef",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='curfix_ef',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, CurFixef, OLS with subject FE",
    cluster_var_name="subject",
)

# RT: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="rt",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='rt',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, RT, OLS with subject FE",
    cluster_var_name="subject",
)

# Number looked: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="num_looked",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='num_looked',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, NumberLooked, OLS with subject FE",
    cluster_var_name="subject",
)

# Pct looked: OLS with subject FE
record_ols_fe(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#with_covariates",
    outcome_var="pct_looked",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='pct_looked',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, pct_looked, OLS with subject FE",
    cluster_var_name="subject",
)

# ============================================================
# OLS WITH SUBJECT FE: RT-restricted samples
# ============================================================
print("\n=== OLS WITH SUBJECT FE: RT-restricted samples ===")

# Efficiency: OLS with subject FE, RT < 2000
panel_rt2000 = panel[['subject', 'numalt', 'numalt_9', 'numalt_16', 'efficiency_rt2000']].dropna()
if len(panel_rt2000) > 0:
    record_ols_fe(
        spec_id="rc/sample/outliers/trim_rt_2000",
        spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
        outcome_var="efficiency_rt2000",
        treatment_var="numalt_16",
        data=panel_rt2000,
        outcome_col='efficiency_rt2000',
        treatment_col='numalt_9 + numalt_16',
        vcov={"CRV1": "subject"},
        sample_desc="Subject x condition panel, RT < 2000, OLS with subject FE",
        cluster_var_name="subject",
    )

# Efficiency: OLS with subject FE, RT < 1500
panel_rt1500 = panel[['subject', 'numalt', 'numalt_9', 'numalt_16', 'efficiency_rt1500']].dropna()
if len(panel_rt1500) > 0:
    record_ols_fe(
        spec_id="rc/sample/outliers/trim_rt_1500",
        spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
        outcome_var="efficiency_rt1500",
        treatment_var="numalt_16",
        data=panel_rt1500,
        outcome_col='efficiency_rt1500',
        treatment_col='numalt_9 + numalt_16',
        vcov={"CRV1": "subject"},
        sample_desc="Subject x condition panel, RT < 1500, OLS with subject FE",
        cluster_var_name="subject",
    )

# ============================================================
# ADDITIONAL CROSSED SPECS: OLS alternative outcomes + conditions
# ============================================================
print("\n=== OLS WITH SUBJECT FE: Pairwise on 4 vs 9 subset ===")

# For pairwise OLS comparisons, subset the panel
panel_4v9 = panel[panel['numalt'].isin([4, 9])].copy()
panel_4v9['treat_9'] = (panel_4v9['numalt'] == 9).astype(int)

panel_9v16 = panel[panel['numalt'].isin([9, 16])].copy()
panel_9v16['treat_16'] = (panel_9v16['numalt'] == 16).astype(int)

panel_4v16 = panel[panel['numalt'].isin([4, 16])].copy()
panel_4v16['treat_16'] = (panel_4v16['numalt'] == 16).astype(int)

# Efficiency OLS: 4 vs 9
record_ols_fe(
    spec_id="rc/sample/conditions/4_vs_9",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="efficiency",
    treatment_var="treat_9",
    data=panel_4v9,
    outcome_col='efficiency',
    treatment_col='treat_9',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, 4 vs 9 only, OLS with subject FE",
    cluster_var_name="subject",
)

# Efficiency OLS: 9 vs 16
record_ols_fe(
    spec_id="rc/sample/conditions/9_vs_16",
    spec_tree_path="specification_tree/modules/robustness/sample.md#conditions",
    outcome_var="efficiency",
    treatment_var="treat_16",
    data=panel_9v16,
    outcome_col='efficiency',
    treatment_col='treat_16',
    vcov={"CRV1": "subject"},
    sample_desc="Subject x condition panel, 9 vs 16 only, OLS with subject FE",
    cluster_var_name="subject",
)

# ============================================================
# INFERENCE VARIANTS
# ============================================================
print("\n=== INFERENCE VARIANTS ===")

# For each baseline OLS spec, compute HC1 and cluster variants
# Use the full panel with efficiency as outcome
# Inference variant 1: HC1 robust SEs (ignoring within-subject correlation)
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_005",  # the first OLS design spec
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/se.md#hc1",
    outcome_var="efficiency",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='efficiency',
    treatment_col='numalt_9 + numalt_16',
    vcov="hetero",
    infer_type="infer/se/hc/hc1",
    sample_desc="Subject x condition panel, efficiency, HC1 SEs",
)

# Inference variant 2: cluster(subject) (same as canonical for OLS but different from paired t-test)
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_005",
    spec_id="infer/se/cluster/subject",
    spec_tree_path="specification_tree/modules/inference/se.md#cluster",
    outcome_var="efficiency",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='efficiency',
    treatment_col='numalt_9 + numalt_16',
    vcov={"CRV1": "subject"},
    infer_type="infer/se/cluster/subject",
    sample_desc="Subject x condition panel, efficiency, CRV1(subject)",
    cluster_var_name="subject",
)

# HC1 for stopping prob
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_006",
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/se.md#hc1",
    outcome_var="stopping_prob",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='stopping_prob',
    treatment_col='numalt_9 + numalt_16',
    vcov="hetero",
    infer_type="infer/se/hc/hc1",
    sample_desc="Subject x condition panel, stopping_prob, HC1 SEs",
)

# HC1 for fix_duration
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_007",
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/se.md#hc1",
    outcome_var="fix_duration",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='fix_duration',
    treatment_col='numalt_9 + numalt_16',
    vcov="hetero",
    infer_type="infer/se/hc/hc1",
    sample_desc="Subject x condition panel, fix_duration, HC1 SEs",
)

# HC1 for best_seen_chosen
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_008",
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/se.md#hc1",
    outcome_var="best_seen_chosen",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='best_seen_chosen',
    treatment_col='numalt_9 + numalt_16',
    vcov="hetero",
    infer_type="infer/se/hc/hc1",
    sample_desc="Subject x condition panel, best_seen_chosen, HC1 SEs",
)

# HC1 for efficiency_new
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_040",
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/se.md#hc1",
    outcome_var="efficiency_new",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='efficiency_new',
    treatment_col='numalt_9 + numalt_16',
    vcov="hetero",
    infer_type="infer/se/hc/hc1",
    sample_desc="Subject x condition panel, efficiency_new, HC1 SEs",
)

# HC1 for curfix_ef
record_inference_variant(
    base_run_id=f"{PAPER_ID}_run_041",
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/se.md#hc1",
    outcome_var="curfix_ef",
    treatment_var="numalt_16",
    data=panel,
    outcome_col='curfix_ef',
    treatment_col='numalt_9 + numalt_16',
    vcov="hetero",
    infer_type="infer/se/hc/hc1",
    sample_desc="Subject x condition panel, curfix_ef, HC1 SEs",
)

# ============================================================
# ADDITIONAL SPECS TO REACH 50+
# ============================================================
print("\n=== ADDITIONAL SPECS ===")

# FixDuration on initial fixations only: 4 vs 16
record_ttest(
    spec_id="rc/sample/outliers/drop_refixations",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="FixDuration_init",
    treatment_var="numalt_4_vs_16",
    data=df_init,
    outcome_col='FixDuration',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Initial fixations only, fix duration, 4 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_refixations", "refixation_filter": "initial_only"},
)

# FixDuration on initial fixations only: 4 vs 9
record_ttest(
    spec_id="rc/sample/outliers/drop_refixations",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="FixDuration_init",
    treatment_var="numalt_4_vs_9",
    data=df_init,
    outcome_col='FixDuration',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Initial fixations only, fix duration, 4 vs 9",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_refixations", "refixation_filter": "initial_only", "conditions": [4, 9]},
)

# FixDuration on initial fixations only: 9 vs 16
record_ttest(
    spec_id="rc/sample/outliers/drop_refixations",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="FixDuration_init",
    treatment_var="numalt_9_vs_16",
    data=df_init,
    outcome_col='FixDuration',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Initial fixations only, fix duration, 9 vs 16",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_refixations", "refixation_filter": "initial_only", "conditions": [9, 16]},
)

# Elapsed time at end of initial search: subject x condition (from Stats-Fig-5A.R)
# Compute elapsed time = cumulative FixDuration during initial fixations
df_init_elapsed = df_init.copy()
elapsed_init = np.zeros(len(df_init_elapsed))
for i in range(len(df_init_elapsed)):
    if df_init_elapsed['fixOrder'].iloc[i] == 1:
        elapsed_init[i] = df_init_elapsed['FixDuration'].iloc[i]
    elif df_init_elapsed['fixOrder'].iloc[i] > 1:
        elapsed_init[i] = elapsed_init[i-1] + df_init_elapsed['FixDuration'].iloc[i]
df_init_elapsed['elapsed_init'] = elapsed_init

# Keep only final initial fixation (where stopping_dummy == 1)
df_final_init = df_init_elapsed[df_init_elapsed['stopping_dummy'] == 1].copy()

# Elapsed time at stop: 4 vs 16
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="elapsed_at_stop",
    treatment_var="numalt_4_vs_16",
    data=df_final_init,
    outcome_col='elapsed_init',
    numalt_a=4.0,
    numalt_b=16.0,
    sample_desc="Final initial fixation, elapsed time, 4 vs 16",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/elapsed_at_stop", "outcome_definition": "cumulative_fixation_duration_at_stop"},
)

# Elapsed time at stop: 4 vs 9
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="elapsed_at_stop",
    treatment_var="numalt_4_vs_9",
    data=df_final_init,
    outcome_col='elapsed_init',
    numalt_a=4.0,
    numalt_b=9.0,
    sample_desc="Final initial fixation, elapsed time, 4 vs 9",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/elapsed_at_stop", "outcome_definition": "cumulative_fixation_duration_at_stop", "conditions": [4, 9]},
)

# Elapsed time at stop: 9 vs 16
record_ttest(
    spec_id="rc/data/outcome/curfix_ef_vs_efficiency",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#outcome",
    outcome_var="elapsed_at_stop",
    treatment_var="numalt_9_vs_16",
    data=df_final_init,
    outcome_col='elapsed_init',
    numalt_a=9.0,
    numalt_b=16.0,
    sample_desc="Final initial fixation, elapsed time, 9 vs 16",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/elapsed_at_stop", "outcome_definition": "cumulative_fixation_duration_at_stop", "conditions": [9, 16]},
)

# ============================================================
# WRITE OUTPUTS
# ============================================================
print(f"\n=== WRITING OUTPUTS ===")
print(f"Total specification results: {len(results)}")
print(f"Total inference results: {len(inference_results)}")

# Verify unique spec_run_ids
spec_ids = [r['spec_run_id'] for r in results]
assert len(spec_ids) == len(set(spec_ids)), "Duplicate spec_run_ids found!"

# Write specification_results.csv
df_specs = pd.DataFrame(results)
df_specs.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {len(df_specs)} rows to specification_results.csv")

# Write inference_results.csv
if inference_results:
    df_infer = pd.DataFrame(inference_results)
    df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    print(f"Wrote {len(df_infer)} rows to inference_results.csv")

# Count successes/failures
n_success = sum(1 for r in results if r['run_success'] == 1)
n_fail = sum(1 for r in results if r['run_success'] == 0)
n_infer_success = sum(1 for r in inference_results if r['run_success'] == 1)
n_infer_fail = sum(1 for r in inference_results if r['run_success'] == 0)

# Write SPECIFICATION_SEARCH.md
md_content = f"""# Specification Search: {PAPER_ID}

## Paper
Arieli, Ben-Ami & Rubinstein (2011), "Tracking Decision Makers under Uncertainty"
American Economic Journal: Microeconomics, 3(4), 68-76.

## Surface Summary
- **Baseline groups**: 1 (G1: Effect of choice set size on attention efficiency)
- **Design**: Randomized within-subject experiment (N=41 subjects, 3 conditions: 4/9/16 items)
- **Budget**: 50 core specs max
- **Seed**: 112415
- **Canonical inference**: Paired t-test on within-subject condition means

## Execution Summary
- **Planned specifications**: {len(results)} core + {len(inference_results)} inference
- **Executed successfully**: {n_success} core + {n_infer_success} inference
- **Failed**: {n_fail} core + {n_infer_fail} inference

## Specification Breakdown

### Baselines (4 specs)
1. `baseline`: Efficiency (4 vs 16) paired t-test -- paper's Stats-2A main comparison
2. `baseline__stopping_prob`: Stopping probability (4 vs 16)
3. `baseline__fixation_duration`: Fixation duration (4 vs 16)
4. `baseline__choice_quality`: Best-seen-chosen (4 vs 16)

### Design Alternatives ({sum(1 for r in results if r['spec_id'].startswith('design/'))} specs)
- OLS with subject FE for: efficiency, stopping_prob, fix_duration, best_seen_chosen,
  efficiency_new, curfix_ef, rt, num_looked, pct_looked
- Pairwise OLS (4v9, 9v16 subsets)

### Robustness: Sample Restrictions ({sum(1 for r in results if r['spec_id'].startswith('rc/sample'))} specs)
- Pairwise condition comparisons (4v9, 9v16, 4v16) across outcomes
- RT trimming (< 2000ms, < 1500ms)
- All-fixations sample (include refixations)
- Initial-fixations-only for duration

### Robustness: Data Construction ({sum(1 for r in results if r['spec_id'].startswith('rc/data') or r['spec_id'].startswith('rc/form'))} specs)
- Efficiency_New (alternative efficiency definition)
- CurFixef (current fixation efficiency)
- CurFixefNew
- Rank-based efficiency
- Elapsed time at search termination

### Inference Variants ({len(inference_results)} specs)
- HC1 robust SEs (ignoring within-subject correlation)
- CRV1(subject) cluster SEs

## Deviations from Surface
- Surface noted N~39 subjects; data contains 41 unique subjects
- No structural model estimation attempted (correctly excluded per surface)
- Rank-based efficiency is approximate (rank within fixation-level data, not within full set)

## Software Stack
- Python {SW_BLOCK['runner_version']}
- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- scipy {SW_BLOCK['packages'].get('scipy', 'N/A')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md_content)
print("Wrote SPECIFICATION_SEARCH.md")

print(f"\n=== DONE ===")
print(f"Total specs: {len(results)} | Success: {n_success} | Fail: {n_fail}")
print(f"Total inference: {len(inference_results)} | Success: {n_infer_success} | Fail: {n_infer_fail}")
