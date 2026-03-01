#!/usr/bin/env python3
"""
Specification Search Script for 149481-V1:
"Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence"
Samek & Longfield, AER

Surface-driven execution:
  - G1: Experiment 1 (Public TV Stations) — ITT of thank-you call on donation outcomes
  - G2: Experiment 2 (National Non-Profit) — same treatment, different population
  - Baseline: diff-in-means (Table 2) + OLS with strata FE and controls (Table A1)
  - 50+ specifications across controls LOO, control sets, sample trimming,
    functional form transforms, FE swaps, and inference variants

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import traceback
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "149481-V1"
DATA_DIR = "data/downloads/extracted/149481-V1"
OUTPUT_DIR = DATA_DIR
RAW_DATA_DIR = f"{DATA_DIR}/thank_you_replication/data"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg1 = surface_obj["baseline_groups"][0]
bg2 = surface_obj["baseline_groups"][1]
design_audit_g1 = bg1["design_audit"]
design_audit_g2 = bg2["design_audit"]
design_block_g1 = {"randomized_experiment": design_audit_g1}
design_block_g2 = {"randomized_experiment": design_audit_g2}
canonical_inference_g1 = bg1["inference_plan"]["canonical"]
canonical_inference_g2 = bg2["inference_plan"]["canonical"]

# ============================================================
# Accumulators and counters
# ============================================================

results = []
inference_results = []
spec_counter = 0
infer_counter = 0


def make_run_id():
    global spec_counter
    spec_counter += 1
    return f"{PAPER_ID}_run_{spec_counter:03d}"


def make_infer_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}_infer_{infer_counter:03d}"


def add_result(spec_id, run_id, spec_tree_path, baseline_group_id,
               outcome_var, treatment_var, coefficient, std_error,
               p_value, ci_lower, ci_upper, n_obs, r_squared,
               coefficient_vector_json, sample_desc, fixed_effects,
               controls_desc, cluster_var, run_success=1, run_error=""):
    results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coefficient,
        "std_error": std_error,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "r_squared": r_squared,
        "coefficient_vector_json": json.dumps(coefficient_vector_json),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": run_success,
        "run_error": run_error,
    })


def add_inference_result(spec_run_id, spec_id, spec_tree_path,
                         baseline_group_id, coefficient, std_error,
                         p_value, ci_lower, ci_upper, n_obs, r_squared,
                         coefficient_vector_json, run_success=1, run_error=""):
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": make_infer_id(),
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "coefficient": coefficient,
        "std_error": std_error,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "r_squared": r_squared,
        "coefficient_vector_json": json.dumps(coefficient_vector_json),
        "run_success": run_success,
        "run_error": run_error,
    })


def make_payload(coefficients, inference_spec, design_blk, extra_blocks=None):
    payload = make_success_payload(
        coefficients=coefficients,
        inference=inference_spec,
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design=design_blk,
        blocks=extra_blocks,
    )
    return payload


def make_fail_payload(error_msg, stage="estimation"):
    return make_failure_payload(
        error=error_msg,
        error_details={"stage": stage, "exception_type": "RuntimeError",
                       "exception_message": error_msg},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
    )


# ============================================================
# Helper: diff-in-means test (ranksum, chi2, ttest)
# ============================================================

def diff_in_means_test(data, outcome_var, treat_var, test_type, condition=None):
    """Run a nonparametric or t-test difference-in-means comparison.

    Returns: (coef, se, pvalue, ci_lower, ci_upper, nobs)
    """
    d = data.copy()
    if condition is not None:
        d = d[condition(d)].copy()
    d = d.dropna(subset=[outcome_var, treat_var])
    treat = d[d[treat_var] == 1][outcome_var]
    ctrl = d[d[treat_var] == 0][outcome_var]
    nobs = len(treat) + len(ctrl)
    coef = treat.mean() - ctrl.mean()

    # Pooled SE for difference in means
    se_treat = treat.std(ddof=1) / np.sqrt(len(treat)) if len(treat) > 1 else 0
    se_ctrl = ctrl.std(ddof=1) / np.sqrt(len(ctrl)) if len(ctrl) > 1 else 0
    se = np.sqrt(se_treat**2 + se_ctrl**2)
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    if test_type == "ranksum":
        stat, pval = stats.mannwhitneyu(treat, ctrl, alternative='two-sided')
    elif test_type == "chi2":
        ct = pd.crosstab(d[treat_var], d[outcome_var])
        stat, pval, dof, expected = stats.chi2_contingency(ct)
    elif test_type == "ttest":
        stat, pval = stats.ttest_ind(treat, ctrl, equal_var=False)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return coef, se, pval, ci_lower, ci_upper, nobs


# ============================================================
# Helper: OLS spec via pyfixest
# ============================================================

def run_ols_spec(data, formula, vcov, focal_var='treat'):
    """Run OLS via pyfixest. Returns (coef, se, pval, ci_lo, ci_hi, nobs, r2, coeffs, model)."""
    m = pf.feols(formula, data=data, vcov=vcov)
    coef = float(m.coef().get(focal_var, np.nan))
    se = float(m.se().get(focal_var, np.nan))
    pval = float(m.pvalue().get(focal_var, np.nan))
    try:
        ci = m.confint()
        ci_lo = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
        ci_hi = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
    except Exception:
        ci_lo, ci_hi = np.nan, np.nan
    nobs = int(m._N)
    try:
        r2 = float(m._r2)
    except Exception:
        r2 = np.nan
    coeffs = {k: float(v) for k, v in m.coef().items()}
    return coef, se, pval, ci_lo, ci_hi, nobs, r2, coeffs, m


# ============================================================
# DATA PREPARATION - EXPERIMENT 1 (replicating Stata step1 + step2)
# ============================================================
print("Loading and preparing data for Experiment 1...")

# --- Transactions ---
trans = pd.read_stata(f"{RAW_DATA_DIR}/transactions.dta")
trans['gift_date_str'] = trans['gift_date'].astype(str).str.split(' ').str[0]
trans['date'] = pd.to_datetime(trans['gift_date_str'], format='%Y-%m-%d', errors='coerce')
trans.sort_values(['account_id', 'date'], inplace=True)

trans['gift_kind1'] = 4
trans.loc[trans['gift_kind'] == 'IN', 'gift_kind1'] = 0
trans.loc[trans['gift_kind'] == 'OP', 'gift_kind1'] = 1
trans.loc[trans['source_category'] == 'SG', 'gift_kind1'] = 2
trans.loc[trans['gift_kind'] == 'NULL', 'gift_kind1'] = 3
trans['pledge'] = (trans['source_category'] == 'P').astype(int)
trans['big_donor_temp'] = (trans['payment_amount'] >= 10000).astype(int)
trans['big_donor'] = trans.groupby('account_id')['big_donor_temp'].transform('sum')

# Collapse: one payment per day per account_id x date x station x gift_date
trans_temp = trans.groupby(['account_id', 'date', 'station', 'gift_date_str']).agg({
    'payment_amount': 'sum',
    'gift_kind1': 'mean',
    'pledge': 'mean',
    'big_donor': 'first'
}).reset_index()
trans_temp.rename(columns={'gift_kind1': 'gift_kind'}, inplace=True)
trans_temp['account_id'] = trans_temp['account_id'].str.upper()

# --- Callog ---
callog = pd.read_stata(f"{RAW_DATA_DIR}/callog.dta")
callog['account_id'] = callog['account_id'].str.upper()
callog['station'] = callog['station'].str.upper()
callog_dup = callog['account_id'].duplicated(keep=False)
callog = callog[~callog_dup].copy()
callog['treat'] = np.where(callog['segment'] == 'Control', 0,
                  np.where(callog['segment'] == 'Active Calls', 1, np.nan))
callog['exec_date_int'] = callog['exec_date'].astype(int)
# Drop exec_dates with insufficient follow-up (17Q4, 18Q1)
callog = callog[~callog['exec_date_int'].isin([1704, 1707, 1710, 1801])].copy()
callog['exec_year'] = 2000 + callog['exec_date_int'] // 100
callog['exec_month'] = callog['exec_date_int'] % 100
callog['edate'] = pd.to_datetime(dict(year=callog['exec_year'],
                                       month=callog['exec_month'], day=1))
callog_temp = callog[['account_id', 'treat', 'exec_date', 'edate',
                       'response_type', 'station']].copy()
callog_temp.rename(columns={'station': 'station_cal'}, inplace=True)

# --- Demographics ---
demo = pd.read_stata(f"{RAW_DATA_DIR}/demographics.dta")
demo['account_id'] = demo['account_id'].str.upper()
demo = demo[demo['account_id'] != 'NULL'].copy()
demo.drop_duplicates(inplace=True)
demo['female'] = np.nan
demo.loc[demo['gender'] == 'M', 'female'] = 0
demo.loc[demo['gender'] == 'F', 'female'] = 1
demo['agecode_num'] = pd.to_numeric(demo['agecode'], errors='coerce')
demo['age_at_gift'] = ''
demo.loc[demo['agecode_num'].isin([1, 2, 3]), 'age_at_gift'] = '18 to 44 years old'
demo.loc[demo['agecode_num'].isin([4, 5]), 'age_at_gift'] = '45 to 64 years old'
demo.loc[demo['agecode_num'].isin([6, 7]), 'age_at_gift'] = '65+ years old'
demo['income'] = ''
demo.loc[demo['hhincomecode'].isin(['A', 'B', 'C']), 'income'] = 'Below $34,000'
demo.loc[demo['hhincomecode'].isin(['D', 'E', 'F']), 'income'] = '$35,000-$99,999'
demo.loc[demo['hhincomecode'].isin(['G', 'H', 'I']), 'income'] = '$100,000-$174,999'
demo.loc[demo['hhincomecode'].isin(['J', 'K', 'L']), 'income'] = '$175,000+'
demo.loc[demo['hhincomecode'] == 'U', 'income'] = ''
demo['lor_num'] = pd.to_numeric(demo['lor'], errors='coerce')
demo['lorgroup'] = ''
demo.loc[demo['lor_num'].isin([2, 3, 4]), 'lorgroup'] = '5 years or less'
demo.loc[demo['lor_num'].isin([5, 6]), 'lorgroup'] = '6-15 years'
demo.loc[demo['lor_num'].isin([7, 8]), 'lorgroup'] = '16+ years'
demo['age_income_missing'] = demo['agecode_num'].isna().astype(int)
demo['female_missing'] = demo['female'].isna().astype(int)
demo_temp = demo[['account_id', 'female', 'age_at_gift', 'income', 'lorgroup',
                   'age_income_missing', 'female_missing']].copy()

# --- Treatments ---
treatments = pd.read_stata(f"{RAW_DATA_DIR}/treatments.dta")
treatments['account_id'] = treatments['account_id'].str.upper()
treatments.drop_duplicates(inplace=True)

# --- Merge (transaction-level) ---
merged = trans_temp.merge(demo_temp, on='account_id', how='left')
merged = merged.merge(callog_temp, on='account_id', how='inner',
                      suffixes=('', '_cal2'))
merged = merged.merge(treatments, on='account_id', how='left')
merged = merged[merged['big_donor'] < 1].copy()

# Timeline (based on date relative to edate)
diff_days = (merged['date'] - merged['edate']).dt.total_seconds() / 86400.0
merged['timeline'] = ''
merged.loc[diff_days < -365.25, 'timeline'] = 'a_existing_donor'
merged.loc[(diff_days > -365.25) & (diff_days <= 0), 'timeline'] = 'b_year before'
merged.loc[(diff_days >= 0) & (diff_days < 365.25), 'timeline'] = 'c_year after'
merged.loc[(diff_days >= 365.25) & (diff_days < 365.25*2), 'timeline'] = 'd_two years after'
merged.loc[(diff_days >= 365.25*2) & (diff_days < 365.25*3), 'timeline'] = 'e_three years after'
merged.loc[(diff_days >= 365.25*3) & (diff_days < 365.25*4), 'timeline'] = 'f_four years after'
merged.loc[(diff_days >= 365.25*4) & (diff_days < 365.25*5), 'timeline'] = 'g_five years after'
merged.loc[diff_days >= 365.25*5, 'timeline'] = 'h_six+ years after'

# Drop sustaining donors
merged['sustaining'] = ((merged['gift_kind'] == 2) &
                        (merged['timeline'] == 'b_year before')).astype(int)
merged['sustaining_max'] = merged.groupby('account_id')['sustaining'].transform('max')
merged = merged[merged['sustaining_max'] < 1].copy()

# Fill NaN in groupby columns to prevent pandas dropping NaN rows
merged['script'] = merged['script'].fillna(-999)
merged['female'] = merged['female'].fillna(-999)

# Collapse: sum payment_amount, count transactions, mean pledge
# by timeline + all person-level attributes
collapse_by = ['timeline', 'account_id', 'station_cal', 'exec_date',
               'edate', 'treat', 'script', 'response_type',
               'female', 'age_at_gift', 'lorgroup', 'income',
               'female_missing', 'age_income_missing']
agg_df = merged.groupby(collapse_by).agg(
    payment_amount=('payment_amount', 'sum'),
    var1=('payment_amount', 'count'),
    pledge=('pledge', 'mean'),
).reset_index()

# Encode timeline to integer (matching Stata's encode alphabetical ordering)
timeline_map = {
    'a_existing_donor': 1, 'b_year before': 2, 'c_year after': 3,
    'd_two years after': 4, 'e_three years after': 5,
    'f_four years after': 6, 'g_five years after': 7,
    'h_six+ years after': 8
}
agg_df['t'] = agg_df['timeline'].map(timeline_map).fillna(0).astype(int)

# Reshape wide: Stata does reshape wide payment_amount var1 pledge, i(account_id) j(t)
# account_id is the only row identifier; other columns are invariant within account_id
# (since callog is deduplicated to one row per account_id)
# First, extract the person-level attributes (take first per account_id)
person_cols = ['account_id', 'station_cal', 'exec_date', 'edate', 'treat',
               'script', 'response_type', 'female', 'age_at_gift', 'lorgroup',
               'income', 'female_missing', 'age_income_missing']
person_attrs = agg_df.drop_duplicates(subset=['account_id'])[person_cols].copy()

# Pivot: one row per account_id, columns are payment_amount{t}, var1{t}, pledge{t}
pivot = agg_df.pivot_table(index='account_id', columns='t',
                           values=['payment_amount', 'var1', 'pledge'],
                           aggfunc='sum').reset_index()
# Flatten multi-level column names
flat_cols = []
for col in pivot.columns:
    if isinstance(col, tuple):
        if col[1] == '' or col[1] == 0:
            flat_cols.append(str(col[0]))
        else:
            flat_cols.append(f"{col[0]}{col[1]}")
    else:
        flat_cols.append(str(col))
pivot.columns = flat_cols

# Merge person attributes back
pivot = pivot.merge(person_attrs, on='account_id', how='left')

# Drop existing donors (those with transactions before the "year before" window)
if 'payment_amount1' in pivot.columns:
    pivot = pivot[pivot['payment_amount1'].isna()].copy()

# Fix sentinel values
pivot.loc[pivot['script'] == -999, 'script'] = 0
pivot.loc[pivot['female'] == -999, 'female'] = np.nan

# Fill zeros for payment and count columns
for c in ['payment_amount2', 'payment_amount3', 'payment_amount4']:
    if c in pivot.columns:
        pivot[c] = pivot[c].fillna(0)
for c in ['var12', 'var13', 'var14']:
    if c in pivot.columns:
        pivot[c] = pivot[c].fillna(0)

# Create outcomes
pivot['retention'] = np.where(pivot['payment_amount2'] > 0,
                              pivot['payment_amount3'] / pivot['payment_amount2'],
                              np.nan)
pivot['renewing'] = ((pivot['var13'] != 0) & pivot['var13'].notna()).astype(int)
pivot['donated'] = np.where((pivot['var13'] > 0) & pivot['var13'].notna(), 1, 0)
pivot['gift_cond'] = np.where(pivot['payment_amount3'] > 0,
                              pivot['payment_amount3'], np.nan)

# Create strata FE
pivot['station_id'] = pd.Categorical(pivot['station_cal']).codes
pivot['edate_str'] = pivot['exec_date'].astype(str)
pivot['edate_dummy'] = pd.Categorical(pivot['edate_str']).codes
pivot['ii'] = pivot.groupby(['station_id', 'edate_dummy']).ngroup()

# Demographic dummies (alphabetical category ordering like Stata's tab, gen())
age_cats = sorted([x for x in pivot['age_at_gift'].unique()
                   if isinstance(x, str) and x != ''])
if len(age_cats) >= 2:
    pivot['age_display2'] = (pivot['age_at_gift'] == age_cats[1]).astype(int) \
        if len(age_cats) > 1 else 0
    pivot['age_display3'] = (pivot['age_at_gift'] == age_cats[2]).astype(int) \
        if len(age_cats) > 2 else 0
else:
    pivot['age_display2'] = 0
    pivot['age_display3'] = 0

inc_cats = sorted([x for x in pivot['income'].unique()
                   if isinstance(x, str) and x != ''])
pivot['inc_display1'] = (pivot['income'] == inc_cats[0]).astype(int) \
    if len(inc_cats) > 0 else 0
pivot['inc_display2'] = (pivot['income'] == inc_cats[1]).astype(int) \
    if len(inc_cats) > 1 else 0
pivot['inc_display3'] = (pivot['income'] == inc_cats[2]).astype(int) \
    if len(inc_cats) > 2 else 0

lor_cats = sorted([x for x in pivot['lorgroup'].unique()
                   if isinstance(x, str) and x != ''])
pivot['lor_display2'] = (pivot['lorgroup'] == lor_cats[1]).astype(int) \
    if len(lor_cats) > 1 else 0

# Fill demographic missing (NaN -> 0 for indicator variables)
pivot['female'] = pivot['female'].fillna(0)
pivot['age_display2'] = pivot['age_display2'].fillna(0)
pivot['age_display3'] = pivot['age_display3'].fillna(0)
pivot['inc_display1'] = pivot['inc_display1'].fillna(0)
pivot['inc_display2'] = pivot['inc_display2'].fillna(0)
pivot['inc_display3'] = pivot['inc_display3'].fillna(0)
pivot['lor_display2'] = pivot['lor_display2'].fillna(0)

# === Step 2 filters (matching Stata step2) ===
# Drop excluded exec_dates for Experiment 1
pivot = pivot[~pivot['exec_date'].astype(int).isin([1610, 1701])].copy()
# Drop stations with no control observations
station_drops = ['24', '55', '64', '61']
pivot = pivot[~pivot['station_cal'].astype(str).isin(station_drops)].copy()
# Drop baseline non-donors (payment_amount2==0)
pivot = pivot[pivot['payment_amount2'] > 0].copy()

# Convert FE vars to str for pyfixest
pivot['ii_str'] = pivot['ii'].astype(str)
pivot['station_id_str'] = pivot['station_id'].astype(str)

exp1 = pivot.copy()
print(f"Experiment 1 sample: N={len(exp1)}")
print(f"  Treat: {int(exp1['treat'].sum())}, Control: {int((exp1['treat']==0).sum())}")

# ============================================================
# DATA PREPARATION - EXPERIMENT 2 (from gift.dta)
# Replicating Stata step1 (gift section) + step2 (exp2 section)
# ============================================================
print("\nLoading and preparing data for Experiment 2...")

gift_raw = pd.read_stata(f"{RAW_DATA_DIR}/gift.dta")

# Convert float32 to float64
for col in gift_raw.columns:
    if gift_raw[col].dtype == np.float32:
        gift_raw[col] = gift_raw[col].astype(np.float64)

# Rename columns to match Stata code
gift_raw.rename(columns={
    'treatment': 'response_type',
    'paymentamounttransactions': 'payment_amount',
    'id': 'account_id',
    'giftdatetransactions': 'gift_date_str',
}, inplace=True)

gift_raw['var1'] = 1  # counter for number of transactions

# Create treat variable
gift_raw['treat'] = np.nan
gift_raw.loc[gift_raw['response_type'] == 'Control', 'treat_str'] = 'Control'
gift_raw.loc[(gift_raw['response_type'] == 'Called: Contacted') |
             (gift_raw['response_type'] == 'Called: Not Contacted'),
             'treat_str'] = 'Call'

# Parse dates
gift_raw['date'] = pd.to_datetime(gift_raw['gift_date_str'], format='%m/%d/%y',
                                   errors='coerce')
gift_raw['edate'] = pd.Timestamp('2013-04-01')  # exec date for Exp 2

# Create timeline
diff_days_g2 = (gift_raw['edate'] - gift_raw['date']).dt.total_seconds() / 86400.0
# Note: in Stata, diff = edate - date (positive if gift before exec_date)
# timeline uses date relative to edate:
gift_raw['timeline'] = ''
gift_raw.loc[gift_raw['date'] < gift_raw['edate'] - pd.Timedelta(days=365.25),
             'timeline'] = 'a_existing_donor'
gift_raw.loc[(gift_raw['date'] > gift_raw['edate'] - pd.Timedelta(days=365.25)) &
             (gift_raw['date'] <= gift_raw['edate']),
             'timeline'] = 'b_year before'
gift_raw.loc[(gift_raw['date'] >= gift_raw['edate']) &
             (gift_raw['date'] < gift_raw['edate'] + pd.Timedelta(days=365.25)),
             'timeline'] = 'c_year after'
gift_raw.loc[gift_raw['date'] >= gift_raw['edate'] + pd.Timedelta(days=365.25),
             'timeline'] = 'two years after or more'
gift_raw.loc[gift_raw['timeline'] == '', 'timeline'] = 'na'

# Collapse and reshape (matching Stata step2 for Exp2)
# collapse (sum) payment_amount var1, by(timeline account_id treat response_type)
gift_agg = gift_raw.groupby(['timeline', 'account_id', 'treat_str',
                              'response_type']).agg(
    payment_amount=('payment_amount', 'sum'),
    var1=('var1', 'sum'),
).reset_index()

# Encode timeline
g2_timeline_map = {
    'a_existing_donor': 1, 'b_year before': 2, 'c_year after': 3,
    'two years after or more': 4, 'na': 5
}
gift_agg['t'] = gift_agg['timeline'].map(g2_timeline_map).fillna(0).astype(int)

# Get person-level attributes (one per account_id)
g2_person = gift_agg.drop_duplicates(subset=['account_id'])[
    ['account_id', 'treat_str', 'response_type']].copy()

# Pivot wide: i(account_id treat) j(t)
g2_pivot = gift_agg.pivot_table(index='account_id', columns='t',
                                 values=['payment_amount', 'var1'],
                                 aggfunc='sum').reset_index()
flat_cols2 = []
for col in g2_pivot.columns:
    if isinstance(col, tuple):
        if col[1] == '' or col[1] == 0:
            flat_cols2.append(str(col[0]))
        else:
            flat_cols2.append(f"{col[0]}{col[1]}")
    else:
        flat_cols2.append(str(col))
g2_pivot.columns = flat_cols2

# Merge person attributes back
g2_pivot = g2_pivot.merge(g2_person, on='account_id', how='left')

# Drop existing donors (payment_amount1 != .)
if 'payment_amount1' in g2_pivot.columns:
    g2_pivot = g2_pivot[g2_pivot['payment_amount1'].isna()].copy()

# Fill missing
for c in ['payment_amount2', 'payment_amount3', 'payment_amount4']:
    if c in g2_pivot.columns:
        g2_pivot[c] = g2_pivot[c].fillna(0)
for c in ['var12', 'var13', 'var14']:
    if c in g2_pivot.columns:
        g2_pivot[c] = g2_pivot[c].fillna(0)

# Create outcomes
g2_pivot['retention'] = np.where(g2_pivot['payment_amount2'] > 0,
                                 g2_pivot['payment_amount3'] / g2_pivot['payment_amount2'],
                                 np.nan)
g2_pivot['renewing'] = 0
g2_pivot.loc[g2_pivot['var13'] != 0, 'renewing'] = 1
g2_pivot['donated'] = np.where(
    (g2_pivot['var13'] > 0) & g2_pivot['var13'].notna(), 1, 0)
g2_pivot['gift_cond'] = np.where(g2_pivot['payment_amount3'] > 0,
                                 g2_pivot['payment_amount3'], np.nan)

# Create treat (encode treatment, then recode: Stata encode makes
# "Call"=1 and "Control"=2, then replace treat=0 if treat==2)
g2_pivot['treat'] = np.where(g2_pivot['treat_str'] == 'Control', 0,
                    np.where(g2_pivot['treat_str'] == 'Call', 1, np.nan))

exp2 = g2_pivot.dropna(subset=['treat']).copy()
print(f"Experiment 2 sample: N={len(exp2)}")
print(f"  Treat: {int(exp2['treat'].sum())}, Control: {int((exp2['treat']==0).sum())}")

# ============================================================
# CONTROL VARIABLE DEFINITIONS
# ============================================================
G1_FULL_CONTROLS = ['payment_amount2', 'var12', 'female', 'age_display2',
                    'age_display3', 'inc_display3', 'inc_display1',
                    'inc_display2', 'lor_display2']
G1_BASELINE_GIVING = ['payment_amount2', 'var12']
G1_DEMOGRAPHICS = ['female', 'age_display2', 'age_display3', 'inc_display3',
                   'inc_display1', 'inc_display2', 'lor_display2']
G2_CONTROLS = ['payment_amount2', 'var12']

CONTINUOUS_MONETARY = ['payment_amount3', 'gift_cond', 'retention']
BINARY_OUTCOMES = ['renewing', 'donated']

# ============================================================
# EXPERIMENT 1 - BASELINE SPECS (Table 2 + Table A1)
# ============================================================
print("\n=== Running G1 baselines ===")

# --- Table 2 difference-in-means baselines ---
table2_specs_g1 = [
    ("baseline__t2_exp1_renewing", "renewing", "chi2", None,
     "Table 2 Exp1 Percent Donating"),
    ("baseline__t2_exp1_payment_amount3", "payment_amount3", "ranksum", None,
     "Table 2 Exp1 Amount Donated"),
    ("baseline__t2_exp1_var13", "var13", "ranksum", None,
     "Table 2 Exp1 Number of Gifts"),
    ("baseline__t2_exp1_gift_cond", "gift_cond", "ranksum",
     lambda d: d['renewing'] == 1, "Table 2 Exp1 Amount|Donated"),
    ("baseline__t2_exp1_retention", "retention", "ranksum", None,
     "Table 2 Exp1 Retention Rate"),
]

for spec_id, outcome, test, cond, desc in table2_specs_g1:
    run_id = make_run_id()
    try:
        coef, se, pv, ci_lo, ci_hi, nobs = diff_in_means_test(
            exp1, outcome, 'treat', test, cond)
        coeffs = {
            "treat_effect": float(coef),
            "treat_mean": float(exp1[exp1['treat'] == 1][outcome].dropna().mean()),
            "ctrl_mean": float(exp1[exp1['treat'] == 0][outcome].dropna().mean()),
        }
        if cond is not None:
            d_cond = exp1[cond(exp1)]
            coeffs["treat_mean"] = float(
                d_cond[d_cond['treat'] == 1][outcome].dropna().mean())
            coeffs["ctrl_mean"] = float(
                d_cond[d_cond['treat'] == 0][outcome].dropna().mean())
        infer = {"spec_id": canonical_inference_g1["spec_id"],
                 "params": {"test": test}}
        payload = make_payload(coeffs, infer, design_block_g1)
        sample = "Exp1 full sample" if cond is None else "Exp1 conditional on donating"
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, np.nan, payload, sample, "", "none", "")
        print(f"  {spec_id}: coef={coef:.4f}, p={pv:.4f}, N={nobs}")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), desc, "", "none", "",
                   0, str(e))

# --- Table A1 OLS baselines ---
ols_baselines_g1 = [
    ("baseline__tA1_exp1_donated_ols", "donated",
     "donated ~ treat + payment_amount2 + var12 + female + age_display2 + "
     "age_display3 + inc_display3 + inc_display1 + inc_display2 + lor_display2 | ii_str",
     None, "Table A1 Exp1 Donated OLS"),
    ("baseline__tA1_exp1_gift_cond_ols", "gift_cond",
     "gift_cond ~ treat + payment_amount2 + var12 + female + age_display2 + "
     "age_display3 + inc_display3 + inc_display1 + inc_display2 + lor_display2 | ii_str",
     None, "Table A1 Exp1 Gift|Donated OLS"),
]

g1_ols_run_ids = {}

for spec_id, outcome, formula, cond, desc in ols_baselines_g1:
    run_id = make_run_id()
    g1_ols_run_ids[spec_id] = run_id
    try:
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp1, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid",
                 "params": {"note": "classical SE matching Stata xtreg, fe default"}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"controls": {"spec_id": "rc/controls/sets/full_with_fe",
                                             "included": G1_FULL_CONTROLS,
                                             "n_controls": 9}})
        sample = ("Exp1 full sample" if outcome != "gift_cond"
                  else "Exp1 conditional on donating")
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample,
                   "ii (station x exec_date)",
                   ", ".join(G1_FULL_CONTROLS), "")
        print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pv:.4f}, "
              f"N={nobs}, R2={r2:.4f}")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), desc, "", "", "",
                   0, str(e))

# ============================================================
# EXPERIMENT 2 - BASELINE SPECS (Table 2 + Table A1)
# ============================================================
print("\n=== Running G2 baselines ===")

table2_specs_g2 = [
    ("baseline__t2_exp2_renewing", "renewing", "chi2", None,
     "Table 2 Exp2 Percent Donating"),
    ("baseline__t2_exp2_payment_amount3", "payment_amount3", "ranksum", None,
     "Table 2 Exp2 Amount Donated"),
    ("baseline__t2_exp2_var13", "var13", "ranksum", None,
     "Table 2 Exp2 Number of Gifts"),
    ("baseline__t2_exp2_gift_cond", "gift_cond", "ranksum",
     lambda d: d['renewing'] == 1, "Table 2 Exp2 Amount|Donated"),
    ("baseline__t2_exp2_retention", "retention", "ranksum", None,
     "Table 2 Exp2 Retention Rate"),
]

for spec_id, outcome, test, cond, desc in table2_specs_g2:
    run_id = make_run_id()
    try:
        coef, se, pv, ci_lo, ci_hi, nobs = diff_in_means_test(
            exp2, outcome, 'treat', test, cond)
        coeffs = {
            "treat_effect": float(coef),
            "treat_mean": float(exp2[exp2['treat'] == 1][outcome].dropna().mean()),
            "ctrl_mean": float(exp2[exp2['treat'] == 0][outcome].dropna().mean()),
        }
        if cond is not None:
            d_cond = exp2[cond(exp2)]
            coeffs["treat_mean"] = float(
                d_cond[d_cond['treat'] == 1][outcome].dropna().mean())
            coeffs["ctrl_mean"] = float(
                d_cond[d_cond['treat'] == 0][outcome].dropna().mean())
        infer = {"spec_id": canonical_inference_g2["spec_id"],
                 "params": {"test": test}}
        payload = make_payload(coeffs, infer, design_block_g2)
        sample = "Exp2 full sample" if cond is None else "Exp2 conditional on donating"
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, np.nan, payload, sample, "", "none", "")
        print(f"  {spec_id}: coef={coef:.4f}, p={pv:.4f}, N={nobs}")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G2", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), desc, "", "none", "",
                   0, str(e))

# G2 OLS baselines (Table A1) - no FE, no demographics
ols_baselines_g2 = [
    ("baseline__tA1_exp2_donated_ols", "donated",
     "donated ~ treat + payment_amount2 + var12",
     None, "Table A1 Exp2 Donated OLS"),
    ("baseline__tA1_exp2_gift_cond_ols", "gift_cond",
     "gift_cond ~ treat + payment_amount2 + var12",
     None, "Table A1 Exp2 Gift|Donated OLS"),
]

g2_ols_run_ids = {}

for spec_id, outcome, formula, cond, desc in ols_baselines_g2:
    run_id = make_run_id()
    g2_ols_run_ids[spec_id] = run_id
    try:
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp2, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid",
                 "params": {"note": "classical SE matching Stata reg default"}}
        payload = make_payload(coeffs, infer, design_block_g2,
                               {"controls": {"spec_id": "rc/controls/sets/baseline_giving_only",
                                             "included": G2_CONTROLS,
                                             "n_controls": 2}})
        sample = ("Exp2 full sample" if outcome != "gift_cond"
                  else "Exp2 conditional on donating")
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample,
                   "none", ", ".join(G2_CONTROLS), "")
        print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pv:.4f}, N={nobs}")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/designs/randomized_experiment.md",
                   "G2", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), desc, "", "", "",
                   0, str(e))

# ============================================================
# G1 DESIGN VARIANTS
# ============================================================
print("\n=== Running G1 design variants ===")

g1_focal_outcomes_dim = [
    ("renewing", None, "Exp1 full sample"),
    ("payment_amount3", None, "Exp1 full sample"),
    ("donated", None, "Exp1 full sample"),
]

# diff_in_means: OLS without controls or FE
for outcome, cond, sample_desc in g1_focal_outcomes_dim:
    run_id = make_run_id()
    try:
        formula = f"{outcome} ~ treat"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp1, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"controls": {"spec_id": "rc/controls/sets/none",
                                             "included": [], "n_controls": 0}})
        add_result("design/randomized_experiment/estimator/diff_in_means",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc, "none", "none", "")
        print(f"  G1 dim {outcome}: coef={coef:.4f}, p={pv:.4f}")
    except Exception as e:
        add_result("design/randomized_experiment/estimator/diff_in_means",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# strata_fe: OLS with strata FE only, no controls
for outcome, cond, sample_desc in g1_focal_outcomes_dim:
    run_id = make_run_id()
    try:
        formula = f"{outcome} ~ treat | ii_str"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp1, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"controls": {"spec_id": "rc/controls/sets/none",
                                             "included": [], "n_controls": 0},
                                "fixed_effects": {"spec_id": "rc/fe/strata/station_x_exec_date",
                                                  "fe": ["ii"]}})
        add_result("design/randomized_experiment/estimator/strata_fe",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc,
                   "ii (station x exec_date)", "none", "")
        print(f"  G1 strata_fe {outcome}: coef={coef:.4f}, p={pv:.4f}")
    except Exception as e:
        add_result("design/randomized_experiment/estimator/strata_fe",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# with_covariates: OLS with controls + FE (focal outcomes)
for outcome, cond, sample_desc in g1_focal_outcomes_dim:
    run_id = make_run_id()
    try:
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp1, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"controls": {"spec_id": "rc/controls/sets/full_with_fe",
                                             "included": G1_FULL_CONTROLS,
                                             "n_controls": 9}})
        add_result("design/randomized_experiment/estimator/with_covariates",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc,
                   "ii (station x exec_date)",
                   ", ".join(G1_FULL_CONTROLS), "")
        print(f"  G1 with_cov {outcome}: coef={coef:.4f}, p={pv:.4f}")
    except Exception as e:
        add_result("design/randomized_experiment/estimator/with_covariates",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# ============================================================
# G1 ROBUSTNESS - CONTROLS (LOO, Sets) — OLS specs only
# ============================================================
print("\n=== Running G1 RC controls ===")

loo_outcomes = [("donated", "Exp1 full sample"),
                ("gift_cond", "Exp1 conditional on donating")]

for drop_ctrl in G1_FULL_CONTROLS:
    remaining = [c for c in G1_FULL_CONTROLS if c != drop_ctrl]
    ctrl_str = " + ".join(remaining) if remaining else "1"
    for outcome, sample_desc in loo_outcomes:
        run_id = make_run_id()
        spec_id = f"rc/controls/loo/drop_{drop_ctrl}"
        try:
            formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(exp1, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g1,
                                   {"controls": {"spec_id": spec_id,
                                                 "family": "loo",
                                                 "dropped": [drop_ctrl],
                                                 "included": remaining,
                                                 "n_controls": len(remaining)}})
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#loo",
                       "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                       nobs, r2, payload, sample_desc,
                       "ii (station x exec_date)",
                       ", ".join(remaining), "")
        except Exception as e:
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#loo",
                       "G1", outcome, "treat", np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

print(f"  G1 LOO: {len(G1_FULL_CONTROLS)} controls x 2 outcomes = "
      f"{len(G1_FULL_CONTROLS) * 2} specs")

# Control sets
control_sets_g1 = [
    ("rc/controls/sets/none", [], "no controls"),
    ("rc/controls/sets/baseline_giving_only", G1_BASELINE_GIVING,
     "payment_amount2, var12"),
    ("rc/controls/sets/demographics_only", G1_DEMOGRAPHICS,
     "demographics only"),
    ("rc/controls/sets/full_with_fe", G1_FULL_CONTROLS,
     "all 9 controls"),
]

for ctrl_spec_id, ctrl_list, ctrl_desc in control_sets_g1:
    for outcome, sample_desc in loo_outcomes:
        run_id = make_run_id()
        try:
            if ctrl_list:
                ctrl_str = " + ".join(ctrl_list)
                formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
            else:
                formula = f"{outcome} ~ treat | ii_str"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(exp1, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g1,
                                   {"controls": {"spec_id": ctrl_spec_id,
                                                 "included": ctrl_list,
                                                 "n_controls": len(ctrl_list)}})
            add_result(ctrl_spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#sets",
                       "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                       nobs, r2, payload, sample_desc,
                       "ii (station x exec_date)", ctrl_desc, "")
        except Exception as e:
            add_result(ctrl_spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#sets",
                       "G1", outcome, "treat", np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

print(f"  G1 control sets: {len(control_sets_g1)} sets x 2 outcomes")

# ============================================================
# G1 ROBUSTNESS - SAMPLE
# ============================================================
print("\n=== Running G1 RC sample ===")

trim_outcomes_g1 = [("payment_amount3", "Exp1 full sample"),
                    ("gift_cond", "Exp1 conditional on donating")]

for pct_lo, pct_hi, spec_suffix in [(1, 99, "trim_y_1_99"),
                                     (5, 95, "trim_y_5_95")]:
    for outcome, sample_desc in trim_outcomes_g1:
        run_id = make_run_id()
        spec_id = f"rc/sample/outliers/{spec_suffix}"
        try:
            d = exp1.copy()
            y = d[outcome].dropna()
            lo_val = np.percentile(y, pct_lo)
            hi_val = np.percentile(y, pct_hi)
            d = d[(d[outcome].isna()) |
                  ((d[outcome] >= lo_val) & (d[outcome] <= hi_val))].copy()
            ctrl_str = " + ".join(G1_FULL_CONTROLS)
            formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(d, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g1,
                                   {"sample": {"spec_id": spec_id,
                                               "trim_lo": pct_lo,
                                               "trim_hi": pct_hi,
                                               "threshold_lo": float(lo_val),
                                               "threshold_hi": float(hi_val)},
                                    "controls": {"included": G1_FULL_CONTROLS,
                                                 "n_controls": 9}})
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/sample.md#outliers",
                       "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                       nobs, r2, payload,
                       f"{sample_desc}, trimmed p{pct_lo}-p{pct_hi}",
                       "ii (station x exec_date)",
                       ", ".join(G1_FULL_CONTROLS), "")
        except Exception as e:
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/sample.md#outliers",
                       "G1", outcome, "treat", np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

# Drop big donors >= 5k
for outcome, sample_desc in loo_outcomes:
    run_id = make_run_id()
    spec_id = "rc/sample/quality/drop_big_donors_5k"
    try:
        d = exp1[exp1['payment_amount2'] < 5000].copy()
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
            run_ols_spec(d, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"sample": {"spec_id": spec_id,
                                           "note": "Drop donors with baseline gift >= 5000"},
                                "controls": {"included": G1_FULL_CONTROLS,
                                             "n_controls": 9}})
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/sample.md#quality",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload,
                   f"{sample_desc}, drop baseline gift >= 5k",
                   "ii (station x exec_date)",
                   ", ".join(G1_FULL_CONTROLS), "")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/sample.md#quality",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# Drop stations with few obs (< 10 in either arm per stratum)
for outcome, sample_desc in loo_outcomes:
    run_id = make_run_id()
    spec_id = "rc/sample/quality/drop_stations_with_few_obs"
    try:
        strata_counts = exp1.groupby(['ii', 'treat']).size().unstack(fill_value=0)
        cols_sc = strata_counts.columns
        if 0.0 in cols_sc and 1.0 in cols_sc:
            keep_strata = strata_counts[(strata_counts[0.0] >= 10) &
                                        (strata_counts[1.0] >= 10)].index
        elif 0 in cols_sc and 1 in cols_sc:
            keep_strata = strata_counts[(strata_counts[0] >= 10) &
                                        (strata_counts[1] >= 10)].index
        else:
            keep_strata = strata_counts.index
        d = exp1[exp1['ii'].isin(keep_strata)].copy()
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
            run_ols_spec(d, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"sample": {"spec_id": spec_id,
                                           "note": "Drop strata with <10 obs per arm",
                                           "n_strata_dropped": int(
                                               len(strata_counts) - len(keep_strata))},
                                "controls": {"included": G1_FULL_CONTROLS,
                                             "n_controls": 9}})
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/sample.md#quality",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload,
                   f"{sample_desc}, drop low-obs strata",
                   "ii (station x exec_date)",
                   ", ".join(G1_FULL_CONTROLS), "")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/sample.md#quality",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# ============================================================
# G1 ROBUSTNESS - FUNCTIONAL FORM (asinh, log1p)
# ============================================================
print("\n=== Running G1 RC functional form ===")

form_outcomes_g1 = [("payment_amount3", "Exp1 full sample"),
                    ("gift_cond", "Exp1 conditional on donating")]

for transform, spec_suffix in [("asinh", "asinh"), ("log1p", "log1p")]:
    for outcome, sample_desc in form_outcomes_g1:
        run_id = make_run_id()
        spec_id = f"rc/form/outcome/{spec_suffix}"
        try:
            d = exp1.copy()
            if transform == "asinh":
                d[f'{outcome}_t'] = np.arcsinh(d[outcome])
                interp = "inverse hyperbolic sine; approximate semi-elasticity"
            else:
                d[f'{outcome}_t'] = np.log1p(d[outcome].clip(lower=0))
                interp = "log(1+y); approximate percent change"
            ctrl_str = " + ".join(G1_FULL_CONTROLS)
            formula = f"{outcome}_t ~ treat + {ctrl_str} | ii_str"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(d, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g1,
                                   {"functional_form": {"spec_id": spec_id,
                                                        "transform": transform,
                                                        "interpretation": interp},
                                    "controls": {"included": G1_FULL_CONTROLS,
                                                 "n_controls": 9}})
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/functional_form.md",
                       "G1", f"{outcome} ({transform})", "treat",
                       coef, se, pv, ci_lo, ci_hi, nobs, r2, payload,
                       sample_desc, "ii (station x exec_date)",
                       ", ".join(G1_FULL_CONTROLS), "")
            print(f"  G1 {spec_suffix} {outcome}: coef={coef:.4f}, p={pv:.4f}")
        except Exception as e:
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/functional_form.md",
                       "G1", f"{outcome} ({transform})", "treat",
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

# ============================================================
# G1 ROBUSTNESS - FE VARIANTS
# ============================================================
print("\n=== Running G1 RC FE variants ===")

# Station only FE (drop exec_date from strata)
exp1['station_only_fe'] = exp1['station_id'].astype(str)
for outcome, sample_desc in loo_outcomes:
    run_id = make_run_id()
    spec_id = "rc/fe/strata/station_only"
    try:
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str} | station_only_fe"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
            run_ols_spec(exp1, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"fixed_effects": {"spec_id": spec_id,
                                                  "fe": ["station_id"]},
                                "controls": {"included": G1_FULL_CONTROLS,
                                             "n_controls": 9}})
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/fixed_effects.md",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc,
                   "station_id only",
                   ", ".join(G1_FULL_CONTROLS), "")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/fixed_effects.md",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# No FE
for outcome, sample_desc in loo_outcomes:
    run_id = make_run_id()
    spec_id = "rc/fe/strata/none"
    try:
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str}"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
            run_ols_spec(exp1, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1,
                               {"fixed_effects": {"spec_id": spec_id,
                                                  "fe": []},
                                "controls": {"included": G1_FULL_CONTROLS,
                                             "n_controls": 9}})
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/fixed_effects.md",
                   "G1", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc,
                   "none", ", ".join(G1_FULL_CONTROLS), "")
    except Exception as e:
        add_result(spec_id, run_id,
                   "specification_tree/modules/robustness/fixed_effects.md",
                   "G1", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# ============================================================
# G2 DESIGN VARIANTS
# ============================================================
print("\n=== Running G2 design variants ===")

g2_focal = [("renewing", None, "Exp2 full sample"),
            ("payment_amount3", None, "Exp2 full sample"),
            ("donated", None, "Exp2 full sample")]

# diff_in_means: OLS without controls
for outcome, cond, sample_desc in g2_focal:
    run_id = make_run_id()
    try:
        formula = f"{outcome} ~ treat"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp2, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g2,
                               {"controls": {"spec_id": "rc/controls/sets/none",
                                             "included": [], "n_controls": 0}})
        add_result("design/randomized_experiment/estimator/diff_in_means",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc, "none", "none", "")
    except Exception as e:
        add_result("design/randomized_experiment/estimator/diff_in_means",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G2", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# with_covariates: OLS with baseline giving controls
for outcome, cond, sample_desc in g2_focal:
    run_id = make_run_id()
    try:
        ctrl_str = " + ".join(G2_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str}"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp2, formula, 'iid', 'treat')
        infer = {"spec_id": "infer/se/iid", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g2,
                               {"controls": {"spec_id": "rc/controls/sets/baseline_giving_only",
                                             "included": G2_CONTROLS,
                                             "n_controls": 2}})
        add_result("design/randomized_experiment/estimator/with_covariates",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                   nobs, r2, payload, sample_desc,
                   "none", ", ".join(G2_CONTROLS), "")
    except Exception as e:
        add_result("design/randomized_experiment/estimator/with_covariates",
                   run_id,
                   "specification_tree/designs/randomized_experiment.md#estimator",
                   "G2", outcome, "treat", np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan,
                   make_fail_payload(str(e)), sample_desc, "", "", "",
                   0, str(e))

# ============================================================
# G2 ROBUSTNESS - CONTROLS (LOO)
# ============================================================
print("\n=== Running G2 RC controls ===")

g2_ols_outcomes = [("donated", "Exp2 full sample"),
                   ("gift_cond", "Exp2 conditional on donating")]

for drop_ctrl in G2_CONTROLS:
    remaining = [c for c in G2_CONTROLS if c != drop_ctrl]
    for outcome, sample_desc in g2_ols_outcomes:
        run_id = make_run_id()
        spec_id = f"rc/controls/loo/drop_{drop_ctrl}"
        try:
            if remaining:
                ctrl_str = " + ".join(remaining)
                formula = f"{outcome} ~ treat + {ctrl_str}"
            else:
                formula = f"{outcome} ~ treat"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(exp2, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g2,
                                   {"controls": {"spec_id": spec_id,
                                                 "family": "loo",
                                                 "dropped": [drop_ctrl],
                                                 "included": remaining,
                                                 "n_controls": len(remaining)}})
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#loo",
                       "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                       nobs, r2, payload, sample_desc,
                       "none",
                       ", ".join(remaining) if remaining else "none", "")
        except Exception as e:
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#loo",
                       "G2", outcome, "treat", np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

# G2 control sets
g2_ctrl_sets = [
    ("rc/controls/sets/none", [], "no controls"),
    ("rc/controls/sets/baseline_giving_only", G2_CONTROLS,
     "payment_amount2, var12"),
]
for ctrl_spec_id, ctrl_list, ctrl_desc in g2_ctrl_sets:
    for outcome, sample_desc in g2_ols_outcomes:
        run_id = make_run_id()
        try:
            if ctrl_list:
                ctrl_str = " + ".join(ctrl_list)
                formula = f"{outcome} ~ treat + {ctrl_str}"
            else:
                formula = f"{outcome} ~ treat"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(exp2, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g2,
                                   {"controls": {"spec_id": ctrl_spec_id,
                                                 "included": ctrl_list,
                                                 "n_controls": len(ctrl_list)}})
            add_result(ctrl_spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#sets",
                       "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                       nobs, r2, payload, sample_desc,
                       "none", ctrl_desc, "")
        except Exception as e:
            add_result(ctrl_spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#sets",
                       "G2", outcome, "treat", np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

# ============================================================
# G2 ROBUSTNESS - SAMPLE
# ============================================================
print("\n=== Running G2 RC sample ===")

g2_trim_outcomes = [("payment_amount3", "Exp2 full sample"),
                    ("gift_cond", "Exp2 conditional on donating")]

for pct_lo, pct_hi, spec_suffix in [(1, 99, "trim_y_1_99"),
                                     (5, 95, "trim_y_5_95")]:
    for outcome, sample_desc in g2_trim_outcomes:
        run_id = make_run_id()
        spec_id = f"rc/sample/outliers/{spec_suffix}"
        try:
            d = exp2.copy()
            y = d[outcome].dropna()
            lo_val = np.percentile(y, pct_lo)
            hi_val = np.percentile(y, pct_hi)
            d = d[(d[outcome].isna()) |
                  ((d[outcome] >= lo_val) & (d[outcome] <= hi_val))].copy()
            ctrl_str = " + ".join(G2_CONTROLS)
            formula = f"{outcome} ~ treat + {ctrl_str}"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(d, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g2,
                                   {"sample": {"spec_id": spec_id,
                                               "trim_lo": pct_lo,
                                               "trim_hi": pct_hi},
                                    "controls": {"included": G2_CONTROLS,
                                                 "n_controls": 2}})
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/sample.md#outliers",
                       "G2", outcome, "treat", coef, se, pv, ci_lo, ci_hi,
                       nobs, r2, payload,
                       f"{sample_desc}, trimmed p{pct_lo}-p{pct_hi}",
                       "none", ", ".join(G2_CONTROLS), "")
        except Exception as e:
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/sample.md#outliers",
                       "G2", outcome, "treat", np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

# ============================================================
# G2 ROBUSTNESS - FUNCTIONAL FORM
# ============================================================
print("\n=== Running G2 RC functional form ===")

for transform, spec_suffix in [("asinh", "asinh"), ("log1p", "log1p")]:
    for outcome, sample_desc in g2_trim_outcomes:
        run_id = make_run_id()
        spec_id = f"rc/form/outcome/{spec_suffix}"
        try:
            d = exp2.copy()
            if transform == "asinh":
                d[f'{outcome}_t'] = np.arcsinh(d[outcome])
                interp = "inverse hyperbolic sine"
            else:
                d[f'{outcome}_t'] = np.log1p(d[outcome].clip(lower=0))
                interp = "log(1+y)"
            ctrl_str = " + ".join(G2_CONTROLS)
            formula = f"{outcome}_t ~ treat + {ctrl_str}"
            coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m_obj = \
                run_ols_spec(d, formula, 'iid', 'treat')
            infer = {"spec_id": "infer/se/iid", "params": {}}
            payload = make_payload(coeffs, infer, design_block_g2,
                                   {"functional_form": {"spec_id": spec_id,
                                                        "transform": transform,
                                                        "interpretation": interp},
                                    "controls": {"included": G2_CONTROLS,
                                                 "n_controls": 2}})
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/functional_form.md",
                       "G2", f"{outcome} ({transform})", "treat",
                       coef, se, pv, ci_lo, ci_hi, nobs, r2, payload,
                       sample_desc, "none", ", ".join(G2_CONTROLS), "")
        except Exception as e:
            add_result(spec_id, run_id,
                       "specification_tree/modules/robustness/functional_form.md",
                       "G2", f"{outcome} ({transform})", "treat",
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       make_fail_payload(str(e)), sample_desc, "", "", "",
                       0, str(e))

# ============================================================
# INFERENCE VARIANTS (for OLS baseline specs)
# ============================================================
print("\n=== Running inference variants ===")

# HC1 for G1 OLS baselines
for base_spec_id, base_run_id in g1_ols_run_ids.items():
    outcome = "donated" if "donated" in base_spec_id else "gift_cond"
    try:
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp1, formula, 'hetero', 'treat')
        infer = {"spec_id": "infer/se/hc/hc1", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g1)
        add_inference_result(base_run_id, "infer/se/hc/hc1",
                             "specification_tree/modules/inference/standard_errors.md#hc",
                             "G1", coef, se, pv, ci_lo, ci_hi, nobs, r2, payload)
    except Exception as e:
        add_inference_result(base_run_id, "infer/se/hc/hc1",
                             "specification_tree/modules/inference/standard_errors.md#hc",
                             "G1", np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan,
                             make_fail_payload(str(e)), 0, str(e))

# Cluster at station level for G1 OLS baselines
for base_spec_id, base_run_id in g1_ols_run_ids.items():
    outcome = "donated" if "donated" in base_spec_id else "gift_cond"
    try:
        ctrl_str = " + ".join(G1_FULL_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str} | ii_str"
        d = exp1.copy()
        d['station_cl'] = d['station_id'].astype(str)
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            d, formula, {"CRV1": "station_cl"}, 'treat')
        infer = {"spec_id": "infer/se/cluster/station",
                 "params": {"cluster_var": "station_id"}}
        payload = make_payload(coeffs, infer, design_block_g1)
        add_inference_result(base_run_id, "infer/se/cluster/station",
                             "specification_tree/modules/inference/standard_errors.md#cluster",
                             "G1", coef, se, pv, ci_lo, ci_hi, nobs, r2, payload)
    except Exception as e:
        add_inference_result(base_run_id, "infer/se/cluster/station",
                             "specification_tree/modules/inference/standard_errors.md#cluster",
                             "G1", np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan,
                             make_fail_payload(str(e)), 0, str(e))

# T-test inference for G1 diff-in-means baselines
ttest_specs_g1 = [
    ("renewing", "chi2", None),
    ("payment_amount3", "ranksum", None),
    ("var13", "ranksum", None),
    ("gift_cond", "ranksum", lambda d: d['renewing'] == 1),
    ("retention", "ranksum", None),
]

# Get the run IDs for the Table 2 baselines (first 5 results)
t2_run_ids_g1 = [r["spec_run_id"] for r in results[:5]]

for i, (outcome, orig_test, cond) in enumerate(ttest_specs_g1):
    if i >= len(t2_run_ids_g1):
        break
    base_run_id = t2_run_ids_g1[i]
    try:
        coef, se, pv, ci_lo, ci_hi, nobs = diff_in_means_test(
            exp1, outcome, 'treat', 'ttest', cond)
        infer = {"spec_id": "infer/test/ttest", "params": {}}
        coeffs = {"treat_effect": float(coef)}
        payload = make_payload(coeffs, infer, design_block_g1)
        add_inference_result(base_run_id, "infer/test/ttest",
                             "specification_tree/modules/inference/tests.md#ttest",
                             "G1", coef, se, pv, ci_lo, ci_hi, nobs,
                             np.nan, payload)
    except Exception as e:
        add_inference_result(base_run_id, "infer/test/ttest",
                             "specification_tree/modules/inference/tests.md#ttest",
                             "G1", np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan,
                             make_fail_payload(str(e)), 0, str(e))

# HC1 for G2 OLS baselines
for base_spec_id, base_run_id in g2_ols_run_ids.items():
    outcome = "donated" if "donated" in base_spec_id else "gift_cond"
    try:
        ctrl_str = " + ".join(G2_CONTROLS)
        formula = f"{outcome} ~ treat + {ctrl_str}"
        coef, se, pv, ci_lo, ci_hi, nobs, r2, coeffs, m = run_ols_spec(
            exp2, formula, 'hetero', 'treat')
        infer = {"spec_id": "infer/se/hc/hc1", "params": {}}
        payload = make_payload(coeffs, infer, design_block_g2)
        add_inference_result(base_run_id, "infer/se/hc/hc1",
                             "specification_tree/modules/inference/standard_errors.md#hc",
                             "G2", coef, se, pv, ci_lo, ci_hi, nobs, r2, payload)
    except Exception as e:
        add_inference_result(base_run_id, "infer/se/hc/hc1",
                             "specification_tree/modules/inference/standard_errors.md#hc",
                             "G2", np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan,
                             make_fail_payload(str(e)), 0, str(e))

# T-test for G2 diff-in-means baselines
t2_g2_run_ids = [r["spec_run_id"] for r in results
                 if r["baseline_group_id"] == "G2"
                 and r["spec_id"].startswith("baseline__t2")][:5]

ttest_specs_g2 = [
    ("renewing", "chi2", None),
    ("payment_amount3", "ranksum", None),
    ("var13", "ranksum", None),
    ("gift_cond", "ranksum", lambda d: d['renewing'] == 1),
    ("retention", "ranksum", None),
]

for i, (outcome, orig_test, cond) in enumerate(ttest_specs_g2):
    if i >= len(t2_g2_run_ids):
        break
    base_run_id = t2_g2_run_ids[i]
    try:
        coef, se, pv, ci_lo, ci_hi, nobs = diff_in_means_test(
            exp2, outcome, 'treat', 'ttest', cond)
        infer = {"spec_id": "infer/test/ttest", "params": {}}
        coeffs = {"treat_effect": float(coef)}
        payload = make_payload(coeffs, infer, design_block_g2)
        add_inference_result(base_run_id, "infer/test/ttest",
                             "specification_tree/modules/inference/tests.md#ttest",
                             "G2", coef, se, pv, ci_lo, ci_hi, nobs,
                             np.nan, payload)
    except Exception as e:
        add_inference_result(base_run_id, "infer/test/ttest",
                             "specification_tree/modules/inference/tests.md#ttest",
                             "G2", np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan,
                             make_fail_payload(str(e)), 0, str(e))

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\n=== Saving outputs ===")

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"specification_results.csv: {len(spec_df)} rows")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    print(f"inference_results.csv: {len(infer_df)} rows")
else:
    infer_df = pd.DataFrame()

# Summary stats
n_success = int(spec_df['run_success'].sum())
n_fail = len(spec_df) - n_success
n_g1 = len(spec_df[spec_df['baseline_group_id'] == 'G1'])
n_g2 = len(spec_df[spec_df['baseline_group_id'] == 'G2'])
print(f"\nTotal specs: {len(spec_df)} (G1: {n_g1}, G2: {n_g2})")
print(f"Successes: {n_success}, Failures: {n_fail}")
if len(infer_df) > 0:
    n_infer_success = int(infer_df['run_success'].sum())
    print(f"Inference variants: {len(infer_df)} ({n_infer_success} success)")

# Verify unique spec_run_ids
assert spec_df['spec_run_id'].is_unique, "spec_run_id not unique!"
print("spec_run_id uniqueness: OK")

# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================
print("\nWriting SPECIFICATION_SEARCH.md...")

successful = spec_df[spec_df['run_success'] == 1].copy()
failed = spec_df[spec_df['run_success'] == 0].copy()

md_lines = []
md_lines.append("# Specification Search Report: 149481-V1")
md_lines.append("")
md_lines.append("**Paper:** Samek & Longfield, \"Do Thank-You Calls Increase "
                "Charitable Giving? Expert Forecasts and Field Experimental "
                "Evidence\", AER")
md_lines.append("")
md_lines.append("## Design")
md_lines.append("")
md_lines.append("- **Design:** Randomized experiment (ITT)")
md_lines.append("- **G1 (Experiment 1):** Public TV stations; randomization within "
                "station x quarter strata")
md_lines.append("- **G2 (Experiment 2):** National non-profit; no stratification")
md_lines.append("- **Treatment:** Thank-you phone call (treat=1 vs control=0)")
md_lines.append("- **Focal outcomes:** renewing (binary), payment_amount3 (continuous)")
md_lines.append("")

# Baseline table
md_lines.append("## Baseline Specifications")
md_lines.append("")
md_lines.append("### G1 Baselines (Experiment 1)")
md_lines.append("")

g1_baselines = successful[
    (successful['baseline_group_id'] == 'G1') &
    (successful['spec_id'].str.startswith('baseline'))
]
if len(g1_baselines) > 0:
    md_lines.append("| Spec ID | Outcome | Coef | SE | p-value | N |")
    md_lines.append("|---------|---------|------|-----|---------|---|")
    for _, row in g1_baselines.iterrows():
        md_lines.append(
            f"| {row['spec_id']} | {row['outcome_var']} | "
            f"{row['coefficient']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {int(row['n_obs']) if not np.isnan(row['n_obs']) else 'NA'} |"
        )
    md_lines.append("")

md_lines.append("### G2 Baselines (Experiment 2)")
md_lines.append("")

g2_baselines = successful[
    (successful['baseline_group_id'] == 'G2') &
    (successful['spec_id'].str.startswith('baseline'))
]
if len(g2_baselines) > 0:
    md_lines.append("| Spec ID | Outcome | Coef | SE | p-value | N |")
    md_lines.append("|---------|---------|------|-----|---------|---|")
    for _, row in g2_baselines.iterrows():
        md_lines.append(
            f"| {row['spec_id']} | {row['outcome_var']} | "
            f"{row['coefficient']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {int(row['n_obs']) if not np.isnan(row['n_obs']) else 'NA'} |"
        )
    md_lines.append("")

# Specification counts
md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {n_success}")
md_lines.append(f"- Failed: {n_fail}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append(f"- G1 specs: {n_g1}")
md_lines.append(f"- G2 specs: {n_g2}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

categories = {
    "Baseline (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('baseline'))],
    "Baseline (G2)": successful[
        (successful['baseline_group_id'] == 'G2') &
        (successful['spec_id'].str.startswith('baseline'))],
    "Design Variants (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('design/'))],
    "Design Variants (G2)": successful[
        (successful['baseline_group_id'] == 'G2') &
        (successful['spec_id'].str.startswith('design/'))],
    "Controls LOO (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('rc/controls/loo/'))],
    "Controls Sets (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('rc/controls/sets/'))],
    "Controls LOO (G2)": successful[
        (successful['baseline_group_id'] == 'G2') &
        (successful['spec_id'].str.startswith('rc/controls/loo/'))],
    "Controls Sets (G2)": successful[
        (successful['baseline_group_id'] == 'G2') &
        (successful['spec_id'].str.startswith('rc/controls/sets/'))],
    "Sample (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('rc/sample/'))],
    "Sample (G2)": successful[
        (successful['baseline_group_id'] == 'G2') &
        (successful['spec_id'].str.startswith('rc/sample/'))],
    "Functional Form (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('rc/form/'))],
    "Functional Form (G2)": successful[
        (successful['baseline_group_id'] == 'G2') &
        (successful['spec_id'].str.startswith('rc/form/'))],
    "FE Variants (G1)": successful[
        (successful['baseline_group_id'] == 'G1') &
        (successful['spec_id'].str.startswith('rc/fe/'))],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = int((cat_df['p_value'] < 0.05).sum())
        coef_range = (f"[{cat_df['coefficient'].min():.4f}, "
                      f"{cat_df['coefficient'].max():.4f}]")
        md_lines.append(
            f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | "
            f"{coef_range} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Group | Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|-------|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(
                f"| {row['baseline_group_id']} | {row['spec_id']} | "
                f"{row['std_error']:.4f} | {row['p_value']:.4f} | "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |")
        else:
            md_lines.append(
                f"| {row['baseline_group_id']} | {row['spec_id']} | "
                f"FAILED | - | - |")
md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")

for group, group_label in [("G1", "Experiment 1"), ("G2", "Experiment 2")]:
    group_df = successful[successful['baseline_group_id'] == group]
    if len(group_df) > 0:
        n_sig_total = int((group_df['p_value'] < 0.05).sum())
        pct_sig = n_sig_total / len(group_df) * 100
        sign_pos = (group_df['coefficient'] > 0).sum()
        sign_neg = (group_df['coefficient'] < 0).sum()
        sign_consistent = (sign_pos == len(group_df)) or \
                          (sign_neg == len(group_df))
        median_coef = group_df['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"### {group_label} ({group})")
        md_lines.append("")
        md_lines.append(
            f"- **Sign consistency:** "
            f"{'All specifications have the same sign' if sign_consistent else f'{sign_pos} positive, {sign_neg} negative'}")
        md_lines.append(
            f"- **Significance stability:** {n_sig_total}/{len(group_df)} "
            f"({pct_sig:.1f}%) specifications significant at 5%")
        md_lines.append(
            f"- **Direction:** Median coefficient is {sign_word} "
            f"({median_coef:.4f})")

        if pct_sig >= 80 and sign_consistent:
            strength = "STRONG"
        elif pct_sig >= 50 and sign_consistent:
            strength = "MODERATE"
        elif pct_sig >= 50:
            strength = "MODERATE (mixed signs)"
        elif pct_sig >= 30:
            strength = "WEAK"
        else:
            strength = "FRAGILE"

        md_lines.append(f"- **Robustness assessment:** {strength}")
        md_lines.append("")

md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\n=== DONE ===")
