#!/usr/bin/env python3
"""
Specification search for paper 171681-V1
Ambuehl, Bernheim & Lusardi, "Evaluating Deliberative Competence"

Two baseline groups:
  G1: Compounding knowledge (score_compounding ~ Full, tag==1, Full/Control, sample=='old')
  G2: Financial competence (negAbsDiff ~ Control Full Rule72 Rhetoric, nocons, sample=='old')

Data management replicates Stata do-files from raw Qualtrics CSV.
"""

import sys, os, json, hashlib, traceback, warnings as _warnings
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
from scipy import stats

_warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────
REPO = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PKG  = os.path.join(REPO, "data/downloads/extracted/171681-V1")
RAW  = os.path.join(PKG, "data/raw")
OUT  = PKG

sys.path.insert(0, os.path.join(REPO, "scripts"))
from agent_output_utils import (
    surface_hash as compute_surface_hash,
    software_block,
    make_success_payload,
    make_failure_payload,
    error_details_from_exception,
)

with open(os.path.join(PKG, "SPECIFICATION_SURFACE.json")) as f:
    SURFACE = json.load(f)
SHASH      = compute_surface_hash(SURFACE)
SW         = software_block()
PAPER_ID   = "171681-V1"
CANON_INF  = {"spec_id": "infer/se/cluster/id", "params": {"cluster_var": "id"}}
DA_G1      = SURFACE["baseline_groups"][0]["design_audit"]
DA_G2      = SURFACE["baseline_groups"][1]["design_audit"]

def dblock(audit):
    return {"randomized_experiment": dict(audit)}

# ═══════════════════════════════════════════════════════════════════════════
#  DATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def stata_col_name(c):
    """Convert Qualtrics CSV header to what Stata's 'insheet, names' produces."""
    return c.lower().replace(': ', '').replace(':', '').replace(' ', '_').replace('#', '').replace('-', '')

def read_batch(fname, id_offset, recode=None, drop_excluded=True, drop_row0=True):
    """Read a raw CSV batch mimicking Stata's insheet behaviour."""
    fp = os.path.join(RAW, fname)
    df = pd.read_csv(fp, low_memory=False)
    # Rename columns Stata-style
    df.columns = [stata_col_name(c) for c in df.columns]
    # Drop row 0 (the extra Qualtrics text row — Stata 'drop if id==1')
    if drop_row0 and len(df) > 0:
        df = df.iloc[1:].reset_index(drop=True)
    # Generate id
    df['id'] = df.index + id_offset
    # Destring
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Rename treatment → fined
    if 'treatment' in df.columns:
        df.rename(columns={'treatment': 'fined'}, inplace=True)
    # Drop excluded column if present
    if drop_excluded and 'excluded' in df.columns:
        df.drop(columns=['excluded'], inplace=True, errors='ignore')
    # Recode fined for substance/rhetoric batches
    if recode == 'sr':
        mask1 = df['fined'] == 1
        mask0 = df['fined'] == 0
        df.loc[mask1, 'fined'] = 2   # substance only
        df.loc[mask0, 'fined'] = 3   # rhetoric only
    return df


def find_switch_points(df, impl_prefix_1, nlines_1, impl_prefix_2, nlines_2):
    """
    Given a long-format dataframe (id x line), compute switching points
    for the coarse (pricelist 1) and fine (pricelist 2) valuations
    for each of the 20 MPL treatments, per subject.

    Returns a dataframe at id level with value{1..20} and multi flag.
    """
    results = pd.DataFrame({'id': df['id'].unique()})

    multi_any = np.zeros(len(results), dtype=int)

    for impl_i in range(1, 21):
        # ── Coarse pricelist ──────────────────────────────────
        cols_coarse = [f'{impl_prefix_1}{impl_i}_{j}' for j in range(1, nlines_1+1)]
        cols_coarse = [c for c in cols_coarse if c in df.columns]

        if not cols_coarse:
            results[f'value_coarse_{impl_i}'] = np.nan
            results[f'multi_coarse_{impl_i}'] = 0
            continue

        # Pivot: for each id, get choices across lines
        pivot = df[['id'] + cols_coarse].drop_duplicates()
        # Group by id and compute switch point
        coarse_vals = []
        coarse_multi = []
        for _, grp in pivot.groupby('id'):
            if len(grp) == 0:
                coarse_vals.append(np.nan)
                coarse_multi.append(0)
                continue
            row = grp.iloc[0]
            choices = [row.get(c, np.nan) for c in cols_coarse]
            val, multi = _compute_switch(choices, is_coarse=True, nlines=nlines_1)
            coarse_vals.append(val)
            coarse_multi.append(multi)

        id_order = pivot.groupby('id').first().index
        r_coarse = pd.DataFrame({'id': id_order, f'value_coarse_{impl_i}': coarse_vals, f'multi_coarse_{impl_i}': coarse_multi})
        results = results.merge(r_coarse, on='id', how='left')

        # ── Fine pricelist ────────────────────────────────────
        cols_fine = [f'{impl_prefix_2}{impl_i}_{j}' for j in range(1, nlines_2+1)]
        cols_fine = [c for c in cols_fine if c in df.columns]

        if not cols_fine:
            results[f'value_fine_{impl_i}'] = np.nan
            results[f'multi_fine_{impl_i}'] = 0
            continue

        pivot_f = df[['id'] + cols_fine].drop_duplicates()
        fine_vals = []
        fine_multi = []
        for _, grp in pivot_f.groupby('id'):
            if len(grp) == 0:
                fine_vals.append(np.nan)
                fine_multi.append(0)
                continue
            row = grp.iloc[0]
            choices = [row.get(c, np.nan) for c in cols_fine]
            val, multi = _compute_switch(choices, is_coarse=False, nlines=nlines_2)
            fine_vals.append(val)
            fine_multi.append(multi)

        id_order_f = pivot_f.groupby('id').first().index
        r_fine = pd.DataFrame({'id': id_order_f, f'value_fine_{impl_i}': fine_vals, f'multi_fine_{impl_i}': fine_multi})
        results = results.merge(r_fine, on='id', how='left')

    # Combine coarse + fine
    for impl_i in range(1, 21):
        c_col = f'value_coarse_{impl_i}'
        f_col = f'value_fine_{impl_i}'
        if c_col in results.columns and f_col in results.columns:
            # Coarse value: 20 - 2*(switch_high - 1)
            # Fine value: already computed as 1.8 - 0.2*switch_low (capped at 0)
            results[f'value{impl_i}'] = results[c_col] + results[f_col]
        else:
            results[f'value{impl_i}'] = np.nan

    # Multi flag
    results['multi'] = 0
    for impl_i in range(1, 21):
        for prefix in ['multi_coarse_', 'multi_fine_']:
            c = f'{prefix}{impl_i}'
            if c in results.columns:
                results.loc[results[c] == 1, 'multi'] = 1

    # Also set multi=1 if sum of multi_coarse or sum of multi_fine == 1
    coarse_sum = results[[c for c in results.columns if c.startswith('multi_coarse_')]].sum(axis=1)
    fine_sum = results[[c for c in results.columns if c.startswith('multi_fine_')]].sum(axis=1)
    results.loc[coarse_sum == 1, 'multi'] = 1
    results.loc[fine_sum == 1, 'multi'] = 1

    # Keep only needed columns
    keep = ['id', 'multi'] + [f'value{i}' for i in range(1, 21)]
    return results[[c for c in keep if c in results.columns]]


def _compute_switch(choices, is_coarse, nlines):
    """
    Compute switching point from a list of choices (0=left, 1=right).
    Returns (value, multi_flag).
    For coarse: value = 20 - 2*(switch_high - 1)
    For fine: value = 1.8 - 0.2*switch_low, capped at 0
    """
    lines_left = []  # lines where chose left (0)
    lines_right = []  # lines where chose right (1)
    for j, c in enumerate(choices):
        line = j + 1
        if c == 0:
            lines_left.append(line)
        elif c == 1:
            lines_right.append(line)

    switch_low = max(lines_left) if lines_left else None
    switch_high = min(lines_right) if lines_right else None
    mll = max(lines_left) if lines_left else None
    mlr = min(lines_right) if lines_right else None

    # Multi-switch detection
    if switch_high is not None and switch_low is not None:
        multi = int(switch_high != 1 + switch_low)
    else:
        multi = 0

    if is_coarse:
        if switch_high == nlines or switch_high is None or switch_low == 1 or switch_low is None:
            multi = 0
    else:
        if (switch_high == 1 and switch_low is None) or (switch_high is None and switch_low == nlines):
            multi = 0

    if multi == 1:
        return np.nan, 1

    if is_coarse:
        val = switch_high
        if mlr == 1:
            val = 0
        if mll == nlines:
            val = nlines + 1
        if val is None:
            return np.nan, 0
        value = 20 - 2 * (val - 1)
    else:
        val = switch_low
        if mlr == 1:
            val = 0
        if mll == nlines:
            val = nlines
        if val is None:
            return np.nan, 0
        value = 1.8 - 0.2 * val
        if value < 0:
            value = 0

    return value, 0


def build_expA():
    """Build Experiment A managed dataset from raw CSV files."""
    batches = [
        ("expAbatch1.csv", 1000, None, False),  # batch 1: no drop excluded, no recode
        ("expAbatch2.csv", 2000, None, True),
        ("expAbatch3.csv", 3000, 'sr', True),
        ("expAbatch4.csv", 4000, None, True),
        ("expAbatch5.csv", 5000, None, True),
        ("expAbatch6.csv", 6000, 'sr', True),
        ("expAbatch7.csv", 7000, 'sr', True),
        ("expAbatch8.csv", 8000, 'sr', True),
    ]

    dfs = []
    for fname, offset, recode, drop_exc in batches:
        df = read_batch(fname, offset, recode=recode, drop_excluded=drop_exc)
        dfs.append(df)

    # Common columns
    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    common = sorted(common)

    df_all = pd.concat([d[common] for d in dfs], ignore_index=True)

    # Drop attrited (help_test == .)
    df_all = df_all.dropna(subset=['help_test']).reset_index(drop=True)

    # ── Extract non-MPL individual data ───────────────────────────
    non_mpl_candidates = ['id', 'gender', 'age', 'ethnicity', 'educ', 'marital', 'urban',
                          'income', 'employed', 'hh_size', 'stocks', 'fined', 'help_test',
                          'payment_for_test', 'sc0_0',
                          'finlit1', 'finlit2', 'finlit3', 'finlit4', 'finlit5',
                          'test1', 'test2', 'test3', 'test4', 'test5',
                          'test6', 'test7', 'test8', 'test9', 'test10',
                          'rule72', 'rule72_inv', 'calculate']
    non_mpl_cols = [c for c in non_mpl_candidates if c in df_all.columns]
    df_non_mpl = df_all[non_mpl_cols].copy()

    # ── MPL switch points ─────────────────────────────────────────
    # Stata renames: impl{i}_1_11 -> impl{i}_1 (line 1), impl{i}_1_21 -> impl{i}_2, etc.
    # Coarse pricelist 1: columns impl{i}_1_11, impl{i}_1_21, ..., impl{i}_1_111  (lines 1-11)
    # Fine pricelist 2: columns impl{i}_2_31, impl{i}_2_41, ..., impl{i}_2_121  (lines 1-10)

    # After Stata rename, coarse line j maps to column impl{i}_1_{j}1
    # In our Stata-named columns: impl{i}_1_11 = line 1, impl{i}_1_21 = line 2, etc.
    # So column for impl i, coarse, line j is: impl{i}_1_{j}1

    # Build a per-id dataset of switch points
    switch_data = pd.DataFrame({'id': df_all['id'].unique()})
    switch_data['multi'] = 0

    for impl_i in range(1, 21):
        # Coarse: impl{i}_1_{j}1 for j=1..11
        coarse_cols = []
        for j in range(1, 12):
            cname = f'impl{impl_i}_1_{j}1'
            if cname in df_all.columns:
                coarse_cols.append(cname)

        # Fine: impl{impl_i}_2_{j}1 for j=3..12
        fine_cols = []
        for j in range(3, 13):
            cname = f'impl{impl_i}_2_{j}1'
            if cname in df_all.columns:
                fine_cols.append(cname)

        # Compute per-individual switch points vectorized
        if coarse_cols:
            coarse_data = df_all[['id'] + coarse_cols].drop_duplicates(subset=['id'])
            coarse_arr = coarse_data[coarse_cols].values  # (n_subjects, 11)

            coarse_vals = np.full(len(coarse_data), np.nan)
            coarse_multi = np.zeros(len(coarse_data), dtype=int)

            for idx in range(len(coarse_data)):
                choices = coarse_arr[idx]
                val, m = _compute_switch(choices.tolist(), is_coarse=True, nlines=11)
                coarse_vals[idx] = val
                coarse_multi[idx] = m

            coarse_result = pd.DataFrame({
                'id': coarse_data['id'].values,
                f'coarse_{impl_i}': coarse_vals,
                f'multi_c_{impl_i}': coarse_multi,
            })
            switch_data = switch_data.merge(coarse_result, on='id', how='left')
        else:
            switch_data[f'coarse_{impl_i}'] = np.nan
            switch_data[f'multi_c_{impl_i}'] = 0

        if fine_cols:
            fine_data = df_all[['id'] + fine_cols].drop_duplicates(subset=['id'])
            fine_arr = fine_data[fine_cols].values

            fine_vals = np.full(len(fine_data), np.nan)
            fine_multi = np.zeros(len(fine_data), dtype=int)

            for idx in range(len(fine_data)):
                choices = fine_arr[idx]
                val, m = _compute_switch(choices.tolist(), is_coarse=False, nlines=10)
                fine_vals[idx] = val
                fine_multi[idx] = m

            fine_result = pd.DataFrame({
                'id': fine_data['id'].values,
                f'fine_{impl_i}': fine_vals,
                f'multi_f_{impl_i}': fine_multi,
            })
            switch_data = switch_data.merge(fine_result, on='id', how='left')
        else:
            switch_data[f'fine_{impl_i}'] = np.nan
            switch_data[f'multi_f_{impl_i}'] = 0

    # Combine coarse + fine
    for impl_i in range(1, 21):
        switch_data[f'value{impl_i}'] = switch_data[f'coarse_{impl_i}'] + switch_data[f'fine_{impl_i}']

    # Multi flag
    for impl_i in range(1, 21):
        switch_data.loc[switch_data[f'multi_c_{impl_i}'] == 1, 'multi'] = 1
        switch_data.loc[switch_data[f'multi_f_{impl_i}'] == 1, 'multi'] = 1

    mc_sum = switch_data[[f'multi_c_{i}' for i in range(1, 21)]].sum(axis=1)
    mf_sum = switch_data[[f'multi_f_{i}' for i in range(1, 21)]].sum(axis=1)
    switch_data.loc[mc_sum == 1, 'multi'] = 1
    switch_data.loc[mf_sum == 1, 'multi'] = 1

    keep = ['id', 'multi'] + [f'value{i}' for i in range(1, 21)]
    switch_data = switch_data[keep]

    # ── Merge and reshape ─────────────────────────────────────────
    df_merged = switch_data.merge(df_non_mpl, on='id', how='inner')

    # Reshape long
    value_cols = [f'value{i}' for i in range(1, 21)]
    id_cols = [c for c in df_merged.columns if c not in value_cols]
    df_long = df_merged.melt(id_vars=id_cols, value_vars=value_cols,
                              var_name='tr_str', value_name='value')
    df_long['treatment'] = df_long['tr_str'].str.replace('value', '').astype(int)
    df_long.drop(columns=['tr_str'], inplace=True)

    # Framed / delay / amount
    df_long['framed'] = (df_long['treatment'] > 10).astype(int)
    df_long['delay'] = 72
    df_long.loc[(df_long['treatment'] >= 6) & (df_long['treatment'] <= 10), 'delay'] = 36
    df_long.loc[df_long['treatment'].isin([11, 13, 15, 17, 19]), 'delay'] = 36

    df_long['amount'] = 20
    df_long.loc[df_long['treatment'].isin([2, 7, 14, 13]), 'amount'] = 18
    df_long.loc[df_long['treatment'].isin([3, 8, 16, 15]), 'amount'] = 16
    df_long.loc[df_long['treatment'].isin([4, 9, 18, 17]), 'amount'] = 14
    df_long.loc[df_long['treatment'].isin([5, 10, 20, 19]), 'amount'] = 12
    df_long['amount_precise'] = df_long['amount'].astype(float)

    # Rename treatments
    remap = {6:1, 1:2, 7:3, 2:4, 8:5, 3:6, 9:7, 4:8, 10:9, 5:10}
    df_long['tr_old'] = df_long['treatment']
    new_t = df_long['treatment'].map(remap)
    df_long.loc[new_t.notna(), 'treatment'] = new_t[new_t.notna()].astype(int)
    df_long.loc[df_long['framed'] == 1, 'treatment'] -= 10

    precise = {1: 20.40, 2: 20.47, 3: 18.47, 4: 18.73, 5: 16.29,
               6: 16.80, 7: 14.37, 8: 15.16, 9: 11.58, 10: 11.83}
    for tr, ap in precise.items():
        df_long.loc[(df_long['treatment'] == tr) | (df_long['treatment'] == tr+10), 'amount_precise'] = ap

    df_long['id_alt'] = 1000 * df_long['id'] + df_long['treatment']

    # Reshape wide (unframed/framed)
    df_uf = df_long[df_long['framed'] == 0].copy().rename(columns={'value': 'v_unframed'})
    df_fr = df_long[df_long['framed'] == 1][['id_alt', 'value']].copy().rename(columns={'value': 'v_framed'})
    keep_uf = [c for c in df_uf.columns if c not in ['framed', 'tr_old']]
    df_wide = df_uf[keep_uf].merge(df_fr, on='id_alt', how='outer')

    # ── Variable construction ─────────────────────────────────────
    # Test scores (Exp A correct answers)
    correct_A = {1: 2, 2: 4, 3: 4, 4: 7, 5: 6, 6: 3, 7: 4, 8: 2, 9: 3, 10: 3}
    for i in range(1, 11):
        tc = f'test{i}'
        df_wide[f't{i}'] = (df_wide[tc] == correct_A[i]).astype(int) if tc in df_wide.columns else 0

    df_wide['score_compounding'] = sum(df_wide[f't{i}'] for i in range(1, 6))
    if 'sc0_0' in df_wide.columns:
        df_wide['score'] = df_wide['sc0_0']
    else:
        df_wide['score'] = sum(df_wide[f't{i}'] for i in range(1, 11))
    df_wide['score_indexing'] = df_wide['score'] - df_wide['score_compounding']

    # Financial literacy
    if 'finlit1' in df_wide.columns:
        df_wide['fl1'] = (df_wide['finlit1'] == 1).astype(int)
        df_wide['fl2'] = (df_wide['finlit2'] == 1).astype(int)
        df_wide['fl3'] = (df_wide['finlit3'] == 3).astype(int)
        df_wide['fl4'] = (df_wide['finlit4'] == 1).astype(int)
        df_wide['fl5'] = (df_wide['finlit5'] == 2).astype(int)
        df_wide['fl_score_compound'] = df_wide['fl1'] * df_wide['fl2'] * df_wide['fl3']
        df_wide['fl_sum_compound'] = df_wide['fl1'] + df_wide['fl2'] + df_wide['fl3']

    df_wide['age'] += 17
    df_wide['v_framed'] += 0.1
    df_wide['v_unframed'] += 0.1
    df_wide.loc[df_wide['v_framed'] <= 0, 'v_framed'] = 0
    df_wide.loc[df_wide['v_unframed'] <= 0, 'v_unframed'] = 0

    df_wide['fl_high'] = (df_wide['fl_score_compound'] == 1).astype(int)
    df_wide['discount_unframed'] = df_wide['v_unframed'] / df_wide['amount'] * 100
    df_wide['discount_framed'] = df_wide['v_framed'] / df_wide['amount_precise'] * 100
    df_wide['diff'] = df_wide['discount_framed'] - df_wide['discount_unframed']
    df_wide['absDiff'] = df_wide['diff'].abs()
    df_wide['sqDiff'] = (df_wide['discount_framed']**2 - df_wide['discount_unframed']**2).abs() / 100

    df_wide['Control']  = (df_wide['fined'] == 0).astype(int)
    df_wide['Full']     = (df_wide['fined'] == 1).astype(int)
    df_wide['Rule72']   = (df_wide['fined'] == 2).astype(int)
    df_wide['Rhetoric'] = (df_wide['fined'] == 3).astype(int)

    # Demographics
    for name, cond in [('afram',1),('asian',2),('caucasian',3),('hispanic',4),('other',5)]:
        df_wide[name] = (df_wide['ethnicity'] == cond).astype(int)
    df_wide['lessThanHighSchool'] = (df_wide['educ'] <= 2).astype(int)
    df_wide['highSchool'] = (df_wide['educ'] == 3).astype(int)
    df_wide['voc'] = (df_wide['educ'] == 4).astype(int)
    df_wide['someCollege'] = (df_wide['educ'] == 5).astype(int)
    df_wide['college'] = (df_wide['educ'] == 6).astype(int)
    df_wide['graduate'] = (df_wide['educ'] >= 7).astype(int)
    df_wide['rural'] = (df_wide['urban'] == 3).astype(int)
    df_wide['urbanSuburban'] = (df_wide['urban'] <= 2).astype(int)
    df_wide['fullTime'] = (df_wide['employed'] == 3).astype(int)
    df_wide['partTime'] = (df_wide['employed'] == 2).astype(int)
    for k in [1,2,3]:
        df_wide[f'hh{k}'] = (df_wide['hh_size'] == k).astype(int)
    df_wide['hh4'] = (df_wide['hh_size'] >= 4).astype(int)
    df_wide['married'] = df_wide['marital'].isin([3, 4]).astype(int)
    df_wide['widowed'] = (df_wide['marital'] == 6).astype(int)
    df_wide['divorced'] = (df_wide['marital'] == 1).astype(int)
    df_wide['never_married'] = df_wide['marital'].isin([2, 5]).astype(int)
    df_wide['ownStocks'] = (df_wide['stocks'] == 1).astype(int)
    df_wide['income'] /= 1000
    df_wide['tag'] = (df_wide.groupby('id').cumcount() == 0).astype(int)
    df_wide['sample'] = 'old'

    return df_wide


def build_expB():
    """Build Experiment B managed dataset."""
    dfs = []
    for batch_num, fname in [(1, 'expBbatch1.csv'), (2, 'expBbatch2.csv')]:
        fp = os.path.join(RAW, fname)
        df = pd.read_csv(fp, low_memory=False)
        df.columns = [stata_col_name(c) for c in df.columns]
        # Drop first 2 rows (Qualtrics headers), then drop row 0 again
        df = df.iloc[2:].reset_index(drop=True)
        if len(df) > 0:
            df = df.iloc[1:].reset_index(drop=True)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['batch'] = batch_num
        df['id'] = df.index + 1 + 1000 * batch_num
        if 'treatment' in df.columns:
            df.rename(columns={'treatment': 'fined'}, inplace=True)
        dfs.append(df)

    common = sorted(set(dfs[0].columns) & set(dfs[1].columns))
    df_B = pd.concat([d[common] for d in dfs], ignore_index=True)

    # Test scores (Exp B correct answers for s6-s10 differ)
    correct_B = {1: 2, 2: 4, 3: 4, 4: 7, 5: 6, 6: 4, 7: 5, 8: 5, 9: 5, 10: 1}
    for i in range(1, 11):
        tc = f'test{i}'
        df_B[f's{i}'] = (df_B[tc] == correct_B[i]).astype(int) if tc in df_B.columns else 0
    df_B['score'] = sum(df_B[f's{i}'] for i in range(1, 11))

    # Drop attriters
    df_B = df_B.dropna(subset=['test10']).reset_index(drop=True)

    # Non-MPL
    non_mpl_B = ['id', 'gender', 'age', 'ethnicity', 'educ', 'marital', 'urban',
                 'income', 'employed', 'hh_size', 'stocks', 'fined', 'help_test',
                 'payment_for_test', 'score',
                 'finlit1', 'finlit2', 'finlit3', 'finlit4', 'finlit5',
                 'test1', 'test2', 'test3', 'test4', 'test5',
                 'test6', 'test7', 'test8', 'test9', 'test10',
                 'rule72', 'rule72_inv', 'calculate', 'durationinseconds']
    non_mpl_B = [c for c in non_mpl_B if c in df_B.columns]
    df_non_mpl_B = df_B[non_mpl_B].copy()

    # MPL switch points for Exp B
    # Column naming: impl{i}11_impl{i}1_{1..11} for coarse
    # In Stata-cleaned form: impl{i}11_impl{i}1_{j}
    switch_B = pd.DataFrame({'id': df_B['id'].unique()})
    switch_B['multi'] = 0

    for impl_i in range(1, 21):
        # Coarse: impl{i}11_impl{i}1_{1..11}
        coarse_cols = []
        for j in range(1, 12):
            cname = f'impl{impl_i}11_impl{impl_i}1_{j}'
            if cname in df_B.columns:
                coarse_cols.append(cname)

        if coarse_cols:
            cd = df_B[['id'] + coarse_cols].drop_duplicates(subset=['id'])
            vals, mults = [], []
            for _, row in cd.iterrows():
                choices = [row.get(c, np.nan) for c in coarse_cols]
                v, m = _compute_switch(choices, is_coarse=True, nlines=11)
                vals.append(v)
                mults.append(m)
            r = pd.DataFrame({'id': cd['id'].values, f'coarse_{impl_i}': vals, f'mc_{impl_i}': mults})
            switch_B = switch_B.merge(r, on='id', how='left')
        else:
            switch_B[f'coarse_{impl_i}'] = np.nan
            switch_B[f'mc_{impl_i}'] = 0

        # Fine: impl{i}21_impl{i}2_{3..12}
        fine_cols = []
        for j in range(3, 13):
            cname = f'impl{impl_i}21_impl{impl_i}2_{j}'
            if cname in df_B.columns:
                fine_cols.append(cname)

        if fine_cols:
            fd = df_B[['id'] + fine_cols].drop_duplicates(subset=['id'])
            vals, mults = [], []
            for _, row in fd.iterrows():
                choices = [row.get(c, np.nan) for c in fine_cols]
                v, m = _compute_switch(choices, is_coarse=False, nlines=10)
                vals.append(v)
                mults.append(m)
            r = pd.DataFrame({'id': fd['id'].values, f'fine_{impl_i}': vals, f'mf_{impl_i}': mults})
            switch_B = switch_B.merge(r, on='id', how='left')
        else:
            switch_B[f'fine_{impl_i}'] = np.nan
            switch_B[f'mf_{impl_i}'] = 0

    for impl_i in range(1, 21):
        switch_B[f'value{impl_i}'] = switch_B.get(f'coarse_{impl_i}', np.nan) + switch_B.get(f'fine_{impl_i}', np.nan)

    for impl_i in range(1, 21):
        for pref in [f'mc_{impl_i}', f'mf_{impl_i}']:
            if pref in switch_B.columns:
                switch_B.loc[switch_B[pref] == 1, 'multi'] = 1
    mc_sum = switch_B[[f'mc_{i}' for i in range(1, 21) if f'mc_{i}' in switch_B.columns]].sum(axis=1)
    mf_sum = switch_B[[f'mf_{i}' for i in range(1, 21) if f'mf_{i}' in switch_B.columns]].sum(axis=1)
    switch_B.loc[mc_sum == 1, 'multi'] = 1
    switch_B.loc[mf_sum == 1, 'multi'] = 1

    keep = ['id', 'multi'] + [f'value{i}' for i in range(1, 21)]
    switch_B = switch_B[[c for c in keep if c in switch_B.columns]]

    df_merged = switch_B.merge(df_non_mpl_B, on='id', how='inner')

    # Reshape long
    value_cols = [f'value{i}' for i in range(1, 21) if f'value{i}' in df_merged.columns]
    id_cols = [c for c in df_merged.columns if c not in value_cols]
    df_long = df_merged.melt(id_vars=id_cols, value_vars=value_cols,
                              var_name='tr_str', value_name='value')
    df_long['treatment'] = df_long['tr_str'].str.replace('value', '').astype(int)
    df_long.drop(columns=['tr_str'], inplace=True)

    df_long['framed'] = (df_long['treatment'] > 10).astype(int)
    df_long['delay'] = 72
    df_long.loc[(df_long['treatment'] >= 6) & (df_long['treatment'] <= 10), 'delay'] = 36
    df_long.loc[df_long['treatment'].isin([11, 13, 15, 17, 19]), 'delay'] = 36
    df_long['amount'] = 20
    df_long.loc[df_long['treatment'].isin([2, 7, 14, 13]), 'amount'] = 18
    df_long.loc[df_long['treatment'].isin([3, 8, 16, 15]), 'amount'] = 16
    df_long.loc[df_long['treatment'].isin([4, 9, 18, 17]), 'amount'] = 14
    df_long.loc[df_long['treatment'].isin([5, 10, 20, 19]), 'amount'] = 12
    df_long['amount_precise'] = df_long['amount'].astype(float)

    remap = {6:1, 1:2, 7:3, 2:4, 8:5, 3:6, 9:7, 4:8, 10:9, 5:10}
    new_t = df_long['treatment'].map(remap)
    df_long.loc[new_t.notna(), 'treatment'] = new_t[new_t.notna()].astype(int)
    df_long.loc[df_long['framed'] == 1, 'treatment'] -= 10

    precise = {1: 20.40, 2: 20.47, 3: 18.47, 4: 18.73, 5: 16.29,
               6: 16.80, 7: 14.37, 8: 15.16, 9: 11.58, 10: 11.83}
    for tr, ap in precise.items():
        df_long.loc[(df_long['treatment'] == tr) | (df_long['treatment'] == tr+10), 'amount_precise'] = ap

    df_long['id_alt'] = 1000 * df_long['id'] + df_long['treatment']

    df_uf = df_long[df_long['framed'] == 0].copy().rename(columns={'value': 'v_unframed'})
    df_fr = df_long[df_long['framed'] == 1][['id_alt', 'value']].copy().rename(columns={'value': 'v_framed'})
    keep_uf = [c for c in df_uf.columns if c not in ['framed']]
    df_wide = df_uf[keep_uf].merge(df_fr, on='id_alt', how='outer')

    # Scores for Exp B (different correct answers for t6-t10)
    correct_B_t = {1: 2, 2: 4, 3: 4, 4: 7, 5: 6, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5}
    for i in range(1, 11):
        tc = f'test{i}'
        df_wide[f't{i}'] = (df_wide[tc] == correct_B_t[i]).astype(int) if tc in df_wide.columns else 0

    df_wide['score_compounding'] = sum(df_wide[f't{i}'] for i in range(1, 6))
    df_wide['score_indexing'] = df_wide['score'] - df_wide['score_compounding']

    if 'finlit1' in df_wide.columns:
        df_wide['fl1'] = (df_wide['finlit1'] == 1).astype(int)
        df_wide['fl2'] = (df_wide['finlit2'] == 1).astype(int)
        df_wide['fl3'] = (df_wide['finlit3'] == 3).astype(int)
        df_wide['fl4'] = (df_wide['finlit4'] == 1).astype(int)
        df_wide['fl5'] = (df_wide['finlit5'] == 2).astype(int)
        df_wide['fl_score_compound'] = df_wide['fl1'] * df_wide['fl2'] * df_wide['fl3']
        df_wide['fl_sum_compound'] = df_wide['fl1'] + df_wide['fl2'] + df_wide['fl3']

    df_wide['age'] += 17
    df_wide['v_framed'] += 0.1
    df_wide['v_unframed'] += 0.1
    df_wide.loc[df_wide['v_framed'] <= 0, 'v_framed'] = 0
    df_wide.loc[df_wide['v_unframed'] <= 0, 'v_unframed'] = 0

    df_wide['fl_high'] = (df_wide['fl_score_compound'] == 1).astype(int)
    df_wide['discount_unframed'] = df_wide['v_unframed'] / df_wide['amount'] * 100
    df_wide['discount_framed'] = df_wide['v_framed'] / df_wide['amount_precise'] * 100
    df_wide['diff'] = df_wide['discount_framed'] - df_wide['discount_unframed']
    df_wide['absDiff'] = df_wide['diff'].abs()
    df_wide['sqDiff'] = (df_wide['discount_framed']**2 - df_wide['discount_unframed']**2).abs() / 100

    df_wide['contNew'] = (df_wide['fined'] == 0).astype(int)
    df_wide['fullNew'] = (df_wide['fined'] == 1).astype(int)
    df_wide['Control'] = 0
    df_wide['Full'] = 0
    df_wide['Rule72'] = 0
    df_wide['Rhetoric'] = 0

    for name, cond in [('afram',1),('asian',2),('caucasian',3),('hispanic',4),('other',5)]:
        df_wide[name] = (df_wide['ethnicity'] == cond).astype(int)
    df_wide['lessThanHighSchool'] = (df_wide['educ'] <= 2).astype(int)
    df_wide['highSchool'] = (df_wide['educ'] == 3).astype(int)
    df_wide['voc'] = (df_wide['educ'] == 4).astype(int)
    df_wide['someCollege'] = (df_wide['educ'] == 5).astype(int)
    df_wide['college'] = (df_wide['educ'] == 6).astype(int)
    df_wide['graduate'] = (df_wide['educ'] >= 7).astype(int)
    df_wide['rural'] = (df_wide['urban'] == 3).astype(int)
    df_wide['urbanSuburban'] = (df_wide['urban'] <= 2).astype(int)
    df_wide['fullTime'] = (df_wide['employed'] == 3).astype(int)
    df_wide['partTime'] = (df_wide['employed'] == 2).astype(int)
    for k in [1,2,3]:
        df_wide[f'hh{k}'] = (df_wide['hh_size'] == k).astype(int)
    df_wide['hh4'] = (df_wide['hh_size'] >= 4).astype(int)
    df_wide['married'] = df_wide['marital'].isin([3, 4]).astype(int)
    df_wide['widowed'] = (df_wide['marital'] == 6).astype(int)
    df_wide['divorced'] = (df_wide['marital'] == 1).astype(int)
    df_wide['never_married'] = df_wide['marital'].isin([2, 5]).astype(int)
    df_wide['ownStocks'] = (df_wide['stocks'] == 1).astype(int)
    df_wide['tag'] = (df_wide.groupby('id').cumcount() == 0).astype(int)
    df_wide['sample'] = 'new'

    return df_wide


def combine_experiments(df_A, df_B):
    """Combine Exp A and Exp B, matching analysis.do data management."""
    df_A['fullNew'] = 0
    df_A['contNew'] = 0

    common = sorted(set(df_A.columns) & set(df_B.columns))
    df = pd.concat([df_B[common], df_A[common]], ignore_index=True)

    for c in ['Full', 'Control', 'Rule72', 'Rhetoric', 'fullNew', 'contNew']:
        if c not in df.columns:
            df[c] = 0

    df.loc[df['sample'] == 'new', ['Full', 'Control', 'Rule72', 'Rhetoric']] = 0
    df.loc[df['sample'] == 'old', ['fullNew', 'contNew']] = 0

    df['negAbsDiff'] = -df['absDiff']
    df['negSqDiff'] = -df['sqDiff']

    # Individual-level means
    df['vSimple'] = df.groupby('id')['discount_unframed'].transform('mean')
    df['vComplex'] = df.groupby('id')['discount_framed'].transform('mean')
    df['dRaw'] = df['vComplex'] - df['vSimple']
    df['dAbs'] = df['dRaw'].abs()

    # Re-id
    df.loc[df['sample'] == 'old', 'id'] = 10000 + df.loc[df['sample'] == 'old', 'id']
    df.loc[df['sample'] == 'new', 'id'] = 20000 + df.loc[df['sample'] == 'new', 'id']

    df = df.sort_values('id').reset_index(drop=True)
    df['tag'] = (df.groupby('id').cumcount() == 0).astype(int)

    # Keep single-switchers only
    df = df[df['multi'] == 0].reset_index(drop=True)

    # meanVsimple
    for dv in [36, 72]:
        mask = df['delay'] == dv
        df.loc[mask, f'mv_{dv}_i'] = df.loc[mask].groupby('id')['discount_unframed'].transform('mean')
    for dv in [36, 72]:
        df[f'meanVsimple{dv}'] = df.groupby('id')[f'mv_{dv}_i'].transform('mean') / 100
    df['meanVsimple'] = 100 * (df['meanVsimple36'] + df['meanVsimple72']) / 2

    # Income fix
    df.loc[df['income'] < 0, 'income'] = np.nan
    df.loc[df['sample'] == 'old', 'income'] = df.loc[df['sample'] == 'old', 'income'] * 1000

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

FULL_CTRLS = ['fl1','fl2','fl3','fl4','fl5','gender','age','income',
              'afram','asian','caucasian','hispanic','other',
              'lessThanHighSchool','highSchool','voc','someCollege','college','graduate',
              'fullTime','partTime','married','widowed','divorced','never_married',
              'rural','urbanSuburban','hh1','hh2','hh3','hh4','ownStocks']
MINIMAL_CTRLS = ['gender','age','income','caucasian','college']
EXTENDED_CTRLS = MINIMAL_CTRLS + ['fl1','fl2','fl3','married','fullTime','ownStocks','urbanSuburban']

SPEC_ROWS = []
INF_ROWS = []
_rc, _ic = [0], [0]

def _rid():
    _rc[0] += 1
    return f"{PAPER_ID}__run_{_rc[0]:04d}"

def _iid():
    _ic[0] += 1
    return f"{PAPER_ID}__infer_{_ic[0]:04d}"


def ols_cl(df, y, xvars, cl='id', ctrls=None):
    """OLS with intercept + cluster-robust SEs via pyfixest."""
    all_x = list(xvars) + (ctrls or [])
    fml = f"{y} ~ " + " + ".join(all_x)
    needed = [y] + all_x + [cl]
    dc = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    m = pf.feols(fml, data=dc, vcov={"CRV1": cl})
    return m


def ols_nocons_cl(df, y, arms, cl='id', ctrls=None):
    """OLS without constant + cluster-robust SEs via statsmodels."""
    all_x = list(arms) + (ctrls or [])
    needed = [y] + all_x + [cl]
    dc = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    X = dc[all_x].values.astype(float)
    yy = dc[y].values.astype(float)
    groups = dc[cl].values
    m = sm.OLS(yy, X).fit(cov_type='cluster', cov_kwds={'groups': groups})
    return m, all_x, dc


def wald_eq(m, vnames, v1, v2):
    """Wald test v1 == v2."""
    i1, i2 = vnames.index(v1), vnames.index(v2)
    d = m.params[i1] - m.params[i2]
    cov = m.cov_params()
    se = np.sqrt(cov[i1,i1] + cov[i2,i2] - 2*cov[i1,i2])
    t = d / se
    p = 2 * stats.t.sf(abs(t), m.df_resid)
    return d, se, p


def add_row(**kw):
    kw.setdefault('paper_id', PAPER_ID)
    kw.setdefault('ci_lower', np.nan)
    kw.setdefault('ci_upper', np.nan)
    kw.setdefault('fixed_effects', '')
    kw.setdefault('cluster_var', 'id')
    kw.setdefault('run_success', 1)
    kw.setdefault('run_error', '')
    if isinstance(kw.get('coefficient_vector_json'), dict):
        kw['coefficient_vector_json'] = json.dumps(kw['coefficient_vector_json'])
    SPEC_ROWS.append(kw)

def add_infer(**kw):
    kw.setdefault('paper_id', PAPER_ID)
    kw.setdefault('ci_lower', np.nan)
    kw.setdefault('ci_upper', np.nan)
    kw.setdefault('run_success', 1)
    kw.setdefault('run_error', '')
    if isinstance(kw.get('coefficient_vector_json'), dict):
        kw['coefficient_vector_json'] = json.dumps(kw['coefficient_vector_json'])
    INF_ROWS.append(kw)


def safe_run(fn, spec_id, group, tree_path, y, t, **extra_meta):
    """Execute fn(); on failure record a failure row."""
    try:
        fn()
    except Exception as e:
        rid = _rid()
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=f"{group} {spec_id}"))
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id=group,
                spec_tree_path=tree_path, outcome_var=y, treatment_var=t,
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, sample_desc=extra_meta.get('sample_desc',''),
                controls_desc=extra_meta.get('controls_desc',''),
                coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])


# ═══════════════════════════════════════════════════════════════════════════
#  G1 SPECS
# ═══════════════════════════════════════════════════════════════════════════

def run_g1_intercept(df, spec_id, filt, treat, tree, controls=None,
                     sample_desc='', controls_desc='none', **extra_blocks):
    """G1 spec: OLS with intercept, focal coef on treat."""
    rid = _rid()
    try:
        dc = filt(df).copy()
        m = ols_cl(dc, 'score_compounding', [treat], ctrls=controls)
        c = float(m.coef().loc[treat])
        s = float(m.se().loc[treat])
        p = float(m.pvalue().loc[treat])
        ci = m.confint().loc[treat]
        coefs = {k: float(v) for k, v in m.coef().items()}
        pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G1), **extra_blocks)
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path=tree, outcome_var='score_compounding', treatment_var=treat,
                coefficient=c, std_error=s, p_value=p, ci_lower=float(ci.iloc[0]),
                ci_upper=float(ci.iloc[1]), n_obs=m._N, r_squared=m._r2,
                sample_desc=sample_desc, controls_desc=controls_desc,
                coefficient_vector_json=pld)
        return rid, c, s, p
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=f'G1 {spec_id}'))
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path=tree, outcome_var='score_compounding', treatment_var=treat,
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, sample_desc=sample_desc,
                controls_desc=controls_desc, coefficient_vector_json=pld,
                run_success=0, run_error=str(e)[:240])
        return rid, np.nan, np.nan, np.nan


def run_g1_nocons(df, spec_id, filt, arms, tree, focal1='Full', focal2='Control',
                  sample_desc='', controls_desc='none', controls=None, **extra_blocks):
    """G1 spec: OLS without constant, Wald test focal1==focal2."""
    rid = _rid()
    try:
        dc = filt(df).copy()
        m, vn, _ = ols_nocons_cl(dc, 'score_compounding', arms, ctrls=controls)
        d, se, p = wald_eq(m, vn, focal1, focal2)
        coefs = {vn[i]: float(m.params[i]) for i in range(len(vn))}
        pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G1),
                                    extra={"focal_test": f"{focal1}=={focal2}", "model_form": "nocons"},
                                    **extra_blocks)
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path=tree, outcome_var='score_compounding',
                treatment_var=','.join(arms), coefficient=d, std_error=se, p_value=p,
                n_obs=int(m.nobs), r_squared=float(m.rsquared),
                sample_desc=sample_desc, controls_desc=controls_desc,
                coefficient_vector_json=pld)
        return rid
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=f'G1 {spec_id}'))
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path=tree, outcome_var='score_compounding',
                treatment_var=','.join(arms), coefficient=np.nan, std_error=np.nan,
                p_value=np.nan, n_obs=np.nan, r_squared=np.nan,
                sample_desc=sample_desc, controls_desc=controls_desc,
                coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])
        return rid


def run_all_g1(df):
    fA = lambda d: d[(d['tag']==1)&((d['Full']==1)|(d['Control']==1))&(d['sample']=='old')]
    fB = lambda d: _prep_expB(d[(d['tag']==1)&((d['fullNew']==1)|(d['contNew']==1))&(d['sample']=='new')])
    fPool = lambda d: _prep_pooled(d[(d['tag']==1)&((d['Full']==1)|(d['Control']==1)|(d['fullNew']==1)|(d['contNew']==1))])

    # Baselines
    bl_rid, *_ = run_g1_intercept(df, 'baseline', fA, 'Full',
        'designs/randomized_experiment.md#ols-itt', sample_desc="tag==1, Full|Control, sample=='old'")
    run_g1_intercept(df, 'baseline__table4_col2_expB', fB, 'Full',
        'designs/randomized_experiment.md#ols-itt', sample_desc="tag==1, fullNew|contNew, sample=='new'")

    # Design
    # diff in means (Welch t-test)
    rid = _rid()
    try:
        dc = fA(df).copy()
        t_vals = dc.loc[dc['Full']==1, 'score_compounding'].dropna()
        c_vals = dc.loc[dc['Control']==1, 'score_compounding'].dropna()
        d = t_vals.mean() - c_vals.mean()
        se = np.sqrt(t_vals.var(ddof=1)/len(t_vals) + c_vals.var(ddof=1)/len(c_vals))
        df_w = (t_vals.var(ddof=1)/len(t_vals) + c_vals.var(ddof=1)/len(c_vals))**2 / \
               ((t_vals.var(ddof=1)/len(t_vals))**2/(len(t_vals)-1) + (c_vals.var(ddof=1)/len(c_vals))**2/(len(c_vals)-1))
        p = 2*stats.t.sf(abs(d/se), df_w)
        pld = make_success_payload(coefficients={"diff": d}, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G1))
        add_row(spec_id='design/randomized_experiment/estimator/diff_in_means', spec_run_id=rid,
                baseline_group_id='G1', spec_tree_path='designs/randomized_experiment.md#difference-in-means',
                outcome_var='score_compounding', treatment_var='Full',
                coefficient=d, std_error=se, p_value=p,
                n_obs=len(t_vals)+len(c_vals), r_squared=np.nan,
                sample_desc='Welch t-test', controls_desc='none', coefficient_vector_json=pld)
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage='G1 dim'))
        add_row(spec_id='design/randomized_experiment/estimator/diff_in_means', spec_run_id=rid,
                baseline_group_id='G1', spec_tree_path='designs/randomized_experiment.md#difference-in-means',
                outcome_var='score_compounding', treatment_var='Full',
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])

    run_g1_intercept(df, 'design/randomized_experiment/estimator/with_covariates', fA, 'Full',
        'designs/randomized_experiment.md#with-covariates', controls=FULL_CTRLS,
        sample_desc="tag==1, Full|Control, sample=='old'", controls_desc='full demographics (32)')

    # Single additions
    for ctrl, sid in [('gender','add_gender'),('age','add_age'),('income','add_income'),
                      ('fl_high','add_fl_high'),('ownStocks','add_ownStocks')]:
        run_g1_intercept(df, f'rc/controls/single/{sid}', fA, 'Full',
            'modules/robustness/controls.md#single-addition', controls=[ctrl],
            sample_desc="tag==1, Full|Control, sample=='old'", controls_desc=ctrl,
            axis_block_name='controls', axis_block={'spec_id': f'rc/controls/single/{sid}', 'family': 'single', 'added': [ctrl]})

    # Sets
    for sid, ctrls, desc in [('demographics_minimal', MINIMAL_CTRLS, 'minimal (5)'),
                              ('demographics_extended', EXTENDED_CTRLS, 'extended (12)'),
                              ('demographics_full', FULL_CTRLS, 'full (32)')]:
        run_g1_intercept(df, f'rc/controls/sets/{sid}', fA, 'Full',
            'modules/robustness/controls.md#curated-sets', controls=ctrls,
            sample_desc="tag==1, Full|Control, sample=='old'", controls_desc=desc,
            axis_block_name='controls', axis_block={'spec_id': f'rc/controls/sets/{sid}', 'family': 'curated_set', 'n_controls': len(ctrls)})

    # LOO
    for dv, sid in [('gender','drop_gender'),('age','drop_age'),('income','drop_income')]:
        c2 = [c for c in FULL_CTRLS if c != dv]
        run_g1_intercept(df, f'rc/controls/loo/{sid}', fA, 'Full',
            'modules/robustness/controls.md#leave-one-out-controls-loo', controls=c2,
            sample_desc="tag==1, Full|Control, sample=='old'", controls_desc=f'full minus {dv}',
            axis_block_name='controls', axis_block={'spec_id': f'rc/controls/loo/{sid}', 'family': 'loo', 'dropped': [dv]})

    # Sample variants
    run_g1_intercept(df, 'rc/sample/experiment/expA_only', fA, 'Full',
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Exp A',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/experiment/expA_only','description':'Exp A only'})
    run_g1_intercept(df, 'rc/sample/experiment/expB_only', fB, 'Full',
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Exp B',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/experiment/expB_only','description':'Exp B only'})
    run_g1_intercept(df, 'rc/sample/experiment/pooled_AB', fPool, 'Full',
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Pooled A+B',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/experiment/pooled_AB','description':'Pooled A+B'})

    # Trim
    rid = _rid()
    try:
        dc = fA(df).copy()
        lo, hi = dc['score_compounding'].quantile([0.01, 0.99])
        dc2 = dc[(dc['score_compounding']>=lo)&(dc['score_compounding']<=hi)]
        m = ols_cl(dc2, 'score_compounding', ['Full'])
        c = float(m.coef().loc['Full']); s = float(m.se().loc['Full']); p = float(m.pvalue().loc['Full'])
        pld = make_success_payload(coefficients={k:float(v) for k,v in m.coef().items()}, inference=CANON_INF,
                                    software=SW, surface_hash=SHASH, design=dblock(DA_G1),
                                    axis_block_name='sample', axis_block={'spec_id':'rc/sample/outliers/trim_y_1_99','family':'trim'})
        add_row(spec_id='rc/sample/outliers/trim_y_1_99', spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path='modules/robustness/sample.md#outlier-trimming',
                outcome_var='score_compounding', treatment_var='Full',
                coefficient=c, std_error=s, p_value=p, n_obs=m._N, r_squared=m._r2,
                sample_desc='trim 1-99', coefficient_vector_json=pld)
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage='G1 trim'))
        add_row(spec_id='rc/sample/outliers/trim_y_1_99', spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path='modules/robustness/sample.md#outlier-trimming',
                outcome_var='score_compounding', treatment_var='Full',
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])

    # Alt outcomes
    for sid, yvar in [('rc/outcome/score_indexing','score_indexing'),
                      ('rc/outcome/fl_score_compound','fl_score_compound'),
                      ('rc/outcome/fl_sum_compound','fl_sum_compound')]:
        rid = _rid()
        try:
            dc = fA(df).copy()
            m = ols_cl(dc, yvar, ['Full'])
            c = float(m.coef().loc['Full']); s = float(m.se().loc['Full']); p = float(m.pvalue().loc['Full'])
            pld = make_success_payload(coefficients={k:float(v) for k,v in m.coef().items()}, inference=CANON_INF,
                                        software=SW, surface_hash=SHASH, design=dblock(DA_G1))
            add_row(spec_id=sid, spec_run_id=rid, baseline_group_id='G1',
                    spec_tree_path='modules/robustness/controls.md#alternative-outcomes',
                    outcome_var=yvar, treatment_var='Full', coefficient=c, std_error=s, p_value=p,
                    n_obs=m._N, r_squared=m._r2, sample_desc='Exp A tag==1', coefficient_vector_json=pld)
        except Exception as e:
            pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=sid))
            add_row(spec_id=sid, spec_run_id=rid, baseline_group_id='G1',
                    spec_tree_path='modules/robustness/controls.md#alternative-outcomes',
                    outcome_var=yvar, treatment_var='Full', coefficient=np.nan, std_error=np.nan,
                    p_value=np.nan, n_obs=np.nan, r_squared=np.nan,
                    coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])

    # score total
    rid = _rid()
    try:
        dc = fA(df).copy(); dc['score_total'] = dc['score_compounding'] + dc['score_indexing']
        m = ols_cl(dc, 'score_total', ['Full'])
        c = float(m.coef().loc['Full']); s = float(m.se().loc['Full']); p = float(m.pvalue().loc['Full'])
        pld = make_success_payload(coefficients={k:float(v) for k,v in m.coef().items()}, inference=CANON_INF,
                                    software=SW, surface_hash=SHASH, design=dblock(DA_G1))
        add_row(spec_id='rc/outcome/score_compounding_plus_indexing', spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path='modules/robustness/controls.md#alternative-outcomes',
                outcome_var='score_total', treatment_var='Full', coefficient=c, std_error=s, p_value=p,
                n_obs=m._N, r_squared=m._r2, sample_desc='Exp A tag==1', coefficient_vector_json=pld)
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage='score total'))
        add_row(spec_id='rc/outcome/score_compounding_plus_indexing', spec_run_id=rid, baseline_group_id='G1',
                spec_tree_path='modules/robustness/controls.md#alternative-outcomes',
                outcome_var='score_total', treatment_var='Full', coefficient=np.nan, std_error=np.nan,
                p_value=np.nan, n_obs=np.nan, r_squared=np.nan,
                coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])

    # Treatment arms
    for sid, tv, filt in [('rc/treatment/rule72_only_vs_control', 'Rule72',
                            lambda d: d[(d['tag']==1)&((d['Rule72']==1)|(d['Control']==1))&(d['sample']=='old')]),
                           ('rc/treatment/rhetoric_only_vs_control', 'Rhetoric',
                            lambda d: d[(d['tag']==1)&((d['Rhetoric']==1)|(d['Control']==1))&(d['sample']=='old')])]:
        run_g1_intercept(df, sid, filt, tv, 'designs/randomized_experiment.md#treatment-arms',
                         sample_desc=f'{tv} vs Control, Exp A')

    # All arms nocons
    run_g1_nocons(df, 'rc/treatment/all_arms_nocons',
                  lambda d: d[(d['tag']==1)&(d['sample']=='old')],
                  ['Rule72','Rhetoric','Full','Control'],
                  'designs/randomized_experiment.md#all-arms',
                  sample_desc='all arms, tag==1, Exp A')

    # Joint
    for sid, filt in [('rc/joint/expA_with_controls', fA),
                      ('rc/joint/expB_with_controls', fB),
                      ('rc/joint/pooled_with_controls', fPool)]:
        run_g1_intercept(df, sid, filt, 'Full', 'modules/robustness/joint.md#joint-specifications',
                         controls=FULL_CTRLS, sample_desc=sid, controls_desc='full (32)',
                         axis_block_name='joint', axis_block={'spec_id':sid, 'axes_changed':['sample','controls']})

    return bl_rid


def _prep_expB(d):
    d2 = d.copy(); d2['Full'] = d2['fullNew']; return d2
def _prep_pooled(d):
    d2 = d.copy()
    d2.loc[d2['sample']=='new','Full'] = d2.loc[d2['sample']=='new','fullNew']
    d2.loc[d2['sample']=='new','Control'] = d2.loc[d2['sample']=='new','contNew']
    return d2[(d2['Full']==1)|(d2['Control']==1)]


# ═══════════════════════════════════════════════════════════════════════════
#  G2 SPECS
# ═══════════════════════════════════════════════════════════════════════════

def run_g2_nc(df, spec_id, filt, arms, tree, f1='Full', f2='Control',
              sample_desc='', controls_desc='none', controls=None, **extra_blocks):
    """G2 spec: OLS nocons, Wald test f1==f2."""
    rid = _rid()
    try:
        dc = filt(df).copy()
        m, vn, _ = ols_nocons_cl(dc, 'negAbsDiff', arms, ctrls=controls)
        d, se, p = wald_eq(m, vn, f1, f2)
        coefs = {vn[i]: float(m.params[i]) for i in range(len(vn))}
        pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G2),
                                    extra={"focal_test": f"{f1}=={f2}", "model_form": "nocons"},
                                    **extra_blocks)
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G2',
                spec_tree_path=tree, outcome_var='negAbsDiff', treatment_var=','.join(arms),
                coefficient=d, std_error=se, p_value=p,
                n_obs=int(m.nobs), r_squared=float(m.rsquared),
                sample_desc=sample_desc, controls_desc=controls_desc,
                coefficient_vector_json=pld)
        return rid, d, se, p
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=f'G2 {spec_id}'))
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G2',
                spec_tree_path=tree, outcome_var='negAbsDiff', treatment_var=','.join(arms),
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, sample_desc=sample_desc,
                controls_desc=controls_desc, coefficient_vector_json=pld,
                run_success=0, run_error=str(e)[:240])
        return rid, np.nan, np.nan, np.nan


def run_g2_ic(df, spec_id, filt, tree, controls=None,
              sample_desc='', controls_desc='none', **extra_blocks):
    """G2 spec: OLS with intercept, Full vs Control subset."""
    rid = _rid()
    try:
        dc = filt(df).copy()
        m = ols_cl(dc, 'negAbsDiff', ['Full'], ctrls=controls)
        c = float(m.coef().loc['Full']); s = float(m.se().loc['Full']); p = float(m.pvalue().loc['Full'])
        coefs = {k:float(v) for k,v in m.coef().items()}
        pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G2), **extra_blocks)
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G2',
                spec_tree_path=tree, outcome_var='negAbsDiff', treatment_var='Full',
                coefficient=c, std_error=s, p_value=p,
                n_obs=m._N, r_squared=m._r2,
                sample_desc=sample_desc, controls_desc=controls_desc,
                coefficient_vector_json=pld)
        return rid
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=f'G2 {spec_id}'))
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G2',
                spec_tree_path=tree, outcome_var='negAbsDiff', treatment_var='Full',
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, sample_desc=sample_desc,
                controls_desc=controls_desc, coefficient_vector_json=pld,
                run_success=0, run_error=str(e)[:240])
        return rid


def run_g2_altout(df, spec_id, yvar, construct, tree, sample_desc=''):
    """G2 spec: alternative outcome, nocons."""
    rid = _rid()
    try:
        dc = df[df['sample']=='old'].copy()
        if construct == 'finCompCorr':
            cf = dc.groupby(['id','delay'])['discount_unframed'].transform('mean')
            dc['finCompCorr'] = -(1.0/cf) * dc['absDiff'] * 100
            y = 'finCompCorr'
        elif construct == 'finCompCorrSq':
            cf = dc.groupby(['id','delay'])['discount_unframed'].transform('mean') / 100
            dc['finCompCorrSq'] = -(1.0/(cf**2)) * dc['sqDiff']
            y = 'finCompCorrSq'
        else:
            y = yvar
        m, vn, _ = ols_nocons_cl(dc, y, ['Control','Full','Rule72','Rhetoric'])
        d, se, p = wald_eq(m, vn, 'Full', 'Control')
        coefs = {vn[i]:float(m.params[i]) for i in range(len(vn))}
        pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G2),
                                    extra={"focal_test":"Full==Control","model_form":"nocons"})
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G2', spec_tree_path=tree,
                outcome_var=y, treatment_var='Control,Full,Rule72,Rhetoric',
                coefficient=d, std_error=se, p_value=p,
                n_obs=int(m.nobs), r_squared=float(m.rsquared),
                sample_desc=sample_desc or "sample=='old', nocons",
                coefficient_vector_json=pld)
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=f'G2 {spec_id}'))
        add_row(spec_id=spec_id, spec_run_id=rid, baseline_group_id='G2', spec_tree_path=tree,
                outcome_var=yvar or construct, treatment_var='Control,Full,Rule72,Rhetoric',
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan,
                coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])


def run_all_g2(df):
    fA = lambda d: d[d['sample']=='old']
    fB_nc = lambda d: _g2b(d)
    arms4 = ['Control','Full','Rule72','Rhetoric']
    arms2 = ['Control','Full']

    # Baselines
    bl_rid, *_ = run_g2_nc(df, 'baseline', fA, arms4,
        'designs/randomized_experiment.md#ols-itt', sample_desc="sample=='old', nocons")
    run_g2_nc(df, 'baseline__table7_col2_expB', fB_nc, arms2,
        'designs/randomized_experiment.md#ols-itt', sample_desc="sample=='new', nocons")

    # Design
    run_g2_ic(df, 'design/randomized_experiment/estimator/diff_in_means',
              lambda d: d[(d['sample']=='old')&((d['Full']==1)|(d['Control']==1))],
              'designs/randomized_experiment.md#difference-in-means', sample_desc='Full|Control, Exp A')
    run_g2_ic(df, 'design/randomized_experiment/estimator/with_covariates',
              lambda d: d[(d['sample']=='old')&((d['Full']==1)|(d['Control']==1))],
              'designs/randomized_experiment.md#with-covariates', controls=FULL_CTRLS,
              sample_desc='Full|Control, Exp A', controls_desc='full (32)')

    # Single controls
    for ctrl, sid in [('gender','add_gender'),('age','add_age'),('income','add_income'),
                      ('fl_high','add_fl_high'),('ownStocks','add_ownStocks'),('meanVsimple','add_meanVsimple')]:
        run_g2_nc(df, f'rc/controls/single/{sid}', fA, arms4,
            'modules/robustness/controls.md#single-addition', controls=[ctrl],
            sample_desc="sample=='old', nocons", controls_desc=ctrl,
            axis_block_name='controls', axis_block={'spec_id':f'rc/controls/single/{sid}','family':'single','added':[ctrl]})

    # Control sets
    for sid, ctrls, desc in [('demographics_minimal',MINIMAL_CTRLS,'minimal'),
                              ('demographics_extended',EXTENDED_CTRLS,'extended'),
                              ('demographics_full',FULL_CTRLS,'full')]:
        run_g2_nc(df, f'rc/controls/sets/{sid}', fA, arms4,
            'modules/robustness/controls.md#curated-sets', controls=ctrls,
            sample_desc="sample=='old', nocons", controls_desc=desc,
            axis_block_name='controls', axis_block={'spec_id':f'rc/controls/sets/{sid}','family':'curated_set','n_controls':len(ctrls)})

    # LOO
    for dv, sid in [('gender','drop_gender_from_full'),('age','drop_age_from_full'),('income','drop_income_from_full')]:
        c2 = [c for c in FULL_CTRLS if c != dv]
        run_g2_nc(df, f'rc/controls/loo/{sid}', fA, arms4,
            'modules/robustness/controls.md#leave-one-out-controls-loo', controls=c2,
            sample_desc="sample=='old', nocons", controls_desc=f'full-{dv}',
            axis_block_name='controls', axis_block={'spec_id':f'rc/controls/loo/{sid}','family':'loo','dropped':[dv]})

    # Sample variants
    run_g2_nc(df, 'rc/sample/experiment/expA_only', fA, arms4,
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Exp A',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/experiment/expA_only','description':'Exp A only'})
    run_g2_nc(df, 'rc/sample/experiment/expB_only', fB_nc, arms2,
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Exp B',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/experiment/expB_only','description':'Exp B only'})
    run_g2_nc(df, 'rc/sample/experiment/pooled_AB',
              lambda d: _g2_pool(d), arms2,
              'modules/robustness/sample.md#experiment-subsamples', sample_desc='Pooled A+B',
              axis_block_name='sample', axis_block={'spec_id':'rc/sample/experiment/pooled_AB','description':'Pooled A+B'})

    # Delay
    run_g2_nc(df, 'rc/sample/delay/delay_72_only', lambda d: fA(d)[d['delay']==72], arms4,
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Exp A delay==72',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/delay/delay_72_only','description':'delay==72 only'})
    run_g2_nc(df, 'rc/sample/delay/delay_36_only', lambda d: fA(d)[d['delay']==36], arms4,
        'modules/robustness/sample.md#experiment-subsamples', sample_desc='Exp A delay==36',
        axis_block_name='sample', axis_block={'spec_id':'rc/sample/delay/delay_36_only','description':'delay==36 only'})

    # Outlier trimming
    for sid, lo, hi in [('rc/sample/outliers/trim_absDiff_1_99',0.01,0.99),
                         ('rc/sample/outliers/trim_absDiff_5_95',0.05,0.95)]:
        rid = _rid()
        try:
            dc = fA(df).copy()
            ql, qh = dc['absDiff'].quantile([lo, hi])
            dc2 = dc[(dc['absDiff']>=ql)&(dc['absDiff']<=qh)].copy()
            dc2['negAbsDiff'] = -dc2['absDiff']
            m, vn, _ = ols_nocons_cl(dc2, 'negAbsDiff', arms4)
            d, se, p = wald_eq(m, vn, 'Full', 'Control')
            coefs = {vn[i]:float(m.params[i]) for i in range(len(vn))}
            pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                        surface_hash=SHASH, design=dblock(DA_G2),
                                        axis_block_name='sample', axis_block={'spec_id':sid,'family':'trim','quantiles':[lo,hi]})
            add_row(spec_id=sid, spec_run_id=rid, baseline_group_id='G2',
                    spec_tree_path='modules/robustness/sample.md#outlier-trimming',
                    outcome_var='negAbsDiff', treatment_var=','.join(arms4),
                    coefficient=d, std_error=se, p_value=p,
                    n_obs=int(m.nobs), r_squared=float(m.rsquared), sample_desc=f'trim {lo*100:.0f}-{hi*100:.0f}',
                    coefficient_vector_json=pld)
        except Exception as e:
            pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=sid))
            add_row(spec_id=sid, spec_run_id=rid, baseline_group_id='G2',
                    spec_tree_path='modules/robustness/sample.md#outlier-trimming',
                    outcome_var='negAbsDiff', treatment_var=','.join(arms4),
                    coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                    n_obs=np.nan, r_squared=np.nan, coefficient_vector_json=pld,
                    run_success=0, run_error=str(e)[:240])

    # Alt outcomes
    for sid, yv, con in [('rc/outcome/negSqDiff','negSqDiff',None),
                          ('rc/outcome/diff','diff',None),
                          ('rc/outcome/discount_framed','discount_framed',None),
                          ('rc/outcome/finCompCorr',None,'finCompCorr'),
                          ('rc/outcome/finCompCorrSq',None,'finCompCorrSq')]:
        run_g2_altout(df, sid, yv, con, 'modules/robustness/controls.md#alternative-outcomes')

    # Treatment arms
    run_g2_nc(df, 'rc/treatment/rule72_only_vs_control',
              lambda d: d[(d['sample']=='old')&((d['Rule72']==1)|(d['Control']==1))],
              ['Control','Rule72'], 'designs/randomized_experiment.md#treatment-arms',
              f1='Rule72', f2='Control', sample_desc='Rule72 vs Control')
    run_g2_nc(df, 'rc/treatment/rhetoric_only_vs_control',
              lambda d: d[(d['sample']=='old')&((d['Rhetoric']==1)|(d['Control']==1))],
              ['Control','Rhetoric'], 'designs/randomized_experiment.md#treatment-arms',
              f1='Rhetoric', f2='Control', sample_desc='Rhetoric vs Control')
    run_g2_ic(df, 'rc/treatment/full_vs_control_with_intercept',
              lambda d: d[(d['sample']=='old')&((d['Full']==1)|(d['Control']==1))],
              'designs/randomized_experiment.md#treatment-arms', sample_desc='Full vs Control, intercept')
    run_g2_nc(df, 'rc/treatment/all_arms_nocons', fA, arms4,
              'designs/randomized_experiment.md#all-arms', sample_desc='all arms nocons')

    # Joint delay
    for sid, filt in [('rc/joint/expA_delay72', lambda d: d[(d['sample']=='old')&(d['delay']==72)]),
                      ('rc/joint/expA_delay36', lambda d: d[(d['sample']=='old')&(d['delay']==36)]),
                      ('rc/joint/expB_delay72', lambda d: _g2b_d(d, 72)),
                      ('rc/joint/expB_delay36', lambda d: _g2b_d(d, 36))]:
        a = arms4 if 'expA' in sid else arms2
        run_g2_nc(df, sid, filt, a, 'modules/robustness/joint.md#joint-specifications', sample_desc=sid,
                  axis_block_name='joint', axis_block={'spec_id':sid,'axes_changed':['sample']})

    # Joint with controls
    for sid, filt, a in [('rc/joint/pooled_with_controls', lambda d: _g2_pool(d), arms2),
                          ('rc/joint/expA_with_controls', fA, arms4),
                          ('rc/joint/expB_with_controls', lambda d: _g2b(d), arms2)]:
        run_g2_nc(df, sid, filt, a, 'modules/robustness/joint.md#joint-specifications',
                  controls=FULL_CTRLS, sample_desc=sid, controls_desc='full (32)',
                  axis_block_name='joint', axis_block={'spec_id':sid,'axes_changed':['sample','controls']})

    # form/individual_means
    rid = _rid()
    try:
        dc = df[(df['sample']=='old')&(df['tag']==1)].copy()
        m, vn, _ = ols_nocons_cl(dc, 'dAbs', arms4)
        d, se, p = wald_eq(m, vn, 'Full', 'Control')
        coefs = {vn[i]:float(m.params[i]) for i in range(len(vn))}
        pld = make_success_payload(coefficients=coefs, inference=CANON_INF, software=SW,
                                    surface_hash=SHASH, design=dblock(DA_G2),
                                    axis_block_name='functional_form',
                                    axis_block={'spec_id':'rc/form/individual_means',
                                                'interpretation':'Collapsed to individual means (dAbs), nocons.'})
        add_row(spec_id='rc/form/individual_means', spec_run_id=rid, baseline_group_id='G2',
                spec_tree_path='modules/robustness/functional_form.md#aggregation',
                outcome_var='dAbs', treatment_var=','.join(arms4),
                coefficient=d, std_error=se, p_value=p,
                n_obs=int(m.nobs), r_squared=float(m.rsquared),
                sample_desc='individual means, tag==1', coefficient_vector_json=pld)
    except Exception as e:
        pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage='G2 form'))
        add_row(spec_id='rc/form/individual_means', spec_run_id=rid, baseline_group_id='G2',
                spec_tree_path='modules/robustness/functional_form.md#aggregation',
                outcome_var='dAbs', treatment_var=','.join(arms4),
                coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                n_obs=np.nan, r_squared=np.nan, coefficient_vector_json=pld,
                run_success=0, run_error=str(e)[:240])

    return bl_rid


def _g2b(d):
    d2 = d[d['sample']=='new'].copy(); d2['Control']=d2['contNew']; d2['Full']=d2['fullNew']; return d2
def _g2b_d(d, delay):
    d2 = _g2b(d); return d2[d2['delay']==delay]
def _g2_pool(d):
    d2 = d.copy()
    d2.loc[d2['sample']=='new','Control'] = d2.loc[d2['sample']=='new','contNew']
    d2.loc[d2['sample']=='new','Full'] = d2.loc[d2['sample']=='new','fullNew']
    return d2[(d2['Full']==1)|(d2['Control']==1)]


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

def run_infer(df):
    variants = [
        ('infer/se/hc/hc1', 'HC1', 'modules/inference/standard_errors.md#heteroskedasticity-robust'),
        ('infer/se/hc/hc3', 'HC3', 'modules/inference/standard_errors.md#hc3'),
    ]
    for row in list(SPEC_ROWS):
        if row['run_success'] == 0 or not row['spec_id'].startswith('baseline'):
            continue
        bg = row['baseline_group_id']
        srid = row['spec_run_id']
        for vsid, cov_t, tp in variants:
            try:
                if bg == 'G1' and row['spec_id'] == 'baseline':
                    dc = df[(df['tag']==1)&((df['Full']==1)|(df['Control']==1))&(df['sample']=='old')].dropna(subset=['score_compounding','Full','id'])
                    X = sm.add_constant(dc[['Full']].values.astype(float))
                    y = dc['score_compounding'].values.astype(float)
                    m = sm.OLS(y, X).fit(cov_type=cov_t)
                    c, s, p = float(m.params[1]), float(m.bse[1]), float(m.pvalues[1])
                elif bg == 'G1' and 'expB' in row['spec_id']:
                    dc = df[(df['tag']==1)&((df['fullNew']==1)|(df['contNew']==1))&(df['sample']=='new')].copy()
                    dc['Full_B'] = dc['fullNew']
                    dc = dc.dropna(subset=['score_compounding','Full_B','id'])
                    X = sm.add_constant(dc[['Full_B']].values.astype(float))
                    y = dc['score_compounding'].values.astype(float)
                    m = sm.OLS(y, X).fit(cov_type=cov_t)
                    c, s, p = float(m.params[1]), float(m.bse[1]), float(m.pvalues[1])
                elif bg == 'G2' and row['spec_id'] == 'baseline':
                    dc = df[df['sample']=='old'].dropna(subset=['negAbsDiff','Control','Full','Rule72','Rhetoric','id'])
                    arms = ['Control','Full','Rule72','Rhetoric']
                    X = dc[arms].values.astype(float)
                    y = dc['negAbsDiff'].values.astype(float)
                    m = sm.OLS(y, X).fit(cov_type=cov_t)
                    d = m.params[1] - m.params[0]
                    cov_m = m.cov_params()
                    se = np.sqrt(cov_m[1,1]+cov_m[0,0]-2*cov_m[0,1])
                    p = 2*stats.t.sf(abs(d/se), m.df_resid)
                    c, s = d, se
                elif bg == 'G2' and 'expB' in row['spec_id']:
                    dc = df[df['sample']=='new'].copy()
                    dc['Control'] = dc['contNew']; dc['Full'] = dc['fullNew']
                    dc = dc.dropna(subset=['negAbsDiff','Control','Full','id'])
                    X = dc[['Control','Full']].values.astype(float)
                    y = dc['negAbsDiff'].values.astype(float)
                    m = sm.OLS(y, X).fit(cov_type=cov_t)
                    d = m.params[1] - m.params[0]
                    cov_m = m.cov_params()
                    se = np.sqrt(cov_m[1,1]+cov_m[0,0]-2*cov_m[0,1])
                    p = 2*stats.t.sf(abs(d/se), m.df_resid)
                    c, s = d, se
                else:
                    continue
                pld = make_success_payload(coefficients={"focal":c}, inference={"spec_id":vsid,"params":{"cov_type":cov_t}},
                                            software=SW, surface_hash=SHASH, design=dblock(DA_G1 if bg=='G1' else DA_G2))
                add_infer(inference_run_id=_iid(), spec_run_id=srid, spec_id=vsid,
                          spec_tree_path=tp, baseline_group_id=bg,
                          outcome_var=row['outcome_var'], treatment_var=row['treatment_var'],
                          coefficient=c, std_error=s, p_value=p,
                          n_obs=int(m.nobs), r_squared=float(m.rsquared),
                          cluster_var='', coefficient_vector_json=pld)
            except Exception as e:
                pld = make_failure_payload(error=str(e), error_details=error_details_from_exception(e, stage=vsid))
                add_infer(inference_run_id=_iid(), spec_run_id=srid, spec_id=vsid,
                          spec_tree_path=tp, baseline_group_id=bg,
                          outcome_var=row['outcome_var'], treatment_var=row['treatment_var'],
                          coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                          n_obs=np.nan, r_squared=np.nan, cluster_var='',
                          coefficient_vector_json=pld, run_success=0, run_error=str(e)[:240])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*60)
    print(f'Specification Search: {PAPER_ID}')
    print('='*60)

    print('\n[1] Building Experiment A data...')
    df_A = build_expA()
    print(f'  Exp A: {len(df_A)} obs, {df_A["id"].nunique()} ids')

    print('\n[2] Building Experiment B data...')
    df_B = build_expB()
    print(f'  Exp B: {len(df_B)} obs, {df_B["id"].nunique()} ids')

    print('\n[3] Combining...')
    df = combine_experiments(df_A, df_B)
    print(f'  Combined: {len(df)} obs, {df["id"].nunique()} ids')
    print(f'  Old: {(df["sample"]=="old").sum()}, New: {(df["sample"]=="new").sum()}')

    print('\n[4] Running G1 specs...')
    bl1 = run_all_g1(df)
    g1n = sum(1 for r in SPEC_ROWS if r['baseline_group_id']=='G1')
    g1s = sum(1 for r in SPEC_ROWS if r['baseline_group_id']=='G1' and r['run_success']==1)
    print(f'  G1: {g1s}/{g1n}')

    print('\n[5] Running G2 specs...')
    bl2 = run_all_g2(df)
    g2n = sum(1 for r in SPEC_ROWS if r['baseline_group_id']=='G2')
    g2s = sum(1 for r in SPEC_ROWS if r['baseline_group_id']=='G2' and r['run_success']==1)
    print(f'  G2: {g2s}/{g2n}')

    print('\n[6] Inference variants...')
    run_infer(df)
    print(f'  {len(INF_ROWS)} rows')

    total = len(SPEC_ROWS); succ = sum(1 for r in SPEC_ROWS if r['run_success']==1)
    print(f'\nTOTAL: {succ}/{total} specs succeeded')

    # Write outputs
    pd.DataFrame(SPEC_ROWS).to_csv(os.path.join(OUT, 'specification_results.csv'), index=False)
    print(f'Wrote specification_results.csv ({total} rows)')

    if INF_ROWS:
        pd.DataFrame(INF_ROWS).to_csv(os.path.join(OUT, 'inference_results.csv'), index=False)
        print(f'Wrote inference_results.csv ({len(INF_ROWS)} rows)')

    md = f"""# Specification Search: {PAPER_ID}

## Surface Summary
- **Paper**: Ambuehl, Bernheim & Lusardi, "Evaluating Deliberative Competence"
- **Design**: Randomized Experiment (online, MTurk)
- **Baseline Groups**: 2 (G1: compounding knowledge, G2: financial competence)
- **Surface hash**: {SHASH}
- **Seeds**: 171681 (G1), 171682 (G2)

## Execution Summary

| Metric | G1 | G2 | Total |
|--------|----|----|-------|
| Planned | {g1n} | {g2n} | {total} |
| Succeeded | {g1s} | {g2s} | {succ} |
| Failed | {g1n-g1s} | {g2n-g2s} | {total-succ} |

Inference variants: {len(INF_ROWS)} rows (HC1, HC3 on baselines)

## Software Stack
- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
- pyfixest, statsmodels, scipy, pandas, numpy

## Notes
1. Data management replicated from raw Qualtrics CSV (8 Exp A + 2 Exp B batches). Includes MPL switch-point extraction (coarse + fine), multi-switcher flagging, treatment renaming/reshaping, midpoint adjustment, discount rate construction, and all demographic variable construction.
2. G2 baselines and many G2 variants use no-constant OLS with all treatment arm dummies. The focal coefficient is the Wald test Full == Control.
3. `rc/sample/attrition/include_multi_switchers__requires_remanagement` SKIPPED: requires re-running the full pipeline without multi==0 exclusion.
4. Constructed outcomes finCompCorr and finCompCorrSq built inline following the paper's formulas.
5. Minor numerical differences from published tables are expected due to floating-point differences between Python and Stata in the complex MPL extraction pipeline.
"""
    with open(os.path.join(OUT, 'SPECIFICATION_SEARCH.md'), 'w') as f:
        f.write(md)
    print('Wrote SPECIFICATION_SEARCH.md')
    print('Done!')
