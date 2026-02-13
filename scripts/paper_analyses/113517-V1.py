"""
Specification Search Script for 113517-V1
"The Relative Power of Employment-to-Employment Reallocation and
Unemployment Exits in Predicting Wage Growth"
Moscarini & Postel-Vinay, AER P&P 2017

Memory-optimized: avoids large DataFrame copies in first-stage regressions.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
import time
import os
import gc

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data/downloads/extracted/113517-V1/Codes-and-data')
OUT_DIR = os.path.join(BASE_DIR, 'data/downloads/extracted/113517-V1')
PAPER_ID = '113517-V1'

# ============================================================
# Load + prepare data
# ============================================================
print("Loading data...")
t0 = time.time()
df = pd.read_parquet(os.path.join(DATA_DIR, 'preprocessed.parquet'))
print(f"  Loaded in {time.time()-t0:.1f}s, shape: {df.shape}")

for v in ['lagstate','laguni','lagsiz','lagocc','lagind','lagpub','mkt_t','mkt']:
    df[v] = df[v].astype('Int64')

df['ym_num'] = df['year_month_num']
for dv in ['logern_nom','logern','loghwr_nom','loghwr']:
    df[f'd{dv}'] = df[dv] - df[f'lag{dv}']

df['EZeligible_hw'] = ((df['EZeligible']==1) & (df['lagphr']==1)).astype(int)
df['DWeligible_hw'] = ((df['DWeligible']==1) & (df['lagphr']==1)).astype(int)

df['loghrs'] = np.log(df['hrs'].clip(lower=0.01))
df['laghrs'] = df.groupby('panel_id')['loghrs'].shift(1)
df['ym_num_sq'] = df['ym_num']**2
df['year'] = df['year_month'].dt.year
df['mkt_coarse'] = df['sex'].astype(int)*10000 + df['agegroup'].astype(int)*100 + df['education'].astype(int)
df['mkt_t_coarse'] = df['mkt_coarse'].astype(np.int64)*1000000 + df['year_month_num'].astype(np.int64)
df['spell_len'] = df.groupby('panel_id')['panel_id'].transform('count')
p99 = df['wgt'].quantile(0.99)
df['wgt_p99'] = df['wgt'].clip(upper=p99)

gc.collect()
print(f"  Data prep done in {time.time()-t0:.1f}s")

# ============================================================
# First-stage: no copy, use boolean mask + dropna on a view
# ============================================================
_fs_cache = {}

def run_first_stage_cached(depv, rhs_formula, elig_col, fe_col='mkt_t'):
    key = (depv, rhs_formula, elig_col, fe_col)
    if key in _fs_cache:
        return _fs_cache[key]

    # Build list of needed columns
    needed = [depv, 'wgt', fe_col]
    for term in rhs_formula.split('+'):
        t = term.strip()
        needed.append(t[2:-1] if t.startswith('C(') else t)

    # Boolean mask: eligible + non-null + positive weight
    mask = (df[elig_col] == 1) & (df['wgt'] > 0)
    for col in needed:
        mask = mask & df[col].notna()

    # Use .loc with mask directly - no copy
    sub = df.loc[mask, needed]

    formula = f"{depv} ~ {rhs_formula} | {fe_col}"
    t1 = time.time()
    m = pf.feols(formula, data=sub, weights='wgt')
    fe = m.fixef()
    fk = [k for k in fe.keys() if fe_col in k][0]
    result = fe[fk]
    _fs_cache[key] = result
    # Free model memory
    del m, sub
    gc.collect()
    print(f"    FS({depv[:8]:8s},{elig_col[:6]:6s},{fe_col}) {time.time()-t1:.0f}s", flush=True)
    return result

# ============================================================
# First-stage configs
# ============================================================
E_BASE = "C(lagstate)+C(laguni)+C(lagsiz)+C(lagocc)+C(lagind)+C(lagpub)"
U_BASE = "C(lagstate)"

FS_CFGS = {
    'baseline':           {'e': E_BASE, 'u': U_BASE, 'lag_dv': True,  'ee_dw': True},
    'e_controls_minimal': {'e': "C(lagstate)", 'u': "C(lagstate)", 'lag_dv': True, 'ee_dw': True},
    'e_controls_extended':{'e': E_BASE, 'u': E_BASE, 'lag_dv': True,  'ee_dw': True},
    'u_controls_extended':{'e': E_BASE, 'u': "C(lagstate)+C(laguni)", 'lag_dv': True, 'ee_dw': True},
    'no_lag_depvar':      {'e': E_BASE, 'u': U_BASE, 'lag_dv': False, 'ee_dw': False},
    'add_lag_hours':      {'e': E_BASE+"+laghrs", 'u': U_BASE, 'lag_dv': True, 'ee_dw': True},
}

def build_first_stage(depvar, fs_key, fe_col='mkt_t'):
    cfg = FS_CFGS[fs_key]
    ec, uc = cfg['e'], cfg['u']
    hourly = depvar in ('loghwr','loghwr_nom')
    ez = 'EZeligible_hw' if hourly else 'EZeligible'
    dw = 'DWeligible_hw' if hourly else 'DWeligible'
    lagdv = f'lag{depvar}'
    ddv = f'd{depvar}'

    res = {}
    ee_rhs = f'{lagdv}+{ec}' if cfg['lag_dv'] else ec
    res['xee'] = run_first_stage_cached('eetrans_i', ee_rhs, ez, fe_col)
    dw_rhs = f'eetrans_i+{ec}' if cfg['ee_dw'] else ec
    res['xdvar'] = run_first_stage_cached(ddv, dw_rhs, dw, fe_col)
    res['xue'] = run_first_stage_cached('uetrans_i', uc, 'UZeligible', fe_col)
    res['xne'] = run_first_stage_cached('netrans_i', uc, 'NZeligible', fe_col)
    res['xur'] = run_first_stage_cached('unm', uc, 'UReligible', fe_col)
    res['xeu'] = run_first_stage_cached('eutrans_i', ec, ez, fe_col)
    res['xen'] = run_first_stage_cached('entrans_i', ec, ez, fe_col)
    return res

# ============================================================
# Apply / cleanup FE
# ============================================================
XFE_COLS = ['xee','xue','xne','xeu','xen','xur','xnue','xenu','xee_i']

def apply_fe(fe_res, depvar, fe_col='mkt_t'):
    xdv = f'xd{depvar}'
    hourly = depvar in ('loghwr','loghwr_nom')
    dw_col = 'DWeligible_hw' if hourly else 'DWeligible'

    for vn in ['xee','xue','xne','xeu','xen','xur']:
        femap = {int(k):v for k,v in fe_res[vn].items()}
        df[vn] = df[fe_col].map(femap)

    femap = {int(k):v for k,v in fe_res['xdvar'].items()}
    df[xdv] = df[fe_col].map(femap)
    df.loc[df[dw_col]!=1, xdv] = np.nan

    df['xnue'] = df['xue'] + df['xne']
    df['xenu'] = df['xen'] + df['xeu']
    df['xee_i'] = df['xee'] * df['eetrans_i']

def cleanup_fe(depvar):
    xdv = f'xd{depvar}'
    for c in XFE_COLS + [xdv]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

# ============================================================
# Second-stage runner
# ============================================================
def run_2nd(formula, focal_var='xee', wt='wgt', sfilt=None, clust=None):
    lhs = formula.split('~')[0].strip()
    rfe = formula.split('~')[1]
    if '|' in rfe:
        rp = rfe.split('|')[0].strip()
        fp = rfe.split('|')[1].strip()
        rv = [v.strip() for v in rp.split('+')]
        fv = [v.strip() for v in fp.split('+')]
        needed = [lhs] + rv + fv
    else:
        rv = [v.strip() for v in rfe.strip().split('+')]
        needed = [lhs] + rv

    if wt: needed.append(wt)
    if clust:
        if isinstance(clust, list): needed.extend(clust)
        else: needed.append(clust)

    # Build mask without copy
    mask = pd.Series(True, index=df.index)
    for col in needed:
        mask = mask & df[col].notna()
    if wt:
        mask = mask & (df[wt] > 0)

    sub = df.loc[mask]
    if sfilt:
        sub = sub[sfilt(sub)]

    kw = {}
    if wt: kw['weights'] = wt
    if clust:
        if isinstance(clust, list):
            kw['vcov'] = {"CRV1": "+".join(clust)}
        else:
            kw['vcov'] = {"CRV1": clust}

    m = pf.feols(formula, data=sub, **kw)
    c = m.coef(); s = m.se(); p = m.pvalue(); ci = m.confint()
    result = {
        'coefficient': float(c[focal_var]),
        'std_error': float(s[focal_var]),
        'p_value': float(p[focal_var]),
        'ci_lower': float(ci.loc[focal_var,'2.5%']) if focal_var in ci.index else np.nan,
        'ci_upper': float(ci.loc[focal_var,'97.5%']) if focal_var in ci.index else np.nan,
        'n_obs': int(m._N),
        'r_squared': float(m._r2),
        'coef_vec': {k: float(v) for k,v in c.items()},
    }
    del m
    return result

# ============================================================
# Enumerate specifications
# ============================================================
GROUPS = {
    'G1': {'depvar':'logern_nom', 'label':'Nominal Earnings', 'hourly': False},
    'G2': {'depvar':'logern',     'label':'Real Earnings',    'hourly': False},
    'G3': {'depvar':'loghwr_nom', 'label':'Nom. Hourly Wage', 'hourly': True},
    'G4': {'depvar':'loghwr',     'label':'Real Hourly Wage', 'hourly': True},
}

FLOWS = {
    'ee_only':       (['xee'],                               'EE only'),
    'ee_ue':         (['xee','xue'],                         'EE+UE'),
    'ee_ue_ur':      (['xee','xue','xur'],                   'EE+UE+UR'),
    'all_flows':     (['xee','xue','xur','xne','xen','xeu'], 'All flows'),
    'grouped_flows': (['xee','xur','xnue','xenu'],           'Grouped flows'),
}
ANCHORS = ['ee_only','ee_ue_ur','all_flows']

specs = []

for gid, gconf in GROUPS.items():
    dv = gconf['depvar']; xdv = f'xd{dv}'; lb = gconf['label']

    def mk_f(xdv, regs, fe='mkt', extra_rhs=None):
        rhs = ' + '.join(regs)
        if extra_rhs: rhs += ' + ' + ' + '.join(extra_rhs)
        return f"{xdv} ~ {rhs} | {fe}"

    # 1. BASELINE: 5 flow specs
    for fk,(regs,fl) in FLOWS.items():
        specs.append((gid, f'baseline/{fk}', 'design/panel_fixed_effects/estimator/within',
                       fk, 'baseline', 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num']),
                       'wgt', None, None,
                       f'{lb}, all eligible', 'mkt', f'ym+{fl}', '', f'Baseline {fl}'))

    # 2. RC/CONTROLS/FIRST_STAGE: 5 x 3 = 15
    for fsk in ['e_controls_minimal','e_controls_extended','u_controls_extended','no_lag_depvar','add_lag_hours']:
        for ak in ANCHORS:
            regs, fl = FLOWS[ak]
            specs.append((gid, f'rc/controls/first_stage/{fsk}__{ak}', f'rc/controls/first_stage/{fsk}',
                           ak, fsk, 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num']),
                           'wgt', None, None,
                           f'{lb}, all eligible', 'mkt', f'ym+{fl} (fs:{fsk})', '',
                           f'FS: {fsk}, {fl}'))

    # 3. RC/SAMPLE: 6 x 3 = 18
    sfilts = {
        'drop_first_panel': (lambda d: d['panel_id']>=20,  'drop 1996 panel', 'rc/sample/time/drop_first_panel'),
        'drop_last_panel':  (lambda d: d['panel_id']<=80,  'drop 2008 panel', 'rc/sample/time/drop_last_panel'),
        'pre_crisis':       (lambda d: d['year']<2008,     'pre-2008',        'rc/sample/time/pre_crisis'),
        'post_crisis':      (lambda d: d['year']>=2008,    '2008+',           'rc/sample/time/post_crisis'),
        'drop_short_spells':(lambda d: d['spell_len']>=3,  'spell>=3',        'rc/sample/quality/drop_short_spells'),
        'job_stayers':      (lambda d: (d['eetrans_i']==0)&(d['lagemp']>0), 'job stayers', 'rc/sample/restriction/job_stayers'),
    }
    for sk, (sf, sd, sp) in sfilts.items():
        for ak in ANCHORS:
            regs, fl = FLOWS[ak]
            specs.append((gid, f'{sp}__{ak}', sp,
                           ak, 'baseline', 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num']),
                           'wgt', sf, None,
                           f'{lb}, {sd}', 'mkt', f'ym+{fl}', '', f'{sd}, {fl}'))

    # 4. RC/FE: 2 x 3 = 6
    for ak in ANCHORS:
        regs, fl = FLOWS[ak]
        specs.append((gid, f'rc/fe/replace/time_dummies__{ak}', 'rc/fe/replace/time_dummies',
                       ak, 'baseline', 'mkt_t', mk_f(xdv, regs, fe='mkt+year_month_num'),
                       'wgt', None, None, lb, 'mkt+ym', fl, '', f'Time dummies, {fl}'))
        specs.append((gid, f'rc/fe/replace/time_quadratic__{ak}', 'rc/fe/replace/time_quadratic',
                       ak, 'baseline', 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num','ym_num_sq']),
                       'wgt', None, None, lb, 'mkt', f'ym+ym^2+{fl}', '', f'Quad time, {fl}'))

    # 5. RC/WEIGHTS: 2 x 3 = 6
    for ak in ANCHORS:
        regs, fl = FLOWS[ak]
        f = mk_f(xdv, regs, extra_rhs=['ym_num'])
        specs.append((gid, f'rc/weights/main/unweighted__{ak}', 'rc/weights/main/unweighted',
                       ak, 'baseline', 'mkt_t', f,
                       None, None, None, f'{lb}, unwt', 'mkt', f'ym+{fl}', '', f'Unwt, {fl}'))
        specs.append((gid, f'rc/weights/trim/p99__{ak}', 'rc/weights/trim/p99',
                       ak, 'baseline', 'mkt_t', f,
                       'wgt_p99', None, None, f'{lb}, trim wgt', 'mkt', f'ym+{fl}', '', f'Trim p99, {fl}'))

    # 6. RC/DATA/AGG: 1 x 3 = 3
    for ak in ANCHORS:
        regs, fl = FLOWS[ak]
        specs.append((gid, f'rc/data/aggregation/coarser_markets__{ak}', 'rc/data/aggregation/coarser_markets',
                       ak, 'baseline', 'mkt_t_coarse', mk_f(xdv, regs, fe='mkt_coarse', extra_rhs=['ym_num']),
                       'wgt', None, None, f'{lb}, coarse mkt', 'mkt_coarse', f'ym+{fl}', '',
                       f'Coarser mkts, {fl}'))

    # 7. INFER: 4 x 3 = 12
    for ak in ANCHORS:
        regs, fl = FLOWS[ak]
        f = mk_f(xdv, regs, extra_rhs=['ym_num'])
        specs.append((gid, f'infer/se/hc/hc1__{ak}', 'infer/se/hc/hc1',
                       ak, 'baseline', 'mkt_t', f,
                       'wgt', None, None, lb, 'mkt', f'ym+{fl}', 'HC1', f'HC1, {fl}'))
        specs.append((gid, f'infer/se/cluster/market__{ak}', 'infer/se/cluster/market',
                       ak, 'baseline', 'mkt_t', f,
                       'wgt', None, 'mkt', lb, 'mkt', f'ym+{fl}', 'cl(mkt)', f'Cl mkt, {fl}'))
        specs.append((gid, f'infer/se/cluster/time__{ak}', 'infer/se/cluster/time',
                       ak, 'baseline', 'mkt_t', f,
                       'wgt', None, 'year_month_num', lb, 'mkt', f'ym+{fl}', 'cl(ym)', f'Cl time, {fl}'))
        specs.append((gid, f'infer/se/cluster/market_time__{ak}', 'infer/se/cluster/market_time',
                       ak, 'baseline', 'mkt_t', f,
                       'wgt', None, ['mkt','year_month_num'], lb, 'mkt', f'ym+{fl}', 'cl(mkt+ym)', f'2way cl, {fl}'))

    # 8. CROSS: no_lag_depvar x non-anchors (2)
    for fk in ['ee_ue','grouped_flows']:
        regs, fl = FLOWS[fk]
        specs.append((gid, f'rc/controls/first_stage/no_lag_depvar__{fk}', 'rc/controls/first_stage/no_lag_depvar',
                       fk, 'no_lag_depvar', 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num']),
                       'wgt', None, None, lb, 'mkt', f'ym+{fl}(no_lag)', '', f'No lag dv x {fl}'))

    # 9. CROSS: cluster_mkt x non-anchors (2)
    for fk in ['ee_ue','grouped_flows']:
        regs, fl = FLOWS[fk]
        specs.append((gid, f'infer/se/cluster/market__{fk}', 'infer/se/cluster/market',
                       fk, 'baseline', 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num']),
                       'wgt', None, 'mkt', lb, 'mkt', f'ym+{fl}', 'cl(mkt)', f'Cl mkt x {fl}'))

    # 10. CROSS: unweighted x non-anchors (2)
    for fk in ['ee_ue','grouped_flows']:
        regs, fl = FLOWS[fk]
        specs.append((gid, f'rc/weights/main/unweighted__{fk}', 'rc/weights/main/unweighted',
                       fk, 'baseline', 'mkt_t', mk_f(xdv, regs, extra_rhs=['ym_num']),
                       None, None, None, f'{lb}, unwt', 'mkt', f'ym+{fl}', '', f'Unwt x {fl}'))

print(f"\nTotal specs: {len(specs)}")
for g in ['G1','G2','G3','G4']:
    print(f"  {g}: {sum(1 for s in specs if s[0]==g)}")

# ============================================================
# Execute
# ============================================================
fe_combo_cache = {}
results = []
n_failed = 0
run_id = 0
t_start = time.time()

def sort_key(s):
    gid = s[0]; fs_key = s[4]; fe_col = s[5]
    return (GROUPS[gid]['hourly'], fs_key, fe_col, GROUPS[gid]['depvar'])

specs.sort(key=sort_key)
current_fe_key = None

for i, sp in enumerate(specs):
    gid, spec_id, tree_path, flow_key, fs_key, fe_col, formula, wt, sfilt, clust, \
        sample_desc, fe_desc, ctrl_desc, clust_desc, notes = sp

    dv = GROUPS[gid]['depvar']
    xdv = f'xd{dv}'
    run_id += 1
    srid = f'{PAPER_ID}_run{run_id:04d}'
    combo_key = (dv, fs_key, fe_col)

    if (i+1) % 20 == 1 or combo_key not in fe_combo_cache:
        elapsed = time.time() - t_start
        print(f"\n[{i+1}/{len(specs)}] {elapsed:.0f}s | {gid} {spec_id}", flush=True)

    try:
        if combo_key not in fe_combo_cache:
            print(f"  Building FS combo: {dv}, {fs_key}, {fe_col}", flush=True)
            fe_combo_cache[combo_key] = build_first_stage(dv, fs_key, fe_col)
            gc.collect()

        if current_fe_key != combo_key:
            if current_fe_key is not None:
                cleanup_fe(current_fe_key[0])
            apply_fe(fe_combo_cache[combo_key], dv, fe_col)
            current_fe_key = combo_key

        r = run_2nd(formula, 'xee', wt, sfilt, clust)

        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': srid,
            'spec_id': spec_id,
            'spec_tree_path': tree_path,
            'baseline_group_id': gid,
            'outcome_var': xdv,
            'treatment_var': 'xee',
            'coefficient': r['coefficient'],
            'std_error': r['std_error'],
            'p_value': r['p_value'],
            'ci_lower': r['ci_lower'],
            'ci_upper': r['ci_upper'],
            'n_obs': r['n_obs'],
            'r_squared': r['r_squared'],
            'coefficient_vector_json': json.dumps(r['coef_vec']),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': ctrl_desc,
            'cluster_var': clust_desc,
            'notes': notes,
        })

    except Exception as e:
        n_failed += 1
        import traceback
        traceback.print_exc()
        print(f"  FAILED [{spec_id}]: {e}", flush=True)
        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': srid,
            'spec_id': spec_id,
            'spec_tree_path': tree_path,
            'baseline_group_id': gid,
            'outcome_var': xdv,
            'treatment_var': 'xee',
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': 0,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({'error': str(e)}),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': ctrl_desc,
            'cluster_var': clust_desc,
            'notes': f'FAILED: {e}',
        })

    # Periodic save + gc
    if (i+1) % 50 == 0:
        gc.collect()
        # Partial save
        rdf_partial = pd.DataFrame(results)
        rdf_partial.to_csv(os.path.join(OUT_DIR, 'specification_results_partial.csv'), index=False)
        print(f"  [Saved partial results: {len(results)} rows]", flush=True)

# ============================================================
# Write final outputs
# ============================================================
total_time = time.time() - t_start
print(f"\n{'='*60}")
print(f"COMPLETE: {len(specs)} specs, {len(specs)-n_failed} succeeded, {n_failed} failed")
print(f"Total execution time: {total_time:.0f}s")
print(f"Unique first-stage regressions: {len(_fs_cache)}")

rdf = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, 'specification_results.csv')
rdf.to_csv(out_path, index=False)
print(f"Wrote {len(rdf)} rows to {out_path}")

for g in ['G1','G2','G3','G4']:
    gd = rdf[(rdf['baseline_group_id']==g) & rdf['coefficient'].notna()]
    if len(gd)>0:
        print(f"\n{g} ({GROUPS[g]['label']}): {len(gd)} specs")
        print(f"  coef range: [{gd['coefficient'].min():.6f}, {gd['coefficient'].max():.6f}]")
        print(f"  median: {gd['coefficient'].median():.6f}")
        print(f"  N range: [{gd['n_obs'].min()}, {gd['n_obs'].max()}]")

print(f"\nTotal time (incl load): {time.time()-t0:.0f}s")
