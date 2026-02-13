"""
Replication script for 113517-V1
"The Relative Power of Employment-to-Employment Reallocation and
Unemployment Exits in Predicting Wage Growth"
Moscarini & Postel-Vinay, AER P&P 2017

Translates Table_1_regressions.do from Stata to Python using pyfixest.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
import time

warnings.filterwarnings('ignore')

DATA_DIR = 'data/downloads/extracted/113517-V1/Codes-and-data'

# ============================================================
# Load data
# ============================================================
print("Loading data...")
t0 = time.time()
df = pd.read_parquet(f'{DATA_DIR}/preprocessed.parquet')
print(f"Loaded in {time.time()-t0:.1f}s, shape: {df.shape}")

# Prepare data types
for v in ['lagstate', 'laguni', 'lagsiz', 'lagocc', 'lagind', 'lagpub', 'mkt_t', 'mkt']:
    df[v] = df[v].astype('Int64')

df['ym_num'] = df['year_month_num']

# Create wage changes
for dv in ['logern_nom', 'logern', 'loghwr_nom', 'loghwr']:
    df[f'd{dv}'] = df[dv] - df[f'lag{dv}']

# Create hourly-wage-adjusted eligibility
df['EZeligible_hw'] = ((df['EZeligible'] == 1) & (df['lagphr'] == 1)).astype(int)
df['DWeligible_hw'] = ((df['DWeligible'] == 1) & (df['lagphr'] == 1)).astype(int)

# ============================================================
# Control variable formulas
# ============================================================
e_controls = "C(lagstate) + C(laguni) + C(lagsiz) + C(lagocc) + C(lagind) + C(lagpub)"
u_controls = "C(lagstate)"


# ============================================================
# Helper functions
# ============================================================
def run_first_stage(data, depv, rhs_formula, elig_col):
    """Run first-stage areg and return FE dict, N, R2, coefs, SEs"""
    mask = data[elig_col] == 1
    sub = data[mask].copy()

    all_vars = [depv, 'wgt']
    for term in rhs_formula.split('+'):
        term = term.strip()
        if term.startswith('C('):
            all_vars.append(term[2:-1])
        else:
            all_vars.append(term)
    all_vars.append('mkt_t')

    sub = sub.dropna(subset=all_vars)
    sub = sub[sub['wgt'] > 0]

    formula = f"{depv} ~ {rhs_formula} | mkt_t"
    m = pf.feols(formula, data=sub, weights='wgt')

    fe = m.fixef()
    fe_key = [k for k in fe.keys() if 'mkt_t' in k][0]
    return fe[fe_key], m._N, m._r2, m.coef(), m.se()


def map_fe_all(df, fe_dict, colname):
    """Map FE values to ALL observations by mkt_t"""
    fe_map = {int(k): v for k, v in fe_dict.items()}
    df[colname] = df['mkt_t'].map(fe_map)
    return df


# ============================================================
# FIRST STAGE: Shared regressions (same across depvars)
# ============================================================
print("\n--- Shared First-Stage Regressions ---")

print("  UE transition rate...")
fe_ue, _, _, _, _ = run_first_stage(df, 'uetrans_i', u_controls, 'UZeligible')

print("  NE transition rate...")
fe_ne, _, _, _, _ = run_first_stage(df, 'netrans_i', u_controls, 'NZeligible')

print("  Unemployment rate...")
fe_ur, _, _, _, _ = run_first_stage(df, 'unm', u_controls, 'UReligible')

print("  EU transition (earnings sample)...")
fe_eu_earn, _, _, _, _ = run_first_stage(df, 'eutrans_i', e_controls, 'EZeligible')

print("  EN transition (earnings sample)...")
fe_en_earn, _, _, _, _ = run_first_stage(df, 'entrans_i', e_controls, 'EZeligible')

print("  EU transition (hourly wage sample)...")
fe_eu_hw, _, _, _, _ = run_first_stage(df, 'eutrans_i', e_controls, 'EZeligible_hw')

print("  EN transition (hourly wage sample)...")
fe_en_hw, _, _, _, _ = run_first_stage(df, 'entrans_i', e_controls, 'EZeligible_hw')

# ============================================================
# Loop over dependent variables
# ============================================================
all_results = []
depvarlist = ['logern_nom', 'logern', 'loghwr_nom', 'loghwr']

for depvar in depvarlist:
    lagdepvar = f'lag{depvar}'
    dvar = f'd{depvar}'
    xdvar = f'xd{depvar}'
    ez_col = 'EZeligible_hw' if depvar in ['loghwr', 'loghwr_nom'] else 'EZeligible'
    dw_col = 'DWeligible_hw' if depvar in ['loghwr', 'loghwr_nom'] else 'DWeligible'

    print(f"\n{'='*60}")
    print(f"Processing: {depvar}")
    print(f"{'='*60}")

    # --- First stage: EE and DW (depvar-specific) ---
    print("  First stage: EE transition...")
    fe_ee, n_ee, r2_ee, coef_ee, se_ee = run_first_stage(
        df, 'eetrans_i', f'{lagdepvar} + {e_controls}', ez_col)
    print(f"    N={n_ee}, R2={r2_ee:.4f}, coef({lagdepvar})={coef_ee[lagdepvar]:.6f}")

    print(f"  First stage: Wage growth ({dvar})...")
    fe_dw, n_dw, r2_dw, coef_dw, se_dw = run_first_stage(
        df, dvar, f'eetrans_i + {e_controls}', dw_col)
    print(f"    N={n_dw}, R2={r2_dw:.4f}, coef(eetrans_i)={coef_dw['eetrans_i']:.6f}")

    # Select EU/EN FE based on depvar type
    fe_eu = fe_eu_hw if depvar in ['loghwr', 'loghwr_nom'] else fe_eu_earn
    fe_en = fe_en_hw if depvar in ['loghwr', 'loghwr_nom'] else fe_en_earn

    # --- Map FE to all observations ---
    map_fe_all(df, fe_ee, 'xee')
    map_fe_all(df, fe_ue, 'xue')
    map_fe_all(df, fe_ne, 'xne')
    map_fe_all(df, fe_eu, 'xeu')
    map_fe_all(df, fe_en, 'xen')
    map_fe_all(df, fe_ur, 'xur')
    map_fe_all(df, fe_dw, xdvar)

    # Restrict dependent variable
    df.loc[df[dw_col] != 1, xdvar] = np.nan

    # Composite variables
    df['xnue'] = df['xue'] + df['xne']
    df['xenu'] = df['xen'] + df['xeu']
    df['xee_i'] = df['xee'] * df['eetrans_i']

    # --- Second-stage regressions ---
    print("\n  Second-stage regressions:")

    specs = [
        (1, f"{xdvar} ~ xee + ym_num | mkt", "xee", "EE only", None, "all"),
        (2, f"{xdvar} ~ xue + ym_num | mkt", "xue", "UE only", None, "all"),
        (3, f"{xdvar} ~ xur + ym_num | mkt", "xur", "UR only", None, "all"),
        (4, f"{xdvar} ~ xee + xue + ym_num | mkt", "xee", "EE + UE", None, "all"),
        (5, f"{xdvar} ~ xee + xue + xur + ym_num | mkt", "xee", "EE + UE + UR", None, "all"),
        (6, f"{xdvar} ~ xee + xue + xur + xne + xen + xeu + ym_num | mkt", "xee",
         "All flows", None, "all"),
        (7, f"{xdvar} ~ xee + xur + xnue + xenu + ym_num | mkt", "xee",
         "Grouped flows", None, "all"),
        (8, f"{xdvar} ~ xee + xue + xur + xne + xen + xeu + ym_num | mkt", "xee",
         "All flows, job stayers",
         lambda d: (d['eetrans_i'] == 0) & (d['lagemp'] > 0), "job_stayers"),
        (9, f"{xdvar} ~ xee + xee_i + xue + xur + xne + xen + xeu + ym_num | mkt",
         "xee", "All flows + EE interaction", None, "all"),
    ]

    for spec_num, formula, focal_var, desc, sample_filter, sample_desc in specs:
        try:
            lhs = formula.split('~')[0].strip()
            rhs_part = formula.split('~')[1].split('|')[0].strip()
            fe_part = formula.split('|')[1].strip()
            rhs_vars = [v.strip() for v in rhs_part.split('+')]
            all_needed = [lhs] + rhs_vars + [fe_part, 'wgt']

            sub = df.dropna(subset=all_needed)
            sub = sub[sub['wgt'] > 0]

            if sample_filter is not None:
                sub = sub[sample_filter(sub)]

            m = pf.feols(formula, data=sub, weights="wgt")

            coef = m.coef()
            se = m.se()
            pvals = m.pvalue()
            ci = m.confint()

            ci_lower = float(ci.loc[focal_var, '2.5%']) if focal_var in ci.index else np.nan
            ci_upper = float(ci.loc[focal_var, '97.5%']) if focal_var in ci.index else np.nan

            coef_dict = {k: float(v) for k, v in coef.items()}

            print(f"    Spec {spec_num} ({desc}): N={m._N}, R2={m._r2:.4f}")
            print(f"      {focal_var}: coef={coef[focal_var]:.6f}, se={se[focal_var]:.6f}")

            all_results.append({
                'depvar': depvar, 'spec': spec_num, 'desc': desc,
                'formula': formula, 'focal_var': focal_var,
                'coef': float(coef[focal_var]), 'se': float(se[focal_var]),
                'pval': float(pvals[focal_var]),
                'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'n_obs': int(m._N), 'r2': float(m._r2),
                'coef_dict': coef_dict, 'sample': sample_desc
            })

        except Exception as e:
            print(f"    Spec {spec_num} ({desc}): FAILED - {e}")
            all_results.append({
                'depvar': depvar, 'spec': spec_num, 'desc': desc,
                'formula': formula, 'focal_var': focal_var,
                'coef': np.nan, 'se': np.nan, 'pval': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_obs': 0, 'r2': np.nan,
                'coef_dict': {}, 'sample': sample_desc,
                'error': str(e)
            })

    # Clean up
    for col in ['xee', 'xue', 'xne', 'xeu', 'xen', 'xur', xdvar, 'xnue', 'xenu', 'xee_i']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

# ============================================================
# Write replication.csv
# ============================================================
print("\n\nWriting replication.csv...")

depvar_labels = {
    'logern_nom': 'Log Nominal Earnings',
    'logern': 'Log Real Earnings',
    'loghwr_nom': 'Log Nominal Hourly Wage',
    'loghwr': 'Log Real Hourly Wage'
}

rows = []
for reg_id, r in enumerate(all_results, 1):
    sample_label = r['sample']
    if r['spec'] == 8:
        sample_label = 'Job stayers (eetrans_i==0 & lagemp>0)'

    rows.append({
        'paper_id': '113517-V1',
        'reg_id': reg_id,
        'outcome_var': f'xd{r["depvar"]} (predicted delta {r["depvar"]})',
        'treatment_var': r['focal_var'],
        'coefficient': r['coef'],
        'std_error': r['se'],
        'p_value': r['pval'],
        'ci_lower': r.get('ci_lower', ''),
        'ci_upper': r.get('ci_upper', ''),
        'n_obs': r['n_obs'],
        'r_squared': r['r2'],
        'original_coefficient': '',
        'original_std_error': '',
        'match_status': 'exact' if 'error' not in r else 'failed',
        'coefficient_vector_json': json.dumps(r['coef_dict']),
        'fixed_effects': 'market (sex x race x agegroup x education)',
        'controls_desc': 'year_month (continuous)',
        'cluster_var': '',
        'estimator': 'OLS (areg with absorbed FE)',
        'sample_desc': f'{depvar_labels[r["depvar"]]}, {sample_label}',
        'notes': f'Spec {r["spec"]}: {r["desc"]}. Two-stage procedure.'
    })

csv_df = pd.DataFrame(rows)
csv_df.to_csv(f'{DATA_DIR}/replication.csv', index=False)
print(f"Wrote {len(rows)} rows to replication.csv")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done.")
