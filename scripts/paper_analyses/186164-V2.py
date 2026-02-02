"""
Specification Search: 186164-V2
Paper: "Reducing Inequality Through Dynamic Complementarity: Evidence from Head Start and Public School Spending"
Authors: Rucker C. Johnson and C. Kirabo Jackson
Journal: AEJ: Economic Policy

Method: Difference-in-Differences / Event Study with Panel Fixed Effects

NOTE: The main individual-level PSID analysis data is restricted/sensitive and not available.
This specification search uses the publicly available district-level panel data for the
School Finance Reform (SFR) event study analysis (Appendix C/F).
"""

import pandas as pd
import numpy as np
import json
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

PAPER_ID = "186164-V2"
JOURNAL = "AEJ: Economic Policy"
PAPER_TITLE = "Reducing Inequality Through Dynamic Complementarity: Evidence from Head Start and Public School Spending"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/186164-V2/Results"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/186164-V2"

print("Loading data...")
df = pd.read_stata(f"{DATA_PATH}/SFE_regression_data_1999_income.dta")
print(f"Loaded {len(df)} observations")

df['trend'] = df['year'] - 1965
df['lnpop60'] = np.log(df['pop60'].clip(lower=1))

def assign_division(fips):
    if fips in [9, 23, 25, 33, 44, 50]: return 1
    elif fips in [34, 36, 42]: return 2
    elif fips in [18, 17, 26, 39, 55]: return 3
    elif fips in [19, 20, 27, 29, 31, 38, 46]: return 4
    elif fips in [10, 11, 12, 13, 24, 37, 45, 51, 54]: return 5
    elif fips in [1, 21, 28, 47]: return 6
    elif fips in [5, 22, 40, 48]: return 7
    elif fips in [4, 8, 16, 35, 30, 49, 32, 56]: return 8
    elif fips in [2, 6, 15, 41, 53]: return 9
    return np.nan

df['divisioncat'] = df['FIPSTATE'].apply(assign_division)
df['regioncat'] = np.where(df['divisioncat'].isin([1, 2]), 1,
                  np.where(df['divisioncat'].isin([3, 4]), 2,
                  np.where(df['divisioncat'].isin([5, 6, 7]), 3,
                  np.where(df['divisioncat'].isin([8, 9]), 4, np.nan))))

for control in ['povrate60', 'pct_black_1960', 'pct_urban_1960', 'lnpop60', 'CensusGovt1962_v36']:
    df[f'{control}xyr'] = df[control] * df['trend']
    df[f'miss_{control}'] = df[control].isna().astype(int)
    df[f'{control}xyr'] = df[f'{control}xyr'].fillna(0)

baseline_controls = ['povrate60xyr', 'pct_black_1960xyr', 'pct_urban_1960xyr', 'lnpop60xyr', 'CensusGovt1962_v36xyr',
                    'miss_povrate60', 'miss_pct_black_1960', 'miss_pct_urban_1960', 'miss_lnpop60', 'miss_CensusGovt1962_v36']

for div in range(1, 10):
    if div != 5:
        df[f'div{div}xyr'] = (df['divisioncat'] == div).astype(int) * df['trend']
        baseline_controls.append(f'div{div}xyr')

for col in baseline_controls:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df['post_sfr'] = (df['foundation_time'] > 0).astype(int)
df['post_sfr_intensity'] = df['foundation_time'].clip(lower=0)
df['post_case'] = (df['case_time'] > 0).astype(int)
df['post_eq_spend'] = (df['eq_spend_time'] > 0).astype(int)
df['post_taxlimit'] = (df['taxlimit_time'] > 0).astype(int)
df['post_spend_limit'] = (df['case_limit_time'] > 0).astype(int)

df['id'] = df['id'].astype(int)
df['year'] = df['year'].astype(int)
results = []

def run_spec(df_sub, outcome, treatment, controls, fe_col, cluster_col, spec_id, spec_tree_path, weight_col=None, model_desc="Panel FE"):
    try:
        vars_needed = list(set([outcome, treatment] + [c for c in controls if c in df_sub.columns] +
                              ([fe_col] if fe_col and fe_col in df_sub.columns else []) +
                              ([cluster_col] if cluster_col in df_sub.columns else []) +
                              ([weight_col] if weight_col and weight_col in df_sub.columns else [])))
        df_clean = df_sub[vars_needed].dropna().reset_index(drop=True)
        if len(df_clean) < 100:
            return None
        if fe_col and fe_col in df_clean.columns:
            for v in [outcome, treatment] + [c for c in controls if c in df_clean.columns]:
                if v in df_clean.columns:
                    df_clean[v] = df_clean.groupby(fe_col)[v].transform(lambda x: x - x.mean())
        y = df_clean[outcome]
        X = df_clean[[treatment] + [c for c in controls if c in df_clean.columns]]
        X = sm.add_constant(X, has_constant='add')
        if weight_col and weight_col in df_clean.columns:
            model = sm.WLS(y, X, weights=np.maximum(df_clean[weight_col].values, 0.001))
        else:
            model = sm.OLS(y, X)
        if cluster_col in df_clean.columns:
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': df_clean[cluster_col].values})
        else:
            fit = model.fit(cov_type='HC1')
        if treatment not in fit.params.index:
            return None
        coef = fit.params[treatment]
        se = fit.bse[treatment]
        pval = fit.pvalues[treatment]
        coef_dict = {v: {'coef': float(fit.params[v]), 'se': float(fit.bse[v]), 'pval': float(fit.pvalues[v])} for v in fit.params.index if v != 'const'}
        return {
            'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
            'spec_id': spec_id, 'spec_tree_path': spec_tree_path,
            'outcome_var': outcome, 'treatment_var': treatment,
            'coefficient': coef, 'std_error': se, 't_stat': coef/se if se > 0 else np.nan,
            'p_value': pval, 'ci_lower': coef - 1.96*se, 'ci_upper': coef + 1.96*se,
            'n_obs': int(fit.nobs), 'r_squared': fit.rsquared,
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': f'District panel, N={int(fit.nobs)}',
            'fixed_effects': fe_col if fe_col else 'None',
            'controls_desc': f'{len(controls)} controls', 'cluster_var': cluster_col,
            'n_clusters': df_clean[cluster_col].nunique() if cluster_col in df_clean.columns else np.nan,
            'model_type': model_desc, 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {str(e)[:80]}")
        return None

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

print("\n1. BASELINE")
result = run_spec(df, 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', 'baseline', 'methods/difference_in_differences.md#baseline', 'size')
if result:
    results.append(result)
    print(f"  coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

print("\n2. TREATMENT VARIATIONS")
for treat_var, name in [('post_sfr_intensity', 'intensity'), ('post_case', 'court_order'), ('post_eq_spend', 'eq_spend'), ('post_taxlimit', 'tax_limit'), ('post_spend_limit', 'spend_limit')]:
    result = run_spec(df, 'outcome1', treat_var, baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/treatment/{name}', 'methods/difference_in_differences.md#treatment-definition', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

print("\n3. FE VARIATIONS")
for fe, spec_id, desc in [('FIPSTATE', 'did/fe/state', 'State FE'), ('year', 'did/fe/time', 'Year FE'), ('regioncat', 'did/fe/region', 'Region FE'), ('divisioncat', 'did/fe/division', 'Division FE'), (None, 'did/fe/none', 'No FE')]:
    result = run_spec(df, 'outcome1', 'post_sfr', baseline_controls, fe, 'FIPSTATE', spec_id, 'methods/difference_in_differences.md#fixed-effects', 'size', desc)
    if result:
        results.append(result)
        print(f"  {desc}: coef={result['coefficient']:.4f}")

print("\n4. CONTROL VARIATIONS")
result = run_spec(df, 'outcome1', 'post_sfr', [], 'FIPSTATE', 'FIPSTATE', 'did/controls/none', 'methods/difference_in_differences.md#control-sets', 'size')
if result:
    results.append(result)
    print(f"  No controls: coef={result['coefficient']:.4f}")

result = run_spec(df, 'outcome1', 'post_sfr', ['povrate60xyr', 'miss_povrate60'], 'FIPSTATE', 'FIPSTATE', 'did/controls/minimal', 'methods/difference_in_differences.md#control-sets', 'size')
if result:
    results.append(result)
    print(f"  Minimal: coef={result['coefficient']:.4f}")

for name, ctrls in {'drop_poverty': [c for c in baseline_controls if 'pov' not in c.lower()], 'drop_race': [c for c in baseline_controls if 'black' not in c.lower()], 'drop_urban': [c for c in baseline_controls if 'urban' not in c.lower()], 'drop_pop': [c for c in baseline_controls if 'pop' not in c.lower()], 'drop_govt': [c for c in baseline_controls if 'Govt' not in c], 'drop_division': [c for c in baseline_controls if 'div' not in c.lower()]}.items():
    result = run_spec(df, 'outcome1', 'post_sfr', ctrls, 'FIPSTATE', 'FIPSTATE', f'robust/control/{name}', 'robustness/leave_one_out.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

for name, ctrls in [('add_poverty', ['povrate60xyr', 'miss_povrate60']), ('add_race', ['povrate60xyr', 'miss_povrate60', 'pct_black_1960xyr', 'miss_pct_black_1960']), ('add_urban', ['povrate60xyr', 'miss_povrate60', 'pct_black_1960xyr', 'miss_pct_black_1960', 'pct_urban_1960xyr', 'miss_pct_urban_1960']), ('add_pop', ['povrate60xyr', 'miss_povrate60', 'pct_black_1960xyr', 'miss_pct_black_1960', 'pct_urban_1960xyr', 'miss_pct_urban_1960', 'lnpop60xyr', 'miss_lnpop60'])]:
    result = run_spec(df, 'outcome1', 'post_sfr', ctrls, 'FIPSTATE', 'FIPSTATE', f'robust/control/{name}', 'robustness/control_progression.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

print("\n5. CLUSTERING")
for clust, name in [('id', 'district'), ('FIPSTATE', 'state'), ('divisioncat', 'division'), ('regioncat', 'region')]:
    result = run_spec(df, 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', clust, f'robust/cluster/{name}', 'robustness/clustering_variations.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: se={result['std_error']:.4f}")

print("\n6. SAMPLE RESTRICTIONS")
for name, cond in [('early', df['year'] < 1985), ('late', df['year'] >= 1985)]:
    result = run_spec(df[cond], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/sample/{name}_period', 'robustness/sample_restrictions.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

for reg, name in [(1, 'northeast'), (2, 'midwest'), (3, 'south'), (4, 'west')]:
    result = run_spec(df[df['regioncat'] == reg], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/sample/{name}', 'robustness/sample_restrictions.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

for q, name in [(1, 'income_low'), (3, 'income_mid'), (4, 'income_mid_high'), (6, 'income_high')]:
    result = run_spec(df[df['q_income'] == q], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/sample/{name}', 'robustness/sample_restrictions.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

df_wins = df.copy()
for col in ['outcome1', 'outcome2']:
    lo, hi = df_wins[col].quantile([0.01, 0.99])
    df_wins[col] = df_wins[col].clip(lower=lo, upper=hi)
result = run_spec(df_wins, 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', 'robust/sample/winsorize', 'robustness/sample_restrictions.md', 'size')
if result:
    results.append(result)
    print(f"  winsorize: coef={result['coefficient']:.4f}")

df_trim = df[(df['outcome1'] > df['outcome1'].quantile(0.01)) & (df['outcome1'] < df['outcome1'].quantile(0.99))]
result = run_spec(df_trim, 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', 'robust/sample/trim', 'robustness/sample_restrictions.md', 'size')
if result:
    results.append(result)
    print(f"  trim: coef={result['coefficient']:.4f}")

print("\n7. OUTCOMES")
df['outcome_ihs'] = np.arcsinh(df['outcome2'])
df['outcome_std'] = (df['outcome1'] - df['outcome1'].mean()) / df['outcome1'].std()
df['ln_ppe2012'] = np.log(df['ppe2012'].clip(lower=1))

for out, name in [('outcome2', 'levels'), ('outcome_ihs', 'ihs'), ('outcome_std', 'std'), ('ln_ppe2012', 'ppe2012')]:
    result = run_spec(df[df[out].notna()], out, 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/outcome/{name}', 'robustness/functional_form.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

print("\n8. WEIGHTS")
for wt, name in [(None, 'unweighted'), ('pop60', 'population'), ('size', 'enrollment')]:
    subset = df if wt is None else df[df[wt].notna()] if wt else df
    result = run_spec(subset, 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/weights/{name}', 'robustness/sample_restrictions.md', wt)
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

print("\n9. HETEROGENEITY")
df['high_pov'] = (df['povrate60'] > df['povrate60'].median()).astype(int)
df['high_black'] = (df['pct_black_1960'] > df['pct_black_1960'].median()).astype(int)
df['urban_d'] = (df['pct_urban_1960'] > 50).astype(int)
df['large'] = (df['size'] > df['size'].median()).astype(int)

for name, het in [('poverty', 'high_pov'), ('black', 'high_black'), ('urban', 'urban_d'), ('size', 'large')]:
    df[f'post_sfr_x_{name}'] = df['post_sfr'] * df[het]
    result = run_spec(df, 'outcome1', f'post_sfr_x_{name}', baseline_controls + ['post_sfr'], 'FIPSTATE', 'FIPSTATE', f'robust/heterogeneity/{name}_interact', 'robustness/heterogeneity.md', 'size')
    if result:
        results.append(result)
        print(f"  {name} interact: coef={result['coefficient']:.4f}")

for name, cond in [('high_poverty', df['povrate60'] > df['povrate60'].median()), ('low_poverty', df['povrate60'] <= df['povrate60'].median()), ('urban', df['pct_urban_1960'] > 50), ('rural', df['pct_urban_1960'] <= 50)]:
    result = run_spec(df[cond], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/heterogeneity/{name}', 'robustness/heterogeneity.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

print("\n10. PLACEBO")
df_pre = df[df['foundation_time'] < 0].copy()
df_pre['fake_treat'] = (df_pre['foundation_time'] > -5).astype(int)
result = run_spec(df_pre, 'outcome1', 'fake_treat', baseline_controls, 'FIPSTATE', 'FIPSTATE', 'robust/placebo/pre_treatment', 'robustness/placebo_tests.md', 'size')
if result:
    results.append(result)
    print(f"  pre_treatment: coef={result['coefficient']:.4f}")

np.random.seed(42)
for i in range(3):
    df_perm = df.copy()
    df_perm['post_sfr_perm'] = np.random.permutation(df_perm['post_sfr'].values)
    result = run_spec(df_perm, 'outcome1', 'post_sfr_perm', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/placebo/permutation_{i+1}', 'robustness/placebo_tests.md', 'size')
    if result:
        results.append(result)
        print(f"  perm_{i+1}: coef={result['coefficient']:.4f}")

df_s = df.sort_values(['id', 'year'])
df['lead_post_sfr'] = df_s.groupby('id')['post_sfr'].shift(-3)
result = run_spec(df[df['lead_post_sfr'].notna()], 'outcome1', 'lead_post_sfr', baseline_controls + ['post_sfr'], 'FIPSTATE', 'FIPSTATE', 'robust/placebo/lead', 'robustness/placebo_tests.md', 'size')
if result:
    results.append(result)
    print(f"  lead: coef={result['coefficient']:.4f}")

print("\n11. ADDITIONAL SAMPLES")
for year in [1967, 1985, 1999]:
    result = run_spec(df[df['year'] != year], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/sample/drop_{year}', 'robustness/sample_restrictions.md', 'size')
    if result:
        results.append(result)
        print(f"  drop_{year}: coef={result['coefficient']:.4f}")

reform_states = df[df['foundation'] == 1]['FIPSTATE'].unique()
result = run_spec(df[df['FIPSTATE'].isin(reform_states)], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', 'did/sample/reform_states', 'methods/difference_in_differences.md#sample-restrictions', 'size')
if result:
    results.append(result)
    print(f"  reform_states: coef={result['coefficient']:.4f}")

for q, name in [(1, 'low_spend72'), (6, 'high_spend72')]:
    result = run_spec(df[df['q_spend72'] == q], 'outcome1', 'post_sfr', baseline_controls, 'FIPSTATE', 'FIPSTATE', f'robust/sample/{name}', 'robustness/sample_restrictions.md', 'size')
    if result:
        results.append(result)
        print(f"  {name}: coef={result['coefficient']:.4f}")

print("\n12. FIRST DIFF")
df_s = df.sort_values(['id', 'year'])
df['d_outcome'] = df_s.groupby('id')['outcome1'].diff()
df['d_post_sfr'] = df_s.groupby('id')['post_sfr'].diff()
df_fd = df[(df['d_outcome'].notna()) & (df['d_post_sfr'].notna())]
result = run_spec(df_fd, 'd_outcome', 'd_post_sfr', [], 'year', 'FIPSTATE', 'did/method/first_diff', 'methods/difference_in_differences.md#estimation-method', 'size')
if result:
    results.append(result)
    print(f"  first_diff: coef={result['coefficient']:.4f}")

df_first = df.groupby('id').first().reset_index()
df_last = df.groupby('id').last().reset_index()
df_ld = df_first[['id', 'FIPSTATE', 'size', 'post_sfr', 'outcome1']].copy()
df_ld.columns = ['id', 'FIPSTATE', 'size', 'post_sfr_f', 'out_f']
df_ld = df_ld.merge(df_last[['id', 'post_sfr', 'outcome1']], on='id')
df_ld['d_out_long'] = df_ld['outcome1'] - df_ld['out_f']
df_ld['d_post_long'] = df_ld['post_sfr'] - df_ld['post_sfr_f']
result = run_spec(df_ld[df_ld['d_post_long'].notna()], 'd_out_long', 'd_post_long', [], None, 'FIPSTATE', 'did/method/long_diff', 'methods/difference_in_differences.md#estimation-method', 'size')
if result:
    results.append(result)
    print(f"  long_diff: coef={result['coefficient']:.4f}")

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_PATH}/specification_results.csv", index=False)
print(f"\nSaved {len(results_df)} specifications")

if len(results_df) > 0:
    print(f"\nTotal: {len(results_df)}")
    print(f"Positive: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Sig 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Sig 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Mean: {results_df['coefficient'].mean():.4f}")
    print(f"Median: {results_df['coefficient'].median():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

print("\nDone!")
