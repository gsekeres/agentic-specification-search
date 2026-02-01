"""
Specification Search: Paper 116347-V1
"Workplace Friendships and Productivity"

This script replicates the main analysis and runs systematic specification searches
for a paper studying peer effects in the workplace using random assignment of workers
to work positions in a fish processing plant in Myanmar.

Main hypothesis: Working alongside friends affects productivity
Main finding: Working alongside friends DECREASES productivity by ~4-6%

Treatment variables:
- along_fr: Friend works alongside (highest proximity) - MAIN TREATMENT
- across_fr: Friend works across the table (medium proximity)
- low_fr: Friend works nearby but not along/across (low proximity)

Instruments:
- assign_along_fr, assign_across_fr, assign_low_fr (random position assignments)

Author: Specification Search Agent
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf

# =============================================================================
# SETUP
# =============================================================================

BASE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_DIR = os.path.join(BASE_DIR, 'data/downloads/extracted/116347-V1/data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/downloads/extracted/116347-V1')

PAPER_ID = '116347-V1'
PAPER_TITLE = 'Workplace Friendships and Productivity'
JOURNAL = 'AEJ: Applied Economics'

# =============================================================================
# DATA LOADING AND CONSTRUCTION
# =============================================================================

def load_data():
    """Load and merge all required data files."""

    # Load base data
    df_work = pd.read_stata(os.path.join(DATA_DIR, 'X1X2X3_WORKREPORT_DEC052016.dta'))
    df_worked_pos = pd.read_stata(os.path.join(DATA_DIR, 'X1X2X3_ID_WORKEDPOS_NOV232016_2.dta'))
    df_assigned_pos = pd.read_stata(os.path.join(DATA_DIR, 'X1X2X3_ID_ASSIGNEDPOS_NOV232016_2.dta'))
    df_pos_table = pd.read_stata(os.path.join(DATA_DIR, 'X1X2X3_POS_TABLEPAIR.dta'))
    df_m1 = pd.read_stata(os.path.join(DATA_DIR, 'Survey_M1_NOV282016.dta'))
    df_m2 = pd.read_stata(os.path.join(DATA_DIR, 'Survey_M2_NOV292016.dta'))
    df_m3 = pd.read_stata(os.path.join(DATA_DIR, 'Survey_M3_DEC012016.dta'))

    return df_work, df_worked_pos, df_assigned_pos, df_pos_table, df_m1, df_m2, df_m3


def build_friendship_network(df_m2):
    """Build friendship network from survey data."""
    friendships = df_m2[['id', 'friendid']].dropna()
    friendships = friendships.astype({'id': int, 'friendid': int})
    friend_set = set()
    for _, row in friendships.iterrows():
        friend_set.add((row['id'], row['friendid']))
    return friend_set


def get_proximity_type(pos1, pos2, table_size):
    """
    Determine proximity type between two positions on the same table.
    Returns: 'along', 'across', 'low', or None if same position
    """
    if pos1 == pos2:
        return None

    p1, p2 = min(pos1, pos2), max(pos1, pos2)

    if table_size == 6:
        along_pairs = {(1,2), (2,3), (4,5), (5,6)}
        across_pairs = {(1,4), (2,5), (3,6)}

        if (p1, p2) in along_pairs or (p2, p1) in along_pairs:
            return 'along'
        elif (p1, p2) in across_pairs or (p2, p1) in across_pairs:
            return 'across'
        else:
            return 'low'

    elif table_size == 4:
        along_pairs = {(1,2), (3,4)}
        across_pairs = {(1,3), (2,4)}

        if (p1, p2) in along_pairs or (p2, p1) in along_pairs:
            return 'along'
        elif (p1, p2) in across_pairs or (p2, p1) in across_pairs:
            return 'across'
        else:
            return 'low'
    else:
        return 'low'


def build_analysis_data(df_work, df_assigned_pos, df_pos_table, friend_set):
    """Build the analysis dataset with proximity measures."""

    # Position-table mapping
    pos_to_table = {}
    table_positions = {}
    for _, row in df_pos_table.dropna().iterrows():
        g = int(row['group'])
        pos = int(row['num_position'])
        tbl = int(row['num_table'])
        pos_to_table[(g, pos)] = tbl
        if (g, tbl) not in table_positions:
            table_positions[(g, tbl)] = []
        table_positions[(g, tbl)].append(pos)

    def is_friend(id1, id2):
        return (id1, id2) in friend_set

    # Process experiment period
    df_exp = df_work[df_work['attend'] == 1].copy()
    df_exp['prod'] = df_exp['work'] / (df_exp['mins'] / 60)
    df_exp['logprod'] = np.log(df_exp['prod'])

    worker_day_data = []

    for (group, date), day_df in df_exp.groupby(['group', 'date']):
        assigned_this_day = df_assigned_pos[(df_assigned_pos['group'] == group) &
                                             (df_assigned_pos['date'] == date)]
        assigned_map = dict(zip(assigned_this_day['num_position'], assigned_this_day['id']))
        worked_map = dict(zip(day_df['num_position'], day_df['id']))

        for _, row in day_df.iterrows():
            worker_id = int(row['id'])
            my_pos = int(row['num_position'])
            my_table = pos_to_table.get((int(group), my_pos))

            if my_table is None:
                continue

            table_poss = table_positions.get((int(group), my_table), [])
            table_size = len(table_poss)
            sorted_table_poss = sorted(table_poss)
            my_pos_in_table = sorted_table_poss.index(my_pos) + 1

            # Initialize proximity indicators
            has_along_friend = has_across_friend = has_low_friend = 0
            assign_along_friend = assign_across_friend = assign_low_friend = 0

            for other_pos in table_poss:
                if other_pos == my_pos:
                    continue

                other_pos_in_table = sorted_table_poss.index(other_pos) + 1
                prox_type = get_proximity_type(my_pos_in_table, other_pos_in_table, table_size)

                if prox_type is None:
                    continue

                # Check actual position
                other_worker = worked_map.get(other_pos)
                if other_worker is not None and is_friend(worker_id, int(other_worker)):
                    if prox_type == 'along':
                        has_along_friend = 1
                    elif prox_type == 'across':
                        has_across_friend = 1
                    elif prox_type == 'low':
                        has_low_friend = 1

                # Check assigned position
                assigned_worker = assigned_map.get(other_pos)
                if assigned_worker is not None and is_friend(worker_id, int(assigned_worker)):
                    if prox_type == 'along':
                        assign_along_friend = 1
                    elif prox_type == 'across':
                        assign_across_friend = 1
                    elif prox_type == 'low':
                        assign_low_friend = 1

            worker_day_data.append({
                'id': worker_id,
                'group': int(group),
                'date': int(date),
                'num_position': my_pos,
                'num_table': my_table,
                'prod': row['prod'],
                'logprod': row['logprod'],
                'along_fr': has_along_friend,
                'across_fr': has_across_friend,
                'low_fr': has_low_friend,
                'assign_along_fr': assign_along_friend,
                'assign_across_fr': assign_across_friend,
                'assign_low_fr': assign_low_friend,
            })

    df_analysis = pd.DataFrame(worker_day_data)
    df_analysis['groupdate'] = df_analysis['group'].astype(str) + '_' + df_analysis['date'].astype(str)
    df_analysis['near_fr'] = ((df_analysis['along_fr'] == 1) |
                              (df_analysis['across_fr'] == 1) |
                              (df_analysis['low_fr'] == 1)).astype(int)
    df_analysis['assign_near_fr'] = ((df_analysis['assign_along_fr'] == 1) |
                                      (df_analysis['assign_across_fr'] == 1) |
                                      (df_analysis['assign_low_fr'] == 1)).astype(int)

    return df_analysis


# =============================================================================
# ESTIMATION FUNCTIONS
# =============================================================================

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   controls_desc, fixed_effects, cluster_var, model_type='OLS'):
    """Extract results from a pyfixest model into standard format."""

    coef = model.coef()[treatment_var]
    se = model.se()[treatment_var]
    tstat = model.tstat()[treatment_var]
    pval = model.pvalue()[treatment_var]
    ci = model.confint()
    ci_lower = ci.loc[treatment_var, '2.5%']
    ci_upper = ci.loc[treatment_var, '97.5%']

    coef_vector = {
        'treatment': {'var': treatment_var, 'coef': float(coef), 'se': float(se), 'pval': float(pval)},
        'controls': [{'var': v, 'coef': float(model.coef()[v]), 'se': float(model.se()[v]),
                     'pval': float(model.pvalue()[v])} for v in model.coef().index if v != treatment_var],
        'fixed_effects': fixed_effects.split(' + ') if fixed_effects else [],
        'diagnostics': {}
    }

    return {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': spec_id, 'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var, 'treatment_var': treatment_var,
        'coefficient': float(coef), 'std_error': float(se), 't_stat': float(tstat),
        'p_value': float(pval), 'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper),
        'n_obs': int(model._N), 'r_squared': float(model._r2),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Experiment period', 'fixed_effects': fixed_effects,
        'controls_desc': controls_desc, 'cluster_var': cluster_var, 'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


def run_specifications(df):
    """Run all specifications and return results."""

    results = []

    # ===================
    # BASELINE
    # ===================
    print("Running baseline specification...")
    baseline = pf.feols("logprod ~ along_fr | id + groupdate", data=df, vcov={'CRV1': 'id'})
    results.append(extract_results(
        baseline, 'baseline', 'methods/panel_fixed_effects.md',
        'logprod', 'along_fr', 'None', 'id + groupdate', 'id', 'Panel FE'
    ))

    # ===================
    # OLS SPECIFICATIONS
    # ===================
    print("Running OLS specifications...")

    # Full with all proximity
    ols_full = pf.feols("logprod ~ low_fr + across_fr + along_fr | id + groupdate",
                        data=df, vcov={'CRV1': 'id'})
    for treat in ['low_fr', 'across_fr', 'along_fr']:
        results.append(extract_results(
            ols_full, f'panel/fe/twoway_{treat}', 'methods/panel_fixed_effects.md#fixed-effects',
            'logprod', treat, 'All proximity measures', 'id + groupdate', 'id', 'Panel FE'
        ))

    # Unit FE only
    ols_unit = pf.feols("logprod ~ low_fr + across_fr + along_fr | id", data=df, vcov={'CRV1': 'id'})
    results.append(extract_results(
        ols_unit, 'panel/fe/unit', 'methods/panel_fixed_effects.md#fixed-effects',
        'logprod', 'along_fr', 'All proximity measures', 'id', 'id', 'Panel FE'
    ))

    # Time FE only
    ols_time = pf.feols("logprod ~ low_fr + across_fr + along_fr | groupdate",
                        data=df, vcov={'CRV1': 'groupdate'})
    results.append(extract_results(
        ols_time, 'panel/fe/time', 'methods/panel_fixed_effects.md#fixed-effects',
        'logprod', 'along_fr', 'All proximity measures', 'groupdate', 'groupdate', 'Panel FE'
    ))

    # Pooled OLS
    ols_pooled = pf.feols("logprod ~ low_fr + across_fr + along_fr", data=df, vcov='hetero')
    results.append(extract_results(
        ols_pooled, 'panel/fe/none', 'methods/panel_fixed_effects.md#fixed-effects',
        'logprod', 'along_fr', 'All proximity measures', 'None', 'robust', 'Pooled OLS'
    ))

    # ===================
    # CLUSTERING VARIATIONS
    # ===================
    print("Running clustering variations...")

    for cluster_var, cluster_id in [('date', 'time'), ('group', 'unit'), (None, 'none')]:
        vcov = {'CRV1': cluster_var} if cluster_var else 'hetero'
        model = pf.feols("logprod ~ low_fr + across_fr + along_fr | id + groupdate",
                        data=df, vcov=vcov)
        results.append(extract_results(
            model, f'robust/cluster/{cluster_id}',
            'robustness/clustering_variations.md#single-level-clustering',
            'logprod', 'along_fr', 'All proximity measures', 'id + groupdate',
            cluster_var if cluster_var else 'robust', 'Panel FE'
        ))

    # ===================
    # FUNCTIONAL FORM
    # ===================
    print("Running functional form variations...")

    # Level outcome
    form_level = pf.feols("prod ~ low_fr + across_fr + along_fr | id + groupdate",
                          data=df, vcov={'CRV1': 'id'})
    results.append(extract_results(
        form_level, 'robust/form/y_level', 'robustness/functional_form.md',
        'prod', 'along_fr', 'All proximity measures', 'id + groupdate', 'id', 'Panel FE'
    ))

    # ===================
    # SAMPLE RESTRICTIONS
    # ===================
    print("Running sample restrictions...")

    median_date = df['date'].median()

    # Early/late period
    for name, sample_df in [('early_period', df[df['date'] <= median_date]),
                            ('late_period', df[df['date'] > median_date])]:
        model = pf.feols("logprod ~ low_fr + across_fr + along_fr | id + groupdate",
                        data=sample_df, vcov={'CRV1': 'id'})
        results.append(extract_results(
            model, f'robust/sample/{name}', 'robustness/sample_restrictions.md',
            'logprod', 'along_fr', 'All proximity measures', 'id + groupdate', 'id', 'Panel FE'
        ))

    # By group
    for g in [1, 2, 3]:
        model = pf.feols("logprod ~ low_fr + across_fr + along_fr | id + groupdate",
                        data=df[df['group'] == g], vcov={'CRV1': 'id'})
        results.append(extract_results(
            model, f'robust/sample/group_{g}', 'robustness/sample_restrictions.md',
            'logprod', 'along_fr', 'All proximity measures', 'id + groupdate', 'id', 'Panel FE'
        ))

    # Trimmed samples
    for pct, name in [(0.01, '1pct'), (0.05, '5pct')]:
        p_lo, p_hi = df['logprod'].quantile([pct, 1-pct])
        df_trim = df[(df['logprod'] >= p_lo) & (df['logprod'] <= p_hi)]
        model = pf.feols("logprod ~ low_fr + across_fr + along_fr | id + groupdate",
                        data=df_trim, vcov={'CRV1': 'id'})
        results.append(extract_results(
            model, f'robust/sample/trim_{name}', 'robustness/sample_restrictions.md',
            'logprod', 'along_fr', 'All proximity measures', 'id + groupdate', 'id', 'Panel FE'
        ))

    # ===================
    # LEAVE-ONE-OUT
    # ===================
    print("Running leave-one-out...")

    for dropped in ['low_fr', 'across_fr']:
        remaining = [v for v in ['low_fr', 'across_fr', 'along_fr'] if v != dropped]
        model = pf.feols(f"logprod ~ {' + '.join(remaining)} | id + groupdate",
                        data=df, vcov={'CRV1': 'id'})
        results.append(extract_results(
            model, f'robust/loo/drop_{dropped}', 'robustness/leave_one_out.md',
            'logprod', 'along_fr', f'Dropped {dropped}', 'id + groupdate', 'id', 'Panel FE'
        ))

    # ===================
    # SINGLE COVARIATE
    # ===================
    print("Running single covariate...")

    for treat in ['along_fr', 'across_fr', 'low_fr', 'near_fr']:
        model = pf.feols(f"logprod ~ {treat} | id + groupdate", data=df, vcov={'CRV1': 'id'})
        results.append(extract_results(
            model, f'robust/single/{treat.replace("_fr", "_only")}',
            'robustness/single_covariate.md',
            'logprod', treat, f'{treat} only', 'id + groupdate', 'id', 'Panel FE'
        ))

    # ===================
    # IV SPECIFICATIONS
    # ===================
    print("Running IV specifications...")

    # IV for each proximity measure
    for treat, inst in [('along_fr', 'assign_along_fr'),
                        ('across_fr', 'assign_across_fr'),
                        ('low_fr', 'assign_low_fr'),
                        ('near_fr', 'assign_near_fr')]:
        try:
            iv = pf.feols(f"logprod ~ 1 | id + groupdate | {treat} ~ {inst}",
                          data=df, vcov={'CRV1': 'id'})
            results.append({
                'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
                'spec_id': f'iv/method/2sls_{treat}',
                'spec_tree_path': 'methods/instrumental_variables.md',
                'outcome_var': 'logprod', 'treatment_var': treat,
                'coefficient': float(iv.coef()[treat]),
                'std_error': float(iv.se()[treat]),
                't_stat': float(iv.tstat()[treat]),
                'p_value': float(iv.pvalue()[treat]),
                'ci_lower': float(iv.confint().loc[treat, '2.5%']),
                'ci_upper': float(iv.confint().loc[treat, '97.5%']),
                'n_obs': int(iv._N), 'r_squared': float(iv._r2),
                'coefficient_vector_json': json.dumps({
                    'treatment': {'var': treat, 'coef': float(iv.coef()[treat])},
                    'first_stage': {'instrument': inst},
                    'fixed_effects': ['id', 'groupdate']
                }),
                'sample_desc': 'Experiment period', 'fixed_effects': 'id + groupdate',
                'controls_desc': f'{treat} instrumented by {inst}',
                'cluster_var': 'id', 'model_type': 'IV 2SLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
        except Exception as e:
            print(f"IV {treat} failed: {e}")

    # First stage
    for treat, inst in [('along_fr', 'assign_along_fr'),
                        ('across_fr', 'assign_across_fr'),
                        ('low_fr', 'assign_low_fr')]:
        fs = pf.feols(f"{treat} ~ {inst} | id + groupdate", data=df, vcov={'CRV1': 'id'})
        fstat = (fs.coef()[inst] / fs.se()[inst]) ** 2
        results.append({
            'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
            'spec_id': f'iv/first_stage/{treat}',
            'spec_tree_path': 'methods/instrumental_variables.md#first-stage',
            'outcome_var': treat, 'treatment_var': inst,
            'coefficient': float(fs.coef()[inst]),
            'std_error': float(fs.se()[inst]),
            't_stat': float(fs.tstat()[inst]),
            'p_value': float(fs.pvalue()[inst]),
            'ci_lower': float(fs.confint().loc[inst, '2.5%']),
            'ci_upper': float(fs.confint().loc[inst, '97.5%']),
            'n_obs': int(fs._N), 'r_squared': float(fs._r2),
            'coefficient_vector_json': json.dumps({
                'treatment': {'var': inst, 'coef': float(fs.coef()[inst])},
                'diagnostics': {'first_stage_F': float(fstat)},
                'fixed_effects': ['id', 'groupdate']
            }),
            'sample_desc': 'Experiment period', 'fixed_effects': 'id + groupdate',
            'controls_desc': f'First stage: {treat} on {inst}',
            'cluster_var': 'id', 'model_type': 'First Stage',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })

    # Reduced form
    rf = pf.feols("logprod ~ assign_along_fr | id + groupdate", data=df, vcov={'CRV1': 'id'})
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'iv/first_stage/reduced_form',
        'spec_tree_path': 'methods/instrumental_variables.md#first-stage',
        'outcome_var': 'logprod', 'treatment_var': 'assign_along_fr',
        'coefficient': float(rf.coef()['assign_along_fr']),
        'std_error': float(rf.se()['assign_along_fr']),
        't_stat': float(rf.tstat()['assign_along_fr']),
        'p_value': float(rf.pvalue()['assign_along_fr']),
        'ci_lower': float(rf.confint().loc['assign_along_fr', '2.5%']),
        'ci_upper': float(rf.confint().loc['assign_along_fr', '97.5%']),
        'n_obs': int(rf._N), 'r_squared': float(rf._r2),
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'assign_along_fr', 'coef': float(rf.coef()['assign_along_fr'])},
            'fixed_effects': ['id', 'groupdate']
        }),
        'sample_desc': 'Experiment period', 'fixed_effects': 'id + groupdate',
        'controls_desc': 'Reduced form: logprod on assign_along_fr',
        'cluster_var': 'id', 'model_type': 'Reduced Form',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # IV no FE
    iv_no_fe = pf.feols("logprod ~ 1 | 0 | along_fr ~ assign_along_fr", data=df, vcov='hetero')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'iv/fe/none',
        'spec_tree_path': 'methods/instrumental_variables.md#fixed-effects',
        'outcome_var': 'logprod', 'treatment_var': 'along_fr',
        'coefficient': float(iv_no_fe.coef()['along_fr']),
        'std_error': float(iv_no_fe.se()['along_fr']),
        't_stat': float(iv_no_fe.tstat()['along_fr']),
        'p_value': float(iv_no_fe.pvalue()['along_fr']),
        'ci_lower': float(iv_no_fe.confint().loc['along_fr', '2.5%']),
        'ci_upper': float(iv_no_fe.confint().loc['along_fr', '97.5%']),
        'n_obs': int(iv_no_fe._N), 'r_squared': float(iv_no_fe._r2),
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'along_fr', 'coef': float(iv_no_fe.coef()['along_fr'])},
            'fixed_effects': []
        }),
        'sample_desc': 'Experiment period', 'fixed_effects': 'None',
        'controls_desc': 'IV without fixed effects',
        'cluster_var': 'robust', 'model_type': 'IV 2SLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # IV sample restrictions
    for name, sample_df in [('early', df[df['date'] <= median_date]),
                            ('late', df[df['date'] > median_date])]:
        try:
            iv = pf.feols("logprod ~ 1 | id + groupdate | along_fr ~ assign_along_fr",
                          data=sample_df, vcov={'CRV1': 'id'})
            results.append({
                'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
                'spec_id': f'iv/sample/{name}_period',
                'spec_tree_path': 'methods/instrumental_variables.md#sample-restrictions',
                'outcome_var': 'logprod', 'treatment_var': 'along_fr',
                'coefficient': float(iv.coef()['along_fr']),
                'std_error': float(iv.se()['along_fr']),
                't_stat': float(iv.tstat()['along_fr']),
                'p_value': float(iv.pvalue()['along_fr']),
                'ci_lower': float(iv.confint().loc['along_fr', '2.5%']),
                'ci_upper': float(iv.confint().loc['along_fr', '97.5%']),
                'n_obs': int(iv._N), 'r_squared': float(iv._r2),
                'coefficient_vector_json': json.dumps({
                    'treatment': {'var': 'along_fr', 'coef': float(iv.coef()['along_fr'])},
                    'fixed_effects': ['id', 'groupdate']
                }),
                'sample_desc': f'{name.capitalize()} period', 'fixed_effects': 'id + groupdate',
                'controls_desc': f'IV on {name} period',
                'cluster_var': 'id', 'model_type': 'IV 2SLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
        except Exception as e:
            print(f"IV {name} period failed: {e}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Specification Search: Paper 116347-V1")
    print("Workplace Friendships and Productivity")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df_work, df_worked_pos, df_assigned_pos, df_pos_table, df_m1, df_m2, df_m3 = load_data()

    # Build friendship network
    print("Building friendship network...")
    friend_set = build_friendship_network(df_m2)
    print(f"  Number of friendship links: {len(friend_set)}")

    # Build analysis dataset
    print("Building analysis dataset...")
    df = build_analysis_data(df_work, df_assigned_pos, df_pos_table, friend_set)
    df = df.dropna(subset=['logprod', 'along_fr', 'across_fr', 'low_fr'])
    print(f"  Sample size: {len(df)}")
    print(f"  Number of workers: {df['id'].nunique()}")
    print(f"  Number of days: {df['date'].nunique()}")

    # Run specifications
    print("\nRunning specifications...")
    results = run_specifications(df)

    # Save results
    print("\nSaving results...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'specification_results.csv'), index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    df_main = df_results[df_results['treatment_var'] == 'along_fr']
    print(f"\nMain treatment (along_fr): {len(df_main)} specifications")
    print(f"  Coefficient range: [{df_main['coefficient'].min():.4f}, {df_main['coefficient'].max():.4f}]")
    print(f"  Median coefficient: {df_main['coefficient'].median():.4f}")
    print(f"  All negative: {(df_main['coefficient'] < 0).all()}")
    print(f"  All significant at 5%: {(df_main['p_value'] < 0.05).all()}")

    print(f"\nTotal specifications: {len(results)}")
    print(f"Results saved to: {OUTPUT_DIR}/specification_results.csv")
