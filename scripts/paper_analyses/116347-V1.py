"""
Specification Search: 116347-V1
Paper: Workplace Friendships and Productivity (Bandiera et al. 2017)

This script replicates and runs robustness checks on the main findings.
The paper examines how working near friends affects productivity using a field experiment
where workers were randomly assigned to positions in a fruit-packing plant.

Main hypothesis: Working alongside friends increases productivity
Treatment variables: has_friend_present, num_friends_present
Outcome: logprod (log productivity)

Note: All workers in this study are female, so gender cannot be used as a control/heterogeneity variable.

Method: Panel FE with worker and group-date FE, clustered SEs
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import pyfixest as pf
    PYFIXEST_AVAILABLE = True
except ImportError:
    PYFIXEST_AVAILABLE = False

# Constants
BASE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
PACKAGE_DIR = f'{BASE_DIR}/data/downloads/extracted/116347-V1'
DATA_DIR = f'{PACKAGE_DIR}/data'

PAPER_ID = '116347-V1'
JOURNAL = 'AEJ-Applied'
PAPER_TITLE = 'Workplace Friendships and Productivity'


def load_and_prepare_data():
    """Load all data files and merge them."""

    # Load experiment period work reports
    work_exp = pd.read_stata(f'{DATA_DIR}/X1X2X3_WORKREPORT_DEC052016.dta')
    work_exp['prior'] = 0

    # Load prior period work reports
    work_prior = pd.read_stata(f'{DATA_DIR}/X1X2X3_PRIOR_WORKREPORT_DEC052016.dta')
    work_prior['prior'] = 1
    work_prior['fullday'] = np.nan

    # Combine
    df = pd.concat([work_exp, work_prior], ignore_index=True)

    # Generate productivity and wage variables
    df['prod'] = (df['work'] / df['mins']) * 60
    df['logprod'] = np.log(df['prod'])
    df['basewage'] = np.where(df['fullday'] == 1, 40000,
                              np.where(df['fullday'] == 0, 20000, np.nan))
    df['piecewage'] = 1500 * df['work']
    df['totalwage'] = df['basewage'] + df['piecewage']
    df['logwage'] = np.log(df['totalwage'])

    # Load Survey Module 1 (demographics)
    m1 = pd.read_stata(f'{DATA_DIR}/Survey_M1_NOV282016.dta')
    df = df.merge(m1, on='id', how='inner')

    # Load Survey Module 3 (personality)
    m3 = pd.read_stata(f'{DATA_DIR}/Survey_M3_DEC012016.dta')
    df = df.merge(m3, on='id', how='inner')

    # Load friendship data
    friends = pd.read_stata(f'{DATA_DIR}/Survey_M2_NOV292016.dta')

    # Create friendship pairs set
    friend_pairs = set()
    for _, row in friends.iterrows():
        if pd.notna(row['friendid']):
            friend_pairs.add((int(row['id']), int(row['friendid'])))

    return df, friend_pairs


def create_friendship_vars(df, friend_pairs):
    """Create friendship proximity variables."""

    df = df.copy()

    # Get workers present each day
    workers_by_date = df.groupby(['group', 'date'])['id'].apply(set).to_dict()

    def count_friends_present(row):
        key = (row['group'], row['date'])
        if key not in workers_by_date:
            return 0
        present = workers_by_date[key]
        count = 0
        for other_id in present:
            if other_id != row['id']:
                if (int(row['id']), int(other_id)) in friend_pairs:
                    count += 1
        return count

    df['num_friends_present'] = df.apply(count_friends_present, axis=1)
    df['has_friend_present'] = (df['num_friends_present'] > 0).astype(int)

    return df


def run_ols_specification(df, outcome, treatment_vars, controls, fe_vars, cluster_var, spec_id, spec_tree_path):
    """Run an OLS/FE specification and return results dictionary.

    Note: cluster_var should be a single string (not a list) for pyfixest compatibility.
    """

    results = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome,
        'treatment_var': treatment_vars[0] if isinstance(treatment_vars, list) else treatment_vars,
    }

    try:
        # Convert treatment_vars to list
        treat_list = treatment_vars if isinstance(treatment_vars, list) else [treatment_vars]

        # Build all variables needed
        all_vars = [outcome] + treat_list
        if controls:
            all_vars.extend(controls)
        if cluster_var:
            all_vars.append(cluster_var)
        if fe_vars:
            all_vars.extend(fe_vars)

        # Remove duplicates while preserving order
        seen = set()
        all_vars = [x for x in all_vars if not (x in seen or seen.add(x))]

        # Filter data
        df_model = df[all_vars].dropna()

        if len(df_model) < 50:
            results['coefficient'] = np.nan
            results['std_error'] = np.nan
            results['t_stat'] = np.nan
            results['p_value'] = np.nan
            results['ci_lower'] = np.nan
            results['ci_upper'] = np.nan
            results['n_obs'] = len(df_model)
            results['r_squared'] = np.nan
            results['model_type'] = 'OLS'
            results['sample_desc'] = f'Insufficient observations (n={len(df_model)})'
            results['fixed_effects'] = str(fe_vars) if fe_vars else 'None'
            results['controls_desc'] = str(controls) if controls else 'None'
            results['cluster_var'] = str(cluster_var) if cluster_var else 'None'
            results['coefficient_vector_json'] = json.dumps({})
            results['estimation_script'] = f'scripts/paper_analyses/{PAPER_ID}.py'
            return results

        # Build formula for pyfixest
        treat_str = ' + '.join(treat_list)
        control_str = ' + '.join(controls) if controls else ''

        if control_str:
            formula = f"{outcome} ~ {treat_str} + {control_str}"
        else:
            formula = f"{outcome} ~ {treat_str}"

        if fe_vars:
            fe_str = ' + '.join(fe_vars)
            formula += f" | {fe_str}"

        # Clustering - must be a string for pyfixest
        if cluster_var:
            vcov = {'CRV1': cluster_var}
        else:
            vcov = 'hetero'

        model = pf.feols(formula, data=df_model, vcov=vcov)

        treat_var = treat_list[0]

        # Check if treatment variable is in results
        if treat_var not in model.coef().index:
            # Treatment was dropped (collinear with FE)
            results['coefficient'] = np.nan
            results['std_error'] = np.nan
            results['t_stat'] = np.nan
            results['p_value'] = np.nan
            results['ci_lower'] = np.nan
            results['ci_upper'] = np.nan
            results['n_obs'] = int(model._N)
            results['r_squared'] = float(model._r2) if hasattr(model, '_r2') else np.nan
            results['model_type'] = 'OLS_FE' if fe_vars else 'OLS'
            results['sample_desc'] = f'Treatment variable dropped (collinear). N={model._N}'
            results['fixed_effects'] = str(fe_vars) if fe_vars else 'None'
            results['controls_desc'] = str(controls) if controls else 'None'
            results['cluster_var'] = str(cluster_var) if cluster_var else 'None'
            results['coefficient_vector_json'] = json.dumps({'note': 'treatment dropped'})
            results['estimation_script'] = f'scripts/paper_analyses/{PAPER_ID}.py'
            return results

        results['coefficient'] = float(model.coef()[treat_var])
        results['std_error'] = float(model.se()[treat_var])
        results['t_stat'] = float(model.tstat()[treat_var])
        results['p_value'] = float(model.pvalue()[treat_var])

        ci = model.confint()
        results['ci_lower'] = float(ci.loc[treat_var, '2.5%'])
        results['ci_upper'] = float(ci.loc[treat_var, '97.5%'])
        results['n_obs'] = int(model._N)
        results['r_squared'] = float(model._r2) if hasattr(model, '_r2') else np.nan

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treat_var,
                'coef': float(model.coef()[treat_var]),
                'se': float(model.se()[treat_var]),
                'pval': float(model.pvalue()[treat_var])
            },
            'controls': [],
            'fixed_effects': fe_vars if fe_vars else [],
            'diagnostics': {}
        }

        # Add other treatment vars if multiple
        for tv in treat_list[1:]:
            if tv in model.coef().index:
                coef_vector['controls'].append({
                    'var': tv,
                    'coef': float(model.coef()[tv]),
                    'se': float(model.se()[tv]),
                    'pval': float(model.pvalue()[tv])
                })

        # Add controls
        if controls:
            for c in controls:
                if c in model.coef().index:
                    coef_vector['controls'].append({
                        'var': c,
                        'coef': float(model.coef()[c]),
                        'se': float(model.se()[c]),
                        'pval': float(model.pvalue()[c])
                    })

        results['coefficient_vector_json'] = json.dumps(coef_vector)
        results['model_type'] = 'OLS_FE' if fe_vars else 'OLS'
        results['sample_desc'] = f'N={results["n_obs"]}'
        results['fixed_effects'] = str(fe_vars) if fe_vars else 'None'
        results['controls_desc'] = str(controls) if controls else 'None'
        results['cluster_var'] = str(cluster_var) if cluster_var else 'None'
        results['estimation_script'] = f'scripts/paper_analyses/{PAPER_ID}.py'

    except Exception as e:
        results['coefficient'] = np.nan
        results['std_error'] = np.nan
        results['t_stat'] = np.nan
        results['p_value'] = np.nan
        results['ci_lower'] = np.nan
        results['ci_upper'] = np.nan
        results['n_obs'] = 0
        results['r_squared'] = np.nan
        results['model_type'] = 'ERROR'
        results['sample_desc'] = f'Error: {str(e)[:100]}'
        results['fixed_effects'] = str(fe_vars) if fe_vars else 'None'
        results['controls_desc'] = str(controls) if controls else 'None'
        results['cluster_var'] = str(cluster_var) if cluster_var else 'None'
        results['coefficient_vector_json'] = json.dumps({'error': str(e)[:200]})
        results['estimation_script'] = f'scripts/paper_analyses/{PAPER_ID}.py'

    return results


def run_all_specifications():
    """Run all specifications for the paper."""

    print("Loading and preparing data...")
    df, friend_pairs = load_and_prepare_data()

    # Create friendship variables
    print("Creating friendship variables...")
    df = create_friendship_vars(df, friend_pairs)

    # Create fixed effects
    df['groupdate'] = df['group'].astype(str) + '_' + df['date'].astype(str)
    df['id_str'] = df['id'].astype(str)
    df['date_str'] = df['date'].astype(str)
    df['group_str'] = df['group'].astype(str)

    # Standardize personality variables
    for var in ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']:
        if var in df.columns:
            df[f'std_{var[:4]}'] = (df[var] - df[var].mean()) / df[var].std()

    # Filter to experiment period with attendance
    df_exp = df[(df['prior'] == 0) & (df['attend'] == 1)].copy()
    df_prior = df[(df['prior'] == 1) & (df['attend'] == 1)].copy()

    print(f"Experiment period observations: {len(df_exp)}")
    print(f"Prior period observations: {len(df_prior)}")
    print(f"Treatment variation: {df_exp['has_friend_present'].value_counts().to_dict()}")

    results = []

    # Define main variables
    main_outcome = 'logprod'
    main_treatment = 'has_friend_present'
    alt_treatment = 'num_friends_present'

    # Controls
    personality_controls = ['std_extr', 'std_agre', 'std_cons', 'std_neur', 'std_open']
    available_pers = [c for c in personality_controls if c in df_exp.columns and df_exp[c].notna().sum() > 100]
    demographic_controls = ['age']
    available_demo = [c for c in demographic_controls if c in df_exp.columns and df_exp[c].notna().sum() > 100]

    print(f"Available controls: {available_demo + available_pers}")

    spec_num = 0

    # =========================================================================
    # BASELINE SPECIFICATIONS
    # =========================================================================
    print("\n1. Running baseline specifications...")

    # 1. Baseline - OLS with worker and group-date FE
    spec_num += 1
    print(f"  Spec {spec_num}: Baseline OLS with FE")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='baseline',
        spec_tree_path='methods/panel_fixed_effects.md#baseline'
    )
    results.append(r)

    # 2. Baseline with continuous treatment
    spec_num += 1
    print(f"  Spec {spec_num}: Baseline with continuous treatment")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[alt_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='baseline/continuous_treatment',
        spec_tree_path='methods/panel_fixed_effects.md#baseline'
    )
    results.append(r)

    # 3. Simple OLS without FE
    spec_num += 1
    print(f"  Spec {spec_num}: Simple OLS without FE")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=available_demo,
        fe_vars=None,
        cluster_var='id_str',
        spec_id='panel/fe/none',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure'
    )
    results.append(r)

    # =========================================================================
    # FIXED EFFECTS VARIATIONS
    # =========================================================================
    print("\n2. Running fixed effects variations...")

    # 4. Worker FE only
    spec_num += 1
    print(f"  Spec {spec_num}: Worker FE only")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str'],
        cluster_var='id_str',
        spec_id='panel/fe/unit',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure'
    )
    results.append(r)

    # 5. Date FE only
    spec_num += 1
    print(f"  Spec {spec_num}: Date FE only")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=available_demo,
        fe_vars=['date_str'],
        cluster_var='id_str',
        spec_id='panel/fe/time',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure'
    )
    results.append(r)

    # 6. Group FE only
    spec_num += 1
    print(f"  Spec {spec_num}: Group FE only")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=available_demo,
        fe_vars=['group_str'],
        cluster_var='id_str',
        spec_id='panel/fe/group',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure'
    )
    results.append(r)

    # 7. Group-date FE only
    spec_num += 1
    print(f"  Spec {spec_num}: Group-date FE only")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=available_demo,
        fe_vars=['groupdate'],
        cluster_var='id_str',
        spec_id='panel/fe/groupdate',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure'
    )
    results.append(r)

    # 8. Worker + Date FE (no group)
    spec_num += 1
    print(f"  Spec {spec_num}: Worker + Date FE")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'date_str'],
        cluster_var='id_str',
        spec_id='panel/fe/twoway',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure'
    )
    results.append(r)

    # =========================================================================
    # CLUSTERING VARIATIONS
    # =========================================================================
    print("\n3. Running clustering variations...")

    # 9. Cluster by worker
    spec_num += 1
    print(f"  Spec {spec_num}: Cluster by worker")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/cluster/unit',
        spec_tree_path='robustness/clustering_variations.md'
    )
    results.append(r)

    # 10. Cluster by date
    spec_num += 1
    print(f"  Spec {spec_num}: Cluster by date")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='date_str',
        spec_id='robust/cluster/time',
        spec_tree_path='robustness/clustering_variations.md'
    )
    results.append(r)

    # 11. Cluster by group
    spec_num += 1
    print(f"  Spec {spec_num}: Cluster by group")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='group_str',
        spec_id='robust/cluster/group',
        spec_tree_path='robustness/clustering_variations.md'
    )
    results.append(r)

    # 12. Robust SE (no clustering)
    spec_num += 1
    print(f"  Spec {spec_num}: Robust SE (no clustering)")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var=None,
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md'
    )
    results.append(r)

    # 13. Cluster by groupdate
    spec_num += 1
    print(f"  Spec {spec_num}: Cluster by group-date")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='groupdate',
        spec_id='robust/cluster/groupdate',
        spec_tree_path='robustness/clustering_variations.md'
    )
    results.append(r)

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n4. Running sample restriction variations...")

    # 14. Early period
    spec_num += 1
    median_date = df_exp['date'].median()
    df_early = df_exp[df_exp['date'] < median_date].copy()
    print(f"  Spec {spec_num}: Early period only (n={len(df_early)})")
    r = run_ols_specification(
        df=df_early,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/early_period',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 15. Late period
    spec_num += 1
    df_late = df_exp[df_exp['date'] >= median_date].copy()
    print(f"  Spec {spec_num}: Late period only (n={len(df_late)})")
    r = run_ols_specification(
        df=df_late,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/late_period',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 16-18. By group
    for grp in df_exp['group'].dropna().unique():
        spec_num += 1
        df_grp = df_exp[df_exp['group'] == grp].copy()
        print(f"  Spec {spec_num}: Group {int(grp)} only (n={len(df_grp)})")
        r = run_ols_specification(
            df=df_grp,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'date_str'],
            cluster_var='id_str',
            spec_id=f'robust/sample/group_{int(grp)}',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

    # 19-21. Drop each group
    for grp in df_exp['group'].dropna().unique():
        spec_num += 1
        df_no_grp = df_exp[df_exp['group'] != grp].copy()
        print(f"  Spec {spec_num}: Excluding group {int(grp)} (n={len(df_no_grp)})")
        r = run_ols_specification(
            df=df_no_grp,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id=f'robust/sample/drop_group_{int(grp)}',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

    # 22-24. Winsorize outcomes
    for pct in [1, 5, 10]:
        spec_num += 1
        df_wins = df_exp.copy()
        lower = df_wins['logprod'].quantile(pct/100)
        upper = df_wins['logprod'].quantile(1 - pct/100)
        df_wins['logprod'] = df_wins['logprod'].clip(lower=lower, upper=upper)
        print(f"  Spec {spec_num}: Winsorize at {pct}%")
        r = run_ols_specification(
            df=df_wins,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id=f'robust/sample/winsorize_{pct}pct',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

    # 25. Trim extremes
    spec_num += 1
    lower = df_exp['logprod'].quantile(0.01)
    upper = df_exp['logprod'].quantile(0.99)
    df_trim = df_exp[(df_exp['logprod'] > lower) & (df_exp['logprod'] < upper)].copy()
    print(f"  Spec {spec_num}: Trim 1% extremes (n={len(df_trim)})")
    r = run_ols_specification(
        df=df_trim,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/trim_1pct',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 26. Full-day workers only
    spec_num += 1
    df_fullday = df_exp[df_exp['fullday'] == 1].copy()
    print(f"  Spec {spec_num}: Full-day workers only (n={len(df_fullday)})")
    r = run_ols_specification(
        df=df_fullday,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/fullday_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 27. Part-day workers
    spec_num += 1
    df_partday = df_exp[df_exp['fullday'] == 0].copy()
    print(f"  Spec {spec_num}: Part-day workers only (n={len(df_partday)})")
    r = run_ols_specification(
        df=df_partday,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/partday_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # =========================================================================
    # AGE SUBGROUPS
    # =========================================================================
    print("\n5. Running age subgroup analyses...")

    if 'age' in df_exp.columns:
        median_age = df_exp['age'].median()

        # 28. Young workers
        spec_num += 1
        df_young = df_exp[df_exp['age'] < median_age].copy()
        print(f"  Spec {spec_num}: Young workers (<{median_age}) (n={len(df_young)})")
        r = run_ols_specification(
            df=df_young,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id='robust/sample/young',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

        # 29. Old workers
        spec_num += 1
        df_old = df_exp[df_exp['age'] >= median_age].copy()
        print(f"  Spec {spec_num}: Older workers (>={median_age}) (n={len(df_old)})")
        r = run_ols_specification(
            df=df_old,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id='robust/sample/old',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

        # 30-32. Age terciles
        age_terciles = df_exp['age'].quantile([0.33, 0.67]).values
        for i, (lower, upper) in enumerate([(0, age_terciles[0]), (age_terciles[0], age_terciles[1]), (age_terciles[1], 100)]):
            spec_num += 1
            df_tercile = df_exp[(df_exp['age'] >= lower) & (df_exp['age'] < upper)].copy()
            print(f"  Spec {spec_num}: Age tercile {i+1} (n={len(df_tercile)})")
            r = run_ols_specification(
                df=df_tercile,
                outcome=main_outcome,
                treatment_vars=[main_treatment],
                controls=None,
                fe_vars=['id_str', 'groupdate'],
                cluster_var='id_str',
                spec_id=f'robust/sample/age_tercile_{i+1}',
                spec_tree_path='robustness/sample_restrictions.md'
            )
            results.append(r)

    # =========================================================================
    # ALTERNATIVE OUTCOMES
    # =========================================================================
    print("\n6. Running alternative outcome specifications...")

    # 33. Level productivity
    spec_num += 1
    print(f"  Spec {spec_num}: Level productivity (not log)")
    r = run_ols_specification(
        df=df_exp,
        outcome='prod',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/outcome/prod_level',
        spec_tree_path='robustness/functional_form.md'
    )
    results.append(r)

    # 34. Work output
    spec_num += 1
    print(f"  Spec {spec_num}: Work output (kg)")
    r = run_ols_specification(
        df=df_exp,
        outcome='work',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/outcome/work',
        spec_tree_path='robustness/measurement.md'
    )
    results.append(r)

    # 35. Log work
    spec_num += 1
    df_exp['logwork'] = np.log(df_exp['work'] + 1)
    print(f"  Spec {spec_num}: Log work output")
    r = run_ols_specification(
        df=df_exp,
        outcome='logwork',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/outcome/logwork',
        spec_tree_path='robustness/functional_form.md'
    )
    results.append(r)

    # 36. Log wage
    spec_num += 1
    print(f"  Spec {spec_num}: Log wage")
    r = run_ols_specification(
        df=df_exp[df_exp['logwage'].notna()],
        outcome='logwage',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/outcome/logwage',
        spec_tree_path='robustness/measurement.md'
    )
    results.append(r)

    # 37. Minutes worked
    spec_num += 1
    print(f"  Spec {spec_num}: Minutes worked")
    r = run_ols_specification(
        df=df_exp,
        outcome='mins',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/outcome/mins',
        spec_tree_path='robustness/measurement.md'
    )
    results.append(r)

    # =========================================================================
    # FUNCTIONAL FORM
    # =========================================================================
    print("\n7. Running functional form variations...")

    # 38. IHS transformation
    spec_num += 1
    df_exp['ihs_prod'] = np.arcsinh(df_exp['prod'])
    print(f"  Spec {spec_num}: IHS productivity")
    r = run_ols_specification(
        df=df_exp,
        outcome='ihs_prod',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/funcform/ihs_outcome',
        spec_tree_path='robustness/functional_form.md'
    )
    results.append(r)

    # 39. Log(prod + 1)
    spec_num += 1
    df_exp['log_prod_plus1'] = np.log(df_exp['prod'] + 1)
    print(f"  Spec {spec_num}: Log(productivity + 1)")
    r = run_ols_specification(
        df=df_exp,
        outcome='log_prod_plus1',
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/funcform/log_plus1',
        spec_tree_path='robustness/functional_form.md'
    )
    results.append(r)

    # =========================================================================
    # HETEROGENEITY ANALYSIS
    # =========================================================================
    print("\n8. Running heterogeneity analysis...")

    # 40. Interaction with age (in non-FE model)
    if 'age' in df_exp.columns:
        spec_num += 1
        df_exp['treat_X_age'] = df_exp[main_treatment] * df_exp['age']
        print(f"  Spec {spec_num}: Heterogeneity by age (non-FE)")
        r = run_ols_specification(
            df=df_exp,
            outcome=main_outcome,
            treatment_vars=[main_treatment, 'treat_X_age'],
            controls=['age'],
            fe_vars=['groupdate'],
            cluster_var='id_str',
            spec_id='robust/heterogeneity/age',
            spec_tree_path='robustness/heterogeneity.md'
        )
        results.append(r)

    # 41-45. Interaction with personality traits
    for pers_var in available_pers:
        spec_num += 1
        interact_name = f'treat_X_{pers_var}'
        df_exp[interact_name] = df_exp[main_treatment] * df_exp[pers_var]
        print(f"  Spec {spec_num}: Heterogeneity by {pers_var}")
        r = run_ols_specification(
            df=df_exp,
            outcome=main_outcome,
            treatment_vars=[main_treatment, interact_name],
            controls=[pers_var],
            fe_vars=['groupdate'],
            cluster_var='id_str',
            spec_id=f'robust/heterogeneity/{pers_var}',
            spec_tree_path='robustness/heterogeneity.md'
        )
        results.append(r)

    # 46. Interaction with group
    spec_num += 1
    for grp in [2, 3]:
        df_exp[f'treat_X_group{grp}'] = df_exp[main_treatment] * (df_exp['group'] == grp).astype(int)
    print(f"  Spec {spec_num}: Heterogeneity by group")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment, 'treat_X_group2', 'treat_X_group3'],
        controls=None,
        fe_vars=['id_str', 'date_str'],
        cluster_var='id_str',
        spec_id='robust/heterogeneity/group',
        spec_tree_path='robustness/heterogeneity.md'
    )
    results.append(r)

    # 47. Intensity effect (continuous)
    spec_num += 1
    print(f"  Spec {spec_num}: Treatment intensity (continuous)")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[alt_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/heterogeneity/intensity',
        spec_tree_path='robustness/heterogeneity.md'
    )
    results.append(r)

    # =========================================================================
    # PLACEBO TESTS
    # =========================================================================
    print("\n9. Running placebo tests...")

    # 48. Prior period
    spec_num += 1
    print(f"  Spec {spec_num}: Prior period (placebo)")
    r = run_ols_specification(
        df=df_prior,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/placebo/prior_period',
        spec_tree_path='robustness/placebo_tests.md'
    )
    results.append(r)

    # 49-51. Fake cutoff dates
    dates_sorted = sorted(df_exp['date'].unique())
    n_dates = len(dates_sorted)
    for i, quartile in enumerate([0.25, 0.5, 0.75]):
        spec_num += 1
        cutoff_idx = int(n_dates * quartile)
        cutoff_date = dates_sorted[cutoff_idx]
        df_exp[f'fake_post_{i}'] = (df_exp['date'] >= cutoff_date).astype(int)
        print(f"  Spec {spec_num}: Fake timing cutoff at {quartile*100}%")
        r = run_ols_specification(
            df=df_exp,
            outcome=main_outcome,
            treatment_vars=[f'fake_post_{i}'],
            controls=None,
            fe_vars=['id_str', 'group_str'],
            cluster_var='id_str',
            spec_id=f'robust/placebo/fake_timing_{int(quartile*100)}',
            spec_tree_path='robustness/placebo_tests.md'
        )
        results.append(r)

    # =========================================================================
    # ALTERNATIVE TREATMENT DEFINITIONS
    # =========================================================================
    print("\n10. Running alternative treatment definitions...")

    # 52-54. Binary at different thresholds
    for thresh in [1, 2, 3]:
        spec_num += 1
        df_exp[f'has_{thresh}_friends'] = (df_exp['num_friends_present'] >= thresh).astype(int)
        print(f"  Spec {spec_num}: At least {thresh} friends present")
        r = run_ols_specification(
            df=df_exp,
            outcome=main_outcome,
            treatment_vars=[f'has_{thresh}_friends'],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id=f'robust/treatment/at_least_{thresh}_friends',
            spec_tree_path='robustness/measurement.md'
        )
        results.append(r)

    # 55. Log of number of friends
    spec_num += 1
    df_exp['log_num_friends'] = np.log(df_exp['num_friends_present'] + 1)
    print(f"  Spec {spec_num}: Log number of friends")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=['log_num_friends'],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/treatment/log_num_friends',
        spec_tree_path='robustness/measurement.md'
    )
    results.append(r)

    # =========================================================================
    # ESTIMATION METHOD VARIATIONS
    # =========================================================================
    print("\n11. Running estimation method variations...")

    # 56. First differences
    spec_num += 1
    df_exp_sorted = df_exp.sort_values(['id', 'date'])
    df_exp_sorted['logprod_diff'] = df_exp_sorted.groupby('id')['logprod'].diff()
    df_exp_sorted['treat_diff'] = df_exp_sorted.groupby('id')[main_treatment].diff()
    df_fd = df_exp_sorted.dropna(subset=['logprod_diff', 'treat_diff'])
    print(f"  Spec {spec_num}: First differences (n={len(df_fd)})")
    r = run_ols_specification(
        df=df_fd,
        outcome='logprod_diff',
        treatment_vars=['treat_diff'],
        controls=None,
        fe_vars=['groupdate'],
        cluster_var='id_str',
        spec_id='panel/method/first_diff',
        spec_tree_path='methods/panel_fixed_effects.md#estimation-method'
    )
    results.append(r)

    # =========================================================================
    # ADDITIONAL SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n12. Running additional sample restrictions...")

    obs_per_worker = df_exp.groupby('id').size()

    # 57. Balanced panel
    spec_num += 1
    max_obs = obs_per_worker.max()
    balanced_workers = obs_per_worker[obs_per_worker == max_obs].index
    df_balanced = df_exp[df_exp['id'].isin(balanced_workers)].copy()
    print(f"  Spec {spec_num}: Balanced panel only (n={len(df_balanced)})")
    r = run_ols_specification(
        df=df_balanced,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/balanced',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 58. Workers with >=5 observations
    spec_num += 1
    multi_obs_workers = obs_per_worker[obs_per_worker >= 5].index
    df_multi = df_exp[df_exp['id'].isin(multi_obs_workers)].copy()
    print(f"  Spec {spec_num}: Workers with >=5 observations (n={len(df_multi)})")
    r = run_ols_specification(
        df=df_multi,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/min_obs_5',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 59. Workers with >=10 observations
    spec_num += 1
    obs10_workers = obs_per_worker[obs_per_worker >= 10].index
    df_obs10 = df_exp[df_exp['id'].isin(obs10_workers)].copy()
    print(f"  Spec {spec_num}: Workers with >=10 observations (n={len(df_obs10)})")
    r = run_ols_specification(
        df=df_obs10,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/min_obs_10',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 60. Drop singletons
    spec_num += 1
    non_singleton_workers = obs_per_worker[obs_per_worker > 1].index
    df_no_single = df_exp[df_exp['id'].isin(non_singleton_workers)].copy()
    print(f"  Spec {spec_num}: Drop singleton workers (n={len(df_no_single)})")
    r = run_ols_specification(
        df=df_no_single,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/drop_singletons',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # 61-63. Time terciles
    date_terciles = np.percentile(df_exp['date'], [33, 67])
    for i, (lower, upper) in enumerate([(df_exp['date'].min(), date_terciles[0]),
                                         (date_terciles[0], date_terciles[1]),
                                         (date_terciles[1], df_exp['date'].max() + 1)]):
        spec_num += 1
        df_time = df_exp[(df_exp['date'] >= lower) & (df_exp['date'] < upper)].copy()
        print(f"  Spec {spec_num}: Time tercile {i+1} (n={len(df_time)})")
        r = run_ols_specification(
            df=df_time,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id=f'robust/sample/time_tercile_{i+1}',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

    # 64-65. High and low productivity workers
    worker_mean_prod = df_exp.groupby('id')['prod'].mean()
    median_prod = worker_mean_prod.median()

    spec_num += 1
    high_prod_workers = worker_mean_prod[worker_mean_prod >= median_prod].index
    df_high = df_exp[df_exp['id'].isin(high_prod_workers)].copy()
    print(f"  Spec {spec_num}: High productivity workers (n={len(df_high)})")
    r = run_ols_specification(
        df=df_high,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/high_prod_workers',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    spec_num += 1
    low_prod_workers = worker_mean_prod[worker_mean_prod < median_prod].index
    df_low = df_exp[df_exp['id'].isin(low_prod_workers)].copy()
    print(f"  Spec {spec_num}: Low productivity workers (n={len(df_low)})")
    r = run_ols_specification(
        df=df_low,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=None,
        fe_vars=['id_str', 'groupdate'],
        cluster_var='id_str',
        spec_id='robust/sample/low_prod_workers',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    results.append(r)

    # =========================================================================
    # CONTROLS WITH NO FE
    # =========================================================================
    print("\n13. Running specifications without worker FE...")

    # 66. With age control
    spec_num += 1
    print(f"  Spec {spec_num}: With age control (no worker FE)")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=['age'],
        fe_vars=['groupdate'],
        cluster_var='id_str',
        spec_id='robust/control/with_age',
        spec_tree_path='robustness/control_progression.md'
    )
    results.append(r)

    # 67. With personality controls
    spec_num += 1
    print(f"  Spec {spec_num}: With personality controls (no worker FE)")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=available_pers,
        fe_vars=['groupdate'],
        cluster_var='id_str',
        spec_id='robust/control/with_personality',
        spec_tree_path='robustness/control_progression.md'
    )
    results.append(r)

    # 68. With all controls
    spec_num += 1
    print(f"  Spec {spec_num}: With all controls (no worker FE)")
    r = run_ols_specification(
        df=df_exp,
        outcome=main_outcome,
        treatment_vars=[main_treatment],
        controls=available_demo + available_pers,
        fe_vars=['groupdate'],
        cluster_var='id_str',
        spec_id='robust/control/all_controls',
        spec_tree_path='robustness/control_progression.md'
    )
    results.append(r)

    # =========================================================================
    # Experience-based subgroups
    # =========================================================================
    if 'monthsonjob' in df_exp.columns:
        print("\n14. Running experience-based analyses...")

        median_exp = df_exp['monthsonjob'].median()

        # 69. Experienced workers
        spec_num += 1
        df_exp_workers = df_exp[df_exp['monthsonjob'] >= median_exp].copy()
        print(f"  Spec {spec_num}: Experienced workers (n={len(df_exp_workers)})")
        r = run_ols_specification(
            df=df_exp_workers,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id='robust/sample/experienced',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

        # 70. Inexperienced workers
        spec_num += 1
        df_inexp_workers = df_exp[df_exp['monthsonjob'] < median_exp].copy()
        print(f"  Spec {spec_num}: Inexperienced workers (n={len(df_inexp_workers)})")
        r = run_ols_specification(
            df=df_inexp_workers,
            outcome=main_outcome,
            treatment_vars=[main_treatment],
            controls=None,
            fe_vars=['id_str', 'groupdate'],
            cluster_var='id_str',
            spec_id='robust/sample/inexperienced',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        results.append(r)

    print(f"\n{'='*60}")
    print(f"TOTAL SPECIFICATIONS RUN: {spec_num}")
    print(f"{'='*60}")

    return results


def main():
    """Main function to run specification search and save results."""

    print("="*60)
    print("SPECIFICATION SEARCH: 116347-V1")
    print("Workplace Friendships and Productivity")
    print("="*60)

    # Run all specifications
    results = run_all_specifications()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = f'{PACKAGE_DIR}/specification_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    valid_results = results_df[results_df['coefficient'].notna()]
    print(f"Total specifications: {len(results_df)}")
    print(f"Valid specifications: {len(valid_results)}")

    if len(valid_results) > 0:
        print(f"\nCoefficient statistics:")
        print(f"  Mean: {valid_results['coefficient'].mean():.4f}")
        print(f"  Median: {valid_results['coefficient'].median():.4f}")
        print(f"  Std: {valid_results['coefficient'].std():.4f}")
        print(f"  Min: {valid_results['coefficient'].min():.4f}")
        print(f"  Max: {valid_results['coefficient'].max():.4f}")

        sig_05 = (valid_results['p_value'] < 0.05).sum()
        sig_01 = (valid_results['p_value'] < 0.01).sum()
        positive = (valid_results['coefficient'] > 0).sum()

        print(f"\nSignificance:")
        print(f"  Significant at 5%: {sig_05} ({100*sig_05/len(valid_results):.1f}%)")
        print(f"  Significant at 1%: {sig_01} ({100*sig_01/len(valid_results):.1f}%)")
        print(f"  Positive coefficients: {positive} ({100*positive/len(valid_results):.1f}%)")

    return results_df


if __name__ == '__main__':
    results_df = main()
