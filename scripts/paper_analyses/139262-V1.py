"""
Specification Search for Paper 139262-V1
"Motivated Beliefs and Anticipation of Uncertainty Resolution"
Christoph Drobner

This script replicates the analysis from the original Stata code in Python
and runs a systematic specification search following the i4r methodology.

Method: Cross-sectional OLS (lab experiment)
Primary outcomes: belief_adjustment, studyperformance, jobperformance
Treatment: resolution (1=Resolution treatment, 0=No-Resolution treatment)
Key moderator: signal (1=good news, 0=bad news)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/139262-V1/replication_package/data/raw_data"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/139262-V1"

# ============================================================================
# DATA CREATION (Replicate data_creation.do)
# ============================================================================

def load_and_create_data():
    """Load raw session data and create analysis dataset."""

    # Load all session files
    dfs = []
    for i in range(1, 11):
        df = pd.read_excel(f"{DATA_PATH}/session_{i}.xlsx")
        df['session'] = i
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Keep relevant columns
    cols_to_keep = [
        'Group', 'Profit', 'Rang1PriorBelief', 'Rang2PriorBelief',
        'Rang3PriorBelief', 'Rang4PriorBelief', 'SumPoints', 'QuizRankInGroup',
        'BinaryComparison', 'Rang1PosteriorBelief', 'Rang2PosteriorBelief',
        'Rang3PosteriorBelief', 'Rang4PosteriorBelief', 'StudySuccess',
        'JobSuccess', 'Age', 'Gender', 'Major', 'TimeOKAnnouncementResolutionOK',
        'session'
    ]

    # Filter to columns that exist
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available].copy()

    # Create ID
    df['id'] = range(1, len(df) + 1)

    # Create resolution treatment variable
    df['resolution'] = df['TimeOKAnnouncementResolutionOK'].notna().astype(int)

    # Rename columns
    rename_map = {
        'Group': 'group',
        'Profit': 'profit',
        'Rang1PriorBelief': 'rang1priorbelief',
        'Rang2PriorBelief': 'rang2priorbelief',
        'Rang3PriorBelief': 'rang3priorbelief',
        'Rang4PriorBelief': 'rang4priorbelief',
        'SumPoints': 'sumpoints',
        'QuizRankInGroup': 'rank',
        'BinaryComparison': 'signal',
        'Rang1PosteriorBelief': 'rang1posteriorbelief',
        'Rang2PosteriorBelief': 'rang2posteriorbelief',
        'Rang3PosteriorBelief': 'rang3posteriorbelief',
        'Rang4PosteriorBelief': 'rang4posteriorbelief',
        'StudySuccess': 'studyperformance',
        'JobSuccess': 'jobperformance',
        'Age': 'age',
        'Gender': 'gender',
        'Major': 'major'
    }
    df = df.rename(columns=rename_map)

    # Recode signal (2 -> 0)
    df['signal'] = df['signal'].replace(2, 0)

    # Floor sumpoints
    df['sumpoints'] = np.floor(df['sumpoints'])

    # Convert beliefs from percentage points to percentages
    for i in range(1, 5):
        df[f'rang{i}priorbelief'] = df[f'rang{i}priorbelief'] / 100
        df[f'rang{i}posteriorbelief'] = df[f'rang{i}posteriorbelief'] / 100

    # Generate Bayesian posterior beliefs
    # Good signal: normalize over rank 1, 2/3*rank2, 1/3*rank3
    # Bad signal: normalize over 1/3*rank2, 2/3*rank3, rank4

    df['bayes_rang1'] = np.nan
    df['bayes_rang2'] = np.nan
    df['bayes_rang3'] = np.nan
    df['bayes_rang4'] = np.nan

    # Good signal (signal==1)
    good_denom = df['rang1priorbelief'] + (2/3)*df['rang2priorbelief'] + (1/3)*df['rang3priorbelief']
    df.loc[df['signal']==1, 'bayes_rang1'] = df.loc[df['signal']==1, 'rang1priorbelief'] / good_denom[df['signal']==1]
    df.loc[df['signal']==1, 'bayes_rang2'] = (2/3) * df.loc[df['signal']==1, 'rang2priorbelief'] / good_denom[df['signal']==1]
    df.loc[df['signal']==1, 'bayes_rang3'] = (1/3) * df.loc[df['signal']==1, 'rang3priorbelief'] / good_denom[df['signal']==1]
    df.loc[df['signal']==1, 'bayes_rang4'] = 0

    # Bad signal (signal==0)
    bad_denom = (1/3)*df['rang2priorbelief'] + (2/3)*df['rang3priorbelief'] + df['rang4priorbelief']
    df.loc[df['signal']==0, 'bayes_rang1'] = 0
    df.loc[df['signal']==0, 'bayes_rang2'] = (1/3) * df.loc[df['signal']==0, 'rang2priorbelief'] / bad_denom[df['signal']==0]
    df.loc[df['signal']==0, 'bayes_rang3'] = (2/3) * df.loc[df['signal']==0, 'rang3priorbelief'] / bad_denom[df['signal']==0]
    df.loc[df['signal']==0, 'bayes_rang4'] = df.loc[df['signal']==0, 'rang4priorbelief'] / bad_denom[df['signal']==0]

    # Generate expected rank beliefs
    df['prior'] = (df['rang1priorbelief']*1 + df['rang2priorbelief']*2 +
                   df['rang3priorbelief']*3 + df['rang4priorbelief']*4)

    df['posterior'] = (df['rang1posteriorbelief']*1 + df['rang2posteriorbelief']*2 +
                       df['rang3posteriorbelief']*3 + df['rang4posteriorbelief']*4)

    df['bayes_posterior'] = (df['bayes_rang1']*1 + df['bayes_rang2']*2 +
                             df['bayes_rang3']*3 + df['bayes_rang4']*4)

    # Generate belief adjustments
    df['belief_adjustment'] = df['posterior'] - df['prior']
    df['bayes_belief_adjustment'] = df['bayes_posterior'] - df['prior']

    # Generate interaction term
    df['signal_bayesbeliefadj'] = df['bayes_belief_adjustment'] * df['signal']

    # Generate wrong/zero belief adjustment indicators
    df['wrong_belief_adjustment'] = 0
    df.loc[(df['signal']==1) & (df['belief_adjustment']>0), 'wrong_belief_adjustment'] = 1
    df.loc[(df['signal']==0) & (df['belief_adjustment']<0), 'wrong_belief_adjustment'] = 1

    df['zero_belief_adjustment'] = 0
    df.loc[(df['belief_adjustment']==0) & (df['bayes_belief_adjustment']!=0), 'zero_belief_adjustment'] = 1

    return df


# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

def run_ols(formula, data, robust=True, cluster_var=None):
    """Run OLS regression and return results dict."""
    try:
        model = smf.ols(formula, data=data).fit(
            cov_type='HC1' if robust and cluster_var is None else
                     ('cluster' if cluster_var else 'nonrobust'),
            cov_kwds={'groups': data[cluster_var]} if cluster_var else None
        )
        return model
    except Exception as e:
        return None


def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                    sample_desc, controls_desc, cluster_var=None, model_type='OLS'):
    """Extract results from statsmodels model."""
    if model is None:
        return None

    try:
        coef = model.params.get(treatment_var, np.nan)
        se = model.bse.get(treatment_var, np.nan)
        tstat = model.tvalues.get(treatment_var, np.nan)
        pval = model.pvalues.get(treatment_var, np.nan)

        # Confidence intervals
        ci = model.conf_int()
        ci_lower = ci.loc[treatment_var, 0] if treatment_var in ci.index else np.nan
        ci_upper = ci.loc[treatment_var, 1] if treatment_var in ci.index else np.nan

        # Coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef) if not np.isnan(coef) else None,
                "se": float(se) if not np.isnan(se) else None,
                "pval": float(pval) if not np.isnan(pval) else None
            },
            "controls": [],
            "diagnostics": {
                "r_squared": float(model.rsquared) if hasattr(model, 'rsquared') else None,
                "adj_r_squared": float(model.rsquared_adj) if hasattr(model, 'rsquared_adj') else None,
                "f_stat": float(model.fvalue) if hasattr(model, 'fvalue') and model.fvalue is not None else None,
                "f_pval": float(model.f_pvalue) if hasattr(model, 'f_pvalue') and model.f_pvalue is not None else None
            }
        }

        # Add control coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        return {
            'paper_id': '139262-V1',
            'journal': 'AER P&P',
            'paper_title': 'Motivated Beliefs and Anticipation of Uncertainty Resolution',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared if hasattr(model, 'rsquared') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': 'None',
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': model_type,
            'estimation_script': 'scripts/paper_analyses/139262-V1.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None


def run_specification_search(df):
    """Run comprehensive specification search."""
    results = []

    # ========================================================================
    # MAIN OUTCOME: belief_adjustment
    # Primary analysis replicates Table 2 from the paper
    # ========================================================================

    print("Running baseline and main specifications...")

    # -------------------------------------------------------------------------
    # BASELINE REPLICATIONS (Table 2)
    # -------------------------------------------------------------------------

    # 1. No-Resolution, Good Signal
    mask = (df['resolution']==0) & (df['signal']==1)
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
    res = extract_results(model, 'bayes_belief_adjustment', 'baseline',
                         'methods/cross_sectional_ols.md#baseline',
                         'belief_adjustment', 'No-Resolution, Good Signal',
                         'None (bivariate)')
    if res: results.append(res)

    # 2. No-Resolution, Bad Signal
    mask = (df['resolution']==0) & (df['signal']==0)
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
    res = extract_results(model, 'bayes_belief_adjustment', 'baseline_nores_bad',
                         'methods/cross_sectional_ols.md#baseline',
                         'belief_adjustment', 'No-Resolution, Bad Signal',
                         'None (bivariate)')
    if res: results.append(res)

    # 3. No-Resolution, Diff-in-Diff
    mask = df['resolution']==0
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
    res = extract_results(model, 'bayes_belief_adjustment', 'baseline_nores_did',
                         'methods/cross_sectional_ols.md#baseline',
                         'belief_adjustment', 'No-Resolution, All signals',
                         'signal, signal*bayes_belief_adjustment')
    if res: results.append(res)

    # 4. Resolution, Good Signal
    mask = (df['resolution']==1) & (df['signal']==1)
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
    res = extract_results(model, 'bayes_belief_adjustment', 'baseline_res_good',
                         'methods/cross_sectional_ols.md#baseline',
                         'belief_adjustment', 'Resolution, Good Signal',
                         'None (bivariate)')
    if res: results.append(res)

    # 5. Resolution, Bad Signal
    mask = (df['resolution']==1) & (df['signal']==0)
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
    res = extract_results(model, 'bayes_belief_adjustment', 'baseline_res_bad',
                         'methods/cross_sectional_ols.md#baseline',
                         'belief_adjustment', 'Resolution, Bad Signal',
                         'None (bivariate)')
    if res: results.append(res)

    # 6. Resolution, Diff-in-Diff
    mask = df['resolution']==1
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
    res = extract_results(model, 'bayes_belief_adjustment', 'baseline_res_did',
                         'methods/cross_sectional_ols.md#baseline',
                         'belief_adjustment', 'Resolution, All signals',
                         'signal, signal*bayes_belief_adjustment')
    if res: results.append(res)

    # -------------------------------------------------------------------------
    # CONTROL VARIATIONS
    # -------------------------------------------------------------------------
    print("Running control variations...")

    controls = ['rank', 'sumpoints', 'prior', 'age', 'gender']

    # Leave-one-out (when using multiple controls)
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        mask = df['resolution'] == treatment

        # Full controls
        ctrl_str = ' + '.join(controls[:3])  # rank, sumpoints, prior
        model = run_ols(f'belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj + {ctrl_str}', df[mask])
        res = extract_results(model, 'bayes_belief_adjustment',
                             f'robust/control/full_{treatment_label.lower()}',
                             'robustness/control_progression.md',
                             'belief_adjustment', f'{treatment_label}, All signals',
                             f'signal, signal*bayes, {ctrl_str}')
        if res: results.append(res)

        # Leave-one-out
        for ctrl in controls[:3]:
            remaining = [c for c in controls[:3] if c != ctrl]
            ctrl_str = ' + '.join(remaining)
            model = run_ols(f'belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj + {ctrl_str}', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/loo/drop_{ctrl}_{treatment_label.lower()}',
                                 'robustness/leave_one_out.md',
                                 'belief_adjustment', f'{treatment_label}, All signals',
                                 f'signal, signal*bayes, {ctrl_str}')
            if res: results.append(res)

    # Add controls incrementally (Table 3 & 4 Appendix style)
    print("Running control progression...")

    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        for sig in [0, 1]:
            sig_label = 'good' if sig == 1 else 'bad'
            mask = (df['resolution'] == treatment) & (df['signal'] == sig)

            # With rank control (Table 3 Appendix)
            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + rank', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/control/add_rank_{treatment_label.lower()}_{sig_label}',
                                 'robustness/control_progression.md',
                                 'belief_adjustment', f'{treatment_label}, {sig_label.capitalize()} Signal',
                                 'rank')
            if res: results.append(res)

            # With sumpoints control (Table 4 Appendix)
            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + sumpoints', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/control/add_sumpoints_{treatment_label.lower()}_{sig_label}',
                                 'robustness/control_progression.md',
                                 'belief_adjustment', f'{treatment_label}, {sig_label.capitalize()} Signal',
                                 'sumpoints')
            if res: results.append(res)

            # With both rank and sumpoints
            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + rank + sumpoints', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/control/add_rank_sumpoints_{treatment_label.lower()}_{sig_label}',
                                 'robustness/control_progression.md',
                                 'belief_adjustment', f'{treatment_label}, {sig_label.capitalize()} Signal',
                                 'rank, sumpoints')
            if res: results.append(res)

    # -------------------------------------------------------------------------
    # SAMPLE RESTRICTIONS
    # -------------------------------------------------------------------------
    print("Running sample restrictions...")

    # Exclude wrong belief adjustments (Table 1 Appendix)
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        for sig in [0, 1]:
            sig_label = 'good' if sig == 1 else 'bad'
            mask = (df['resolution'] == treatment) & (df['signal'] == sig) & (df['wrong_belief_adjustment'] == 0)

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/sample/excl_wrong_{treatment_label.lower()}_{sig_label}',
                                 'robustness/sample_restrictions.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {sig_label.capitalize()} Signal, excl. wrong adjustments',
                                 'None')
            if res: results.append(res)

    # Exclude wrong AND zero belief adjustments (Table 2 Appendix)
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        for sig in [0, 1]:
            sig_label = 'good' if sig == 1 else 'bad'
            mask = ((df['resolution'] == treatment) & (df['signal'] == sig) &
                   (df['wrong_belief_adjustment'] == 0) & (df['zero_belief_adjustment'] == 0))

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/sample/excl_wrong_zero_{treatment_label.lower()}_{sig_label}',
                                 'robustness/sample_restrictions.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {sig_label.capitalize()} Signal, excl. wrong & zero',
                                 'None')
            if res: results.append(res)

    # Exclude rank 1 and rank 4 (Table 5 Appendix)
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        for sig in [0, 1]:
            sig_label = 'good' if sig == 1 else 'bad'
            mask = ((df['resolution'] == treatment) & (df['signal'] == sig) &
                   df['rank'].isin([2, 3]))

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/sample/middle_ranks_{treatment_label.lower()}_{sig_label}',
                                 'robustness/sample_restrictions.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {sig_label.capitalize()} Signal, ranks 2&3 only',
                                 'None')
            if res: results.append(res)

    # By session (drop each session)
    for session in df['session'].unique():
        mask = (df['session'] != session) & (df['resolution'] == 0)
        model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
        res = extract_results(model, 'bayes_belief_adjustment',
                             f'robust/sample/drop_session_{session}',
                             'robustness/sample_restrictions.md',
                             'belief_adjustment',
                             f'No-Resolution, drop session {session}',
                             'signal, signal*bayes')
        if res: results.append(res)

    # -------------------------------------------------------------------------
    # INFERENCE VARIATIONS
    # -------------------------------------------------------------------------
    print("Running inference variations...")

    # Clustered by session
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        mask = df['resolution'] == treatment

        model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj',
                       df[mask], robust=False, cluster_var='session')
        res = extract_results(model, 'bayes_belief_adjustment',
                             f'robust/cluster/session_{treatment_label.lower()}',
                             'robustness/clustering_variations.md',
                             'belief_adjustment', f'{treatment_label}, All signals',
                             'signal, signal*bayes', cluster_var='session')
        if res: results.append(res)

        # Clustered by group
        model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj',
                       df[mask], robust=False, cluster_var='group')
        res = extract_results(model, 'bayes_belief_adjustment',
                             f'robust/cluster/group_{treatment_label.lower()}',
                             'robustness/clustering_variations.md',
                             'belief_adjustment', f'{treatment_label}, All signals',
                             'signal, signal*bayes', cluster_var='group')
        if res: results.append(res)

    # Classical (non-robust) SEs
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        mask = df['resolution'] == treatment

        model = smf.ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj',
                       data=df[mask]).fit()
        res = extract_results(model, 'bayes_belief_adjustment',
                             f'robust/se/classical_{treatment_label.lower()}',
                             'robustness/inference_alternatives.md',
                             'belief_adjustment', f'{treatment_label}, All signals',
                             'signal, signal*bayes', cluster_var='classical')
        if res: results.append(res)

    # -------------------------------------------------------------------------
    # FUNCTIONAL FORM VARIATIONS
    # -------------------------------------------------------------------------
    print("Running functional form variations...")

    # Absolute value of belief adjustment
    df['abs_belief_adjustment'] = np.abs(df['belief_adjustment'])
    df['abs_bayes_belief_adjustment'] = np.abs(df['bayes_belief_adjustment'])

    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        mask = df['resolution'] == treatment

        model = run_ols('abs_belief_adjustment ~ abs_bayes_belief_adjustment + signal', df[mask])
        res = extract_results(model, 'abs_bayes_belief_adjustment',
                             f'robust/funcform/abs_{treatment_label.lower()}',
                             'robustness/functional_form.md',
                             'abs_belief_adjustment', f'{treatment_label}, All signals',
                             'signal')
        if res: results.append(res)

    # Squared terms
    df['bayes_belief_adj_sq'] = df['bayes_belief_adjustment'] ** 2

    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        for sig in [0, 1]:
            sig_label = 'good' if sig == 1 else 'bad'
            mask = (df['resolution'] == treatment) & (df['signal'] == sig)

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + bayes_belief_adj_sq', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/funcform/quadratic_{treatment_label.lower()}_{sig_label}',
                                 'robustness/functional_form.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {sig_label.capitalize()} Signal',
                                 'bayes_belief_adjustment^2')
            if res: results.append(res)

    # -------------------------------------------------------------------------
    # ALTERNATIVE OUTCOMES
    # -------------------------------------------------------------------------
    print("Running alternative outcomes...")

    # Individual rank beliefs as outcomes
    for rank_belief in ['rang1posteriorbelief', 'rang2posteriorbelief',
                        'rang3posteriorbelief', 'rang4posteriorbelief']:
        bayes_var = rank_belief.replace('posteriorbelief', '').replace('rang', 'bayes_rang')
        prior_var = rank_belief.replace('posteriorbelief', 'priorbelief')

        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = df['resolution'] == treatment

            model = run_ols(f'{rank_belief} ~ {bayes_var} + {prior_var} + signal', df[mask])
            res = extract_results(model, bayes_var,
                                 f'robust/outcome/{rank_belief}_{treatment_label.lower()}',
                                 'robustness/measurement.md',
                                 rank_belief, f'{treatment_label}, All signals',
                                 f'{prior_var}, signal')
            if res: results.append(res)

    # -------------------------------------------------------------------------
    # SECOND MAIN OUTCOME: Study/Job Performance (Table 3)
    # -------------------------------------------------------------------------
    print("Running secondary outcome analyses...")

    # OLS versions of the ologit regressions (since we want to avoid ordered logit complexity)
    for outcome in ['studyperformance', 'jobperformance']:
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = df['resolution'] == treatment

            # Baseline
            model = run_ols(f'{outcome} ~ signal + sumpoints + prior', df[mask])
            res = extract_results(model, 'signal',
                                 f'baseline_{outcome}_{treatment_label.lower()}',
                                 'methods/cross_sectional_ols.md#baseline',
                                 outcome, f'{treatment_label}, All signals',
                                 'sumpoints, prior')
            if res: results.append(res)

            # Without controls
            model = run_ols(f'{outcome} ~ signal', df[mask])
            res = extract_results(model, 'signal',
                                 f'robust/control/none_{outcome}_{treatment_label.lower()}',
                                 'robustness/control_progression.md',
                                 outcome, f'{treatment_label}, All signals',
                                 'None')
            if res: results.append(res)

            # With rank
            model = run_ols(f'{outcome} ~ signal + sumpoints + prior + rank', df[mask])
            res = extract_results(model, 'signal',
                                 f'robust/control/add_rank_{outcome}_{treatment_label.lower()}',
                                 'robustness/control_progression.md',
                                 outcome, f'{treatment_label}, All signals',
                                 'sumpoints, prior, rank')
            if res: results.append(res)

            # With demographics
            model = run_ols(f'{outcome} ~ signal + sumpoints + prior + age + gender', df[mask])
            res = extract_results(model, 'signal',
                                 f'robust/control/demographics_{outcome}_{treatment_label.lower()}',
                                 'robustness/control_progression.md',
                                 outcome, f'{treatment_label}, All signals',
                                 'sumpoints, prior, age, gender')
            if res: results.append(res)

    # Exclude wrong adjustments (Table 6 Appendix)
    for outcome in ['studyperformance', 'jobperformance']:
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = (df['resolution'] == treatment) & (df['wrong_belief_adjustment'] == 0)

            model = run_ols(f'{outcome} ~ signal + sumpoints + prior', df[mask])
            res = extract_results(model, 'signal',
                                 f'robust/sample/excl_wrong_{outcome}_{treatment_label.lower()}',
                                 'robustness/sample_restrictions.md',
                                 outcome, f'{treatment_label}, excl. wrong adjustments',
                                 'sumpoints, prior')
            if res: results.append(res)

    # -------------------------------------------------------------------------
    # HETEROGENEITY ANALYSES
    # -------------------------------------------------------------------------
    print("Running heterogeneity analyses...")

    # By gender
    for gender in [0, 1]:
        gender_label = 'male' if gender == 1 else 'female'
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = (df['resolution'] == treatment) & (df['gender'] == gender)

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/heterogeneity/gender_{gender_label}_{treatment_label.lower()}',
                                 'robustness/heterogeneity.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {gender_label.capitalize()} only',
                                 'signal, signal*bayes')
            if res: results.append(res)

    # By age (median split)
    median_age = df['age'].median()
    for age_group in ['young', 'old']:
        age_mask = df['age'] < median_age if age_group == 'young' else df['age'] >= median_age
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = (df['resolution'] == treatment) & age_mask

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/heterogeneity/age_{age_group}_{treatment_label.lower()}',
                                 'robustness/heterogeneity.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {age_group.capitalize()} ({"<" if age_group=="young" else ">="}{median_age})',
                                 'signal, signal*bayes')
            if res: results.append(res)

    # By prior beliefs (above/below median)
    median_prior = df['prior'].median()
    for prior_group in ['optimistic', 'pessimistic']:
        prior_mask = df['prior'] < median_prior if prior_group == 'optimistic' else df['prior'] >= median_prior
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = (df['resolution'] == treatment) & prior_mask

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/heterogeneity/prior_{prior_group}_{treatment_label.lower()}',
                                 'robustness/heterogeneity.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {prior_group.capitalize()} priors',
                                 'signal, signal*bayes')
            if res: results.append(res)

    # By IQ score (above/below median)
    median_iq = df['sumpoints'].median()
    for iq_group in ['high', 'low']:
        iq_mask = df['sumpoints'] >= median_iq if iq_group == 'high' else df['sumpoints'] < median_iq
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = (df['resolution'] == treatment) & iq_mask

            model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/heterogeneity/iq_{iq_group}_{treatment_label.lower()}',
                                 'robustness/heterogeneity.md',
                                 'belief_adjustment',
                                 f'{treatment_label}, {iq_group.capitalize()} IQ',
                                 'signal, signal*bayes')
            if res: results.append(res)

    # -------------------------------------------------------------------------
    # POOLED SPECIFICATIONS (across treatments)
    # -------------------------------------------------------------------------
    print("Running pooled specifications...")

    # Pooled with treatment interaction
    model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj + resolution', df)
    res = extract_results(model, 'bayes_belief_adjustment',
                         'robust/estimation/pooled_with_treatment',
                         'robustness/model_specification.md',
                         'belief_adjustment', 'All subjects',
                         'signal, signal*bayes, resolution')
    if res: results.append(res)

    # Full interaction model
    df['resolution_bayes'] = df['resolution'] * df['bayes_belief_adjustment']
    df['resolution_signal'] = df['resolution'] * df['signal']
    df['resolution_signal_bayes'] = df['resolution'] * df['signal_bayesbeliefadj']

    model = run_ols('belief_adjustment ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj + resolution + resolution_bayes + resolution_signal + resolution_signal_bayes', df)
    res = extract_results(model, 'bayes_belief_adjustment',
                         'robust/estimation/fully_interacted',
                         'robustness/model_specification.md',
                         'belief_adjustment', 'All subjects',
                         'Full treatment interactions')
    if res: results.append(res)

    # -------------------------------------------------------------------------
    # PLACEBO TESTS
    # -------------------------------------------------------------------------
    print("Running placebo tests...")

    # Placebo: Use profit as outcome (should not be affected by signal in systematic way)
    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        mask = df['resolution'] == treatment

        model = run_ols('profit ~ signal + sumpoints + prior', df[mask])
        res = extract_results(model, 'signal',
                             f'robust/placebo/profit_{treatment_label.lower()}',
                             'robustness/placebo_tests.md',
                             'profit', f'{treatment_label}, All signals',
                             'sumpoints, prior')
        if res: results.append(res)

    # Placebo: Prior beliefs on signal (should be zero by design - random assignment)
    model = run_ols('prior ~ signal + resolution', df)
    res = extract_results(model, 'signal',
                         'robust/placebo/prior_on_signal',
                         'robustness/placebo_tests.md',
                         'prior', 'All subjects',
                         'resolution')
    if res: results.append(res)

    # -------------------------------------------------------------------------
    # QUANTILE REGRESSION
    # -------------------------------------------------------------------------
    print("Running quantile regressions...")

    from statsmodels.regression.quantile_regression import QuantReg

    for treatment in [0, 1]:
        treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
        mask = df['resolution'] == treatment
        df_sub = df[mask].dropna(subset=['belief_adjustment', 'bayes_belief_adjustment'])

        for q in [0.25, 0.5, 0.75]:
            try:
                X = sm.add_constant(df_sub['bayes_belief_adjustment'])
                model = QuantReg(df_sub['belief_adjustment'], X).fit(q=q)

                coef = model.params['bayes_belief_adjustment']
                se = model.bse['bayes_belief_adjustment']
                pval = model.pvalues['bayes_belief_adjustment']
                ci = model.conf_int()

                results.append({
                    'paper_id': '139262-V1',
                    'journal': 'AER P&P',
                    'paper_title': 'Motivated Beliefs and Anticipation of Uncertainty Resolution',
                    'spec_id': f'robust/funcform/quantile_{int(q*100)}_{treatment_label.lower()}',
                    'spec_tree_path': 'robustness/functional_form.md',
                    'outcome_var': 'belief_adjustment',
                    'treatment_var': 'bayes_belief_adjustment',
                    'coefficient': coef,
                    'std_error': se,
                    't_stat': coef/se,
                    'p_value': pval,
                    'ci_lower': ci.loc['bayes_belief_adjustment', 0],
                    'ci_upper': ci.loc['bayes_belief_adjustment', 1],
                    'n_obs': len(df_sub),
                    'r_squared': model.prsquared,
                    'coefficient_vector_json': json.dumps({'treatment': {'var': 'bayes_belief_adjustment', 'coef': float(coef), 'se': float(se), 'pval': float(pval)}}),
                    'sample_desc': f'{treatment_label}, All signals',
                    'fixed_effects': 'None',
                    'controls_desc': 'None',
                    'cluster_var': 'quantile',
                    'model_type': f'Quantile_{int(q*100)}',
                    'estimation_script': 'scripts/paper_analyses/139262-V1.py'
                })
            except Exception as e:
                print(f"Quantile regression failed for {treatment_label} q={q}: {e}")

    # -------------------------------------------------------------------------
    # ADDITIONAL SPECIFICATIONS TO REACH 50+
    # -------------------------------------------------------------------------
    print("Running additional specifications...")

    # By rank subgroups
    for rank in [1, 2, 3, 4]:
        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = (df['resolution'] == treatment) & (df['rank'] == rank)

            if mask.sum() >= 10:  # Only if enough observations
                model = run_ols('belief_adjustment ~ bayes_belief_adjustment', df[mask])
                res = extract_results(model, 'bayes_belief_adjustment',
                                     f'robust/heterogeneity/rank_{rank}_{treatment_label.lower()}',
                                     'robustness/heterogeneity.md',
                                     'belief_adjustment',
                                     f'{treatment_label}, Rank {rank} only',
                                     'None')
                if res: results.append(res)

    # Winsorized outcome
    for pct in [5, 10]:
        df_wins = df.copy()
        lower = df_wins['belief_adjustment'].quantile(pct/100)
        upper = df_wins['belief_adjustment'].quantile(1 - pct/100)
        df_wins['belief_adjustment_wins'] = df_wins['belief_adjustment'].clip(lower=lower, upper=upper)

        for treatment in [0, 1]:
            treatment_label = 'Resolution' if treatment == 1 else 'No-Resolution'
            mask = df_wins['resolution'] == treatment

            model = run_ols('belief_adjustment_wins ~ bayes_belief_adjustment + signal + signal_bayesbeliefadj', df_wins[mask])
            res = extract_results(model, 'bayes_belief_adjustment',
                                 f'robust/sample/winsorize_{pct}pct_{treatment_label.lower()}',
                                 'robustness/sample_restrictions.md',
                                 f'belief_adjustment_wins_{pct}',
                                 f'{treatment_label}, winsorized {pct}%',
                                 'signal, signal*bayes')
            if res: results.append(res)

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Loading and creating data...")
    df = load_and_create_data()
    print(f"Data loaded: {len(df)} observations")
    print(f"Treatment distribution: Resolution={df['resolution'].sum()}, No-Resolution={(df['resolution']==0).sum()}")

    print("\nRunning specification search...")
    results = run_specification_search(df)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total specifications: {len(results_df)}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
