"""
Specification Search for Paper 214201-V1
Mission Motivation and Public Sector Performance in Pakistan

Authors: Khan et al.
Method: Randomized Controlled Trial with Panel Fixed Effects

This script replicates the main results and runs systematic specification checks.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest for high-dimensional FE
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("pyfixest not available, using statsmodels")

# Path configuration
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/214201-V1/replication_khan_mission/data"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/214201-V1"


def load_data():
    """Load the main dataset."""
    df = pd.read_stata(f"{DATA_DIR}/master_long.dta")
    return df


def run_feols(formula, data, cluster_var, weights_col=None):
    """
    Run fixed effects regression using pyfixest.

    Parameters:
    -----------
    formula : str - pyfixest formula
    data : DataFrame
    cluster_var : str - clustering variable name
    weights_col : str or None - name of weights column

    Returns:
    --------
    model result object
    """
    vcov = {'CRV1': cluster_var}

    if weights_col:
        model = pf.feols(formula, data=data, vcov=vcov, weights=weights_col)
    else:
        model = pf.feols(formula, data=data, vcov=vcov)

    return model


def get_model_stats(model):
    """Extract statistics from pyfixest model."""
    try:
        n_obs = model._N
    except:
        try:
            n_obs = len(model._Y)
        except:
            n_obs = None

    try:
        r2 = model._r2
    except:
        try:
            r2 = model.r2()
        except:
            r2 = None

    try:
        n_clust = model._N_clust
    except:
        n_clust = None

    return n_obs, r2, n_clust


def specification_search(df):
    """
    Run the full specification search.
    """
    results = []

    # Define the main treatment variables
    treatments = ['treat_mission_nobonus', 'treat_bonus_pr', 'treat5', 'treat_social_all']

    # Define the main outcome
    outcome = 'lhw_visit'

    # Create analysis samples
    # Main sample: exclude wave 0 (baseline) and wave 4 (post-endline)
    df_main = df[(df['data1'] != 1) & (df['data5'] != 1)].copy()

    print(f"Main sample size: {len(df_main)}")
    print(f"Number of LHWs: {df_main['lhw_id'].nunique()}")

    # ============================================
    # BASELINE REPLICATION (Table 1 main result)
    # ============================================
    print("\n=== BASELINE REPLICATION ===")

    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
        model = run_feols(formula, df_main, 'lhw_id', 'pw')

        n_obs, r2, n_clust = get_model_stats(model)

        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'baseline',
            'spec_tree_path': 'baseline',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'n_clusters': int(n_clust) if n_clust else df_main['lhw_id'].nunique(),
            'r_squared': float(r2) if r2 else None,
            'fe_absorbed': ['block', 'wave'],
            'cluster_var': 'lhw_id',
            'weights': 'pw',
            'coefficients': {}
        }

        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }

        results.append(res)
        print(f"Baseline N={res['n_obs']}, Clusters={res['n_clusters']}")
        for treat, vals in res['coefficients'].items():
            print(f"  {treat}: coef={vals['coef']:.4f}, se={vals['se']:.4f}, p={vals['pval']:.4f}")

    except Exception as e:
        print(f"Baseline replication failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================
    # PANEL FE VARIATIONS
    # ============================================
    print("\n=== PANEL FE VARIATIONS ===")

    # panel/fe/unit - Only block FE (within randomization strata)
    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | block"
        model = run_feols(formula, df_main, 'lhw_id', 'pw')
        n_obs, r2, n_clust = get_model_stats(model)
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'panel/fe/unit',
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r2) if r2 else None,
            'fe_absorbed': ['block'],
            'coefficients': {}
        }
        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
        results.append(res)
        print(f"panel/fe/unit: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
    except Exception as e:
        print(f"panel/fe/unit failed: {e}")

    # panel/fe/time - Only wave FE
    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | wave"
        model = run_feols(formula, df_main, 'lhw_id', 'pw')
        n_obs, r2, n_clust = get_model_stats(model)
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'panel/fe/time',
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r2) if r2 else None,
            'fe_absorbed': ['wave'],
            'coefficients': {}
        }
        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
        results.append(res)
        print(f"panel/fe/time: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
    except Exception as e:
        print(f"panel/fe/time failed: {e}")

    # panel/fe/none - Pooled OLS (no FE)
    try:
        formula = f"{outcome} ~ {' + '.join(treatments)}"
        model = run_feols(formula, df_main, 'lhw_id', 'pw')
        n_obs, r2, n_clust = get_model_stats(model)
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'panel/fe/none',
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r2) if r2 else None,
            'fe_absorbed': [],
            'coefficients': {}
        }
        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
        results.append(res)
        print(f"panel/fe/none: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
    except Exception as e:
        print(f"panel/fe/none failed: {e}")

    # ============================================
    # CLUSTERING VARIATIONS
    # ============================================
    print("\n=== CLUSTERING VARIATIONS ===")

    # robust/cluster/none - Robust SE (no clustering)
    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
        model = pf.feols(formula, data=df_main, vcov='hetero', weights='pw')
        n_obs, r2, n_clust = get_model_stats(model)
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'robust/cluster/none',
            'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r2) if r2 else None,
            'cluster_var': None,
            'coefficients': {}
        }
        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
        results.append(res)
        print(f"robust/cluster/none: N={res['n_obs']}, mission se={res['coefficients'].get('treat_mission_nobonus', {}).get('se', 'N/A'):.4f}")
    except Exception as e:
        print(f"robust/cluster/none failed: {e}")

    # robust/cluster/region - Cluster by block
    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
        model = run_feols(formula, df_main, 'block', 'pw')
        n_obs, r2, n_clust = get_model_stats(model)
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'robust/cluster/region',
            'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'n_clusters': df_main['block'].nunique(),
            'cluster_var': 'block',
            'coefficients': {}
        }
        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
        results.append(res)
        print(f"robust/cluster/region (block): N={res['n_obs']}, Clusters={res['n_clusters']}, mission se={res['coefficients'].get('treat_mission_nobonus', {}).get('se', 'N/A'):.4f}")
    except Exception as e:
        print(f"robust/cluster/region failed: {e}")

    # ============================================
    # SAMPLE RESTRICTIONS
    # ============================================
    print("\n=== SAMPLE RESTRICTIONS ===")

    # By wave - each wave separately
    for wave_val in [2, 3, 4]:
        try:
            df_wave = df[df['wave'] == wave_val].copy()
            if len(df_wave) > 100:
                formula = f"{outcome} ~ {' + '.join(treatments)} | block"
                model = run_feols(formula, df_wave, 'lhw_id', 'pw')
                n_obs, r2, n_clust = get_model_stats(model)
                coefs = model.coef()
                ses = model.se()
                pvals = model.pvalue()

                res = {
                    'spec_id': f'robust/sample/wave{wave_val}',
                    'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
                    'outcome': outcome,
                    'n_obs': int(n_obs) if n_obs else None,
                    'sample_description': f'Wave {wave_val} only',
                    'coefficients': {}
                }
                for treat in treatments:
                    if treat in coefs.index:
                        res['coefficients'][treat] = {
                            'coef': float(coefs[treat]),
                            'se': float(ses[treat]),
                            'pval': float(pvals[treat])
                        }
                results.append(res)
                print(f"robust/sample/wave{wave_val}: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
        except Exception as e:
            print(f"Wave {wave_val} subsample failed: {e}")

    # Early period (wave 2)
    try:
        df_early = df[df['data2'] == 1].copy()
        if len(df_early) > 100:
            formula = f"{outcome} ~ {' + '.join(treatments)} | block"
            model = run_feols(formula, df_early, 'lhw_id', 'pw')
            n_obs, r2, n_clust = get_model_stats(model)
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()

            res = {
                'spec_id': 'robust/sample/early_period',
                'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
                'outcome': outcome,
                'n_obs': int(n_obs) if n_obs else None,
                'sample_description': 'Early period (wave 2)',
                'coefficients': {}
            }
            for treat in treatments:
                if treat in coefs.index:
                    res['coefficients'][treat] = {
                        'coef': float(coefs[treat]),
                        'se': float(ses[treat]),
                        'pval': float(pvals[treat])
                    }
            results.append(res)
            print(f"robust/sample/early_period: N={res['n_obs']}")
    except Exception as e:
        print(f"Early period failed: {e}")

    # Late period (waves 3,4)
    try:
        df_late = df[(df['data3'] == 1) | (df['data4'] == 1)].copy()
        if len(df_late) > 100:
            formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
            model = run_feols(formula, df_late, 'lhw_id', 'pw')
            n_obs, r2, n_clust = get_model_stats(model)
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()

            res = {
                'spec_id': 'robust/sample/late_period',
                'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
                'outcome': outcome,
                'n_obs': int(n_obs) if n_obs else None,
                'sample_description': 'Late period (waves 3,4)',
                'coefficients': {}
            }
            for treat in treatments:
                if treat in coefs.index:
                    res['coefficients'][treat] = {
                        'coef': float(coefs[treat]),
                        'se': float(ses[treat]),
                        'pval': float(pvals[treat])
                    }
            results.append(res)
            print(f"robust/sample/late_period: N={res['n_obs']}")
    except Exception as e:
        print(f"Late period failed: {e}")

    # Baseline high performers vs low performers
    try:
        df_high = df_main[df_main['baseline_high'] == 1].copy()
        if len(df_high) > 100:
            formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
            model = run_feols(formula, df_high, 'lhw_id', 'pw')
            n_obs, r2, n_clust = get_model_stats(model)
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()

            res = {
                'spec_id': 'robust/sample/high_baseline',
                'spec_tree_path': 'robustness/sample_restrictions.md#demographic-subgroups',
                'outcome': outcome,
                'n_obs': int(n_obs) if n_obs else None,
                'sample_description': 'High baseline performers',
                'coefficients': {}
            }
            for treat in treatments:
                if treat in coefs.index:
                    res['coefficients'][treat] = {
                        'coef': float(coefs[treat]),
                        'se': float(ses[treat]),
                        'pval': float(pvals[treat])
                    }
            results.append(res)
            print(f"robust/sample/high_baseline: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
    except Exception as e:
        print(f"High baseline failed: {e}")

    try:
        df_low = df_main[df_main['baseline_high'] == 0].copy()
        if len(df_low) > 100:
            formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
            model = run_feols(formula, df_low, 'lhw_id', 'pw')
            n_obs, r2, n_clust = get_model_stats(model)
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()

            res = {
                'spec_id': 'robust/sample/low_baseline',
                'spec_tree_path': 'robustness/sample_restrictions.md#demographic-subgroups',
                'outcome': outcome,
                'n_obs': int(n_obs) if n_obs else None,
                'sample_description': 'Low baseline performers',
                'coefficients': {}
            }
            for treat in treatments:
                if treat in coefs.index:
                    res['coefficients'][treat] = {
                        'coef': float(coefs[treat]),
                        'se': float(ses[treat]),
                        'pval': float(pvals[treat])
                    }
            results.append(res)
            print(f"robust/sample/low_baseline: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
    except Exception as e:
        print(f"Low baseline failed: {e}")

    # ============================================
    # UNWEIGHTED SPECIFICATION
    # ============================================
    print("\n=== UNWEIGHTED SPECIFICATIONS ===")

    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
        model = pf.feols(formula, data=df_main, vcov={'CRV1': 'lhw_id'})
        n_obs, r2, n_clust = get_model_stats(model)
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        res = {
            'spec_id': 'ols/method/ols',
            'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
            'outcome': outcome,
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r2) if r2 else None,
            'weights': None,
            'coefficients': {}
        }
        for treat in treatments:
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
        results.append(res)
        print(f"ols/method/ols (unweighted): N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
    except Exception as e:
        print(f"Unweighted OLS failed: {e}")

    # ============================================
    # SINGLE TREATMENT REGRESSIONS
    # ============================================
    print("\n=== SINGLE TREATMENT REGRESSIONS ===")

    for treat in treatments:
        try:
            formula = f"{outcome} ~ {treat} | block + wave"
            model = run_feols(formula, df_main, 'lhw_id', 'pw')
            n_obs, r2, n_clust = get_model_stats(model)
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()

            res = {
                'spec_id': f'robust/single/{treat}',
                'spec_tree_path': 'robustness/single_covariate.md',
                'outcome': outcome,
                'n_obs': int(n_obs) if n_obs else None,
                'single_treatment': treat,
                'coefficients': {}
            }
            if treat in coefs.index:
                res['coefficients'][treat] = {
                    'coef': float(coefs[treat]),
                    'se': float(ses[treat]),
                    'pval': float(pvals[treat])
                }
            results.append(res)
            print(f"robust/single/{treat}: coef={res['coefficients'][treat]['coef']:.4f}, p={res['coefficients'][treat]['pval']:.4f}")
        except Exception as e:
            print(f"Single treatment {treat} failed: {e}")

    # ============================================
    # ALTERNATIVE OUTCOMES
    # ============================================
    print("\n=== ALTERNATIVE OUTCOMES ===")

    alt_outcomes = ['were_preg_served', 'were_child_served', 'tb_check']

    for alt_out in alt_outcomes:
        try:
            df_alt = df_main[df_main[alt_out].notna()].copy()
            if len(df_alt) > 100:
                formula = f"{alt_out} ~ {' + '.join(treatments)} | block + wave"
                model = pf.feols(formula, data=df_alt, vcov={'CRV1': 'lhw_id'})
                n_obs, r2, n_clust = get_model_stats(model)
                coefs = model.coef()
                ses = model.se()
                pvals = model.pvalue()

                res = {
                    'spec_id': f'custom/outcome_{alt_out}',
                    'spec_tree_path': 'custom',
                    'outcome': alt_out,
                    'n_obs': int(n_obs) if n_obs else None,
                    'coefficients': {}
                }
                for treat in treatments:
                    if treat in coefs.index:
                        res['coefficients'][treat] = {
                            'coef': float(coefs[treat]),
                            'se': float(ses[treat]),
                            'pval': float(pvals[treat])
                        }
                results.append(res)
                print(f"custom/outcome_{alt_out}: N={res['n_obs']}, mission coef={res['coefficients'].get('treat_mission_nobonus', {}).get('coef', 'N/A'):.4f}")
        except Exception as e:
            print(f"Alternative outcome {alt_out} failed: {e}")

    # ============================================
    # TREATMENT COMPARISONS (linear combinations)
    # ============================================
    print("\n=== TREATMENT COMPARISONS ===")

    try:
        formula = f"{outcome} ~ {' + '.join(treatments)} | block + wave"
        model = run_feols(formula, df_main, 'lhw_id', 'pw')
        n_obs, r2, n_clust = get_model_stats(model)

        coefs = model.coef()
        ses = model.se()

        # Get variance-covariance matrix as DataFrame
        try:
            vcov_arr = model._vcov
            vcov_mat = pd.DataFrame(vcov_arr, index=coefs.index, columns=coefs.index)
        except:
            # If we can't get vcov, compute SE of difference manually using formula
            # For now, skip the vcov and compute from SEs (assumes no covariance)
            vcov_mat = None

        # Mission - Placebo (treat_mission_nobonus - treat_social_all)
        if 'treat_mission_nobonus' in coefs.index and 'treat_social_all' in coefs.index:
            diff = coefs['treat_mission_nobonus'] - coefs['treat_social_all']
            if vcov_mat is not None:
                var_diff = (vcov_mat.loc['treat_mission_nobonus', 'treat_mission_nobonus'] +
                           vcov_mat.loc['treat_social_all', 'treat_social_all'] -
                           2*vcov_mat.loc['treat_mission_nobonus', 'treat_social_all'])
            else:
                # Approximate: assume zero covariance (conservative)
                var_diff = ses['treat_mission_nobonus']**2 + ses['treat_social_all']**2
            se_diff = np.sqrt(var_diff)
            tstat = diff / se_diff
            pval = 2 * (1 - stats.t.cdf(abs(tstat), n_obs - len(coefs)))

            res = {
                'spec_id': 'custom/lincom_mission_vs_placebo',
                'spec_tree_path': 'custom',
                'outcome': outcome,
                'comparison': 'treat_mission_nobonus - treat_social_all',
                'n_obs': int(n_obs) if n_obs else None,
                'coefficients': {
                    'difference': {
                        'coef': float(diff),
                        'se': float(se_diff),
                        'pval': float(pval)
                    }
                }
            }
            results.append(res)
            print(f"Mission vs Placebo: diff={diff:.4f}, se={se_diff:.4f}, p={pval:.4f}")

        # Mission-plus - Placebo (treat5 - treat_social_all)
        if 'treat5' in coefs.index and 'treat_social_all' in coefs.index:
            diff = coefs['treat5'] - coefs['treat_social_all']
            if vcov_mat is not None:
                var_diff = (vcov_mat.loc['treat5', 'treat5'] +
                           vcov_mat.loc['treat_social_all', 'treat_social_all'] -
                           2*vcov_mat.loc['treat5', 'treat_social_all'])
            else:
                var_diff = ses['treat5']**2 + ses['treat_social_all']**2
            se_diff = np.sqrt(var_diff)
            tstat = diff / se_diff
            pval = 2 * (1 - stats.t.cdf(abs(tstat), n_obs - len(coefs)))

            res = {
                'spec_id': 'custom/lincom_missionplus_vs_placebo',
                'spec_tree_path': 'custom',
                'outcome': outcome,
                'comparison': 'treat5 - treat_social_all',
                'n_obs': int(n_obs) if n_obs else None,
                'coefficients': {
                    'difference': {
                        'coef': float(diff),
                        'se': float(se_diff),
                        'pval': float(pval)
                    }
                }
            }
            results.append(res)
            print(f"Mission-plus vs Placebo: diff={diff:.4f}, se={se_diff:.4f}, p={pval:.4f}")

        # Mission - Financial Incentive (treat_mission_nobonus - treat_bonus_pr)
        if 'treat_mission_nobonus' in coefs.index and 'treat_bonus_pr' in coefs.index:
            diff = coefs['treat_mission_nobonus'] - coefs['treat_bonus_pr']
            if vcov_mat is not None:
                var_diff = (vcov_mat.loc['treat_mission_nobonus', 'treat_mission_nobonus'] +
                           vcov_mat.loc['treat_bonus_pr', 'treat_bonus_pr'] -
                           2*vcov_mat.loc['treat_mission_nobonus', 'treat_bonus_pr'])
            else:
                var_diff = ses['treat_mission_nobonus']**2 + ses['treat_bonus_pr']**2
            se_diff = np.sqrt(var_diff)
            tstat = diff / se_diff
            pval = 2 * (1 - stats.t.cdf(abs(tstat), n_obs - len(coefs)))

            res = {
                'spec_id': 'custom/lincom_mission_vs_incentive',
                'spec_tree_path': 'custom',
                'outcome': outcome,
                'comparison': 'treat_mission_nobonus - treat_bonus_pr',
                'n_obs': int(n_obs) if n_obs else None,
                'coefficients': {
                    'difference': {
                        'coef': float(diff),
                        'se': float(se_diff),
                        'pval': float(pval)
                    }
                }
            }
            results.append(res)
            print(f"Mission vs Financial: diff={diff:.4f}, se={se_diff:.4f}, p={pval:.4f}")

        # Mission-plus - Financial Incentive (treat5 - treat_bonus_pr)
        if 'treat5' in coefs.index and 'treat_bonus_pr' in coefs.index:
            diff = coefs['treat5'] - coefs['treat_bonus_pr']
            if vcov_mat is not None:
                var_diff = (vcov_mat.loc['treat5', 'treat5'] +
                           vcov_mat.loc['treat_bonus_pr', 'treat_bonus_pr'] -
                           2*vcov_mat.loc['treat5', 'treat_bonus_pr'])
            else:
                var_diff = ses['treat5']**2 + ses['treat_bonus_pr']**2
            se_diff = np.sqrt(var_diff)
            tstat = diff / se_diff
            pval = 2 * (1 - stats.t.cdf(abs(tstat), n_obs - len(coefs)))

            res = {
                'spec_id': 'custom/lincom_missionplus_vs_incentive',
                'spec_tree_path': 'custom',
                'outcome': outcome,
                'comparison': 'treat5 - treat_bonus_pr',
                'n_obs': int(n_obs) if n_obs else None,
                'coefficients': {
                    'difference': {
                        'coef': float(diff),
                        'se': float(se_diff),
                        'pval': float(pval)
                    }
                }
            }
            results.append(res)
            print(f"Mission-plus vs Financial: diff={diff:.4f}, se={se_diff:.4f}, p={pval:.4f}")

    except Exception as e:
        print(f"Treatment comparisons failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================
    # CONTROL MEAN
    # ============================================
    print("\n=== CONTROL MEAN ===")
    try:
        control_mean = df_main[df_main['treat1'] == 1][outcome].mean()
        print(f"Control group mean: {control_mean:.4f}")
    except:
        control_mean = None

    return results, control_mean


def results_to_csv(results, output_path):
    """Convert results to CSV format."""
    rows = []

    for res in results:
        base_row = {
            'spec_id': res.get('spec_id', ''),
            'spec_tree_path': res.get('spec_tree_path', ''),
            'outcome': res.get('outcome', 'lhw_visit'),
            'n_obs': res.get('n_obs', ''),
            'n_clusters': res.get('n_clusters', ''),
            'r_squared': res.get('r_squared', ''),
            'sample_description': res.get('sample_description', ''),
            'cluster_var': res.get('cluster_var', 'lhw_id'),
            'weights': res.get('weights', 'pw'),
            'fe_absorbed': json.dumps(res.get('fe_absorbed', ['block', 'wave'])),
            'coefficient_vector_json': json.dumps(res.get('coefficients', {}))
        }

        # Add individual coefficient columns for main treatment
        coeffs = res.get('coefficients', {})

        # Primary treatment of interest is mission
        if 'treat_mission_nobonus' in coeffs:
            base_row['coef_main'] = coeffs['treat_mission_nobonus'].get('coef', '')
            base_row['se_main'] = coeffs['treat_mission_nobonus'].get('se', '')
            base_row['pval_main'] = coeffs['treat_mission_nobonus'].get('pval', '')
        elif 'difference' in coeffs:
            base_row['coef_main'] = coeffs['difference'].get('coef', '')
            base_row['se_main'] = coeffs['difference'].get('se', '')
            base_row['pval_main'] = coeffs['difference'].get('pval', '')
        else:
            # Use first available coefficient
            for key, val in coeffs.items():
                base_row['coef_main'] = val.get('coef', '')
                base_row['se_main'] = val.get('se', '')
                base_row['pval_main'] = val.get('pval', '')
                break

        rows.append(base_row)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    return df_results


def main():
    """Main function to run the specification search."""
    print("=" * 60)
    print("SPECIFICATION SEARCH: 214201-V1")
    print("Mission Motivation and Public Sector Performance in Pakistan")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations for {df['lhw_id'].nunique()} LHWs")

    # Run specification search
    print("\nRunning specification search...")
    results, control_mean = specification_search(df)

    # Save results
    output_path = f"{OUTPUT_DIR}/specification_results.csv"
    df_results = results_to_csv(results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total specifications run: {len(results)}")
    print(f"Output saved to: {output_path}")
    if control_mean:
        print(f"Control group mean: {control_mean:.4f}")

    # Show coefficient stability
    baseline_coef = None
    for res in results:
        if res.get('spec_id') == 'baseline':
            if 'treat_mission_nobonus' in res.get('coefficients', {}):
                baseline_coef = res['coefficients']['treat_mission_nobonus']['coef']
                break

    if baseline_coef is not None:
        print(f"\nBaseline mission coefficient: {baseline_coef:.4f}")
        print("\nCoefficient variation across specifications:")
        for res in results:
            coeffs = res.get('coefficients', {})
            if 'treat_mission_nobonus' in coeffs:
                coef = coeffs['treat_mission_nobonus']['coef']
                pct_change = (coef - baseline_coef) / abs(baseline_coef) * 100
                pval = coeffs['treat_mission_nobonus']['pval']
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"  {res['spec_id']}: {coef:.4f} ({pct_change:+.1f}%) {sig}")

    return results


if __name__ == "__main__":
    results = main()
