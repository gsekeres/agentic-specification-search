#!/usr/bin/env python3
"""
Specification Search: Does Electoral Competition Curb Party Favoritism?
Paper ID: 113710-V1
Authors: Curto-Grau, Sole-Olle, and Sorribas-Navarro (2018, AEJ: Applied Economics)

This script runs a systematic specification search following the specification tree
methodology from i4r (Institute for Replication).

Method: Fuzzy Regression Discontinuity / Instrumental Variables
- Outcome: tk (transfers per capita from regional to local government)
- Treatment: ab (political alignment dummy)
- Instrument: dab (discontinuous treatment assignment at 50% vote share)
- Running variable: dist1 (distance to 50% margin)
- Key heterogeneity: ecs1 (regional electoral competition)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search')
DATA_PATH = BASE_PATH / 'data/downloads/extracted/113710-V1/data/data'
OUTPUT_PATH = BASE_PATH / 'data/downloads/extracted/113710-V1'

# Import pyfixest for IV estimation
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("pyfixest not available, using statsmodels")

import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVLIML


def load_data():
    """Load and prepare the main dataset."""
    df = pd.read_stata(DATA_PATH / 'db_main.dta')

    # Create needed interaction variables if not present
    # Regional FE dummies
    df['region_fe'] = df['codccaa'].astype(str)

    # Create variables for DiD specification (municipality centered variables)
    # Following the Stata code which uses "center" command to within-demean
    for var in ['tk', 'ab', 'dt1', 'dt2', 'esas1', 'ecs1']:
        if var in df.columns:
            df[f'c_{var}'] = df.groupby('codiine')[var].transform(lambda x: x - x.mean())

    # Create squared and cubic terms for running variable
    df['dist2'] = df['dist1'] ** 2
    df['dist3'] = df['dist1'] ** 3

    # Create interaction terms
    df['vsa'] = df['dist1'] * df['ab']  # running var * treatment
    df['vda'] = df['dist1'] * df['dab']  # running var * instrument
    df['vsa2'] = df['dist2'] * df['ab']
    df['vda2'] = df['dist2'] * df['dab']

    return df


def run_iv_regression(df, outcome, endog, instruments, exog_controls, fe_vars=None,
                      cluster_var='codiine', sample_filter=None, spec_id='',
                      spec_tree_path=''):
    """
    Run an IV/2SLS regression and return standardized results.

    This implements fuzzy RD as IV where:
    - endog: endogenous variable (ab = alignment)
    - instruments: excluded instruments (dab = discontinuity indicator, etc.)
    - exog_controls: controls in both stages (running variable polynomial, region FE)
    """
    # Apply sample filter if provided
    df_reg = df.copy()
    if sample_filter is not None:
        df_reg = df_reg[sample_filter].copy()

    # Drop missing values
    all_vars = [outcome] + endog + instruments + exog_controls
    if fe_vars:
        all_vars += fe_vars
    all_vars.append(cluster_var)
    df_reg = df_reg.dropna(subset=[v for v in all_vars if v in df_reg.columns])

    if len(df_reg) < 50:
        return None

    try:
        # Prepare variables
        y = df_reg[outcome]

        # Create formula for linearmodels
        # Endogenous vars and instruments
        endog_str = ' + '.join(endog)
        instr_str = ' + '.join(instruments)

        # Exogenous controls (create dummies for region FE)
        if fe_vars and 'codccaa' in fe_vars:
            # Add region dummies
            region_dummies = pd.get_dummies(df_reg['codccaa'], prefix='reg', drop_first=True)
            df_reg = pd.concat([df_reg, region_dummies], axis=1)
            exog_with_fe = exog_controls + list(region_dummies.columns)
        else:
            exog_with_fe = exog_controls

        exog_str = ' + '.join(exog_with_fe) if exog_with_fe else '1'

        # Build formula
        formula = f"{outcome} ~ 1 + {exog_str} + [{endog_str} ~ {instr_str}]"

        # Run IV2SLS
        model = IV2SLS.from_formula(formula, data=df_reg)

        # Fit with clustered standard errors
        if cluster_var and cluster_var in df_reg.columns:
            results = model.fit(cov_type='clustered', clusters=df_reg[cluster_var])
        else:
            results = model.fit(cov_type='robust')

        # Extract results for treatment variable
        treat_var = endog[0]  # Main treatment variable

        coef = results.params.get(treat_var, np.nan)
        se = results.std_errors.get(treat_var, np.nan)
        pval = results.pvalues.get(treat_var, np.nan)

        if np.isnan(coef):
            return None

        tstat = coef / se if se > 0 else np.nan
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treat_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fe_vars if fe_vars else [],
            'diagnostics': {
                'first_stage_F': None,  # Would need separate first stage
                'n_instruments': len(instruments),
                'n_endogenous': len(endog)
            }
        }

        # Add other coefficients to controls
        for var in results.params.index:
            if var != treat_var and var != 'Intercept' and not var.startswith('reg_'):
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(results.params[var]),
                    'se': float(results.std_errors[var]),
                    'pval': float(results.pvalues[var])
                })

        return {
            'paper_id': '113710-V1',
            'journal': 'AEJ-Applied',
            'paper_title': 'Does Electoral Competition Curb Party Favoritism?',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treat_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(results.nobs),
            'r_squared': float(results.rsquared) if hasattr(results, 'rsquared') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(results.nobs)}',
            'fixed_effects': ', '.join(fe_vars) if fe_vars else 'None',
            'controls_desc': ', '.join(exog_controls[:5]) + ('...' if len(exog_controls) > 5 else ''),
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': 'IV-2SLS',
            'estimation_script': 'scripts/paper_analyses/113710-V1.py'
        }

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None


def run_ols_regression(df, outcome, treatment, controls, fe_vars=None,
                       cluster_var='codiine', sample_filter=None, spec_id='',
                       spec_tree_path=''):
    """Run an OLS regression (for comparison with IV)."""
    df_reg = df.copy()
    if sample_filter is not None:
        df_reg = df_reg[sample_filter].copy()

    all_vars = [outcome, treatment] + controls
    if cluster_var:
        all_vars.append(cluster_var)
    df_reg = df_reg.dropna(subset=[v for v in all_vars if v in df_reg.columns])

    if len(df_reg) < 50:
        return None

    try:
        # Create region dummies if needed
        if fe_vars and 'codccaa' in fe_vars:
            region_dummies = pd.get_dummies(df_reg['codccaa'], prefix='reg', drop_first=True)
            df_reg = pd.concat([df_reg, region_dummies], axis=1)
            controls_with_fe = controls + list(region_dummies.columns)
        else:
            controls_with_fe = controls

        y = df_reg[outcome]
        X = df_reg[[treatment] + controls_with_fe]
        X = sm.add_constant(X)

        model = sm.OLS(y, X)

        if cluster_var and cluster_var in df_reg.columns:
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
        else:
            results = model.fit(cov_type='HC1')

        coef = results.params.get(treatment, np.nan)
        se = results.bse.get(treatment, np.nan)
        pval = results.pvalues.get(treatment, np.nan)

        if np.isnan(coef):
            return None

        tstat = coef / se if se > 0 else np.nan
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        coef_vector = {
            'treatment': {
                'var': treatment,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fe_vars if fe_vars else [],
            'diagnostics': {}
        }

        return {
            'paper_id': '113710-V1',
            'journal': 'AEJ-Applied',
            'paper_title': 'Does Electoral Competition Curb Party Favoritism?',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(results.nobs),
            'r_squared': float(results.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(results.nobs)}',
            'fixed_effects': ', '.join(fe_vars) if fe_vars else 'None',
            'controls_desc': ', '.join(controls[:5]) + ('...' if len(controls) > 5 else ''),
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': 'OLS',
            'estimation_script': 'scripts/paper_analyses/113710-V1.py'
        }

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None


def run_first_stage(df, endog, instruments, exog_controls, fe_vars=None,
                    cluster_var='codiine', sample_filter=None, spec_id='',
                    spec_tree_path=''):
    """Run first stage regression for fuzzy RD."""
    df_reg = df.copy()
    if sample_filter is not None:
        df_reg = df_reg[sample_filter].copy()

    all_vars = [endog] + instruments + exog_controls
    if cluster_var:
        all_vars.append(cluster_var)
    df_reg = df_reg.dropna(subset=[v for v in all_vars if v in df_reg.columns])

    if len(df_reg) < 50:
        return None

    try:
        # Create region dummies if needed
        if fe_vars and 'codccaa' in fe_vars:
            region_dummies = pd.get_dummies(df_reg['codccaa'], prefix='reg', drop_first=True)
            df_reg = pd.concat([df_reg, region_dummies], axis=1)
            controls_with_fe = exog_controls + instruments + list(region_dummies.columns)
        else:
            controls_with_fe = exog_controls + instruments

        y = df_reg[endog]
        X = df_reg[controls_with_fe]
        X = sm.add_constant(X)

        model = sm.OLS(y, X)

        if cluster_var and cluster_var in df_reg.columns:
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
        else:
            results = model.fit(cov_type='HC1')

        # Get coefficient on main instrument
        instr_var = instruments[0]
        coef = results.params.get(instr_var, np.nan)
        se = results.bse.get(instr_var, np.nan)
        pval = results.pvalues.get(instr_var, np.nan)

        if np.isnan(coef):
            return None

        tstat = coef / se if se > 0 else np.nan
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Calculate F-statistic for instrument
        f_stat = (coef / se) ** 2 if se > 0 else np.nan

        coef_vector = {
            'treatment': {
                'var': instr_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'diagnostics': {
                'first_stage_F': float(f_stat)
            }
        }

        return {
            'paper_id': '113710-V1',
            'journal': 'AEJ-Applied',
            'paper_title': 'Does Electoral Competition Curb Party Favoritism?',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': endog,
            'treatment_var': instr_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(results.nobs),
            'r_squared': float(results.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(results.nobs)}, F={f_stat:.1f}',
            'fixed_effects': ', '.join(fe_vars) if fe_vars else 'None',
            'controls_desc': 'First stage',
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': 'First Stage OLS',
            'estimation_script': 'scripts/paper_analyses/113710-V1.py'
        }

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None


def run_reduced_form(df, outcome, instruments, exog_controls, fe_vars=None,
                     cluster_var='codiine', sample_filter=None, spec_id='',
                     spec_tree_path=''):
    """Run reduced form regression (outcome on instrument)."""
    return run_ols_regression(df, outcome, instruments[0],
                              exog_controls + instruments[1:],
                              fe_vars=fe_vars, cluster_var=cluster_var,
                              sample_filter=sample_filter, spec_id=spec_id,
                              spec_tree_path=spec_tree_path)


def main():
    """Run the full specification search."""
    print("Loading data...")
    df = load_data()

    results = []

    # ===========================================
    # BASELINE SPECIFICATION (Table 1, Column 1)
    # ===========================================
    # Global RD with second-order polynomial
    # ivregress 2sls tk (ab vsa vsa2 = dab vda vda2) dist1 dist2 i.codccaa, vce(cluster codiine)

    print("\n=== Running Baseline Specifications ===")

    # Baseline: Global polynomial RD
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa', 'vsa2'],
        instruments=['dab', 'vda', 'vda2'],
        exog_controls=['dist1', 'dist2'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='baseline',
        spec_tree_path='methods/regression_discontinuity.md#baseline'
    )
    if res:
        results.append(res)
        print(f"Baseline: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}, p={res['p_value']:.4f}")

    # ===========================================
    # RD BANDWIDTH VARIATIONS
    # ===========================================
    print("\n=== Running Bandwidth Variations ===")

    bandwidths = {
        'bw_full': 1.0,  # Full sample
        'bw_386': 0.386,  # Paper's optimal
        'bw_193': 0.193,  # Half optimal
        'bw_0965': 0.0965,  # Quarter optimal
        'bw_048': 0.048  # Eighth optimal
    }

    for bw_name, bw in bandwidths.items():
        sample_filter = np.abs(df['dist1']) < bw if bw < 1.0 else None

        # Linear polynomial within bandwidth
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'rd/bandwidth/{bw_name}',
            spec_tree_path='methods/regression_discontinuity.md#bandwidth-selection'
        )
        if res:
            results.append(res)
            print(f"  {bw_name}: coef={res['coefficient']:.3f}, n={res['n_obs']}")

    # ===========================================
    # POLYNOMIAL ORDER VARIATIONS
    # ===========================================
    print("\n=== Running Polynomial Order Variations ===")

    # Linear (order 1)
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='rd/poly/local_linear',
        spec_tree_path='methods/regression_discontinuity.md#polynomial-order'
    )
    if res:
        results.append(res)
        print(f"  Linear: coef={res['coefficient']:.3f}")

    # Quadratic (order 2) - baseline
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa', 'vsa2'],
        instruments=['dab', 'vda', 'vda2'],
        exog_controls=['dist1', 'dist2'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='rd/poly/local_quadratic',
        spec_tree_path='methods/regression_discontinuity.md#polynomial-order'
    )
    if res:
        results.append(res)
        print(f"  Quadratic: coef={res['coefficient']:.3f}")

    # Cubic (order 3)
    df['vsa3'] = df['dist3'] * df['ab']
    df['vda3'] = df['dist3'] * df['dab']
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa', 'vsa2', 'vsa3'],
        instruments=['dab', 'vda', 'vda2', 'vda3'],
        exog_controls=['dist1', 'dist2', 'dist3'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='rd/poly/local_cubic',
        spec_tree_path='methods/regression_discontinuity.md#polynomial-order'
    )
    if res:
        results.append(res)
        print(f"  Cubic: coef={res['coefficient']:.3f}")

    # ===========================================
    # FIRST STAGE SPECIFICATIONS
    # ===========================================
    print("\n=== Running First Stage Specifications ===")

    for bw_name, bw in [('full', 1.0), ('optimal', 0.386), ('half', 0.193)]:
        sample_filter = np.abs(df['dist1']) < bw if bw < 1.0 else None

        res = run_first_stage(
            df,
            endog='ab',
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'iv/first_stage/{bw_name}',
            spec_tree_path='methods/instrumental_variables.md#first-stage'
        )
        if res:
            results.append(res)
            print(f"  First stage {bw_name}: coef={res['coefficient']:.3f}")

    # ===========================================
    # REDUCED FORM
    # ===========================================
    print("\n=== Running Reduced Form ===")

    res = run_reduced_form(
        df,
        outcome='tk',
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='iv/first_stage/reduced_form',
        spec_tree_path='methods/instrumental_variables.md#first-stage'
    )
    if res:
        results.append(res)
        print(f"  Reduced form: coef={res['coefficient']:.3f}")

    # ===========================================
    # OLS COMPARISON (ignoring endogeneity)
    # ===========================================
    print("\n=== Running OLS Comparison ===")

    res = run_ols_regression(
        df,
        outcome='tk',
        treatment='ab',
        controls=['dist1', 'dist2'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='iv/method/ols',
        spec_tree_path='methods/instrumental_variables.md#estimation-method'
    )
    if res:
        results.append(res)
        print(f"  OLS: coef={res['coefficient']:.3f}")

    # ===========================================
    # CLUSTERING VARIATIONS
    # ===========================================
    print("\n=== Running Clustering Variations ===")

    cluster_vars = {
        'municipality': 'codiine',
        'region': 'codccaa',
        'province': 'cprov'
    }

    for cluster_name, cluster_var in cluster_vars.items():
        if cluster_var in df.columns:
            res = run_iv_regression(
                df,
                outcome='tk',
                endog=['ab', 'vsa', 'vsa2'],
                instruments=['dab', 'vda', 'vda2'],
                exog_controls=['dist1', 'dist2'],
                fe_vars=['codccaa'],
                cluster_var=cluster_var,
                spec_id=f'robust/cluster/{cluster_name}',
                spec_tree_path='robustness/clustering_variations.md#single-level-clustering'
            )
            if res:
                results.append(res)
                print(f"  Cluster by {cluster_name}: se={res['std_error']:.3f}")

    # Robust SE (no clustering)
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa', 'vsa2'],
        instruments=['dab', 'vda', 'vda2'],
        exog_controls=['dist1', 'dist2'],
        fe_vars=['codccaa'],
        cluster_var=None,
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering'
    )
    if res:
        results.append(res)
        print(f"  Robust SE: se={res['std_error']:.3f}")

    # ===========================================
    # CONTROL VARIATIONS
    # ===========================================
    print("\n=== Running Control Variations ===")

    controls_list = ['lpob', 'density', 'debt', 'vcp', 'tipo']

    # No controls
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='rd/controls/none',
        spec_tree_path='methods/regression_discontinuity.md#control-sets'
    )
    if res:
        results.append(res)
        print(f"  No controls: coef={res['coefficient']:.3f}")

    # Add controls incrementally
    for i, ctrl in enumerate(controls_list):
        if ctrl in df.columns:
            controls_so_far = controls_list[:i+1]
            res = run_iv_regression(
                df,
                outcome='tk',
                endog=['ab', 'vsa'],
                instruments=['dab', 'vda'],
                exog_controls=['dist1'] + controls_so_far,
                fe_vars=['codccaa'],
                cluster_var='codiine',
                spec_id=f'robust/control/add_{ctrl}',
                spec_tree_path='robustness/control_progression.md'
            )
            if res:
                results.append(res)
                print(f"  Add {ctrl}: coef={res['coefficient']:.3f}")

    # Full controls
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'] + controls_list,
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='rd/controls/full',
        spec_tree_path='methods/regression_discontinuity.md#control-sets'
    )
    if res:
        results.append(res)
        print(f"  Full controls: coef={res['coefficient']:.3f}")

    # Leave-one-out for each control
    for ctrl in controls_list:
        if ctrl in df.columns:
            remaining = [c for c in controls_list if c != ctrl and c in df.columns]
            res = run_iv_regression(
                df,
                outcome='tk',
                endog=['ab', 'vsa'],
                instruments=['dab', 'vda'],
                exog_controls=['dist1'] + remaining,
                fe_vars=['codccaa'],
                cluster_var='codiine',
                spec_id=f'robust/loo/drop_{ctrl}',
                spec_tree_path='robustness/leave_one_out.md'
            )
            if res:
                results.append(res)
                print(f"  Drop {ctrl}: coef={res['coefficient']:.3f}")

    # ===========================================
    # SAMPLE RESTRICTIONS
    # ===========================================
    print("\n=== Running Sample Restrictions ===")

    # By time period
    for t in df['t'].dropna().unique():
        sample_filter = df['t'] == t
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'robust/sample/period_{int(t)}',
            spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions'
        )
        if res:
            results.append(res)
            print(f"  Period {int(t)}: coef={res['coefficient']:.3f}, n={res['n_obs']}")

    # Exclude each period
    for t in df['t'].dropna().unique():
        sample_filter = df['t'] != t
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'robust/sample/exclude_period_{int(t)}',
            spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions'
        )
        if res:
            results.append(res)
            print(f"  Excl period {int(t)}: coef={res['coefficient']:.3f}")

    # By region (drop each region)
    for region in df['codccaa'].dropna().unique()[:5]:  # First 5 regions
        sample_filter = df['codccaa'] != region
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'robust/sample/drop_region_{int(region)}',
            spec_tree_path='robustness/sample_restrictions.md#geographic-restrictions'
        )
        if res:
            results.append(res)
            print(f"  Drop region {int(region)}: coef={res['coefficient']:.3f}")

    # Outlier handling - winsorize
    for pct in [1, 5, 10]:
        df_wins = df.copy()
        lower = df_wins['tk'].quantile(pct/100)
        upper = df_wins['tk'].quantile(1 - pct/100)
        df_wins['tk'] = df_wins['tk'].clip(lower=lower, upper=upper)

        res = run_iv_regression(
            df_wins,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            spec_id=f'robust/sample/winsorize_{pct}pct',
            spec_tree_path='robustness/sample_restrictions.md#outlier-handling'
        )
        if res:
            results.append(res)
            print(f"  Winsorize {pct}%: coef={res['coefficient']:.3f}")

    # Trim outliers
    for pct in [1, 5]:
        sample_filter = (df['tk'] > df['tk'].quantile(pct/100)) & (df['tk'] < df['tk'].quantile(1 - pct/100))
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'robust/sample/trim_{pct}pct',
            spec_tree_path='robustness/sample_restrictions.md#outlier-handling'
        )
        if res:
            results.append(res)
            print(f"  Trim {pct}%: coef={res['coefficient']:.3f}")

    # ===========================================
    # DONUT HOLE SPECIFICATIONS
    # ===========================================
    print("\n=== Running Donut Hole Specifications ===")

    for donut_size in [0.01, 0.02, 0.05]:
        sample_filter = np.abs(df['dist1']) > donut_size
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'rd/donut/exclude_{int(donut_size*100)}pct',
            spec_tree_path='methods/regression_discontinuity.md#donut-hole-specifications'
        )
        if res:
            results.append(res)
            print(f"  Donut {donut_size}: coef={res['coefficient']:.3f}")

    # ===========================================
    # FUNCTIONAL FORM VARIATIONS
    # ===========================================
    print("\n=== Running Functional Form Variations ===")

    # Log outcome (add 1 to handle zeros)
    df['tk_log'] = np.log(df['tk'] + 1)
    res = run_iv_regression(
        df,
        outcome='tk_log',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='robust/form/y_log',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations'
    )
    if res:
        results.append(res)
        print(f"  Log outcome: coef={res['coefficient']:.3f}")

    # IHS transformation
    df['tk_asinh'] = np.arcsinh(df['tk'])
    res = run_iv_regression(
        df,
        outcome='tk_asinh',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        spec_id='robust/form/y_asinh',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations'
    )
    if res:
        results.append(res)
        print(f"  Asinh outcome: coef={res['coefficient']:.3f}")

    # ===========================================
    # FIXED EFFECTS VARIATIONS
    # ===========================================
    print("\n=== Running Fixed Effects Variations ===")

    # No fixed effects
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=None,
        cluster_var='codiine',
        spec_id='robust/estimation/no_fe',
        spec_tree_path='robustness/model_specification.md'
    )
    if res:
        results.append(res)
        print(f"  No FE: coef={res['coefficient']:.3f}")

    # ===========================================
    # HETEROGENEITY: ELECTORAL COMPETITION
    # ===========================================
    print("\n=== Running Heterogeneity Analysis (Electoral Competition) ===")

    # This is the main result from Table 2 - interaction with electoral competition
    # The paper's key finding is that alignment effect is moderated by electoral competition

    # Create interaction terms for HLATE analysis
    df['esas1'] = df['ecs1'] * df['ab']
    df['edas1'] = df['ecs1'] * df['dab']
    df['vsa_ecs1'] = df['vsa'] * df['ecs1']
    df['vda_ecs1'] = df['vda'] * df['ecs1']
    df['dist1_ecs1'] = df['dist1'] * df['ecs1']

    # Run HLATE specification at optimal bandwidth
    sample_filter = np.abs(df['dist1']) < 0.193
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'esas1', 'vsa', 'vsa_ecs1'],
        instruments=['dab', 'edas1', 'vda', 'vda_ecs1'],
        exog_controls=['dist1', 'dist1_ecs1', 'ecs1'],
        fe_vars=['codccaa'],
        cluster_var='codccaa',  # Region-level clustering for HLATE
        sample_filter=sample_filter,
        spec_id='robust/heterogeneity/electoral_competition',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)
        print(f"  HLATE (competition): coef={res['coefficient']:.3f}")

    # Split by competition level
    median_comp = df['ecs1'].median()

    # High competition regions
    sample_filter_high = df['ecs1'] > median_comp
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        sample_filter=sample_filter_high,
        spec_id='robust/heterogeneity/high_competition',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)
        print(f"  High competition: coef={res['coefficient']:.3f}")

    # Low competition regions
    sample_filter_low = df['ecs1'] <= median_comp
    res = run_iv_regression(
        df,
        outcome='tk',
        endog=['ab', 'vsa'],
        instruments=['dab', 'vda'],
        exog_controls=['dist1'],
        fe_vars=['codccaa'],
        cluster_var='codiine',
        sample_filter=sample_filter_low,
        spec_id='robust/heterogeneity/low_competition',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)
        print(f"  Low competition: coef={res['coefficient']:.3f}")

    # ===========================================
    # HETEROGENEITY: OTHER DIMENSIONS
    # ===========================================
    print("\n=== Running Other Heterogeneity Analyses ===")

    # By population size
    median_pop = df['pob'].median()
    for name, filt in [('large_muni', df['pob'] > median_pop), ('small_muni', df['pob'] <= median_pop)]:
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=filt,
            spec_id=f'robust/heterogeneity/{name}',
            spec_tree_path='robustness/heterogeneity.md'
        )
        if res:
            results.append(res)
            print(f"  {name}: coef={res['coefficient']:.3f}")

    # By density
    median_dens = df['density'].median()
    for name, filt in [('high_density', df['density'] > median_dens), ('low_density', df['density'] <= median_dens)]:
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=filt,
            spec_id=f'robust/heterogeneity/{name}',
            spec_tree_path='robustness/heterogeneity.md'
        )
        if res:
            results.append(res)
            print(f"  {name}: coef={res['coefficient']:.3f}")

    # By debt level
    median_debt = df['debt'].median()
    for name, filt in [('high_debt', df['debt'] > median_debt), ('low_debt', df['debt'] <= median_debt)]:
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=filt,
            spec_id=f'robust/heterogeneity/{name}',
            spec_tree_path='robustness/heterogeneity.md'
        )
        if res:
            results.append(res)
            print(f"  {name}: coef={res['coefficient']:.3f}")

    # ===========================================
    # PLACEBO TESTS
    # ===========================================
    print("\n=== Running Placebo Tests ===")

    # Placebo cutoffs (fake discontinuities away from true cutoff)
    for placebo_cutoff in [-0.15, 0.15, -0.25, 0.25]:
        df['dist1_placebo'] = df['dist1'] - placebo_cutoff
        df['dab_placebo'] = (df['dist1_placebo'] > 0).astype(float)
        df['vda_placebo'] = df['dist1_placebo'] * df['dab_placebo']

        # Only use observations away from true cutoff
        sample_filter = np.abs(df['dist1']) > 0.1  # Away from true cutoff

        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab_placebo', 'vda_placebo'],
            exog_controls=['dist1_placebo'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            sample_filter=sample_filter,
            spec_id=f'rd/placebo/cutoff_{placebo_cutoff:.2f}'.replace('.', '_').replace('-', 'neg'),
            spec_tree_path='methods/regression_discontinuity.md#placebo-cutoff-tests'
        )
        if res:
            results.append(res)
            print(f"  Placebo cutoff {placebo_cutoff}: coef={res['coefficient']:.3f}")

    # ===========================================
    # ALTERNATIVE TREATMENT DEFINITIONS
    # ===========================================
    print("\n=== Running Alternative Treatment Definitions ===")

    # Alternative alignment: include coalition partners
    if 'abcd' in df.columns:
        df['vsbcd'] = df['dist1'] * df['abcd']
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['abcd', 'vsbcd'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            spec_id='robust/treatment/coalition_alignment',
            spec_tree_path='robustness/measurement.md'
        )
        if res:
            results.append(res)
            print(f"  Coalition alignment: coef={res['coefficient']:.3f}")

    # Bloc alignment
    if 'bloc' in df.columns:
        df['vsbloc'] = df['dist1'] * df['bloc']
        res = run_iv_regression(
            df,
            outcome='tk',
            endog=['bloc', 'vsbloc'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            spec_id='robust/treatment/bloc_alignment',
            spec_tree_path='robustness/measurement.md'
        )
        if res:
            results.append(res)
            print(f"  Bloc alignment: coef={res['coefficient']:.3f}")

    # ===========================================
    # ALTERNATIVE OUTCOMES
    # ===========================================
    print("\n=== Running Alternative Outcomes ===")

    # Alternative outcome: tc (total transfers)
    if 'tc' in df.columns and df['tc'].notna().sum() > 100:
        res = run_iv_regression(
            df,
            outcome='tc',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog_controls=['dist1'],
            fe_vars=['codccaa'],
            cluster_var='codiine',
            spec_id='robust/outcome/total_transfers',
            spec_tree_path='robustness/measurement.md'
        )
        if res:
            results.append(res)
            print(f"  Total transfers: coef={res['coefficient']:.3f}")

    # ===========================================
    # LIML ESTIMATION (weak IV robust)
    # ===========================================
    print("\n=== Running LIML Estimation ===")

    try:
        df_reg = df.dropna(subset=['tk', 'ab', 'dab', 'dist1', 'vsa', 'vda', 'codccaa', 'codiine'])

        region_dummies = pd.get_dummies(df_reg['codccaa'], prefix='reg', drop_first=True)
        df_reg = pd.concat([df_reg, region_dummies], axis=1)
        exog_cols = ['dist1'] + list(region_dummies.columns)

        formula = f"tk ~ 1 + {' + '.join(exog_cols)} + [ab + vsa ~ dab + vda]"
        model = IVLIML.from_formula(formula, data=df_reg)
        res_liml = model.fit(cov_type='clustered', clusters=df_reg['codiine'])

        coef = res_liml.params.get('ab', np.nan)
        se = res_liml.std_errors.get('ab', np.nan)
        pval = res_liml.pvalues.get('ab', np.nan)

        if not np.isnan(coef):
            results.append({
                'paper_id': '113710-V1',
                'journal': 'AEJ-Applied',
                'paper_title': 'Does Electoral Competition Curb Party Favoritism?',
                'spec_id': 'iv/method/liml',
                'spec_tree_path': 'methods/instrumental_variables.md#estimation-method',
                'outcome_var': 'tk',
                'treatment_var': 'ab',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(coef/se),
                'p_value': float(pval),
                'ci_lower': float(coef - 1.96*se),
                'ci_upper': float(coef + 1.96*se),
                'n_obs': int(res_liml.nobs),
                'r_squared': np.nan,
                'coefficient_vector_json': json.dumps({'treatment': {'var': 'ab', 'coef': float(coef), 'se': float(se), 'pval': float(pval)}}),
                'sample_desc': f'N={int(res_liml.nobs)}',
                'fixed_effects': 'codccaa',
                'controls_desc': 'dist1',
                'cluster_var': 'codiine',
                'model_type': 'IV-LIML',
                'estimation_script': 'scripts/paper_analyses/113710-V1.py'
            })
            print(f"  LIML: coef={coef:.3f}")
    except Exception as e:
        print(f"  LIML failed: {e}")

    # ===========================================
    # DID SPECIFICATION (Within-municipality)
    # ===========================================
    print("\n=== Running DiD Specifications ===")

    # Create within-municipality demeaned variables
    for var in ['tk', 'ab', 'dt1', 'dt2']:
        if var in df.columns:
            df[f'c_{var}'] = df.groupby('codiine')[var].transform(lambda x: x - x.mean())

    # DiD specification (within-municipality variation)
    res = run_ols_regression(
        df,
        outcome='c_tk',
        treatment='c_ab',
        controls=['c_dt1', 'c_dt2'],
        fe_vars=None,
        cluster_var='codiine',
        spec_id='custom/did_within_municipality',
        spec_tree_path='custom'
    )
    if res:
        results.append(res)
        print(f"  DiD within-municipality: coef={res['coefficient']:.3f}")

    # ===========================================
    # SUMMARY AND SAVE
    # ===========================================
    print(f"\n=== Total specifications: {len(results)} ===")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(OUTPUT_PATH / 'specification_results.csv', index=False)
    print(f"\nResults saved to {OUTPUT_PATH / 'specification_results.csv'}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.3f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.3f}")
    print(f"Coefficient range: [{results_df['coefficient'].min():.3f}, {results_df['coefficient'].max():.3f}]")

    return results_df


if __name__ == '__main__':
    results_df = main()
