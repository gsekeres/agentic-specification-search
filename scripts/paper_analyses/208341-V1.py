"""
Specification Search: AER 208341-V1

Paper: "Land Rental Subsidies and Agricultural Productivity" (RCT in Kenya)
Journal: American Economic Review

Method: Instrumental Variables (2SLS)
- Instruments: Random assignment (rental_subsidy, cash_drop)
- Endogenous: Actual take-up (any_PUsubsidy_paid, any_PUcash_paid)
- Main Outcomes: Cultivation, inputs, labor, output, value added on target plot

Main hypothesis: Rental subsidies increase agricultural productivity and investment
compared to cash transfers (testing whether land constraints matter).
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "208341-V1"
JOURNAL = "AER"
PAPER_TITLE = "Land Rental Subsidies and Agricultural Productivity"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/208341-V1/AER_dta_cleaned/merged_data_target_plot_AER.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/208341-V1/"

# Method classification
METHOD_CODE = "instrumental_variables"
METHOD_TREE_PATH = "specification_tree/methods/instrumental_variables.md"


def load_and_prepare_data():
    """Load and prepare the target plot data."""
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    return df


def winsorize_variable(series, level=0.99, bottom=False):
    """Winsorize a series at given percentile."""
    p99 = series.quantile(level)
    p1 = series.quantile(1 - level)

    result = series.copy()
    result = result.clip(upper=p99)
    if bottom:
        result = result.clip(lower=p1)

    return result


def run_iv_2sls_simple(df, outcome_var, endog_vars, instrument_vars, control_vars=None,
                       include_strata_fe=True, include_round_fe=True, cluster_var=None,
                       sample_filter=None):
    """
    Run 2SLS IV regression using a simplified approach that handles multicollinearity.
    """
    # Prepare sample
    if sample_filter is not None:
        sample_df = df.loc[sample_filter].copy()
    else:
        sample_df = df.copy()

    # Drop missing values
    all_vars = [outcome_var] + endog_vars + instrument_vars
    if control_vars:
        all_vars += control_vars
    if cluster_var:
        all_vars += [cluster_var]

    sample_df = sample_df.dropna(subset=all_vars)

    if len(sample_df) == 0:
        return None

    # Build matrices
    y = sample_df[outcome_var].values.astype(float)

    # Endogenous variables
    X_endog = sample_df[endog_vars].values.astype(float)

    # Instruments
    Z = sample_df[instrument_vars].values.astype(float)

    # Build exogenous variable matrix
    X_exog_list = [np.ones((len(sample_df), 1))]  # constant

    if control_vars:
        X_exog_list.append(sample_df[control_vars].values.astype(float))

    if include_strata_fe:
        # Use fewer strata categories to avoid singularity
        strata = sample_df['stratum_reg'].values
        unique_strata = np.unique(strata[~np.isnan(strata)])
        if len(unique_strata) > 1:
            strata_dummies = np.zeros((len(sample_df), len(unique_strata) - 1))
            for i, s in enumerate(unique_strata[1:]):
                strata_dummies[:, i] = (strata == s).astype(float)
            X_exog_list.append(strata_dummies)

    if include_round_fe:
        rounds = sample_df['endline_round'].values
        unique_rounds = np.unique(rounds[~np.isnan(rounds)])
        if len(unique_rounds) > 1:
            round_dummies = np.zeros((len(sample_df), len(unique_rounds) - 1))
            for i, r in enumerate(unique_rounds[1:]):
                round_dummies[:, i] = (rounds == r).astype(float)
            X_exog_list.append(round_dummies)

    X_exog = np.column_stack(X_exog_list) if len(X_exog_list) > 1 else X_exog_list[0]

    # Remove constant columns and check for multicollinearity
    col_std = X_exog.std(axis=0)
    valid_cols = col_std > 1e-10
    X_exog = X_exog[:, valid_cols]

    # Full instrument matrix (exogenous controls + excluded instruments)
    Z_full = np.column_stack([X_exog, Z])
    X_full = np.column_stack([X_exog, X_endog])

    n = len(y)
    k_endog = X_endog.shape[1]
    k_exog = X_exog.shape[1]
    k = k_exog + k_endog
    n_instruments = Z.shape[1]

    try:
        # First stage - regress each endogenous on all instruments
        X_hat_list = [X_exog]
        first_stage_results = []

        for i, endog in enumerate(endog_vars):
            # First stage regression
            try:
                ZtZ_inv = np.linalg.pinv(Z_full.T @ Z_full)  # Use pseudo-inverse
            except:
                return None

            beta_fs = ZtZ_inv @ Z_full.T @ X_endog[:, i]
            X_hat_i = Z_full @ beta_fs
            X_hat_list.append(X_hat_i.reshape(-1, 1))

            # First stage F-stat for excluded instruments
            resid_full = X_endog[:, i] - X_hat_i
            try:
                resid_restricted = X_endog[:, i] - X_exog @ np.linalg.pinv(X_exog.T @ X_exog) @ X_exog.T @ X_endog[:, i]
            except:
                resid_restricted = X_endog[:, i] - np.mean(X_endog[:, i])

            SSR_full = np.sum(resid_full**2)
            SSR_restricted = np.sum(resid_restricted**2)

            if SSR_full > 0 and n > k_exog + n_instruments:
                F_stat = ((SSR_restricted - SSR_full) / n_instruments) / (SSR_full / (n - k_exog - n_instruments))
            else:
                F_stat = np.nan

            first_stage_results.append({
                'endog_var': endog,
                'F_stat': float(F_stat) if not np.isnan(F_stat) else None
            })

        X_hat = np.column_stack(X_hat_list)

        # Second stage
        try:
            XtX_hat_inv = np.linalg.pinv(X_hat.T @ X_hat)
        except:
            return None

        beta_2sls = XtX_hat_inv @ X_hat.T @ y

        # Residuals (using original X, not X_hat)
        resid = y - X_full @ beta_2sls

        # Variance estimation
        if cluster_var:
            clusters = sample_df[cluster_var].values
            unique_clusters = np.unique(clusters[~pd.isna(clusters)])
            n_clusters = len(unique_clusters)

            # Cluster-robust variance (simplified)
            meat = np.zeros((k, k))
            for c in unique_clusters:
                mask = clusters == c
                X_c = X_full[mask]
                e_c = resid[mask]
                score_c = (X_c.T @ e_c).reshape(-1, 1)
                meat += score_c @ score_c.T

            # Finite sample correction
            correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

            try:
                XfXf_inv = np.linalg.pinv(X_full.T @ X_full)
                XhXf = X_hat.T @ X_full
                V = correction * XtX_hat_inv @ XhXf @ XfXf_inv @ meat @ XfXf_inv @ XhXf.T @ XtX_hat_inv
            except:
                # Fallback to simpler variance
                sigma2 = np.sum(resid**2) / (n - k)
                V = sigma2 * XtX_hat_inv
                n_clusters = None
        else:
            sigma2 = np.sum(resid**2) / (n - k)
            V = sigma2 * XtX_hat_inv
            n_clusters = None

        se = np.sqrt(np.abs(np.diag(V)))

        # Extract coefficients for endogenous variables (last k_endog)
        coefs = {}
        for i, endog in enumerate(endog_vars):
            idx = k_exog + i
            coef = beta_2sls[idx]
            se_i = se[idx]
            t_stat = coef / se_i if se_i > 0 else np.nan
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), max(n - k, 1))) if not np.isnan(t_stat) else np.nan
            ci_lower = coef - 1.96 * se_i
            ci_upper = coef + 1.96 * se_i

            coefs[endog] = {
                'coef': float(coef),
                'se': float(se_i),
                't_stat': float(t_stat) if not np.isnan(t_stat) else None,
                'p_val': float(p_val) if not np.isnan(p_val) else None,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            }

        # R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean)**2)
        ss_res = np.sum(resid**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            'coefficients': coefs,
            'first_stage': first_stage_results,
            'n_obs': int(n),
            'n_clusters': int(n_clusters) if n_clusters else None,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else None
        }

    except Exception as e:
        print(f"Error in IV estimation: {e}")
        return None


def run_ols_simple(df, outcome_var, treatment_vars, control_vars=None,
                   include_strata_fe=True, include_round_fe=True,
                   cluster_var=None, sample_filter=None):
    """Run OLS regression for comparison."""
    if sample_filter is not None:
        sample_df = df.loc[sample_filter].copy()
    else:
        sample_df = df.copy()

    # Drop missing
    all_vars = [outcome_var] + treatment_vars
    if control_vars:
        all_vars += control_vars
    if cluster_var:
        all_vars += [cluster_var]

    sample_df = sample_df.dropna(subset=all_vars)

    if len(sample_df) == 0:
        return None

    # Build matrices
    y = sample_df[outcome_var].values.astype(float)

    X_list = [np.ones((len(sample_df), 1))]  # constant
    X_list.append(sample_df[treatment_vars].values.astype(float))

    if control_vars:
        X_list.append(sample_df[control_vars].values.astype(float))

    if include_strata_fe:
        strata = sample_df['stratum_reg'].values
        unique_strata = np.unique(strata[~np.isnan(strata)])
        if len(unique_strata) > 1:
            strata_dummies = np.zeros((len(sample_df), len(unique_strata) - 1))
            for i, s in enumerate(unique_strata[1:]):
                strata_dummies[:, i] = (strata == s).astype(float)
            X_list.append(strata_dummies)

    if include_round_fe:
        rounds = sample_df['endline_round'].values
        unique_rounds = np.unique(rounds[~np.isnan(rounds)])
        if len(unique_rounds) > 1:
            round_dummies = np.zeros((len(sample_df), len(unique_rounds) - 1))
            for i, r in enumerate(unique_rounds[1:]):
                round_dummies[:, i] = (rounds == r).astype(float)
            X_list.append(round_dummies)

    X = np.column_stack(X_list)

    # Remove constant columns
    col_std = X.std(axis=0)
    valid_cols = col_std > 1e-10
    valid_cols[0] = True  # Keep constant
    X = X[:, valid_cols]

    n, k = X.shape
    n_treat = len(treatment_vars)

    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        resid = y - X @ beta

        if cluster_var:
            clusters = sample_df[cluster_var].values
            unique_clusters = np.unique(clusters[~pd.isna(clusters)])
            n_clusters = len(unique_clusters)

            meat = np.zeros((k, k))
            for c in unique_clusters:
                mask = clusters == c
                X_c = X[mask]
                e_c = resid[mask]
                score_c = (X_c.T @ e_c).reshape(-1, 1)
                meat += score_c @ score_c.T

            correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
            V = correction * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = np.sum(resid**2) / (n - k)
            V = sigma2 * XtX_inv
            n_clusters = None

        se = np.sqrt(np.abs(np.diag(V)))

        coefs = {}
        var_names = ['const'] + treatment_vars
        for i, var in enumerate(var_names):
            if i < len(beta):
                coef = beta[i]
                se_i = se[i]
                t_stat = coef / se_i if se_i > 0 else np.nan
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), max(n - k, 1))) if not np.isnan(t_stat) else np.nan
                ci_lower = coef - 1.96 * se_i
                ci_upper = coef + 1.96 * se_i

                coefs[var] = {
                    'coef': float(coef),
                    'se': float(se_i),
                    't_stat': float(t_stat) if not np.isnan(t_stat) else None,
                    'p_val': float(p_val) if not np.isnan(p_val) else None,
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper)
                }

        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean)**2)
        ss_res = np.sum(resid**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            'coefficients': coefs,
            'n_obs': int(n),
            'n_clusters': int(n_clusters) if n_clusters else None,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else None
        }

    except Exception as e:
        print(f"Error in OLS: {e}")
        return None


def format_result(result, spec_id, spec_tree_path, outcome_var, treatment_var,
                  sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
                  first_stage_F=None):
    """Format result for CSV output."""
    if result is None:
        return None

    if 'coefficients' not in result:
        return None

    # Get treatment coefficient
    if treatment_var in result['coefficients']:
        coef_info = result['coefficients'][treatment_var]
    else:
        return None

    if coef_info['p_val'] is None:
        return None

    # Build coefficient vector JSON
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': coef_info['coef'],
            'se': coef_info['se'],
            'pval': coef_info['p_val']
        },
        'controls': [],
        'fixed_effects': fixed_effects.split(', ') if fixed_effects else [],
        'diagnostics': {
            'first_stage_F': first_stage_F
        }
    }

    # Add first stage info if IV
    if 'first_stage' in result:
        coef_vector['first_stage'] = result['first_stage']

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef_info['coef'],
        'std_error': coef_info['se'],
        't_stat': coef_info['t_stat'] if coef_info['t_stat'] else np.nan,
        'p_value': coef_info['p_val'],
        'ci_lower': coef_info['ci_lower'],
        'ci_upper': coef_info['ci_upper'],
        'n_obs': result['n_obs'],
        'r_squared': result['r_squared'],
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


def main():
    """Run the full specification search."""
    print(f"Loading data from {DATA_PATH}...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} observations")

    results = []

    # Define key variables
    outcome_vars = [
        ('ETwadj_ag_va1_r6_qaB_1', 'Value Added'),
        ('ETd2_1_plot_use_cltvtd_1', 'Cultivated'),
        ('ETd34_ag_inputs1_B_1', 'Input Value'),
        ('ETL_val_1', 'Labor Value'),
        ('ETe1_3_h_value1_qa_1', 'Output Value'),
    ]

    endog_vars = ['any_PUsubsidy_paid', 'any_PUcash_paid']
    instrument_vars = ['rental_subsidy', 'cash_drop']

    # Baseline controls (from the do file)
    baseline_controls_va = [
        'Be2_8_SR_h_value_1',  # baseline output
        'Bd34_ag_inputs1_1',   # baseline inputs
        'Bwadj_vd6_3_L_hhdays_r6_1',  # baseline household labor
        'Bvd6_6_L_hire_days_1',  # baseline hired labor
        'L_target_plot_size_mean'  # plot size
    ]

    # Verify baseline controls exist
    available_controls = [c for c in baseline_controls_va if c in df.columns]
    print(f"Available baseline controls: {available_controls}")

    # ============================================================
    # PREPARE DATA
    # ============================================================

    # Main outcome: Value Added (primary result)
    outcome_var = 'ETwadj_ag_va1_r6_qaB_1'

    # Winsorize outcome at p99 (and p1 for value added)
    df['outcome_W99'] = winsorize_variable(df[outcome_var], level=0.99, bottom=True)

    # Handle missing baseline controls: replace with mean and add missing indicator
    for ctrl in available_controls:
        # Create missing indicator
        missing_col = f'missing_{ctrl}'
        df[missing_col] = df[ctrl].isna().astype(float)
        # Fill missing with column mean
        df[ctrl] = df[ctrl].fillna(df[ctrl].mean())

    missing_cols = [f'missing_{c}' for c in available_controls]
    all_controls = available_controls + missing_cols

    # ============================================================
    # BASELINE SPECIFICATION
    # ============================================================
    print("\n=== Running Baseline Specifications ===")

    # Baseline: 2SLS with strata and round FE, clustered by fin
    print("Running baseline 2SLS...")

    baseline_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=all_controls,
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if baseline_result:
        # Report rental subsidy effect
        fs_F = baseline_result['first_stage'][0]['F_stat'] if baseline_result['first_stage'] else None
        res = format_result(
            baseline_result,
            'baseline',
            'methods/instrumental_variables.md#baseline',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled across endline rounds',
            'strata + round',
            f'Baseline controls: {", ".join(available_controls)}',
            'fin',
            '2SLS',
            first_stage_F=fs_F
        )
        if res:
            results.append(res)
            print(f"  Rental subsidy: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}, p={res['p_value']:.4f}")
            print(f"  First-stage F: {fs_F:.1f}" if fs_F else "  First-stage F: N/A")

        # Also report cash drop effect
        fs_F_cash = baseline_result['first_stage'][1]['F_stat'] if len(baseline_result['first_stage']) > 1 else None
        res_cash = format_result(
            baseline_result,
            'baseline_cash',
            'methods/instrumental_variables.md#baseline',
            outcome_var,
            'any_PUcash_paid',
            'All strata, pooled across endline rounds',
            'strata + round',
            f'Baseline controls: {", ".join(available_controls)}',
            'fin',
            '2SLS',
            first_stage_F=fs_F_cash
        )
        if res_cash:
            results.append(res_cash)
            print(f"  Cash drop: coef={res_cash['coefficient']:.3f}, se={res_cash['std_error']:.3f}, p={res_cash['p_value']:.4f}")

    # ============================================================
    # IV METHOD VARIATIONS
    # ============================================================
    print("\n=== Running IV Method Variations ===")

    # OLS (ignoring endogeneity)
    print("Running OLS for comparison...")
    ols_result = run_ols_simple(
        df,
        'outcome_W99',
        endog_vars,
        control_vars=all_controls,
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if ols_result:
        res = format_result(
            ols_result,
            'iv/method/ols',
            'methods/instrumental_variables.md#estimation-method',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'strata + round',
            f'Baseline controls',
            'fin',
            'OLS'
        )
        if res:
            results.append(res)
            print(f"  OLS rental subsidy: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # Reduced form (instrument on outcome)
    print("Running reduced form...")
    rf_result = run_ols_simple(
        df,
        'outcome_W99',
        instrument_vars,
        control_vars=all_controls,
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if rf_result:
        res = format_result(
            rf_result,
            'iv/first_stage/reduced_form',
            'methods/instrumental_variables.md#first-stage',
            outcome_var,
            'rental_subsidy',
            'All strata, pooled',
            'strata + round',
            f'Baseline controls',
            'fin',
            'OLS (reduced form)'
        )
        if res:
            results.append(res)
            print(f"  Reduced form rental_subsidy: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

        # Also report cash drop reduced form
        res_cash = format_result(
            rf_result,
            'iv/first_stage/reduced_form_cash',
            'methods/instrumental_variables.md#first-stage',
            outcome_var,
            'cash_drop',
            'All strata, pooled',
            'strata + round',
            f'Baseline controls',
            'fin',
            'OLS (reduced form)'
        )
        if res_cash:
            results.append(res_cash)
            print(f"  Reduced form cash_drop: coef={res_cash['coefficient']:.3f}, se={res_cash['std_error']:.3f}")

    # ============================================================
    # SAMPLE VARIATIONS (by stratum)
    # ============================================================
    print("\n=== Running Sample Variations ===")

    for stratum_val, stratum_name in [(1, 'Stratum_C'), (2, 'Stratum_NC')]:
        print(f"Running for {stratum_name}...")

        stratum_result = run_iv_2sls_simple(
            df,
            'outcome_W99',
            endog_vars,
            instrument_vars,
            control_vars=all_controls,
            include_strata_fe=True,
            include_round_fe=True,
            cluster_var='fin',
            sample_filter=(df['stratum_C_NC'] == stratum_val)
        )

        if stratum_result:
            res = format_result(
                stratum_result,
                f'iv/sample/{stratum_name.lower()}',
                'methods/instrumental_variables.md#sample-restrictions',
                outcome_var,
                'any_PUsubsidy_paid',
                stratum_name,
                'strata + round',
                f'Baseline controls',
                'fin',
                '2SLS',
                first_stage_F=stratum_result['first_stage'][0]['F_stat'] if stratum_result['first_stage'] else None
            )
            if res:
                results.append(res)
                print(f"  {stratum_name}: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # ============================================================
    # OUTCOME VARIATIONS
    # ============================================================
    print("\n=== Running Outcome Variations ===")

    for out_var, out_name in outcome_vars[1:]:  # Skip value added (already done)
        print(f"Running for {out_name}...")

        # Winsorize
        df[f'{out_var}_W99'] = winsorize_variable(df[out_var], level=0.99)

        out_result = run_iv_2sls_simple(
            df,
            f'{out_var}_W99',
            endog_vars,
            instrument_vars,
            control_vars=all_controls,
            include_strata_fe=True,
            include_round_fe=True,
            cluster_var='fin',
            sample_filter=(df['stratum_C_NC'] >= 1)
        )

        if out_result:
            res = format_result(
                out_result,
                f'iv/outcome/{out_name.lower().replace(" ", "_")}',
                'methods/instrumental_variables.md#baseline',
                out_var,
                'any_PUsubsidy_paid',
                'All strata, pooled',
                'strata + round',
                f'Baseline controls',
                'fin',
                '2SLS',
                first_stage_F=out_result['first_stage'][0]['F_stat'] if out_result['first_stage'] else None
            )
            if res:
                results.append(res)
                print(f"  {out_name}: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # ============================================================
    # FIXED EFFECTS VARIATIONS
    # ============================================================
    print("\n=== Running Fixed Effects Variations ===")

    # No FE
    print("Running without fixed effects...")
    no_fe_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=all_controls,
        include_strata_fe=False,
        include_round_fe=False,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if no_fe_result:
        res = format_result(
            no_fe_result,
            'iv/fe/none',
            'methods/instrumental_variables.md#fixed-effects',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'none',
            f'Baseline controls',
            'fin',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  No FE: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # Only strata FE
    print("Running with strata FE only...")
    strata_only_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=all_controls,
        include_strata_fe=True,
        include_round_fe=False,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if strata_only_result:
        res = format_result(
            strata_only_result,
            'iv/fe/strata_only',
            'methods/instrumental_variables.md#fixed-effects',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'strata',
            f'Baseline controls',
            'fin',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  Strata FE only: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # Only round FE
    print("Running with round FE only...")
    round_only_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=all_controls,
        include_strata_fe=False,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if round_only_result:
        res = format_result(
            round_only_result,
            'iv/fe/round_only',
            'methods/instrumental_variables.md#fixed-effects',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'round',
            f'Baseline controls',
            'fin',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  Round FE only: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # ============================================================
    # CONTROL VARIATIONS
    # ============================================================
    print("\n=== Running Control Variations ===")

    # No controls
    print("Running without controls...")
    no_ctrl_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=None,
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if no_ctrl_result:
        res = format_result(
            no_ctrl_result,
            'iv/controls/none',
            'methods/instrumental_variables.md#control-sets',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'strata + round',
            'No controls',
            'fin',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  No controls: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # Only plot size control
    print("Running with minimal controls (plot size only)...")
    minimal_ctrl_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=['L_target_plot_size_mean'],
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if minimal_ctrl_result:
        res = format_result(
            minimal_ctrl_result,
            'iv/controls/minimal',
            'methods/instrumental_variables.md#control-sets',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'strata + round',
            'Plot size only',
            'fin',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  Minimal controls: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # ============================================================
    # LEAVE-ONE-OUT ROBUSTNESS
    # ============================================================
    print("\n=== Running Leave-One-Out Robustness ===")

    for ctrl in available_controls:
        print(f"Dropping {ctrl}...")
        # Keep missing indicators but drop the actual control
        loo_controls = [c for c in all_controls if c != ctrl]

        loo_result = run_iv_2sls_simple(
            df,
            'outcome_W99',
            endog_vars,
            instrument_vars,
            control_vars=loo_controls,
            include_strata_fe=True,
            include_round_fe=True,
            cluster_var='fin',
            sample_filter=(df['stratum_C_NC'] >= 1)
        )

        if loo_result:
            res = format_result(
                loo_result,
                f'robust/loo/drop_{ctrl}',
                'robustness/leave_one_out.md',
                outcome_var,
                'any_PUsubsidy_paid',
                'All strata, pooled',
                'strata + round',
                f'Dropped: {ctrl}',
                'fin',
                '2SLS'
            )
            if res:
                results.append(res)
                print(f"  Drop {ctrl}: coef={res['coefficient']:.3f}")

    # ============================================================
    # SINGLE COVARIATE ROBUSTNESS
    # ============================================================
    print("\n=== Running Single Covariate Analysis ===")

    # Bivariate (no controls beyond FE)
    print("Running bivariate...")
    bivariate_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=None,
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var='fin',
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if bivariate_result:
        res = format_result(
            bivariate_result,
            'robust/single/none',
            'robustness/single_covariate.md',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'strata + round',
            'No controls (bivariate)',
            'fin',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  Bivariate: coef={res['coefficient']:.3f}")

    for ctrl in available_controls:
        print(f"Single control: {ctrl}...")
        single_result = run_iv_2sls_simple(
            df,
            'outcome_W99',
            endog_vars,
            instrument_vars,
            control_vars=[ctrl],
            include_strata_fe=True,
            include_round_fe=True,
            cluster_var='fin',
            sample_filter=(df['stratum_C_NC'] >= 1)
        )

        if single_result:
            res = format_result(
                single_result,
                f'robust/single/{ctrl}',
                'robustness/single_covariate.md',
                outcome_var,
                'any_PUsubsidy_paid',
                'All strata, pooled',
                'strata + round',
                f'Single control: {ctrl}',
                'fin',
                '2SLS'
            )
            if res:
                results.append(res)
                print(f"  Single {ctrl}: coef={res['coefficient']:.3f}")

    # ============================================================
    # CLUSTERING VARIATIONS
    # ============================================================
    print("\n=== Running Clustering Variations ===")

    # No clustering (robust SE)
    print("Running with robust (unclustered) SE...")
    robust_result = run_iv_2sls_simple(
        df,
        'outcome_W99',
        endog_vars,
        instrument_vars,
        control_vars=all_controls,
        include_strata_fe=True,
        include_round_fe=True,
        cluster_var=None,
        sample_filter=(df['stratum_C_NC'] >= 1)
    )

    if robust_result:
        res = format_result(
            robust_result,
            'robust/cluster/none',
            'robustness/clustering_variations.md',
            outcome_var,
            'any_PUsubsidy_paid',
            'All strata, pooled',
            'strata + round',
            f'Baseline controls',
            'none (robust)',
            '2SLS'
        )
        if res:
            results.append(res)
            print(f"  Robust SE: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # ============================================================
    # BY ENDLINE ROUND
    # ============================================================
    print("\n=== Running By Endline Round ===")

    for round_val in [1, 2, 3, 4]:
        print(f"Running for round {round_val}...")

        round_result = run_iv_2sls_simple(
            df,
            'outcome_W99',
            endog_vars,
            instrument_vars,
            control_vars=all_controls,
            include_strata_fe=True,
            include_round_fe=False,  # No round FE for single round
            cluster_var='fin',
            sample_filter=(df['endline_round'] == round_val)
        )

        if round_result:
            res = format_result(
                round_result,
                f'iv/sample/round_{round_val}',
                'methods/instrumental_variables.md#sample-restrictions',
                outcome_var,
                'any_PUsubsidy_paid',
                f'Endline round {round_val}',
                'strata',
                f'Baseline controls',
                'fin',
                '2SLS'
            )
            if res:
                results.append(res)
                print(f"  Round {round_val}: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print(f"\n=== Saving {len(results)} Results ===")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_PATH}specification_results.csv", index=False)
    print(f"Saved to {OUTPUT_PATH}specification_results.csv")

    # Print summary
    print("\n=== Summary Statistics ===")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.3f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.3f}")
    print(f"Range: [{results_df['coefficient'].min():.3f}, {results_df['coefficient'].max():.3f}]")

    return results_df


if __name__ == "__main__":
    results_df = main()
