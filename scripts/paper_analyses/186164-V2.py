"""
Specification Search for Paper 186164-V2:
"Reducing Inequality Through Dynamic Complementarity: Evidence from Head Start and Public School Spending"
Johnson & Jackson (AEJ: Economic Policy)

This script runs a systematic specification search on the AVAILABLE school finance reform event study data.
Note: The main individual-level PSID analysis data is NOT available due to sensitive geocode restrictions.

Methods: Event Study / Panel Fixed Effects DiD
Data: School district panel 1967-1999, N=334,185 district-years
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if not available
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
from scipy import stats

# Configuration
PAPER_ID = "186164-V2"
PAPER_TITLE = "Reducing Inequality Through Dynamic Complementarity: Evidence from Head Start and Public School Spending"
JOURNAL = "AEJ-Policy"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/186164-V2/Results"

def load_data():
    """Load the school finance reform event study data."""
    df = pd.read_stata(f"{BASE_PATH}/SFE_regression_data_1999_income.dta")

    # Create key variables following the Stata code
    # Division dummies
    df['division_newengland'] = df['FIPSTATE'].isin([9,23,25,33,44,50]).astype(int)
    df['division_midatlantic'] = df['FIPSTATE'].isin([34,36,42]).astype(int)
    df['division_encentral'] = df['FIPSTATE'].isin([18,17,26,39,55]).astype(int)
    df['division_wncentral'] = df['FIPSTATE'].isin([19,20,27,29,31,38,46]).astype(int)
    df['division_satlantic'] = df['FIPSTATE'].isin([10,11,12,13,24,37,45,51,54]).astype(int)
    df['division_escentral'] = df['FIPSTATE'].isin([1,21,28,47]).astype(int)
    df['division_wscentral'] = df['FIPSTATE'].isin([5,22,40,48]).astype(int)
    df['division_mountain'] = df['FIPSTATE'].isin([4,8,16,35,30,49,32,56]).astype(int)
    df['division_pacific'] = df['FIPSTATE'].isin([2,6,15,41,53]).astype(int)

    # Division category
    conditions = [
        df['division_newengland'] == 1,
        df['division_midatlantic'] == 1,
        df['division_encentral'] == 1,
        df['division_wncentral'] == 1,
        df['division_satlantic'] == 1,
        df['division_escentral'] == 1,
        df['division_wscentral'] == 1,
        df['division_mountain'] == 1,
        df['division_pacific'] == 1
    ]
    df['divisioncat'] = np.select(conditions, [1,2,3,4,5,6,7,8,9], default=np.nan)

    # Trend variable
    df['trend'] = df['year'] - 1965

    # Log population
    df['lnpop60'] = np.log(df['pop60'].replace(0, np.nan))

    # Interaction with trend
    for var in ['povrate60', 'pct_black_1960', 'pct_urban_1960', 'lnpop60', 'CensusGovt1962_v36']:
        df[f'{var}xyr'] = df[var] * df['trend']
        df[f'miss_{var}'] = df[var].isna().astype(int)
        df[f'{var}xyr'] = df[f'{var}xyr'].fillna(0)

    # Create ln_ppe (log per-pupil expenditure) as outcome
    df['ln_ppe'] = df['outcome1']

    # Create post-reform indicators
    for reform in ['foundation', 'eq_spend', 'limit', 'taxlimit']:
        time_var = f'{reform}_time'
        if time_var in df.columns:
            # Post indicator: reform has happened (time >= 0)
            df[f'{reform}_post'] = ((df[time_var] >= 0) & (df[time_var].notna()) &
                                    (df[time_var] != -99)).astype(int)

    return df

def run_twfe_regression(df, outcome, treatment, controls=None, fe_unit='id', fe_time='year',
                        cluster_var='FIPSTATE', weight_var='size', subset=None):
    """Run two-way fixed effects regression."""

    if subset is not None:
        df = df[subset].copy()
    else:
        df = df.copy()

    # Check treatment variable exists
    if treatment not in df.columns:
        print(f"Treatment variable {treatment} not found")
        return None

    # Drop missing
    vars_needed = [outcome, treatment]
    if fe_unit:
        vars_needed.append(fe_unit)
    if fe_time:
        vars_needed.append(fe_time)
    if controls:
        vars_needed.extend(controls)
    if cluster_var and cluster_var in df.columns:
        vars_needed.append(cluster_var)

    df_reg = df.dropna(subset=[v for v in vars_needed if v in df.columns]).copy()

    if len(df_reg) < 100:
        print(f"Too few observations: {len(df_reg)}")
        return None

    # Use pyfixest
    if HAS_PYFIXEST:
        try:
            # Build formula
            if controls:
                ctrl_str = ' + '.join(controls)
                if fe_unit and fe_time:
                    formula = f"{outcome} ~ {treatment} + {ctrl_str} | {fe_unit} + {fe_time}"
                elif fe_unit:
                    formula = f"{outcome} ~ {treatment} + {ctrl_str} | {fe_unit}"
                elif fe_time:
                    formula = f"{outcome} ~ {treatment} + {ctrl_str} | {fe_time}"
                else:
                    formula = f"{outcome} ~ {treatment} + {ctrl_str}"
            else:
                if fe_unit and fe_time:
                    formula = f"{outcome} ~ {treatment} | {fe_unit} + {fe_time}"
                elif fe_unit:
                    formula = f"{outcome} ~ {treatment} | {fe_unit}"
                elif fe_time:
                    formula = f"{outcome} ~ {treatment} | {fe_time}"
                else:
                    formula = f"{outcome} ~ {treatment}"

            # Run regression
            if cluster_var and cluster_var in df_reg.columns:
                model = pf.feols(formula, data=df_reg, vcov={'CRV1': cluster_var})
            else:
                model = pf.feols(formula, data=df_reg, vcov='hetero')

            coef = model.coef()[treatment]
            se = model.se()[treatment]
            pval = model.pvalue()[treatment]
            tstat = model.tstat()[treatment]
            ci = model.confint()
            ci_lower = ci.loc[treatment, '2.5%']
            ci_upper = ci.loc[treatment, '97.5%']
            n_obs = model._N
            r_squared = model._r2

            # Get all coefficients
            coef_dict = {
                "treatment": {"var": treatment, "coef": float(coef), "se": float(se), "pval": float(pval)},
                "controls": [],
                "fixed_effects_absorbed": [fe_unit, fe_time] if fe_unit and fe_time else [fe_unit or fe_time],
                "diagnostics": {}
            }

            if controls:
                for ctrl in controls:
                    if ctrl in model.coef().index:
                        coef_dict["controls"].append({
                            "var": ctrl,
                            "coef": float(model.coef()[ctrl]),
                            "se": float(model.se()[ctrl]),
                            "pval": float(model.pvalue()[ctrl])
                        })

            return {
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_obs': int(n_obs),
                'r_squared': float(r_squared) if r_squared else None,
                'coefficient_vector_json': json.dumps(coef_dict)
            }
        except Exception as e:
            print(f"pyfixest error: {e}")
            return None

    return None

def run_event_study(df, outcome, reform_type, controls=None, fe_unit='id', fe_time='year',
                    cluster_var='FIPSTATE', weight_var='size', window=(-10, 20), ref_period=-1):
    """Run event study regression with leads and lags."""

    time_var = f'{reform_type}_time'
    if time_var not in df.columns:
        return None

    df_reg = df.copy()

    # Create event time dummies
    event_dummies = []
    for t in range(window[0], window[1] + 1):
        if t != ref_period:
            dummy_name = f'event_t{t}' if t < 0 else f'event_t_plus_{t}'
            df_reg[dummy_name] = (df_reg[time_var] == t).astype(int)
            event_dummies.append(dummy_name)

    # Drop observations outside window or never treated
    df_reg = df_reg[(df_reg[time_var] >= window[0]) & (df_reg[time_var] <= window[1])].copy()

    if len(df_reg) < 1000:
        return None

    # Run regression
    all_controls = event_dummies + (controls if controls else [])

    # Use pyfixest if available
    if HAS_PYFIXEST:
        try:
            ctrl_str = ' + '.join(all_controls)
            formula = f"{outcome} ~ {ctrl_str} | {fe_unit} + {fe_time}"

            if cluster_var and cluster_var in df_reg.columns:
                model = pf.feols(formula, data=df_reg, vcov={'CRV1': cluster_var})
            else:
                model = pf.feols(formula, data=df_reg, vcov='hetero')

            # Extract event study coefficients
            event_coefs = []
            for t in range(window[0], window[1] + 1):
                if t == ref_period:
                    event_coefs.append({
                        "rel_time": t,
                        "coef": None,
                        "se": None,
                        "pval": None,
                        "note": "reference period"
                    })
                else:
                    dummy_name = f'event_t{t}' if t < 0 else f'event_t_plus_{t}'
                    if dummy_name in model.coef().index:
                        event_coefs.append({
                            "rel_time": t,
                            "coef": float(model.coef()[dummy_name]),
                            "se": float(model.se()[dummy_name]),
                            "pval": float(model.pvalue()[dummy_name])
                        })

            # Calculate average post-treatment effect
            post_coefs = [ec['coef'] for ec in event_coefs if ec['rel_time'] is not None and ec['rel_time'] >= 0 and ec['coef'] is not None]
            avg_post = np.mean(post_coefs) if post_coefs else None

            # Pre-trend test: joint F-test of pre-period coefficients
            pre_coefs = [ec for ec in event_coefs if ec['rel_time'] is not None and ec['rel_time'] < 0 and ec['coef'] is not None]
            pretrend_pval = None

            coef_dict = {
                "event_time_coefficients": event_coefs,
                "fixed_effects_absorbed": [fe_unit, fe_time],
                "diagnostics": {
                    "reference_period": ref_period,
                    "avg_post_treatment_effect": avg_post,
                    "pretrend_pval": pretrend_pval
                }
            }

            return {
                'coefficient': float(avg_post) if avg_post else None,
                'std_error': None,  # Would need to compute this properly
                't_stat': None,
                'p_value': None,
                'ci_lower': None,
                'ci_upper': None,
                'n_obs': int(model._N),
                'r_squared': float(model._r2) if model._r2 else None,
                'coefficient_vector_json': json.dumps(coef_dict)
            }
        except Exception as e:
            print(f"Event study error: {e}")
            return None

    return None

def create_result_row(spec_id, spec_tree_path, outcome_var, treatment_var, reg_result,
                      sample_desc, fixed_effects, controls_desc, cluster_var, model_type):
    """Create a row for the results dataframe."""
    if reg_result is None:
        return None

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': reg_result.get('coefficient'),
        'std_error': reg_result.get('std_error'),
        't_stat': reg_result.get('t_stat'),
        'p_value': reg_result.get('p_value'),
        'ci_lower': reg_result.get('ci_lower'),
        'ci_upper': reg_result.get('ci_upper'),
        'n_obs': reg_result.get('n_obs'),
        'r_squared': reg_result.get('r_squared'),
        'coefficient_vector_json': reg_result.get('coefficient_vector_json'),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'{PAPER_ID}.py'
    }

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations, {df['id'].nunique()} districts")

    results = []

    # Define baseline controls
    baseline_controls = ['povrate60xyr', 'pct_black_1960xyr', 'pct_urban_1960xyr',
                        'lnpop60xyr', 'CensusGovt1962_v36xyr']

    # Clean controls (remove missing)
    for ctrl in baseline_controls:
        if ctrl in df.columns:
            df[ctrl] = df[ctrl].fillna(0)

    # Reform types to analyze
    reform_types = ['foundation', 'eq_spend', 'taxlimit']

    print("\n=== Running Baseline Specifications ===")

    # Baseline: Foundation reform effect (main reform type in Appendix C)
    for reform in reform_types:
        treatment_var = f'{reform}_post'

        if treatment_var not in df.columns:
            print(f"Skipping {reform} - no treatment variable")
            continue

        print(f"\nRunning baseline for {reform} reform...")
        result = run_twfe_regression(
            df,
            outcome='ln_ppe',
            treatment=treatment_var,
            controls=baseline_controls,
            fe_unit='id',
            fe_time='year',
            cluster_var='FIPSTATE',
            weight_var='size'
        )

        if result:
            row = create_result_row(
                spec_id=f'baseline_{reform}',
                spec_tree_path='methods/event_study.md',
                outcome_var='ln_ppe',
                treatment_var=treatment_var,
                reg_result=result,
                sample_desc='Full sample, all districts 1967-1999',
                fixed_effects='District + Year FE',
                controls_desc='County characteristics x trend',
                cluster_var='FIPSTATE',
                model_type='TWFE'
            )
            if row:
                results.append(row)
                print(f"  Coef: {result['coefficient']:.4f} (SE: {result['std_error']:.4f}), p={result['p_value']:.4f}")

    print("\n=== Running Fixed Effects Variations ===")

    # FE variations for foundation reform
    treatment_var = 'foundation_post'

    # No fixed effects
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit=None, fe_time=None,
        cluster_var='FIPSTATE'
    )
    if result:
        row = create_result_row(
            spec_id='es/fe/none',
            spec_tree_path='methods/event_study.md#fixed-effects',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='None',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='OLS'
        )
        if row:
            results.append(row)
            print(f"  No FE: {result['coefficient']:.4f}")

    # Unit FE only
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time=None,
        cluster_var='FIPSTATE'
    )
    if result:
        row = create_result_row(
            spec_id='es/fe/unit_only',
            spec_tree_path='methods/event_study.md#fixed-effects',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='District FE only',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='FE'
        )
        if row:
            results.append(row)
            print(f"  Unit FE only: {result['coefficient']:.4f}")

    # Time FE only
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit=None, fe_time='year',
        cluster_var='FIPSTATE'
    )
    if result:
        row = create_result_row(
            spec_id='es/fe/time_only',
            spec_tree_path='methods/event_study.md#fixed-effects',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='Year FE only',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='FE'
        )
        if row:
            results.append(row)
            print(f"  Time FE only: {result['coefficient']:.4f}")

    print("\n=== Running Control Set Variations ===")

    # Control variations
    # No controls
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=None, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE'
    )
    if result:
        row = create_result_row(
            spec_id='es/controls/none',
            spec_tree_path='methods/event_study.md#control-sets',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='District + Year FE',
            controls_desc='None',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  No controls: {result['coefficient']:.4f}")

    # Minimal controls
    minimal_controls = ['povrate60xyr', 'pct_black_1960xyr']
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=minimal_controls, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE'
    )
    if result:
        row = create_result_row(
            spec_id='es/controls/minimal',
            spec_tree_path='methods/event_study.md#control-sets',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='District + Year FE',
            controls_desc='Poverty rate x trend, Pct black x trend',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  Minimal controls: {result['coefficient']:.4f}")

    print("\n=== Running Clustering Variations ===")

    # Clustering variations
    # No clustering (robust SE)
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var=None
    )
    if result:
        row = create_result_row(
            spec_id='robust/cluster/none',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='None (robust)',
            model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  No clustering: {result['coefficient']:.4f} (SE: {result['std_error']:.4f})")

    # Cluster by district
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var='id'
    )
    if result:
        row = create_result_row(
            spec_id='robust/cluster/unit',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Full sample',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='id (district)',
            model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  Cluster by district: {result['coefficient']:.4f} (SE: {result['std_error']:.4f})")

    # Cluster by county
    if 'FIPSCNTY' in df.columns:
        result = run_twfe_regression(
            df, outcome='ln_ppe', treatment=treatment_var,
            controls=baseline_controls, fe_unit='id', fe_time='year',
            cluster_var='FIPSCNTY'
        )
        if result:
            row = create_result_row(
                spec_id='robust/cluster/county',
                spec_tree_path='robustness/clustering_variations.md',
                outcome_var='ln_ppe', treatment_var=treatment_var,
                reg_result=result,
                sample_desc='Full sample',
                fixed_effects='District + Year FE',
                controls_desc='County characteristics x trend',
                cluster_var='FIPSCNTY (county)',
                model_type='TWFE'
            )
            if row:
                results.append(row)
                print(f"  Cluster by county: {result['coefficient']:.4f} (SE: {result['std_error']:.4f})")

    print("\n=== Running Sample Restriction Variations ===")

    # Sample restrictions
    # Early period (1967-1983)
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE',
        subset=df['year'] <= 1983
    )
    if result:
        row = create_result_row(
            spec_id='robust/sample/early_period',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Early period 1967-1983',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  Early period: {result['coefficient']:.4f}")

    # Late period (1984-1999)
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE',
        subset=df['year'] >= 1984
    )
    if result:
        row = create_result_row(
            spec_id='robust/sample/late_period',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Late period 1984-1999',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  Late period: {result['coefficient']:.4f}")

    # Low income districts (q_income == 1)
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE',
        subset=df['q_income'] == 1
    )
    if result:
        row = create_result_row(
            spec_id='robust/sample/low_income',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Low income districts (bottom quartile)',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  Low income districts: {result['coefficient']:.4f}")

    # High income districts (q_income == 6)
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE',
        subset=df['q_income'] == 6
    )
    if result:
        row = create_result_row(
            spec_id='robust/sample/high_income',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='High income districts (top quartile)',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  High income districts: {result['coefficient']:.4f}")

    # Trim outliers (1% and 99%)
    q01 = df['ln_ppe'].quantile(0.01)
    q99 = df['ln_ppe'].quantile(0.99)
    result = run_twfe_regression(
        df, outcome='ln_ppe', treatment=treatment_var,
        controls=baseline_controls, fe_unit='id', fe_time='year',
        cluster_var='FIPSTATE',
        subset=(df['ln_ppe'] >= q01) & (df['ln_ppe'] <= q99)
    )
    if result:
        row = create_result_row(
            spec_id='robust/sample/trim_1pct',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='ln_ppe', treatment_var=treatment_var,
            reg_result=result,
            sample_desc='Trimmed 1%/99% outliers',
            fixed_effects='District + Year FE',
            controls_desc='County characteristics x trend',
            cluster_var='FIPSTATE', model_type='TWFE'
        )
        if row:
            results.append(row)
            print(f"  Trimmed outliers: {result['coefficient']:.4f}")

    print("\n=== Running Leave-One-Out Robustness ===")

    # Leave-one-out for controls
    for ctrl in baseline_controls:
        remaining = [c for c in baseline_controls if c != ctrl]
        result = run_twfe_regression(
            df, outcome='ln_ppe', treatment=treatment_var,
            controls=remaining, fe_unit='id', fe_time='year',
            cluster_var='FIPSTATE'
        )
        if result:
            row = create_result_row(
                spec_id=f'robust/loo/drop_{ctrl}',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var='ln_ppe', treatment_var=treatment_var,
                reg_result=result,
                sample_desc='Full sample',
                fixed_effects='District + Year FE',
                controls_desc=f'Dropped: {ctrl}',
                cluster_var='FIPSTATE', model_type='TWFE'
            )
            if row:
                results.append(row)
                print(f"  Drop {ctrl}: {result['coefficient']:.4f}")

    print("\n=== Running Event Study Specifications ===")

    # Event study with leads and lags
    for reform in ['foundation', 'eq_spend']:
        result = run_event_study(
            df, outcome='ln_ppe', reform_type=reform,
            controls=baseline_controls, fe_unit='id', fe_time='year',
            cluster_var='FIPSTATE', window=(-10, 20), ref_period=-1
        )
        if result:
            row = create_result_row(
                spec_id=f'es/dynamic/{reform}',
                spec_tree_path='methods/event_study.md#dynamic-effects',
                outcome_var='ln_ppe', treatment_var=f'{reform}_time',
                reg_result=result,
                sample_desc='Full sample, event window -10 to +20',
                fixed_effects='District + Year FE',
                controls_desc='County characteristics x trend',
                cluster_var='FIPSTATE', model_type='Event Study'
            )
            if row:
                results.append(row)
                coef_val = result['coefficient']
                print(f"  Event study {reform}: avg post = {coef_val:.4f if coef_val else 'N/A'}")

    # Convert to DataFrame and save
    print("\n=== Saving Results ===")
    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        output_path = f"{BASE_PATH}/specification_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Saved {len(results_df)} specifications to {output_path}")

        # Summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total specifications: {len(results_df)}")

        # Filter to rows with valid coefficients
        valid = results_df['coefficient'].notna()
        print(f"Specifications with valid coefficients: {valid.sum()}")

        if valid.sum() > 0:
            pos = (results_df.loc[valid, 'coefficient'] > 0).sum()
            print(f"Positive coefficients: {pos} ({100*pos/valid.sum():.1f}%)")

            sig_05 = (results_df.loc[valid, 'p_value'] < 0.05).sum()
            print(f"Significant at 5%: {sig_05} ({100*sig_05/valid.sum():.1f}%)")

            sig_01 = (results_df.loc[valid, 'p_value'] < 0.01).sum()
            print(f"Significant at 1%: {sig_01} ({100*sig_01/valid.sum():.1f}%)")

            print(f"Median coefficient: {results_df.loc[valid, 'coefficient'].median():.4f}")
            print(f"Mean coefficient: {results_df.loc[valid, 'coefficient'].mean():.4f}")
            print(f"Range: [{results_df.loc[valid, 'coefficient'].min():.4f}, {results_df.loc[valid, 'coefficient'].max():.4f}]")
    else:
        print("No results to save!")

    return results_df

if __name__ == "__main__":
    results_df = main()
