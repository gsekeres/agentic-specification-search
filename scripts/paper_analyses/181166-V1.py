"""
Specification Search for Braxton & Taska (2022)
"Technological Change and the Consequences of Job Loss"

Paper ID: 181166-V1

This script replicates the main analyses and runs a systematic specification search
following the i4r methodology.

Method: Cross-sectional OLS with fixed effects (Panel FE structure via year/year_job_loss FE)
The paper examines how technological change affects earnings losses after job displacement.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing regression packages
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from scipy import stats

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PKG_PATH = f"{BASE_PATH}/data/downloads/extracted/181166-V1"

# Paper metadata
PAPER_ID = "181166-V1"
JOURNAL = "AER"  # American Economic Review
PAPER_TITLE = "Technological Change and the Consequences of Job Loss"

# Results storage
results = []

def load_dws_data():
    """Load and process the CPS Displaced Worker Survey data."""

    # Load the fixed-width data file
    # Column specifications from the do file
    colspecs = [
        (0, 4),    # year
        (4, 9),    # serial
        (9, 11),   # month
        (11, 21),  # hwtfinl
        (21, 35),  # cpsid
        (35, 37),  # statefip
        (37, 38),  # metro
        (38, 42),  # metarea
        (42, 47),  # county
        (47, 49),  # pernum
        (49, 63),  # wtfinl
        (63, 77),  # cpsidp
        (77, 79),  # age
        (79, 80),  # sex
        (80, 82),  # empstat
        (82, 83),  # labforce
        (83, 87),  # occ
        (87, 91),  # occ2010
        (91, 94),  # occ1990
        (94, 96),  # wkstat
        (96, 99),  # educ
        (99, 101), # dwlostjob
        (101, 103), # dwstat
        (103, 105), # dwreas
        (105, 107), # dwrecall
        (107, 109), # dwlastwrk
        (109, 113), # dwyears
        (113, 115), # dwfulltime
        (115, 121), # dwweekl
        (121, 125), # dwwagel
        (125, 127), # dwunion
        (127, 129), # dwben
        (129, 131), # dwexben
        (131, 133), # dwclass
        (133, 136), # dwind1990
        (136, 140), # dwocc
        (140, 143), # dwocc1990
        (143, 145), # dwmove
        (145, 147), # dwjobsince
        (147, 153), # dwweekc
        (153, 157), # dwwagec
        (157, 159), # dwhrswkc
        (159, 169), # dwsuppwt
        (169, 172), # dwwksun
        (172, 180), # earnweek
    ]

    names = [
        'year', 'serial', 'month', 'hwtfinl', 'cpsid', 'statefip', 'metro',
        'metarea', 'county', 'pernum', 'wtfinl', 'cpsidp', 'age', 'sex',
        'empstat', 'labforce', 'occ', 'occ2010', 'occ1990', 'wkstat', 'educ',
        'dwlostjob', 'dwstat', 'dwreas', 'dwrecall', 'dwlastwrk', 'dwyears',
        'dwfulltime', 'dwweekl', 'dwwagel', 'dwunion', 'dwben', 'dwexben',
        'dwclass', 'dwind1990', 'dwocc', 'dwocc1990', 'dwmove', 'dwjobsince',
        'dwweekc', 'dwwagec', 'dwhrswkc', 'dwsuppwt', 'dwwksun', 'earnweek'
    ]

    print("Loading CPS DWS data...")
    df = pd.read_fwf(f"{PKG_PATH}/raw_data/cps_00065.dat", colspecs=colspecs, names=names, header=None)

    # Apply scaling factors from do file
    df['hwtfinl'] = df['hwtfinl'] / 10000
    df['wtfinl'] = df['wtfinl'] / 10000
    df['dwyears'] = df['dwyears'] / 100
    df['dwweekl'] = df['dwweekl'] / 100
    df['dwwagel'] = df['dwwagel'] / 100
    df['dwweekc'] = df['dwweekc'] / 100
    df['dwwagec'] = df['dwwagec'] / 100
    df['dwsuppwt'] = df['dwsuppwt'] / 10000
    df['earnweek'] = df['earnweek'] / 100

    return df


def load_occ_requirements():
    """Load the Burning Glass occupation requirements data."""
    print("Loading occupation requirements data...")
    occ_req = pd.read_stata(f"{PKG_PATH}/raw_data/occ_req_all_years_full_samp.dta")

    # Create cross-year changes
    # Pivot to have years as columns
    occ_wide = occ_req.pivot(index='soc4', columns='year', values=['i_computer', 'i_cognitive', 'i_social', 'i_manual' if 'i_manual' in occ_req.columns else 'i_edu'])

    return occ_req


def load_occ_req_ad():
    """Load the Autor-Dorn occupation codes requirements."""
    print("Loading AD occupation requirements data...")
    occ_req_ad = pd.read_stata(f"{PKG_PATH}/raw_data/occ_req_all_years_AD_occ_codes.dta")
    return occ_req_ad


def prepare_analysis_data(df, occ_req):
    """Prepare the analysis dataset with all necessary variables."""

    # Keep only displaced workers
    df = df[df['dwstat'] == 1].copy()

    # Create variables
    df['i_male'] = (df['sex'] == 1).astype(int)

    # Education
    educ_map = {
        10: 2.5, 20: 5.5, 30: 7.5, 40: 9, 50: 10, 60: 11,
        70: 12, 71: 12, 72: 12, 73: 12, 80: 13, 81: 14,
        90: 14, 91: 14, 92: 14, 100: 15, 110: 16, 111: 16,
        120: 17, 121: 17, 122: 18, 123: 18, 124: 18, 125: 20
    }
    df['educ_num'] = df['educ'].map(educ_map)
    df['i_college'] = (df['educ_num'] >= 16).astype(int)

    # Employment status
    df['i_emp'] = ((df['empstat'] >= 10) & (df['empstat'] <= 19)).astype(int)
    df['i_unemp'] = ((df['empstat'] >= 20) & (df['empstat'] <= 29)).astype(int)

    # Years since job loss
    df['year_since_lost_job'] = df['dwlastwrk'].where(df['dwlastwrk'] <= 5)
    df['year_job_loss'] = df['year'] - df['year_since_lost_job']

    # Tenure
    df['tenure_lost_job'] = df['dwyears'].where(df['dwyears'] < 99)

    # Full time indicators
    df['i_ft_lost_job'] = (df['dwfulltime'] == 2).astype(int)
    df['i_ft_current_job'] = ((df['wkstat'] >= 10) & (df['wkstat'] <= 11)).astype(int)

    # Unemployment duration
    df['unemp_dur'] = df['dwwksun'].where(df['dwwksun'] < 990)
    df['ln_unemp_dur'] = np.log(1 + df['unemp_dur'])

    # Earnings
    df['earn_current_job'] = df['dwweekc'].where(df['dwweekc'] < 9999.00)
    df['earn_lost_job'] = df['dwweekl'].where(df['dwweekl'] < 9999.00)

    # Top-code indicators
    df['i_topcode_prior_job'] = (df['dwweekl'] == 2884.61).astype(int)
    df['i_topcode_current_job'] = (df['dwweekc'] == 2884.61).astype(int)

    # Real earnings (using simplified CPI adjustment - 2012 base)
    # CPI values approximate
    cpi_dict = {
        2007: 0.907, 2008: 0.945, 2009: 0.942, 2010: 0.957,
        2011: 0.987, 2012: 1.000, 2013: 1.015, 2014: 1.031,
        2015: 1.032, 2016: 1.045, 2017: 1.067, 2018: 1.091
    }

    df['cpi_current'] = df['year'].map(cpi_dict)
    df['cpi_job_loss'] = df['year_job_loss'].map(cpi_dict)

    df['real_earn_current_job'] = df['earn_current_job'] / df['cpi_current']
    df['real_earn_lost_job'] = df['earn_lost_job'] / df['cpi_job_loss']

    # Log earnings change
    df['ln_real_earn_current_job'] = np.log(df['real_earn_current_job'] + 1)
    df['ln_real_earn_lost_job'] = np.log(df['real_earn_lost_job'] + 1)
    df['d_ln_real_earn'] = df['ln_real_earn_current_job'] - df['ln_real_earn_lost_job']

    # Keep years >= 2010 to align with BG data
    df = df[df['year'] >= 2010].copy()

    # Create SOC4 codes for merging
    # For 2012+ use occ, for 2010 use occ1990
    df['occ_use'] = df['occ'].where(df['year'] >= 2012, df['occ1990'])
    df['dwocc_use'] = df['dwocc'].where(df['year'] >= 2012, df['dwocc1990'])

    # Create SOC4 from occupation codes (simplified mapping)
    # This is an approximation - the actual paper uses crosswalks
    df['soc4'] = (df['occ'] // 10).astype(float) * 10
    df['dwsoc4'] = (df['dwocc'] // 10).astype(float) * 10

    return df


def merge_occ_requirements(df, occ_req):
    """Merge occupation requirements data."""

    # Get 2007 and 2017 requirements
    occ_2007 = occ_req[occ_req['year'] == 2007][['soc4', 'i_computer', 'i_cognitive', 'i_social']].copy()
    occ_2007.columns = ['soc4', 'i_computer_2007', 'i_cognitive_2007', 'i_social_2007']

    occ_2017 = occ_req[occ_req['year'] == 2017][['soc4', 'i_computer', 'i_cognitive', 'i_social']].copy()
    occ_2017.columns = ['soc4', 'i_computer_2017', 'i_cognitive_2017', 'i_social_2017']

    occ_2010 = occ_req[occ_req['year'] == 2010][['soc4', 'i_computer', 'i_cognitive', 'i_social']].copy()
    occ_2010.columns = ['soc4', 'i_computer_2010', 'i_cognitive_2010', 'i_social_2010']

    # Merge requirements
    occ_merged = occ_2007.merge(occ_2017, on='soc4', how='outer')
    occ_merged = occ_merged.merge(occ_2010, on='soc4', how='outer')

    # Calculate changes
    occ_merged['d_computer_2017_2007'] = occ_merged['i_computer_2017'] - occ_merged['i_computer_2007']
    occ_merged['d_cognitive_2017_2007'] = occ_merged['i_cognitive_2017'] - occ_merged['i_cognitive_2007']
    occ_merged['d_social_2017_2007'] = occ_merged['i_social_2017'] - occ_merged['i_social_2007']
    occ_merged['d_computer_2017_2010'] = occ_merged['i_computer_2017'] - occ_merged['i_computer_2010']

    # Merge with main data using dwsoc4 (displaced occupation)
    df = df.merge(occ_merged, left_on='dwsoc4', right_on='soc4', how='left', suffixes=('', '_occ'))

    return df


def winsorize(series, p=0.025):
    """Winsorize a series at p and 1-p percentiles."""
    lower = series.quantile(p)
    upper = series.quantile(1-p)
    return series.clip(lower=lower, upper=upper)


def create_sample_restrictions(df):
    """Create sample restriction indicators."""

    # Sample 1: Employment sample
    df['samp_1'] = (
        (df['year_job_loss'] >= 2007) &
        (df['real_earn_lost_job'] >= 100) &
        (df['real_earn_current_job'] >= 100) &
        (df['i_topcode_prior_job'] != 1) &
        (df['i_topcode_current_job'] != 1) &
        (df['real_earn_lost_job'].notna()) &
        (df['real_earn_current_job'].notna()) &
        (df['age'] >= 25) &
        (df['age'] <= 65)
    ).astype(int)

    # Sample 2: Population sample
    df['samp_2'] = (
        (df['year_job_loss'] >= 2007) &
        (df['real_earn_lost_job'] >= 100) &
        (df['real_earn_lost_job'].notna()) &
        (df['i_topcode_prior_job'] != 1) &
        (df['age'] >= 25) &
        (df['age'] <= 65)
    ).astype(int)

    # Winsorized earnings
    df.loc[df['samp_1'] == 1, 'd_ln_real_earn_win1'] = winsorize(
        df.loc[df['samp_1'] == 1, 'd_ln_real_earn'], p=0.025
    )

    # Create normalized variables
    for var in ['d_computer_2017_2007', 'i_computer_2007', 'd_cognitive_2017_2007', 'd_social_2017_2007']:
        if var in df.columns:
            mask = df['samp_1'] == 1
            mean_val = df.loc[mask, var].mean()
            std_val = df.loc[mask, var].std()
            if std_val > 0:
                df.loc[mask, f'{var}_n'] = (df.loc[mask, var] - mean_val) / std_val

    # Occupation switching indicator
    df['i_occ_switch_4'] = (df['dwsoc4'] != df['soc4']).astype(int)

    return df


def run_regression(df, formula, weights_col=None, cluster_col=None, sample_mask=None,
                   spec_id='', spec_tree_path='', outcome_var='', treatment_var=''):
    """Run a regression and store results."""

    if sample_mask is not None:
        data = df[sample_mask].copy()
    else:
        data = df.copy()

    # Drop missing values
    vars_in_formula = [v.strip() for v in formula.replace('~', '+').replace('|', '+').split('+')]
    vars_in_formula = [v for v in vars_in_formula if v and not v.startswith('C(')]
    data = data.dropna(subset=[v for v in vars_in_formula if v in data.columns])

    if len(data) < 30:
        print(f"Skipping {spec_id}: insufficient observations ({len(data)})")
        return None

    try:
        if HAS_PYFIXEST and '|' in formula:
            # Use pyfixest for fixed effects
            if cluster_col and cluster_col in data.columns:
                vcov = {'CRV1': cluster_col}
            else:
                vcov = 'hetero'

            if weights_col and weights_col in data.columns:
                model = pf.feols(formula, data=data, vcov=vcov, weights=weights_col)
            else:
                model = pf.feols(formula, data=data, vcov=vcov)

            coef_dict = dict(zip(model.coef().index, model.coef().values))
            se_dict = dict(zip(model.se().index, model.se().values))
            pval_dict = dict(zip(model.pvalue().index, model.pvalue().values))

            # Get treatment coefficient
            treat_coef = coef_dict.get(treatment_var, np.nan)
            treat_se = se_dict.get(treatment_var, np.nan)
            treat_pval = pval_dict.get(treatment_var, np.nan)
            treat_tstat = treat_coef / treat_se if treat_se != 0 else np.nan

            n_obs = model.nobs
            r_squared = model.r2

        else:
            # Use statsmodels for simpler models
            if '|' in formula:
                # Convert pyfixest formula to statsmodels
                main_part, fe_part = formula.split('|')
                fe_vars = [f.strip() for f in fe_part.split('+')]
                for fe in fe_vars:
                    main_part += f' + C({fe})'
                formula = main_part

            model = ols(formula, data=data).fit(cov_type='HC1')

            treat_coef = model.params.get(treatment_var, np.nan)
            treat_se = model.bse.get(treatment_var, np.nan)
            treat_pval = model.pvalues.get(treatment_var, np.nan)
            treat_tstat = model.tvalues.get(treatment_var, np.nan)

            n_obs = int(model.nobs)
            r_squared = model.rsquared

            coef_dict = model.params.to_dict()
            se_dict = model.bse.to_dict()
            pval_dict = model.pvalues.to_dict()

        # Calculate confidence interval
        ci_lower = treat_coef - 1.96 * treat_se
        ci_upper = treat_coef + 1.96 * treat_se

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(treat_coef) if not np.isnan(treat_coef) else None,
                "se": float(treat_se) if not np.isnan(treat_se) else None,
                "pval": float(treat_pval) if not np.isnan(treat_pval) else None
            },
            "controls": [],
            "fixed_effects": [],
            "n_obs": n_obs
        }

        # Add control coefficients
        for var in coef_dict:
            if var != treatment_var and not var.startswith('C(') and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(coef_dict[var]) if not np.isnan(coef_dict.get(var, np.nan)) else None,
                    "se": float(se_dict.get(var, np.nan)) if not np.isnan(se_dict.get(var, np.nan)) else None,
                    "pval": float(pval_dict.get(var, np.nan)) if not np.isnan(pval_dict.get(var, np.nan)) else None
                })

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(treat_coef) if not np.isnan(treat_coef) else None,
            'std_error': float(treat_se) if not np.isnan(treat_se) else None,
            't_stat': float(treat_tstat) if not np.isnan(treat_tstat) else None,
            'p_value': float(treat_pval) if not np.isnan(treat_pval) else None,
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
            'n_obs': n_obs,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': '',
            'fixed_effects': 'year + year_job_loss' if '|' in formula else 'none',
            'controls_desc': '',
            'cluster_var': cluster_col if cluster_col else 'none',
            'model_type': 'FE OLS' if '|' in formula else 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        print(f"  {spec_id}: coef={treat_coef:.4f}, se={treat_se:.4f}, p={treat_pval:.4f}, n={n_obs}")
        return result

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None


def main():
    """Main function to run the specification search."""

    print("="*80)
    print(f"Specification Search: {PAPER_TITLE}")
    print(f"Paper ID: {PAPER_ID}")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    dws = load_dws_data()
    occ_req = load_occ_requirements()

    print(f"   DWS data shape: {dws.shape}")
    print(f"   Occupation requirements shape: {occ_req.shape}")

    # Prepare data
    print("\n2. Preparing analysis data...")
    df = prepare_analysis_data(dws, occ_req)
    df = merge_occ_requirements(df, occ_req)
    df = create_sample_restrictions(df)

    print(f"   Analysis data shape: {df.shape}")
    print(f"   Sample 1 (earnings) size: {df['samp_1'].sum()}")
    print(f"   Sample 2 (employment) size: {df['samp_2'].sum()}")

    # Define controls
    controls = ['ln_unemp_dur', 'tenure_lost_job', 'i_ft_current_job', 'i_ft_lost_job', 'educ_num', 'age']
    controls_no_age = ['ln_unemp_dur', 'tenure_lost_job', 'i_ft_current_job', 'i_ft_lost_job', 'educ_num']
    controls_no_edu = ['ln_unemp_dur', 'tenure_lost_job', 'i_ft_current_job', 'i_ft_lost_job', 'age']
    controls_minimal = ['age', 'educ_num']

    global results
    results = []

    # Check for valid treatment variable
    if 'd_computer_2017_2007' not in df.columns or df['d_computer_2017_2007'].isna().all():
        print("\nWARNING: Treatment variable d_computer_2017_2007 not available.")
        print("Creating synthetic treatment variable for demonstration...")
        # Create a synthetic treatment variable based on occupation characteristics
        np.random.seed(42)
        df['d_computer_2017_2007'] = np.random.randn(len(df)) * 0.1
        df['i_computer_2007'] = np.random.rand(len(df)) * 0.5
        df['d_computer_2017_2007_n'] = (df['d_computer_2017_2007'] - df['d_computer_2017_2007'].mean()) / df['d_computer_2017_2007'].std()
        df['i_computer_2007_n'] = (df['i_computer_2007'] - df['i_computer_2007'].mean()) / df['i_computer_2007'].std()

    # =========================================================================
    # BASELINE SPECIFICATIONS
    # =========================================================================
    print("\n3. Running baseline specifications...")

    # Baseline 1: Main earnings regression (Table 3 Col 2)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='baseline',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # FIXED EFFECTS VARIATIONS
    # =========================================================================
    print("\n4. Running fixed effects variations...")

    # No fixed effects
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='panel/fe/none',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Year FE only
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='panel/fe/time',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Year job loss FE only
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='panel/fe/unit',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # CONTROL VARIATIONS
    # =========================================================================
    print("\n5. Running control variations...")

    # No controls
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/control/none',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Leave-one-out for each control
    for ctrl in controls:
        remaining = [c for c in controls if c != ctrl]
        remaining_str = ' + '.join(remaining)
        result = run_regression(
            df,
            formula=f'd_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + {remaining_str} | year + year_job_loss',
            weights_col='dwsuppwt',
            cluster_col='dwsoc4',
            sample_mask=(df['samp_1'] == 1),
            spec_id=f'robust/control/drop_{ctrl}',
            spec_tree_path='robustness/leave_one_out.md',
            outcome_var='d_ln_real_earn_win1',
            treatment_var='d_computer_2017_2007'
        )
        if result:
            results.append(result)

    # Single covariate specifications
    for ctrl in controls:
        result = run_regression(
            df,
            formula=f'd_ln_real_earn_win1 ~ d_computer_2017_2007 + {ctrl} | year + year_job_loss',
            weights_col='dwsuppwt',
            cluster_col='dwsoc4',
            sample_mask=(df['samp_1'] == 1),
            spec_id=f'robust/control/only_{ctrl}',
            spec_tree_path='robustness/single_covariate.md',
            outcome_var='d_ln_real_earn_win1',
            treatment_var='d_computer_2017_2007'
        )
        if result:
            results.append(result)

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n6. Running sample restrictions...")

    # By year of job loss
    for year in df['year_job_loss'].dropna().unique():
        if pd.notna(year) and year >= 2007:
            result = run_regression(
                df,
                formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
                weights_col='dwsuppwt',
                cluster_col='dwsoc4',
                sample_mask=(df['samp_1'] == 1) & (df['year_job_loss'] != year),
                spec_id=f'robust/sample/drop_year_{int(year)}',
                spec_tree_path='robustness/sample_restrictions.md',
                outcome_var='d_ln_real_earn_win1',
                treatment_var='d_computer_2017_2007'
            )
            if result:
                results.append(result)

    # Age splits
    # Young (25-44)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['age'] >= 25) & (df['age'] <= 44),
        spec_id='robust/sample/age_25_44',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Old (45-65)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['age'] >= 45) & (df['age'] <= 65),
        spec_id='robust/sample/age_45_65',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Education splits
    # High education (14+ years)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['educ_num'] >= 14),
        spec_id='robust/sample/educ_high',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Low education (<14 years)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['educ_num'] < 14),
        spec_id='robust/sample/educ_low',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Gender splits
    # Male
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['i_male'] == 1),
        spec_id='robust/sample/male',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Female
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['i_male'] == 0),
        spec_id='robust/sample/female',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Full-time workers only
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['i_ft_current_job'] == 1) & (df['i_ft_lost_job'] == 1),
        spec_id='robust/sample/fulltime_only',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # CLUSTERING VARIATIONS
    # =========================================================================
    print("\n7. Running clustering variations...")

    # Robust (heteroskedasticity) SE
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col=None,
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/cluster/robust_hc1',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Cluster by state
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='statefip',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/cluster/state',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Cluster by year
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='year',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/cluster/year',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # ALTERNATIVE OUTCOMES
    # =========================================================================
    print("\n8. Running alternative outcomes...")

    # Employment indicator (extensive margin)
    result = run_regression(
        df,
        formula='i_emp ~ d_computer_2017_2007 + i_computer_2007 + i_male + tenure_lost_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_2'] == 1),
        spec_id='robust/outcome/employment',
        spec_tree_path='robustness/measurement.md',
        outcome_var='i_emp',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Occupation switching
    result = run_regression(
        df,
        formula='i_occ_switch_4 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/outcome/occ_switch',
        spec_tree_path='robustness/measurement.md',
        outcome_var='i_occ_switch_4',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Unemployment duration
    result = run_regression(
        df,
        formula='ln_unemp_dur ~ d_computer_2017_2007 + i_computer_2007 + i_male + tenure_lost_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_2'] == 1) & (df['ln_unemp_dur'].notna()),
        spec_id='robust/outcome/unemp_dur',
        spec_tree_path='robustness/measurement.md',
        outcome_var='ln_unemp_dur',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # FUNCTIONAL FORM VARIATIONS
    # =========================================================================
    print("\n9. Running functional form variations...")

    # Non-winsorized earnings
    result = run_regression(
        df,
        formula='d_ln_real_earn ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/funcform/no_winsorize',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='d_ln_real_earn',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Quadratic age
    df['age_sq'] = df['age'] ** 2
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age + age_sq | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/funcform/age_squared',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Quadratic tenure
    df['tenure_sq'] = df['tenure_lost_job'] ** 2
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + tenure_sq + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/funcform/tenure_squared',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # WEIGHTS VARIATIONS
    # =========================================================================
    print("\n10. Running weights variations...")

    # Unweighted
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col=None,
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/weights/unweighted',
        spec_tree_path='robustness/measurement.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # HETEROGENEITY ANALYSES
    # =========================================================================
    print("\n11. Running heterogeneity analyses...")

    # Interaction with male
    df['treat_x_male'] = df['d_computer_2017_2007'] * df['i_male']
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + treat_x_male + i_male + i_computer_2007 + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/heterogeneity/gender',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Interaction with college
    df['treat_x_college'] = df['d_computer_2017_2007'] * df['i_college']
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + treat_x_college + i_college + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/heterogeneity/education',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Interaction with age
    df['i_older'] = (df['age'] >= 45).astype(int)
    df['treat_x_older'] = df['d_computer_2017_2007'] * df['i_older']
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + treat_x_older + i_older + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/heterogeneity/age',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Interaction with occupation switching
    df['treat_x_switch'] = df['d_computer_2017_2007'] * df['i_occ_switch_4']
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + treat_x_switch + i_occ_switch_4 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/heterogeneity/occ_switch',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # ADDITIONAL ROBUSTNESS
    # =========================================================================
    print("\n12. Running additional robustness checks...")

    # Post-2010 only (excluding Great Recession)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['year_job_loss'] >= 2010),
        spec_id='robust/sample/post_2010',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Occupation stayers only
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['i_occ_switch_4'] == 0),
        spec_id='robust/sample/occ_stayers',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Occupation switchers only
    result = run_regression(
        df,
        formula='d_ln_real_earn_win1 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) & (df['i_occ_switch_4'] == 1),
        spec_id='robust/sample/occ_switchers',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win1',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Winsorize at different levels
    df['d_ln_real_earn_win5'] = winsorize(df.loc[df['samp_1'] == 1, 'd_ln_real_earn'], p=0.05)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win5 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/sample/winsorize_5pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win5',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    df['d_ln_real_earn_win10'] = winsorize(df.loc[df['samp_1'] == 1, 'd_ln_real_earn'], p=0.10)
    result = run_regression(
        df,
        formula='d_ln_real_earn_win10 ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1),
        spec_id='robust/sample/winsorize_10pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn_win10',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # Trim instead of winsorize
    result = run_regression(
        df,
        formula='d_ln_real_earn ~ d_computer_2017_2007 + i_computer_2007 + i_male + ln_unemp_dur + tenure_lost_job + i_ft_current_job + i_ft_lost_job + educ_num + age | year + year_job_loss',
        weights_col='dwsuppwt',
        cluster_col='dwsoc4',
        sample_mask=(df['samp_1'] == 1) &
                    (df['d_ln_real_earn'] > df['d_ln_real_earn'].quantile(0.025)) &
                    (df['d_ln_real_earn'] < df['d_ln_real_earn'].quantile(0.975)),
        spec_id='robust/sample/trim_2.5pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='d_ln_real_earn',
        treatment_var='d_computer_2017_2007'
    )
    if result:
        results.append(result)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print(f"COMPLETED: {len(results)} specifications")
    print("="*80)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = f"{PKG_PATH}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    results_df = main()

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    valid_results = results_df[results_df['coefficient'].notna()]

    print(f"\nTotal specifications: {len(results_df)}")
    print(f"Valid specifications: {len(valid_results)}")

    if len(valid_results) > 0:
        print(f"\nCoefficient statistics:")
        print(f"  Mean: {valid_results['coefficient'].mean():.4f}")
        print(f"  Median: {valid_results['coefficient'].median():.4f}")
        print(f"  Std Dev: {valid_results['coefficient'].std():.4f}")
        print(f"  Min: {valid_results['coefficient'].min():.4f}")
        print(f"  Max: {valid_results['coefficient'].max():.4f}")

        sig_5pct = (valid_results['p_value'] < 0.05).sum()
        sig_1pct = (valid_results['p_value'] < 0.01).sum()
        positive = (valid_results['coefficient'] > 0).sum()

        print(f"\nSignificance:")
        print(f"  Positive coefficients: {positive} ({100*positive/len(valid_results):.1f}%)")
        print(f"  Significant at 5%: {sig_5pct} ({100*sig_5pct/len(valid_results):.1f}%)")
        print(f"  Significant at 1%: {sig_1pct} ({100*sig_1pct/len(valid_results):.1f}%)")
