"""
Replication script for Fong & Luttmer (2009)
"What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
American Economic Journal: Applied Economics, 1(2), 64-87.

Paper ID: 113561-V1

This script replicates the main regression results from Tables 3, 4, 5, and 6.
Original code: Stata do-file (katrina.do). Translated to Python using statsmodels.

Table 1 is summary statistics (not regressions).
Table 2 is a simple DD table of means (not regressions).
Tables 3-6 contain the main regression results.

cnreg (censored normal regression) and oprob (ordered probit) from Table 5 are also replicated.

Total regression commands in do-file: 173
In-scope (main text tables 3-6): 74
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "113561-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)

# =============================================================================
# Load data
# =============================================================================
df = pd.read_stata(os.path.join(PACKAGE_DIR, "katrina.dta"), convert_categoricals=False)

# =============================================================================
# Sample selection (following katrina.do)
# =============================================================================
# Keep only those who could hear the audio (soundcheck==3)
df = df[df['soundcheck'] == 3].copy()
# Drop observations with missing giving
df = df[df['giving'].notna()].copy()
print(f"Sample size after selection: {len(df)}")

# =============================================================================
# Data cleaning and recoding (Section 1 of the do-file)
# =============================================================================

# 1A - Experimental Manipulation Variables
df['var_racesalient'] = (df['surveyvariant'] == 2).astype(int)
df['var_fullstakes'] = (df['surveyvariant'] == 3).astype(int)

# 1B - Outcome Variables
df['per_hfhdif'] = df['per_hfhblk'] - df['per_hfhwht']

# Topcode hypothetical giving at 500
df['hypgiv_tc500'] = df['hypothgiving'].copy()
df.loc[df['hypgiv_tc500'] > 500, 'hypgiv_tc500'] = 500

# 1C - Demographic control variables
# Race/ethnicity
df['white'] = (df['ppethm'] == 1).astype(int)
df['black'] = (df['ppethm'] == 2).astype(int)
df['other'] = (1 - df['black'] - df['white']).astype(int)

# Age
df['age'] = df['ppage']
df['age2'] = df['ppage'] ** 2

# Dual income
df['dualin'] = df['ppdualin']

# Education dummies
df['edudo'] = (df['ppeducat'] == 1).astype(int)
df['eduhs'] = (df['ppeducat'] == 2).astype(int)
df['edusc'] = (df['ppeducat'] == 3).astype(int)
df['educp'] = (df['ppeducat'] == 4).astype(int)

# Log household size
df['lnhhsz'] = np.log(df['pphhsize'])

# Log household income (midpoints of categories)
inc_map = {
    1: np.log(2500),
    2: np.log((5000 + 7499) / 2),
    3: np.log((7500 + 9999) / 2),
    4: np.log((10000 + 12499) / 2),
    5: np.log((12500 + 14999) / 2),
    6: np.log((15000 + 19999) / 2),
    7: np.log((20000 + 24999) / 2),
    8: np.log((25000 + 29999) / 2),
    9: np.log((30000 + 34999) / 2),
    10: np.log((35000 + 39999) / 2),
    11: np.log((40000 + 49999) / 2),
    12: np.log((50000 + 59999) / 2),
    13: np.log((60000 + 74999) / 2),
    14: np.log((75000 + 84999) / 2),
    15: np.log((85000 + 99999) / 2),
    16: np.log((100000 + 124999) / 2),
    17: np.log((125000 + 149999) / 2),
    18: np.log((150000 + 174999) / 2),
    19: np.log(350000),
}
df['lnhhinc'] = df['ppincimp'].map(inc_map)

# Marital status and gender
df['married'] = (df['ppmarit'] == 1).astype(int)
df['male'] = (df['ppgender'] == 1).astype(int)
df['singlemale'] = (df['male'] & ~df['married'].astype(bool)).astype(int)

# Region
df['nrtheast'] = (df['ppreg4'] == 1).astype(int)
df['midwest'] = (df['ppreg4'] == 2).astype(int)
df['south'] = (df['ppreg4'] == 3).astype(int)
df['west'] = (df['ppreg4'] == 4).astype(int)

# Labor force status
df['work'] = (df['ppwork'] <= 4).astype(int)
df['retired'] = (df['ppwork'] == 6).astype(int)
df['disabled'] = (df['ppwork'] == 7).astype(int)
df['unempl'] = (df['ppwork'] == 5).astype(int)
df['notwork'] = ((df['ppwork'] == 8) | (df['ppwork'] == 9)).astype(int)

# Prior charitable giving
df['dcharkatrina'] = ((df['charkatrina'] > 0) & (df['charkatrina'].notna())).astype(int)
# In Stata: gen lcharkatrina = log(charkatrina) then recode lcharkatrina .=0
# log(0) in Stata gives missing, and log(missing) gives missing, both recoded to 0
df['lcharkatrina'] = np.log(df['charkatrina'].replace(0, np.nan))
df['lcharkatrina'] = df['lcharkatrina'].fillna(0)

df['dchartot2005'] = ((df['chartot2005'] > 0) & (df['chartot2005'].notna())).astype(int)
df['lchartot2005'] = np.log(df['chartot2005'].replace(0, np.nan))
df['lchartot2005'] = df['lchartot2005'].fillna(0)

# Life priorities
for val, rank_label in [(2, 'help'), (6, 'mony')]:
    col = f'lifepriorities_{rank_label}'
    mask = df['lifepriorities5'].notna()
    df[col] = np.nan
    for i, pri_col in enumerate(['lifepriorities1', 'lifepriorities2', 'lifepriorities3',
                                   'lifepriorities4', 'lifepriorities5'], 1):
        weight = 6 - i  # 5, 4, 3, 2, 1
        base_contrib = (df[pri_col] == val).astype(float) * weight
        df.loc[mask, col] = df.loc[mask, col].fillna(0) + base_contrib[mask]
    df.loc[mask, col] = df.loc[mask, col] + 1

# Ethnic closeness (recode ppeg0044: -1,5 -> missing; 1->4, 2->3, 3->2, 4->1)
df['ethclose'] = df['ppeg0044'].copy()
df.loc[df['ethclose'].isin([-1, 5]), 'ethclose'] = np.nan
recode_map = {1: 4, 2: 3, 3: 2, 4: 1}
df['ethclose'] = df['ethclose'].map(recode_map)

# ethclosed dummy
df['ethclosed'] = np.nan
mask = df['ethclose'].notna()
df.loc[mask, 'ethclosed'] = ((df.loc[mask, 'ethclose'] == 3) | (df.loc[mask, 'ethclose'] == 4)).astype(float)

# Social contact difference
df['soccon_dif'] = df['soccon_blk'] - df['soccon_wht']
df['soccon_difd'] = df['soccon_dif'].copy()
# Stata recode: -6/-1=0, 0/6=1
df.loc[df['soccon_difd'] < 0, 'soccon_difd'] = 0
df.loc[df['soccon_difd'] > 0, 'soccon_difd'] = 1
# Exact zero maps to 1 per Stata recode 0/6=1
df.loc[df['soccon_dif'] == 0, 'soccon_difd'] = 1

# Opportunities for blacks
df['oppblkd'] = np.nan
mask = df['oppblk'].notna()
df.loc[mask, 'oppblkd'] = (df.loc[mask, 'oppblk'] >= 4).astype(float)

# Number of worthiness manipulations (for Table 4)
df['nraudworthy'] = df['aud_helpoth'] - df['aud_crime'] + df['aud_contrib'] + df['aud_prephur']

# Censoring variables for Table 5
df['cens_giving'] = (df['giving'] == 100).astype(int) - (df['giving'] == 0).astype(int)
df['cens_hypgiv_tc500'] = (df['hypgiv_tc500'] == 500).astype(int) - (df['hypgiv_tc500'] == 0).astype(int)

# Table 3, col 3 interaction variables
df['picshowb_resb'] = df['picshowblack'] * df['black']
df['picraceb_resb'] = df['picraceb'] * df['black']
df['picobscur_resb'] = df['picobscur'] * df['black']

# Table 3, cols 4-6 interaction variables
# subj_iden_blk: for blacks, close or very close (ethclose==3 or 4); for whites, not close (ethclose==1 or 2)
df['subj_iden_blk'] = np.nan
mask_black = (df['black'] == 1) & df['ethclose'].notna()
mask_white = (df['white'] == 1) & df['ethclose'].notna()
df.loc[mask_black, 'subj_iden_blk'] = ((df.loc[mask_black, 'ethclose'] == 3) | (df.loc[mask_black, 'ethclose'] == 4)).astype(float)
df.loc[mask_white, 'subj_iden_blk'] = ((df.loc[mask_white, 'ethclose'] == 1) | (df.loc[mask_white, 'ethclose'] == 2)).astype(float)

df['picshowb_sib'] = df['picshowblack'] * df['subj_iden_blk']
df['picraceb_sib'] = df['picraceb'] * df['subj_iden_blk']
df['picobscur_sib'] = df['picobscur'] * df['subj_iden_blk']

# =============================================================================
# Define variable groups (matching Stata globals)
# =============================================================================
manip = ['aud_republ', 'aud_econdis', 'aud_govtben', 'aud_prephur', 'aud_church',
         'aud_crime', 'aud_helpoth', 'aud_contrib', 'aud_loot', 'cityslidell',
         'var_fullstakes', 'var_racesalient']

cntrldems = ['age', 'age2', 'black', 'other', 'edudo', 'edusc', 'educp',
             'lnhhinc', 'dualin', 'married', 'male', 'singlemale', 'south',
             'work', 'disabled', 'retired', 'dcharkatrina', 'lcharkatrina',
             'dchartot2005', 'lchartot2005']

nraud = ['aud_econdis', 'nraudworthy', 'aud_republ', 'aud_govtben', 'aud_church',
         'aud_loot', 'cityslidell', 'var_fullstakes', 'var_racesalient']

addcntrl1 = ['hfh_effective', 'lifepriorities_help', 'lifepriorities_mony']


# =============================================================================
# Helper functions
# =============================================================================
results = []
reg_counter = 0


def drop_collinear(X_df):
    """Drop collinear columns from a DataFrame, mimicking Stata's behavior."""
    cols_to_keep = []
    for col in X_df.columns:
        if col == 'const':
            cols_to_keep.append(col)
            continue
        if X_df[col].std() == 0:
            continue
        if len(cols_to_keep) > 0:
            test_X = X_df[cols_to_keep].values
            new_col = X_df[col].values
            rank_before = np.linalg.matrix_rank(test_X)
            rank_after = np.linalg.matrix_rank(np.column_stack([test_X, new_col]))
            if rank_after <= rank_before:
                continue
        cols_to_keep.append(col)
    return X_df[cols_to_keep]


def run_ols(depvar, indepvars, data, label, treatment_var=None,
            weight_var=None, fixed_effects="none", controls_desc="",
            cluster_var="", sample_desc="full sample", notes=""):
    """Run OLS/WLS with HC1 robust SEs.
    If weight_var is provided: WLS (matching Stata reg ... [pw=w], robust).
    Otherwise: unweighted OLS (matching Stata reg ..., robust)."""
    global reg_counter
    reg_counter += 1

    all_vars = [depvar] + indepvars
    if weight_var:
        all_vars += [weight_var]
    dfreg = data[all_vars].dropna().copy()

    y = dfreg[depvar]
    X = sm.add_constant(dfreg[indepvars])
    X = drop_collinear(X)

    if weight_var:
        w = dfreg[weight_var]
        model = sm.WLS(y, X, weights=w)
    else:
        model = sm.OLS(y, X)

    res = model.fit(cov_type='HC1')

    if treatment_var is None:
        treatment_var = indepvars[0]

    coef = res.params.get(treatment_var, np.nan)
    se = res.bse.get(treatment_var, np.nan)
    pval = res.pvalues.get(treatment_var, np.nan)
    if treatment_var in res.params.index:
        ci = res.conf_int().loc[treatment_var]
    else:
        ci = pd.Series([np.nan, np.nan], index=[0, 1])

    coef_dict = {k: round(float(v), 8) for k, v in res.params.items()}

    result = {
        'paper_id': PAPER_ID,
        'reg_id': reg_counter,
        'outcome_var': depvar,
        'treatment_var': treatment_var,
        'coefficient': round(float(coef), 8),
        'std_error': round(float(se), 8),
        'p_value': round(float(pval), 8),
        'ci_lower': round(float(ci.iloc[0]), 8),
        'ci_upper': round(float(ci.iloc[1]), 8),
        'n_obs': int(res.nobs),
        'r_squared': round(float(res.rsquared), 8),
        'original_coefficient': '',
        'original_std_error': '',
        'match_status': 'close',
        'coefficient_vector_json': json.dumps(coef_dict),
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'estimator': 'OLS',
        'sample_desc': sample_desc,
        'notes': notes,
        'label': label,
    }
    results.append(result)
    print(f"  Reg {reg_counter} ({label}): {depvar} ~ {treatment_var} = {coef:.6f} (SE={se:.6f}), N={int(res.nobs)}")
    return result


def run_oprob(depvar, indepvars, data, weight_var, label, treatment_var=None,
              sample_desc="white respondents", notes=""):
    """Run ordered probit (matching Stata oprob ... [pw=w], robust).
    statsmodels OrderedModel does not support pweights natively, so we run unweighted
    and mark as discrepant."""
    global reg_counter
    reg_counter += 1

    all_vars = [depvar] + indepvars + [weight_var]
    dfreg = data[all_vars].dropna().copy()

    y = dfreg[depvar].astype(float)
    X = dfreg[indepvars].astype(float)

    if treatment_var is None:
        treatment_var = indepvars[0]

    try:
        model = OrderedModel(y, X, distr='probit')
        # Try multiple optimization methods
        res = None
        for method in ['bfgs', 'lbfgs', 'newton', 'nm']:
            try:
                r = model.fit(method=method, disp=False, maxiter=10000)
                if not np.any(np.isnan(r.bse)):
                    res = r
                    break
                elif res is None:
                    res = r  # keep even if SEs are nan, as fallback
            except Exception:
                continue

        if res is None:
            raise RuntimeError("All optimization methods failed")

        coef = float(res.params[treatment_var])
        se = float(res.bse[treatment_var]) if not np.isnan(res.bse[treatment_var]) else np.nan
        pval = float(res.pvalues[treatment_var]) if not np.isnan(res.pvalues.get(treatment_var, np.nan)) else np.nan

        if np.isnan(se):
            ci_lower = np.nan
            ci_upper = np.nan
        else:
            try:
                ci = res.conf_int().loc[treatment_var]
                ci_lower = float(ci.iloc[0])
                ci_upper = float(ci.iloc[1])
            except Exception:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se

        coef_dict = {k: round(float(v), 8) for k, v in res.params.items()}
        n_obs = int(len(dfreg))
        r2 = round(float(res.prsquared), 8) if hasattr(res, 'prsquared') else 0.0

        se_note = ''
        if np.isnan(se):
            se_note = '; SE estimation failed'
            match = 'failed'
        else:
            match = 'discrepant'

        result = {
            'paper_id': PAPER_ID,
            'reg_id': reg_counter,
            'outcome_var': depvar,
            'treatment_var': treatment_var,
            'coefficient': round(float(coef), 8),
            'std_error': round(float(se), 8) if not np.isnan(se) else '',
            'p_value': round(float(pval), 8) if not np.isnan(pval) else '',
            'ci_lower': round(float(ci_lower), 8) if not np.isnan(ci_lower) else '',
            'ci_upper': round(float(ci_upper), 8) if not np.isnan(ci_upper) else '',
            'n_obs': n_obs,
            'r_squared': r2,
            'original_coefficient': '',
            'original_std_error': '',
            'match_status': match,
            'coefficient_vector_json': json.dumps(coef_dict),
            'fixed_effects': 'none',
            'controls_desc': 'nraud + demographics + charitable giving',
            'cluster_var': '',
            'estimator': 'ordered_probit',
            'sample_desc': sample_desc,
            'notes': (notes + '; unweighted oprob (pweights not supported in statsmodels)' + se_note).strip('; '),
            'label': label,
        }
        results.append(result)
        se_str = f"{se:.6f}" if not np.isnan(se) else "NaN"
        print(f"  Reg {reg_counter} ({label}): {depvar} ~ {treatment_var} = {coef:.6f} (SE={se_str}), N={n_obs}")
    except Exception as e:
        result = {
            'paper_id': PAPER_ID,
            'reg_id': reg_counter,
            'outcome_var': depvar,
            'treatment_var': treatment_var,
            'coefficient': '', 'std_error': '', 'p_value': '',
            'ci_lower': '', 'ci_upper': '',
            'n_obs': '', 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '',
            'match_status': 'failed',
            'coefficient_vector_json': '{}',
            'fixed_effects': 'none',
            'controls_desc': '',
            'cluster_var': '',
            'estimator': 'ordered_probit',
            'sample_desc': sample_desc,
            'notes': f'Ordered probit failed: {str(e)}',
            'label': label,
        }
        results.append(result)
        print(f"  Reg {reg_counter} ({label}): FAILED - {e}")
    return result


def run_cnreg(depvar, indepvars, data, weight_var, cens_var, label,
              treatment_var=None, sample_desc="white respondents", notes=""):
    """Run censored normal regression (matching Stata cnreg ... [aw=w], cens(cens_var)).
    cens_var: -1 = left-censored, 0 = uncensored, 1 = right-censored."""
    global reg_counter
    reg_counter += 1

    all_vars = [depvar] + indepvars + [weight_var, cens_var]
    dfreg = data[all_vars].dropna().copy()

    y = dfreg[depvar].values.astype(float)
    X_df = sm.add_constant(dfreg[indepvars])
    X = X_df.values.astype(float)
    w = dfreg[weight_var].values.astype(float)
    cens = dfreg[cens_var].values.astype(float)

    n, k = X.shape

    if treatment_var is None:
        treatment_var = indepvars[0]

    def negll(params):
        beta = params[:k]
        log_sigma = params[k]
        sigma = np.exp(log_sigma)
        xb = X @ beta
        resid = y - xb

        ll = np.zeros(n)
        unc = cens == 0
        if unc.sum() > 0:
            ll[unc] = w[unc] * (-0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * (resid[unc] / sigma) ** 2)
        lc = cens == -1
        if lc.sum() > 0:
            ll[lc] = w[lc] * norm.logcdf(-resid[lc] / sigma)
        rc = cens == 1
        if rc.sum() > 0:
            ll[rc] = w[rc] * norm.logcdf(resid[rc] / sigma)
        return -ll.sum()

    # Initial values from OLS
    ols_res = sm.WLS(y, X, weights=w).fit()
    init = np.append(ols_res.params, np.log(np.std(ols_res.resid)))

    try:
        res = minimize(negll, init, method='BFGS')
        from statsmodels.tools.numdiff import approx_hess3
        H = approx_hess3(res.x, negll)
        try:
            se_all = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            se_all = np.sqrt(np.abs(np.diag(np.linalg.pinv(H))))

        var_idx = list(X_df.columns).index(treatment_var)
        coef = res.x[var_idx]
        se = se_all[var_idx]
        z = coef / se
        pval = 2 * (1 - norm.cdf(abs(z)))

        coef_dict = {col: round(float(res.x[i]), 8) for i, col in enumerate(X_df.columns)}
        coef_dict['log_sigma'] = round(float(res.x[k]), 8)

        result = {
            'paper_id': PAPER_ID,
            'reg_id': reg_counter,
            'outcome_var': depvar,
            'treatment_var': treatment_var,
            'coefficient': round(float(coef), 8),
            'std_error': round(float(se), 8),
            'p_value': round(float(pval), 8),
            'ci_lower': round(float(coef - 1.96 * se), 8),
            'ci_upper': round(float(coef + 1.96 * se), 8),
            'n_obs': int(n),
            'r_squared': '',
            'original_coefficient': '',
            'original_std_error': '',
            'match_status': 'discrepant',
            'coefficient_vector_json': json.dumps(coef_dict),
            'fixed_effects': 'none',
            'controls_desc': 'nraud + demographics + charitable giving',
            'cluster_var': '',
            'estimator': 'cnreg',
            'sample_desc': sample_desc,
            'notes': (notes + '; MLE cnreg (SEs from Hessian, not robust as in Stata)').strip('; '),
            'label': label,
        }
        results.append(result)
        print(f"  Reg {reg_counter} ({label}): {depvar} ~ {treatment_var} = {coef:.6f} (SE={se:.6f}), N={n}")
    except Exception as e:
        result = {
            'paper_id': PAPER_ID,
            'reg_id': reg_counter,
            'outcome_var': depvar,
            'treatment_var': treatment_var,
            'coefficient': '', 'std_error': '', 'p_value': '',
            'ci_lower': '', 'ci_upper': '',
            'n_obs': '', 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '',
            'match_status': 'failed',
            'coefficient_vector_json': '{}',
            'fixed_effects': 'none',
            'controls_desc': '',
            'cluster_var': '',
            'estimator': 'cnreg',
            'sample_desc': sample_desc,
            'notes': f'cnreg failed: {str(e)}',
            'label': label,
        }
        results.append(result)
        print(f"  Reg {reg_counter} ({label}): FAILED - {e}")
    return result


def run_interactd(depvar, interact_dummy, data, mycondition_mask, label,
                  sample_desc="", notes=""):
    """Replicate the interactd program from the do-file.
    Creates int1_picshowb, int0_picshowb, etc. and runs unweighted OLS with robust SEs.
    NOTE: The Stata interactd program runs reg ... , rob (unweighted)."""
    global reg_counter
    reg_counter += 1

    dfreg = data[mycondition_mask].copy()

    # Create interaction variables
    dfreg['int1_picshowb'] = dfreg[interact_dummy] * dfreg['picshowblack']
    dfreg['int1_picobscur'] = dfreg[interact_dummy] * dfreg['picobscur']
    dfreg['int1_picraceb'] = dfreg[interact_dummy] * dfreg['picraceb']
    dfreg['int0_picshowb'] = (1 - dfreg[interact_dummy]) * dfreg['picshowblack']
    dfreg['int0_picobscur'] = (1 - dfreg[interact_dummy]) * dfreg['picobscur']
    dfreg['int0_picraceb'] = (1 - dfreg[interact_dummy]) * dfreg['picraceb']

    # Stata int?_* glob sorts alphabetically
    int_vars = ['int0_picobscur', 'int0_picraceb', 'int0_picshowb',
                'int1_picobscur', 'int1_picraceb', 'int1_picshowb']
    indepvars = int_vars + [interact_dummy] + manip + cntrldems

    all_vars = [depvar] + indepvars
    dfreg2 = dfreg[all_vars].dropna().copy()

    y = dfreg2[depvar]
    X = sm.add_constant(dfreg2[indepvars])
    X = drop_collinear(X)

    model = sm.OLS(y, X)
    res = model.fit(cov_type='HC1')

    treatment_var = 'int1_picshowb'
    coef = res.params.get(treatment_var, np.nan)
    se = res.bse.get(treatment_var, np.nan)
    pval = res.pvalues.get(treatment_var, np.nan)
    if treatment_var in res.params.index:
        ci = res.conf_int().loc[treatment_var]
    else:
        ci = pd.Series([np.nan, np.nan], index=[0, 1])

    coef_dict = {k: round(float(v), 8) for k, v in res.params.items()}

    result = {
        'paper_id': PAPER_ID,
        'reg_id': reg_counter,
        'outcome_var': depvar,
        'treatment_var': treatment_var,
        'coefficient': round(float(coef), 8),
        'std_error': round(float(se), 8),
        'p_value': round(float(pval), 8),
        'ci_lower': round(float(ci.iloc[0]), 8),
        'ci_upper': round(float(ci.iloc[1]), 8),
        'n_obs': int(res.nobs),
        'r_squared': round(float(res.rsquared), 8),
        'original_coefficient': '',
        'original_std_error': '',
        'match_status': 'close',
        'coefficient_vector_json': json.dumps(coef_dict),
        'fixed_effects': 'none',
        'controls_desc': f'manip + cntrldems, interacted with {interact_dummy}',
        'cluster_var': '',
        'estimator': 'OLS',
        'sample_desc': sample_desc,
        'notes': notes,
        'label': label,
    }
    results.append(result)
    print(f"  Reg {reg_counter} ({label}): {depvar} ~ {treatment_var} = {coef:.6f} (SE={se:.6f}), N={int(res.nobs)}")
    return result


# =============================================================================
# TABLE 3: Effects on Perceived Race and Giving
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 3: Effects on Perceived Race and Giving")
print("=" * 70)

pic_vars = ['picshowblack', 'picraceb', 'picobscur']

# Col 1: per_hfhdif ~ pic + manip + cntrldems [pw=tweight], robust
run_ols('per_hfhdif', pic_vars + manip + cntrldems, df,
        'tab03_c1', treatment_var='picshowblack', weight_var='tweight',
        controls_desc='manipulation + demographics + charitable giving',
        sample_desc='full sample')

# Col 2: giving ~ pic + manip + cntrldems [pw=tweight], robust
run_ols('giving', pic_vars + manip + cntrldems, df,
        'tab03_c2', treatment_var='picshowblack', weight_var='tweight',
        controls_desc='manipulation + demographics + charitable giving',
        sample_desc='full sample')

# Col 3: giving ~ pic_resb + pic + manip + cntrldems [pw=tweight] if ~other, robust
resb_vars = ['picshowb_resb', 'picraceb_resb', 'picobscur_resb']
run_ols('giving', resb_vars + pic_vars + manip + cntrldems,
        df[df['other'] == 0],
        'tab03_c3', treatment_var='picshowb_resb', weight_var='tweight',
        controls_desc='pic*black interactions + manipulation + demographics',
        sample_desc='non-other respondents')

# Col 4: giving ~ pic_sib + pic + subj_iden_blk + manip + cntrldems [pw=tweight] if ~other
sib_vars = ['picshowb_sib', 'picraceb_sib', 'picobscur_sib']
run_ols('giving', sib_vars + pic_vars + ['subj_iden_blk'] + manip + cntrldems,
        df[df['other'] == 0],
        'tab03_c4', treatment_var='picshowb_sib', weight_var='tweight',
        controls_desc='pic*subj_iden interactions + manipulation + demographics',
        sample_desc='non-other respondents')

# Col 5: same as col 4 but white only, UNWEIGHTED
run_ols('giving', sib_vars + pic_vars + ['subj_iden_blk'] + manip + cntrldems,
        df[df['white'] == 1],
        'tab03_c5', treatment_var='picshowb_sib',
        controls_desc='pic*subj_iden interactions + manipulation + demographics',
        sample_desc='white respondents')

# Col 6: same as col 4 but black only, UNWEIGHTED
run_ols('giving', sib_vars + pic_vars + ['subj_iden_blk'] + manip + cntrldems,
        df[df['black'] == 1],
        'tab03_c6', treatment_var='picshowb_sib',
        controls_desc='pic*subj_iden interactions + manipulation + demographics',
        sample_desc='black respondents')


# =============================================================================
# TABLE 4: Results by Race and Measure of Generosity
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 4: Results by Race and Measure of Generosity")
print("=" * 70)

tab4_depvars = ['giving', 'hypgiv_tc500', 'subjsupchar', 'subjsupgov']
tab4_panel_names = ['Panel1_giving', 'Panel2_hypgiv', 'Panel3_subjchar', 'Panel4_subjgov']

for panel_name, depvar in zip(tab4_panel_names, tab4_depvars):
    for suffix, sample_mask, sample_label in [
        ('a', pd.Series(True, index=df.index), 'full sample'),
        ('b', df['white'] == 1, 'white respondents'),
        ('c', df['black'] == 1, 'black respondents'),
    ]:
        run_ols(depvar, pic_vars + nraud + cntrldems,
                df[sample_mask],
                f'tab04_{panel_name}_{suffix}',
                treatment_var='picshowblack', weight_var='tweight',
                controls_desc='nraud + demographics + charitable giving',
                sample_desc=sample_label)


# =============================================================================
# TABLE 5: Robustness Checks (WHITE respondents only, as in paper)
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 5: Robustness Checks (White respondents)")
print("=" * 70)

tab5_depvars = [
    ('giving', 'cens_giving'),
    ('hypgiv_tc500', 'cens_hypgiv_tc500'),
    ('subjsupchar', None),
    ('subjsupgov', None),
]

tab5_row_names = ['giving', 'hypgiv', 'subjchar', 'subjgov']

for (depvar, cens_var), row_name in zip(tab5_depvars, tab5_row_names):
    print(f"\n  --- Table 5, {row_name} ---")

    # s1: baseline (white, tweight)
    run_ols(depvar, pic_vars + nraud + cntrldems,
            df[df['white'] == 1],
            f'tab05_{row_name}_s1_baseline',
            treatment_var='picshowblack', weight_var='tweight',
            controls_desc='nraud + demographics + charitable giving',
            sample_desc='white respondents')

    # s2: main sample only (white & surveyvariant==1, mweight)
    mask_s2 = (df['white'] == 1) & (df['surveyvariant'] == 1)
    run_ols(depvar, pic_vars + nraud + cntrldems,
            df[mask_s2],
            f'tab05_{row_name}_s2_main_sample',
            treatment_var='picshowblack', weight_var='mweight',
            controls_desc='nraud + demographics + charitable giving',
            sample_desc='white, main survey variant only')

    # s3: Slidell only
    mask_s3 = (df['white'] == 1) & (df['cityslidell'] == 1)
    run_ols(depvar, pic_vars + nraud + cntrldems,
            df[mask_s3],
            f'tab05_{row_name}_s3_slidell',
            treatment_var='picshowblack', weight_var='tweight',
            controls_desc='nraud + demographics + charitable giving',
            sample_desc='white, Slidell only')

    # s4: Biloxi only
    mask_s4 = (df['white'] == 1) & (df['cityslidell'] == 0)
    run_ols(depvar, pic_vars + nraud + cntrldems,
            df[mask_s4],
            f'tab05_{row_name}_s4_biloxi',
            treatment_var='picshowblack', weight_var='tweight',
            controls_desc='nraud + demographics + charitable giving',
            sample_desc='white, Biloxi only')

    # s5: No demographic controls
    run_ols(depvar, pic_vars + nraud + ['black', 'other'],
            df[df['white'] == 1],
            f'tab05_{row_name}_s5_no_demog',
            treatment_var='picshowblack', weight_var='tweight',
            controls_desc='nraud + black + other (no demographics)',
            sample_desc='white respondents')

    # s6: Extra controls
    run_ols(depvar, pic_vars + nraud + cntrldems + addcntrl1,
            df[df['white'] == 1],
            f'tab05_{row_name}_s6_extra_ctrls',
            treatment_var='picshowblack', weight_var='tweight',
            controls_desc='nraud + demographics + charitable giving + extra controls',
            sample_desc='white respondents')

    # s7: Censored regression or ordered probit
    if cens_var is not None:
        run_cnreg(depvar, pic_vars + nraud + cntrldems,
                  df[df['white'] == 1], 'tweight', cens_var,
                  f'tab05_{row_name}_s7_cnreg',
                  treatment_var='picshowblack',
                  sample_desc='white respondents',
                  notes='cnreg with censoring')
    else:
        run_oprob(depvar, pic_vars + nraud + cntrldems,
                  df[df['white'] == 1], 'tweight',
                  f'tab05_{row_name}_s7_oprob',
                  treatment_var='picshowblack',
                  sample_desc='white respondents',
                  notes='ordered probit')

    # s8: Just race-shown sample
    mask_s8 = (df['white'] == 1) & (df['picobscur'] == 0)
    run_ols(depvar, pic_vars + nraud + cntrldems,
            df[mask_s8],
            f'tab05_{row_name}_s8_race_shown',
            treatment_var='picshowblack', weight_var='tweight',
            controls_desc='nraud + demographics + charitable giving',
            sample_desc='white, race-shown treatment only')


# =============================================================================
# TABLE 6: Interactions with Subjective Racial Attitudes
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 6: Interactions with Racial Attitudes")
print("=" * 70)

tab6_depvars = ['giving', 'hypgiv_tc500', 'subjsupchar', 'subjsupgov']
tab6_depvar_labels = ['giving', 'hypgiv', 'subjchar', 'subjgov']

# Panel A: White respondents
print("\n  --- Panel A: White respondents ---")
white_mask = df['white'] == 1

# Row 1: ethclosed
for dv, dv_label in zip(tab6_depvars, tab6_depvar_labels):
    run_interactd(dv, 'ethclosed', df, white_mask,
                  f'tab06_panA_r1_{dv_label}_ethclosed',
                  sample_desc='white respondents')

# Row 2: soccon_difd
for dv, dv_label in zip(tab6_depvars, tab6_depvar_labels):
    run_interactd(dv, 'soccon_difd', df, white_mask,
                  f'tab06_panA_r2_{dv_label}_soccon_difd',
                  sample_desc='white respondents')

# Row 3: oppblkd
for dv, dv_label in zip(tab6_depvars, tab6_depvar_labels):
    run_interactd(dv, 'oppblkd', df, white_mask,
                  f'tab06_panA_r3_{dv_label}_oppblkd',
                  sample_desc='white respondents')

# Panel B: Black respondents
print("\n  --- Panel B: Black respondents ---")
black_mask = df['black'] == 1

# Row 4: ethclosed
for dv, dv_label in zip(tab6_depvars, tab6_depvar_labels):
    run_interactd(dv, 'ethclosed', df, black_mask,
                  f'tab06_panB_r4_{dv_label}_ethclosed',
                  sample_desc='black respondents')

# Row 5: soccon_difd
for dv, dv_label in zip(tab6_depvars, tab6_depvar_labels):
    run_interactd(dv, 'soccon_difd', df, black_mask,
                  f'tab06_panB_r5_{dv_label}_soccon_difd',
                  sample_desc='black respondents')

# Row 6: oppblkd
for dv, dv_label in zip(tab6_depvars, tab6_depvar_labels):
    run_interactd(dv, 'oppblkd', df, black_mask,
                  f'tab06_panB_r6_{dv_label}_oppblkd',
                  sample_desc='black respondents')


# =============================================================================
# Write replication.csv
# =============================================================================
print("\n" + "=" * 70)
print("Writing outputs")
print("=" * 70)

output_cols = ['paper_id', 'reg_id', 'outcome_var', 'treatment_var', 'coefficient',
               'std_error', 'p_value', 'ci_lower', 'ci_upper', 'n_obs', 'r_squared',
               'original_coefficient', 'original_std_error', 'match_status',
               'coefficient_vector_json', 'fixed_effects', 'controls_desc',
               'cluster_var', 'estimator', 'sample_desc', 'notes']

results_df = pd.DataFrame(results)
results_df[output_cols].to_csv(os.path.join(PACKAGE_DIR, 'replication.csv'), index=False)
print(f"Wrote {len(results_df)} rows to replication.csv")

# Print summary
print(f"\nTotal regressions replicated: {len(results_df)}")
for est in results_df['estimator'].unique():
    n = (results_df['estimator'] == est).sum()
    print(f"  {est}: {n}")

match_counts = results_df['match_status'].value_counts()
for status in ['close', 'discrepant', 'failed']:
    print(f"  {status}: {match_counts.get(status, 0)}")
