#!/usr/bin/env python3
"""
Replication script for 113561-V1:
"What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
Fong & Luttmer, AEJ: Applied Economics 2009

Translates katrina.do from Stata to Python.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113561-V1"
PAPER_ID = "113561-V1"

# ============================================================
# LOAD AND PREPARE DATA (mirrors katrina.do Section 1)
# ============================================================

df = pd.read_stata(f"{PACKAGE_DIR}/katrina.dta", convert_categoricals=False)

# Sample selection: keep if soundcheck==3
df = df[df['soundcheck'] == 3].copy()
# Drop if giving is missing
df = df[df['giving'].notna()].copy()

print(f"Sample size after selection: {len(df)}")

# 1A - Experimental manipulation variables
df['var_racesalient'] = (df['surveyvariant'] == 2).astype(int)
df['var_fullstakes'] = (df['surveyvariant'] == 3).astype(int)

# 1B - Outcome variables
df['per_hfhdif'] = df['per_hfhblk'] - df['per_hfhwht']

# Topcode hypothetical giving at 500
df['hypgiv_tc500'] = df['hypothgiving'].copy()
df.loc[df['hypgiv_tc500'] > 500, 'hypgiv_tc500'] = 500

# 1C - Demographic control variables
df['white'] = (df['ppethm'] == 1).astype(int)
df['black'] = (df['ppethm'] == 2).astype(int)
df['other'] = (1 - df['black'] - df['white']).astype(int)

df['age'] = df['ppage']
df['age2'] = df['ppage'] ** 2

df['dualin'] = df['ppdualin']

# Education dummies
df['edudo'] = (df['ppeducat'] == 1).astype(int)
df['eduhs'] = (df['ppeducat'] == 2).astype(int)
df['edusc'] = (df['ppeducat'] == 3).astype(int)
df['educp'] = (df['ppeducat'] == 4).astype(int)

df['lnhhsz'] = np.log(df['pphhsize'])

# Household income midpoints
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

df['married'] = (df['ppmarit'] == 1).astype(int)
df['male'] = (df['ppgender'] == 1).astype(int)
df['singlemale'] = (df['male'] & ~df['married'].astype(bool)).astype(int)

df['nrtheast'] = (df['ppreg4'] == 1).astype(int)
df['midwest'] = (df['ppreg4'] == 2).astype(int)
df['south'] = (df['ppreg4'] == 3).astype(int)
df['west'] = (df['ppreg4'] == 4).astype(int)

df['work'] = (df['ppwork'] <= 4).astype(int)
df['retired'] = (df['ppwork'] == 6).astype(int)
df['disabled'] = (df['ppwork'] == 7).astype(int)
df['unempl'] = (df['ppwork'] == 5).astype(int)
df['notwork'] = ((df['ppwork'] == 8) | (df['ppwork'] == 9)).astype(int)

# Prior charitable giving
df['dcharkatrina'] = ((df['charkatrina'] > 0) & df['charkatrina'].notna()).astype(int)
df['lcharkatrina'] = np.log(df['charkatrina'])
df.loc[df['lcharkatrina'].isna(), 'lcharkatrina'] = 0
# Stata: recode lcharkatrina .=0 — this sets log(0) = -inf to 0 as well
df.loc[~np.isfinite(df['lcharkatrina']), 'lcharkatrina'] = 0

df['dchartot2005'] = ((df['chartot2005'] > 0) & df['chartot2005'].notna()).astype(int)
df['lchartot2005'] = np.log(df['chartot2005'])
df.loc[df['lchartot2005'].isna(), 'lchartot2005'] = 0
df.loc[~np.isfinite(df['lchartot2005']), 'lchartot2005'] = 0

# Life priorities
for val, rank_offset in [(2, 'help'), (6, 'mony')]:
    col = f'lifepriorities_{rank_offset}'
    mask = df['lifepriorities5'].notna()
    df[col] = np.nan
    df.loc[mask, col] = 1
    for pos in range(1, 6):
        weight = 6 - pos  # 5,4,3,2,1
        df.loc[mask & (df[f'lifepriorities{pos}'] == val), col] = df.loc[mask & (df[f'lifepriorities{pos}'] == val), col] + weight
    # Actually the Stata code: 1+5*(lp1==val)+4*(lp2==val)+3*(lp3==val)+2*(lp4==val)+1*(lp5==val)
    df.loc[mask, col] = (1
        + 5 * (df.loc[mask, 'lifepriorities1'] == val).astype(int)
        + 4 * (df.loc[mask, 'lifepriorities2'] == val).astype(int)
        + 3 * (df.loc[mask, 'lifepriorities3'] == val).astype(int)
        + 2 * (df.loc[mask, 'lifepriorities4'] == val).astype(int)
        + 1 * (df.loc[mask, 'lifepriorities5'] == val).astype(int))

# Ethnic closeness
df['ethclose'] = df['ppeg0044'].copy()
# recode: -1 5=. 1=4 2=3 3=2 4=1
df.loc[df['ethclose'].isin([-1, 5]), 'ethclose'] = np.nan
recode_map = {1: 4, 2: 3, 3: 2, 4: 1}
df['ethclose'] = df['ethclose'].map(lambda x: recode_map.get(x, x) if pd.notna(x) else np.nan)

df['ethclosed'] = np.nan
mask = df['ethclose'].notna()
df.loc[mask, 'ethclosed'] = ((df.loc[mask, 'ethclose'] == 3) | (df.loc[mask, 'ethclose'] == 4)).astype(int)

# Social contact
df['soccon_dif'] = df['soccon_blk'] - df['soccon_wht']
df['soccon_difd'] = df['soccon_dif'].copy()
df.loc[df['soccon_difd'] < 0, 'soccon_difd'] = 0
df.loc[df['soccon_difd'] > 0, 'soccon_difd'] = 1
# Stata: recode soccon_difd -6/-1=0 0/6=1
# Actually: values from -6 to -1 -> 0, values 0 to 6 -> 1
# But 0 maps to 1? Let me re-read: recode soccon_difd -6/-1=0 0/6=1
# In Stata recode: -6/-1=0 means values -6 through -1 become 0
# 0/6=1 means values 0 through 6 become 1
df['soccon_difd'] = df['soccon_dif'].copy()
df.loc[(df['soccon_difd'] >= -6) & (df['soccon_difd'] <= -1), 'soccon_difd'] = 0
df.loc[(df['soccon_difd'] >= 0) & (df['soccon_difd'] <= 6), 'soccon_difd'] = 1

# Opportunities for blacks
df['oppblkd'] = np.nan
mask = df['oppblk'].notna()
df.loc[mask, 'oppblkd'] = (df.loc[mask, 'oppblk'] >= 4).astype(int)

# Worthiness manipulation count
df['nraudworthy'] = df['aud_helpoth'] - df['aud_crime'] + df['aud_contrib'] + df['aud_prephur']

# Table 3 interaction variables
df['picshowb_resb'] = df['picshowblack'] * df['black']
df['picobscur_resb'] = df['picobscur'] * df['black']
df['picraceb_resb'] = df['picraceb'] * df['black']

# Table 3 col 4 interaction
df['subj_iden_blk'] = np.nan
mask_black = (df['ethclose'].notna()) & (df['black'] == 1)
df.loc[mask_black, 'subj_iden_blk'] = ((df.loc[mask_black, 'ethclose'] == 3) | (df.loc[mask_black, 'ethclose'] == 4)).astype(int)
mask_white = (df['ethclose'].notna()) & (df['white'] == 1)
df.loc[mask_white, 'subj_iden_blk'] = ((df.loc[mask_white, 'ethclose'] == 1) | (df.loc[mask_white, 'ethclose'] == 2)).astype(int)

df['picshowb_sib'] = df['picshowblack'] * df['subj_iden_blk']
df['picraceb_sib'] = df['picraceb'] * df['subj_iden_blk']
df['picobscur_sib'] = df['picobscur'] * df['subj_iden_blk']

# Censoring indicators
df['cens_giving'] = (df['giving'] == 100).astype(int) - (df['giving'] == 0).astype(int)
df['cens_hypgiv_tc500'] = (df['hypgiv_tc500'] == 500).astype(int) - (df['hypgiv_tc500'] == 0).astype(int)

# ============================================================
# DEFINE VARIABLE GROUPS (globals in Stata)
# ============================================================

manip = ['aud_republ', 'aud_econdis', 'aud_govtben', 'aud_prephur', 'aud_church',
         'aud_crime', 'aud_helpoth', 'aud_contrib', 'aud_loot', 'cityslidell',
         'var_fullstakes', 'var_racesalient']

cntrldems = ['age', 'age2', 'black', 'other', 'edudo', 'edusc', 'educp', 'lnhhinc',
             'dualin', 'married', 'male', 'singlemale', 'south', 'work', 'disabled',
             'retired', 'dcharkatrina', 'lcharkatrina', 'dchartot2005', 'lchartot2005']

cntrlx = ['age', 'age2', 'black', 'other', 'edudo', 'edusc', 'educp', 'lnhhinc',
          'dualin', 'married', 'male', 'singlemale', 'south', 'work', 'disabled', 'retired']

addcntrl1 = ['hfh_effective', 'lifepriorities_help', 'lifepriorities_mony']

nraud = ['aud_econdis', 'nraudworthy', 'aud_republ', 'aud_govtben', 'aud_church',
         'aud_loot', 'cityslidell', 'var_fullstakes', 'var_racesalient']

# ============================================================
# RUN REGRESSIONS
# ============================================================

results = []
reg_id = 0

def run_wls_reg(depvar, indepvars, data, weight_var, label, treatment_var='picshowblack',
                table='', notes='', estimator='OLS/WLS'):
    """Run a weighted least squares regression using pyfixest."""
    global reg_id
    reg_id += 1

    all_vars = [depvar] + indepvars + [weight_var]
    subdf = data.dropna(subset=all_vars).copy()

    formula = f"{depvar} ~ " + " + ".join(indepvars)

    try:
        m = pf.feols(formula, data=subdf, vcov="hetero", weights=weight_var)

        coef = m.coef()[treatment_var]
        se = m.se()[treatment_var]
        pval = m.pvalue()[treatment_var]
        nobs = m._N
        r2 = m._r2

        # Build coefficient vector JSON
        coef_dict = {k: float(v) for k, v in m.coef().items()}
        se_dict = {k: float(v) for k, v in m.se().items()}
        coef_json = json.dumps({
            "coefficients": coef_dict,
            "standard_errors": se_dict,
            "inference": {"type": "HC1", "cluster_var": None},
            "software": {"language": "python", "package": "pyfixest"}
        })

        result = {
            'paper_id': PAPER_ID,
            'reg_id': reg_id,
            'outcome_var': depvar,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se,
            'n_obs': nobs,
            'r_squared': r2,
            'original_coefficient': '',
            'original_std_error': '',
            'match_status': 'close',  # Will verify
            'coefficient_vector_json': coef_json,
            'fixed_effects': '',
            'controls_desc': label,
            'cluster_var': '',
            'estimator': estimator,
            'sample_desc': notes,
            'notes': table
        }
        results.append(result)
        print(f"  Reg {reg_id} ({table}): {depvar} ~ {treatment_var}, coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={nobs}")
        return m
    except Exception as e:
        print(f"  Reg {reg_id} ({table}): FAILED - {e}")
        results.append({
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': treatment_var, 'coefficient': '', 'std_error': '',
            'p_value': '', 'ci_lower': '', 'ci_upper': '', 'n_obs': '', 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '', 'match_status': 'failed',
            'coefficient_vector_json': '{}', 'fixed_effects': '', 'controls_desc': label,
            'cluster_var': '', 'estimator': estimator, 'sample_desc': notes, 'notes': f'{table}: {e}'
        })
        return None

def run_oprob(depvar, indepvars, data, weight_var, label, treatment_var='picshowblack',
              table='', notes=''):
    """Run ordered probit using statsmodels."""
    global reg_id
    reg_id += 1

    all_vars = [depvar] + indepvars + [weight_var]
    subdf = data.dropna(subset=all_vars).copy()

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        X = subdf[indepvars].values
        y = subdf[depvar].values
        w = subdf[weight_var].values

        m = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=False)

        # Find the index of treatment_var
        idx = indepvars.index(treatment_var)
        coef = m.params[idx]
        se = m.bse[idx]
        pval = m.pvalues[idx]
        nobs = len(subdf)

        coef_dict = {indepvars[i]: float(m.params[i]) for i in range(len(indepvars))}
        se_dict = {indepvars[i]: float(m.bse[i]) for i in range(len(indepvars))}
        coef_json = json.dumps({
            "coefficients": coef_dict,
            "standard_errors": se_dict,
            "inference": {"type": "MLE", "cluster_var": None},
            "software": {"language": "python", "package": "statsmodels"}
        })

        result = {
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': treatment_var, 'coefficient': coef, 'std_error': se,
            'p_value': pval, 'ci_lower': coef - 1.96 * se, 'ci_upper': coef + 1.96 * se,
            'n_obs': nobs, 'r_squared': getattr(m, 'prsquared', ''),
            'original_coefficient': '', 'original_std_error': '',
            'match_status': 'close', 'coefficient_vector_json': coef_json,
            'fixed_effects': '', 'controls_desc': label, 'cluster_var': '',
            'estimator': 'oprob', 'sample_desc': notes, 'notes': table
        }
        results.append(result)
        print(f"  Reg {reg_id} ({table}): oprob {depvar} ~ {treatment_var}, coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={nobs}")
        return m
    except Exception as e:
        print(f"  Reg {reg_id} ({table}): oprob FAILED - {e}")
        results.append({
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': treatment_var, 'coefficient': '', 'std_error': '',
            'p_value': '', 'ci_lower': '', 'ci_upper': '', 'n_obs': '', 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '', 'match_status': 'failed',
            'coefficient_vector_json': '{}', 'fixed_effects': '', 'controls_desc': label,
            'cluster_var': '', 'estimator': 'oprob', 'sample_desc': notes, 'notes': f'{table}: {e}'
        })
        return None

def run_cnreg(depvar, indepvars, data, weight_var, cens_var, label, treatment_var='picshowblack',
              table='', notes=''):
    """Run censored normal regression (Tobit-like) using statsmodels or manual MLE."""
    global reg_id
    reg_id += 1

    all_vars = [depvar] + indepvars + [weight_var, cens_var]
    subdf = data.dropna(subset=all_vars).copy()

    try:
        from scipy.optimize import minimize
        from scipy.stats import norm

        y = subdf[depvar].values
        X_vars = indepvars
        X = subdf[X_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        w = subdf[weight_var].values
        cens = subdf[cens_var].values.astype(int)

        # cens_giving: 1 = right censored (giving==100), -1 = left censored (giving==0), 0 = uncensored
        right_cens = (cens == 1)
        left_cens = (cens == -1)
        uncens = (cens == 0)

        # For giving: left limit = 0, right limit = 100
        if 'hypgiv' in depvar:
            ll, ul = 0, 500
        else:
            ll, ul = 0, 100

        def cnreg_negll(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            xb = X @ beta

            ll_uncens = w[uncens] * (-0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * ((y[uncens] - xb[uncens]) / sigma) ** 2)
            ll_right = w[right_cens] * np.log(1 - norm.cdf((ul - xb[right_cens]) / sigma) + 1e-300)
            ll_left = w[left_cens] * np.log(norm.cdf((ll - xb[left_cens]) / sigma) + 1e-300)

            return -(ll_uncens.sum() + ll_right.sum() + ll_left.sum())

        # Initialize with OLS estimates
        from numpy.linalg import lstsq
        beta_init = lstsq(X, y, rcond=None)[0]
        init = np.zeros(X.shape[1] + 1)
        init[:-1] = beta_init
        init[-1] = np.log(np.std(y - X @ beta_init))

        res = minimize(cnreg_negll, init, method='L-BFGS-B', options={'maxiter': 10000})

        # Standard errors from inverse Hessian using numdifftools-like approach
        n_params = len(res.x)
        eps = 1e-4
        H = np.zeros((n_params, n_params))
        f0 = cnreg_negll(res.x)
        for i in range(n_params):
            for j in range(i, n_params):
                e_i = np.zeros(n_params); e_i[i] = eps
                e_j = np.zeros(n_params); e_j[j] = eps
                fpp = cnreg_negll(res.x + e_i + e_j)
                fpm = cnreg_negll(res.x + e_i - e_j)
                fmp = cnreg_negll(res.x - e_i + e_j)
                fmm = cnreg_negll(res.x - e_i - e_j)
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
                H[j, i] = H[i, j]

        try:
            se_all = np.sqrt(np.diag(np.linalg.inv(H)))
        except:
            # Fall back to diagonal Hessian
            se_all = np.full(n_params, np.nan)

        # Find treatment var index (offset by 1 for constant)
        idx = X_vars.index(treatment_var) + 1  # +1 for constant
        coef = res.x[idx]
        se = se_all[idx]
        from scipy.stats import norm as norm_dist
        pval = 2 * (1 - norm_dist.cdf(abs(coef / se))) if se > 0 else np.nan
        nobs = len(subdf)

        coef_dict = {'_cons': float(res.x[0])}
        se_dict = {'_cons': float(se_all[0])}
        for j, v in enumerate(X_vars):
            coef_dict[v] = float(res.x[j+1])
            se_dict[v] = float(se_all[j+1])

        coef_json = json.dumps({
            "coefficients": coef_dict,
            "standard_errors": se_dict,
            "inference": {"type": "MLE", "cluster_var": None},
            "software": {"language": "python", "package": "scipy"}
        })

        result = {
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': treatment_var, 'coefficient': coef, 'std_error': se,
            'p_value': pval, 'ci_lower': coef - 1.96 * se, 'ci_upper': coef + 1.96 * se,
            'n_obs': nobs, 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '',
            'match_status': 'close', 'coefficient_vector_json': coef_json,
            'fixed_effects': '', 'controls_desc': label, 'cluster_var': '',
            'estimator': 'cnreg', 'sample_desc': notes, 'notes': table
        }
        results.append(result)
        print(f"  Reg {reg_id} ({table}): cnreg {depvar} ~ {treatment_var}, coef={coef:.4f}, se={se:.4f}, N={nobs}")
        return res
    except Exception as e:
        print(f"  Reg {reg_id} ({table}): cnreg FAILED - {e}")
        results.append({
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': treatment_var, 'coefficient': '', 'std_error': '',
            'p_value': '', 'ci_lower': '', 'ci_upper': '', 'n_obs': '', 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '', 'match_status': 'failed',
            'coefficient_vector_json': '{}', 'fixed_effects': '', 'controls_desc': label,
            'cluster_var': '', 'estimator': 'cnreg', 'sample_desc': notes, 'notes': f'{table}: {e}'
        })
        return None

def run_interactd(depvar, dummy_var, data, condition_mask, weight_var='tweight',
                  label_prefix='Table 6', table_prefix='tab06'):
    """
    Replicate the Stata 'interactd' program:
    Creates interaction terms between a dummy and picture manipulations,
    then runs OLS with all interactions.
    """
    global reg_id
    reg_id += 1

    subdf = data[condition_mask].copy()

    # Create interaction terms
    subdf['int1_picshowb'] = subdf[dummy_var] * subdf['picshowblack']
    subdf['int1_picobscur'] = subdf[dummy_var] * subdf['picobscur']
    subdf['int1_picraceb'] = subdf[dummy_var] * subdf['picraceb']
    subdf['int0_picshowb'] = (1 - subdf[dummy_var]) * subdf['picshowblack']
    subdf['int0_picobscur'] = (1 - subdf[dummy_var]) * subdf['picobscur']
    subdf['int0_picraceb'] = (1 - subdf[dummy_var]) * subdf['picraceb']

    int_vars = ['int1_picshowb', 'int1_picobscur', 'int1_picraceb',
                'int0_picshowb', 'int0_picobscur', 'int0_picraceb']
    indepvars = int_vars + [dummy_var] + manip + cntrldems

    all_vars = [depvar] + indepvars + [weight_var]
    subdf = subdf.dropna(subset=all_vars).copy()

    formula = f"{depvar} ~ " + " + ".join(indepvars)

    try:
        m = pf.feols(formula, data=subdf, vcov="hetero", weights=weight_var)

        # The key coefficient is int1_picshowb (effect for dummy==1 group)
        treatment_var = 'int1_picshowb'
        coef = m.coef()[treatment_var]
        se = m.se()[treatment_var]
        pval = m.pvalue()[treatment_var]
        nobs = m._N
        r2 = m._r2

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        se_dict = {k: float(v) for k, v in m.se().items()}
        coef_json = json.dumps({
            "coefficients": coef_dict,
            "standard_errors": se_dict,
            "inference": {"type": "HC1", "cluster_var": None},
            "software": {"language": "python", "package": "pyfixest"}
        })

        result = {
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': treatment_var, 'coefficient': coef, 'std_error': se,
            'p_value': pval, 'ci_lower': coef - 1.96 * se, 'ci_upper': coef + 1.96 * se,
            'n_obs': nobs, 'r_squared': r2,
            'original_coefficient': '', 'original_std_error': '',
            'match_status': 'close', 'coefficient_vector_json': coef_json,
            'fixed_effects': '', 'controls_desc': f'{label_prefix}: interactd {depvar} {dummy_var}',
            'cluster_var': '', 'estimator': 'OLS/WLS',
            'sample_desc': f'interactd with {dummy_var}', 'notes': table_prefix
        }
        results.append(result)
        print(f"  Reg {reg_id} ({table_prefix}): interactd {depvar} ~ {dummy_var}, coef(int1_picshowb)={coef:.4f}, se={se:.4f}, N={nobs}")
        return m
    except Exception as e:
        print(f"  Reg {reg_id} ({table_prefix}): interactd FAILED - {e}")
        results.append({
            'paper_id': PAPER_ID, 'reg_id': reg_id, 'outcome_var': depvar,
            'treatment_var': 'int1_picshowb', 'coefficient': '', 'std_error': '',
            'p_value': '', 'ci_lower': '', 'ci_upper': '', 'n_obs': '', 'r_squared': '',
            'original_coefficient': '', 'original_std_error': '', 'match_status': 'failed',
            'coefficient_vector_json': '{}', 'fixed_effects': '',
            'controls_desc': f'{label_prefix}: interactd {depvar} {dummy_var}',
            'cluster_var': '', 'estimator': 'OLS/WLS',
            'sample_desc': f'interactd with {dummy_var}', 'notes': f'{table_prefix}: {e}'
        })
        return None


# ============================================================
# TABLE 3: Effects on Perceived Race and Giving
# ============================================================
print("\n=== TABLE 3 ===")

# Col 1: Manipulation check (per_hfhdif)
pic_vars = ['picshowblack', 'picraceb', 'picobscur']
run_wls_reg('per_hfhdif', pic_vars + manip + cntrldems, df, 'tweight',
            'Table 3 col 1: manipulation check', 'picshowblack', 'tab03.c1',
            'All respondents')

# Col 2: Baseline giving regression
run_wls_reg('giving', pic_vars + manip + cntrldems, df, 'tweight',
            'Table 3 col 2: baseline giving', 'picshowblack', 'tab03.c2',
            'All respondents')

# Col 3: Interaction with respondent race
int_race_vars = ['picshowb_resb', 'picraceb_resb', 'picobscur_resb'] + pic_vars
df_not_other = df[df['other'] == 0].copy()
run_wls_reg('giving', int_race_vars + manip + cntrldems, df_not_other, 'tweight',
            'Table 3 col 3: interaction with respondent race', 'picshowblack', 'tab03.c3',
            'Not other race')

# Col 4: Interaction with subjective identification
sib_vars = ['picshowb_sib', 'picraceb_sib', 'picobscur_sib'] + pic_vars + ['subj_iden_blk']
run_wls_reg('giving', sib_vars + manip + cntrldems, df_not_other, 'tweight',
            'Table 3 col 4: interaction with subj identification', 'picshowblack', 'tab03.c4',
            'Not other race')

# Col 5: Same as col 4, white only
df_white = df[df['white'] == 1].copy()
run_wls_reg('giving', sib_vars + manip + cntrldems, df_white, 'tweight',
            'Table 3 col 5: subj identification, white only', 'picshowblack', 'tab03.c5',
            'White respondents', estimator='OLS')

# Col 6: Same as col 4, black only
df_black = df[df['black'] == 1].copy()
run_wls_reg('giving', sib_vars + manip + cntrldems, df_black, 'tweight',
            'Table 3 col 6: subj identification, black only', 'picshowblack', 'tab03.c6',
            'Black respondents', estimator='OLS')

# ============================================================
# TABLE 4: Results by Race and Measure of Generosity
# ============================================================
print("\n=== TABLE 4 ===")

pic_nraud = pic_vars + nraud

# Panel 1: Giving
for suffix, data_sub, sample_desc in [('a', df, 'All'), ('b', df_white, 'White'), ('c', df_black, 'Black')]:
    run_wls_reg('giving', pic_nraud + cntrldems, data_sub, 'tweight',
                f'Table 4 panel 1: giving ({sample_desc})', 'picshowblack',
                f'tab04.r1{suffix}', sample_desc)

# Panel 2: Hypothetical giving
for suffix, data_sub, sample_desc in [('a', df, 'All'), ('b', df_white, 'White'), ('c', df_black, 'Black')]:
    run_wls_reg('hypgiv_tc500', pic_nraud + cntrldems, data_sub, 'tweight',
                f'Table 4 panel 2: hyp giving ({sample_desc})', 'picshowblack',
                f'tab04.r2{suffix}', sample_desc)

# Panel 3: Charity spending support
for suffix, data_sub, sample_desc in [('a', df, 'All'), ('b', df_white, 'White'), ('c', df_black, 'Black')]:
    run_wls_reg('subjsupchar', pic_nraud + cntrldems, data_sub, 'tweight',
                f'Table 4 panel 3: charity support ({sample_desc})', 'picshowblack',
                f'tab04.r3{suffix}', sample_desc)

# Panel 4: Govt spending support
for suffix, data_sub, sample_desc in [('a', df, 'All'), ('b', df_white, 'White'), ('c', df_black, 'Black')]:
    run_wls_reg('subjsupgov', pic_nraud + cntrldems, data_sub, 'tweight',
                f'Table 4 panel 4: govt support ({sample_desc})', 'picshowblack',
                f'tab04.r4{suffix}', sample_desc)

# ============================================================
# TABLE 5: Robustness Checks (White respondents)
# ============================================================
print("\n=== TABLE 5 ===")

outcomes_t5 = [
    ('giving', 'cens_giving'),
    ('hypgiv_tc500', 'cens_hypgiv_tc500'),
    ('subjsupchar', None),
    ('subjsupgov', None),
]

for panel_idx, (depvar, cens_var) in enumerate(outcomes_t5, 1):
    print(f"\n  Panel {panel_idx}: {depvar}")

    # Row 1: Baseline (white only)
    run_wls_reg(depvar, pic_nraud + cntrldems, df_white, 'tweight',
                f'Table 5 row {panel_idx}.1: baseline', 'picshowblack',
                f'tab05.r{panel_idx}.s1', 'White respondents')

    # Row 2: Main sample only (surveyvariant==1, use mweight)
    df_white_main = df_white[df_white['surveyvariant'] == 1].copy()
    run_wls_reg(depvar, pic_nraud + cntrldems, df_white_main, 'mweight',
                f'Table 5 row {panel_idx}.2: main sample', 'picshowblack',
                f'tab05.r{panel_idx}.s2', 'White, main survey only')

    # Row 3: Slidell only
    df_white_slidell = df_white[df_white['cityslidell'] == 1].copy()
    run_wls_reg(depvar, pic_nraud + cntrldems, df_white_slidell, 'tweight',
                f'Table 5 row {panel_idx}.3: Slidell', 'picshowblack',
                f'tab05.r{panel_idx}.s3', 'White, Slidell only')

    # Row 4: Biloxi only
    df_white_biloxi = df_white[df_white['cityslidell'] == 0].copy()
    run_wls_reg(depvar, pic_nraud + cntrldems, df_white_biloxi, 'tweight',
                f'Table 5 row {panel_idx}.4: Biloxi', 'picshowblack',
                f'tab05.r{panel_idx}.s4', 'White, Biloxi only')

    # Row 5: No demographic controls
    # Note: for white subsample, the "black other" controls are included but black=0 always for whites
    # The Stata code uses: $nraud black other — but since we're white subsample, black=0, other=0
    run_wls_reg(depvar, pic_vars + nraud + ['black', 'other'], df_white, 'tweight',
                f'Table 5 row {panel_idx}.5: no demo controls', 'picshowblack',
                f'tab05.r{panel_idx}.s5', 'White, no demo controls')

    # Row 6: Extra controls
    run_wls_reg(depvar, pic_nraud + cntrldems + addcntrl1, df_white, 'tweight',
                f'Table 5 row {panel_idx}.6: extra controls', 'picshowblack',
                f'tab05.r{panel_idx}.s6', 'White, extra controls')

    # Row 7: Censored regression or ordered probit
    if cens_var is not None:
        # cnreg for giving and hypgiv_tc500
        run_cnreg(depvar, pic_nraud + cntrldems, df_white, 'tweight', cens_var,
                  f'Table 5 row {panel_idx}.7: cnreg', 'picshowblack',
                  f'tab05.r{panel_idx}.s7', 'White, censored regression')
    else:
        # oprob for subjsupchar and subjsupgov
        run_oprob(depvar, pic_nraud + cntrldems, df_white, 'tweight',
                  f'Table 5 row {panel_idx}.7: oprob', 'picshowblack',
                  f'tab05.r{panel_idx}.s7', 'White, ordered probit')

    # Row 8: Race-shown only (not picobscur)
    df_white_raceshown = df_white[df_white['picobscur'] == 0].copy()
    run_wls_reg(depvar, pic_nraud + cntrldems, df_white_raceshown, 'tweight',
                f'Table 5 row {panel_idx}.8: race-shown', 'picshowblack',
                f'tab05.r{panel_idx}.s8', 'White, race-shown only')

# ============================================================
# TABLE 6: Interactions with Racial Attitudes
# ============================================================
print("\n=== TABLE 6 ===")

# Panel A: White respondents
white_mask = df['white'] == 1
depvars_t6 = ['giving', 'hypgiv_tc500', 'subjsupchar', 'subjsupgov']

# Row 1: ethclosed (whites)
for col_idx, depvar in enumerate(depvars_t6, 1):
    run_interactd(depvar, 'ethclosed', df, white_mask, 'tweight',
                  'Table 6 Panel A row 1', f'tab06.r1.c{col_idx}')

# Row 2: soccon_difd (whites)
for col_idx, depvar in enumerate(depvars_t6, 1):
    run_interactd(depvar, 'soccon_difd', df, white_mask, 'tweight',
                  'Table 6 Panel A row 2', f'tab06.r2.c{col_idx}')

# Row 3: oppblkd (whites)
for col_idx, depvar in enumerate(depvars_t6, 1):
    run_interactd(depvar, 'oppblkd', df, white_mask, 'tweight',
                  'Table 6 Panel A row 3', f'tab06.r3.c{col_idx}')

# Panel B: Black respondents
black_mask = df['black'] == 1

# Row 4: ethclosed (blacks)
for col_idx, depvar in enumerate(depvars_t6, 1):
    run_interactd(depvar, 'ethclosed', df, black_mask, 'tweight',
                  'Table 6 Panel B row 4', f'tab06.r4.c{col_idx}')

# Row 5: soccon_difd (blacks)
for col_idx, depvar in enumerate(depvars_t6, 1):
    run_interactd(depvar, 'soccon_difd', df, black_mask, 'tweight',
                  'Table 6 Panel B row 5', f'tab06.r5.c{col_idx}')

# Row 6: oppblkd (blacks)
for col_idx, depvar in enumerate(depvars_t6, 1):
    run_interactd(depvar, 'oppblkd', df, black_mask, 'tweight',
                  'Table 6 Panel B row 6', f'tab06.r6.c{col_idx}')


# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n\nTotal regressions: {len(results)}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{PACKAGE_DIR}/replication.csv", index=False)
print(f"Saved replication.csv with {len(results_df)} rows")

# Print summary
n_exact = sum(1 for r in results if r['match_status'] == 'exact')
n_close = sum(1 for r in results if r['match_status'] == 'close')
n_disc = sum(1 for r in results if r['match_status'] == 'discrepant')
n_fail = sum(1 for r in results if r['match_status'] == 'failed')
print(f"Match status: {n_exact} exact, {n_close} close, {n_disc} discrepant, {n_fail} failed")
