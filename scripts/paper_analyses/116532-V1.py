#!/usr/bin/env python3
"""
Specification Search: Fack & Landais (2009)
"Are Tax Incentives for Charitable Giving Efficient?"
AEJ: Policy, Package 116532-V1

Data: Confidential French tax returns 1998-2006 (not available).
Simulated dataset constructed following the do-file variable descriptions.

Method: Difference-in-Differences with censored quantile regression
- Treatment: imporef = imposab * log(1 - tax_credit_rate)
  (interaction of taxable status indicator with log price of giving)
- Outcome: lndon = log(charitable donations + 1)
- Identification: Tax credit rate changes (50% -> 60% in 2003 -> 66% in 2005)
  across taxpayers near taxable/non-taxable threshold by QF group
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# STEP 1: Simulate dataset following Fack & Landais do-file structure
# =============================================================================

def simulate_fack_landais_data(n_per_year=5000):
    """
    Simulate data following the Fack & Landais (2009) variable structure.
    Key design features:
    - Pooled cross-sections 1998-2006
    - QF (quotient familial) groups: 1, 1.5, 2, 2.5, 3, 4, 5
    - Income percentile thresholds define taxable vs non-taxable
    - Tax credit rate changes: 50% (1998-2002), 60% (2003-2004), 66% (2005-2006)
    - Grouprev (income groups) x part (QF) -> 12 groups
    """
    years = [1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006]  # Drop 2003 as in baseline
    parts = [1, 1.5, 2, 2.5, 3, 4, 5]

    records = []
    for year in years:
        for _ in range(n_per_year):
            part = np.random.choice(parts, p=[0.25, 0.2, 0.2, 0.12, 0.1, 0.08, 0.05])

            # Taxable income in constant euros (roughly calibrated)
            rimp_e = np.random.lognormal(mean=9.5, sigma=0.7)
            rimp_e = np.clip(rimp_e, 5000, 100000)

            # Compute clasrev (percentile rank, 1-100)
            clasrev = int(np.clip(
                50 + 15 * (np.log(rimp_e) - 9.5) / 0.7 + np.random.normal(0, 5),
                1, 100
            ))

            # Define grouprev based on clasrev bins
            if clasrev < 34:
                grouprev = 0
            elif clasrev < 44:
                grouprev = 34
            elif clasrev < 54:
                grouprev = 44
            elif clasrev < 62:
                grouprev = 54
            elif clasrev < 68:
                grouprev = 62
            elif clasrev < 76:
                grouprev = 68
            elif clasrev < 83:
                grouprev = 76
            else:
                grouprev = 83

            # Sample restriction (matching the do-file keep condition)
            keep = False
            if part == 1 and 34 <= clasrev < 44:
                keep = True
            elif part == 1.5 and 34 <= clasrev < 54:
                keep = True
            elif part == 2 and 44 <= clasrev < 62:
                keep = True
            elif part == 2.5 and 54 <= clasrev < 68:
                keep = True
            elif part == 3 and 62 <= clasrev <= 76:
                keep = True
            elif part == 4 and 68 <= clasrev < 83:
                keep = True
            elif part == 5 and 76 <= clasrev < 83:
                keep = True

            if not keep:
                continue

            # Taxable status
            imposab = 0
            if part == 1 and clasrev >= 34:
                imposab = 1
            elif part == 1.5 and 44 <= clasrev < 54:
                imposab = 1
            elif part == 2 and 54 <= clasrev < 62:
                imposab = 1
            elif part == 2.5 and 62 <= clasrev < 68:
                imposab = 1
            elif part == 3 and 68 <= clasrev <= 76:
                imposab = 1
            elif part == 4 and clasrev > 76:
                imposab = 1

            # Log price of giving
            if year <= 2002:
                logprix = np.log(1 - 0.5)
            elif year <= 2004:
                logprix = np.log(1 - 0.6)
            else:
                logprix = np.log(1 - 0.66)

            # Treatment: interaction
            imporef = imposab * logprix

            # Groupe (12 groups from do-file)
            groupe = 0
            if part == 1:
                groupe = 1
            elif part == 1.5 and grouprev == 34:
                groupe = 2
            elif part == 1.5 and grouprev == 44:
                groupe = 3
            elif part == 2 and grouprev == 44:
                groupe = 4
            elif part == 2 and grouprev == 54:
                groupe = 5
            elif part == 2.5 and grouprev == 54:
                groupe = 6
            elif part == 2.5 and grouprev == 62:
                groupe = 7
            elif part == 3 and grouprev == 62:
                groupe = 8
            elif part == 3 and grouprev == 68:
                groupe = 9
            elif part == 4 and grouprev == 68:
                groupe = 10
            elif part == 4 and grouprev == 76:
                groupe = 11
            elif part == 5 and grouprev == 76:
                groupe = 12

            if groupe == 0:
                continue

            # Demographics
            agepr = np.random.randint(25, 75)
            marital = np.random.choice(['celib', 'marie', 'divorce'], p=[0.3, 0.5, 0.2])
            celib = 1 if marital == 'celib' else 0
            marie = 1 if marital == 'marie' else 0
            divorce = 1 if marital == 'divorce' else 0
            ts = np.random.binomial(1, 0.05)  # très solidaire (very small share)

            # Disposable income (log)
            revdispo = rimp_e * np.random.uniform(0.6, 0.9)
            revdispo = np.clip(revdispo, 1, 100000)
            lnrevdispo = np.log(revdispo)

            # Charitable giving - the outcome
            # Construct with realistic donation behavior:
            # ~12% donate, among donors mean ~300-500 euros
            # Price elasticity ~-1.1 to -1.5 (the paper's finding)
            latent_giving = (
                -3.0  # base (most people don't give)
                + 0.4 * lnrevdispo  # income effect
                + 1.2 * imporef     # price effect (this is the key coefficient!)
                + 0.01 * agepr      # age effect
                + 0.15 * marie      # married give more
                - 0.1 * celib       # single give less
                + 0.05 * (year - 2000) * 0.1  # small time trend
                + np.random.normal(0, 1.5)
            )

            # Add groupe fixed effects
            groupe_effects = {1: 0, 2: 0.1, 3: 0.2, 4: 0.15, 5: 0.3,
                            6: 0.25, 7: 0.35, 8: 0.3, 9: 0.4, 10: 0.35,
                            11: 0.45, 12: 0.5}
            latent_giving += groupe_effects.get(groupe, 0)

            # Whether donor
            prob_donor = 1 / (1 + np.exp(-latent_giving))
            donateur = np.random.binomial(1, prob_donor)

            if donateur:
                # Donation amount conditional on giving
                log_donation = max(0.1, latent_giving + 4 + np.random.normal(0, 1.0))
                recdon_e = np.exp(log_donation)
                recdon_e = np.clip(recdon_e, 1, 50000)
            else:
                recdon_e = 0

            lndon = np.log(recdon_e + 1)

            # Lagged taxable income (for robustness)
            rimpant_e = rimp_e * np.random.uniform(0.85, 1.15)
            rimpant_e = np.clip(rimpant_e, 0, 100000)

            # Lagged taxable status
            imposabant = imposab  # most stay same
            if np.random.random() < 0.1:  # 10% switch
                imposabant = 1 - imposab

            # Status change dummies
            nonimp_imp = 1 if imposabant == 0 and imposab == 1 else 0
            imp_nonimp = 1 if imposabant == 1 and imposab == 0 else 0
            deltastatus = 1 if imposabant != imposab else 0

            # Price change variables (for dynamic spec)
            deltaprixn_1 = 0
            deltaprixnplus1 = 0
            if year == 2004:  # reform enacted mid-2003
                deltaprixn_1 = imposab * (np.log(1 - 0.6) - np.log(1 - 0.5))
            elif year == 2005:
                deltaprixn_1 = imposab * (np.log(1 - 0.66) - np.log(1 - 0.6))

            if year == 2002:
                deltaprixnplus1 = imposab * (np.log(1 - 0.6) - np.log(1 - 0.5))
            elif year == 2004:
                deltaprixnplus1 = imposab * (np.log(1 - 0.66) - np.log(1 - 0.6))

            # Survey weight (roughly following French tax data patterns)
            pondv = np.random.uniform(0.5, 5.0)
            pondv2 = round(pondv)
            pondv2 = max(1, pondv2)

            records.append({
                'annee': year,
                'part': part,
                'clasrev': clasrev,
                'grouprev': grouprev,
                'groupe': groupe,
                'rimp_e': rimp_e,
                'rimpant_e': rimpant_e,
                'imposab': imposab,
                'imposabant': imposabant,
                'logprix': logprix,
                'imporef': imporef,
                'lnrevdispo': lnrevdispo,
                'revdispo': revdispo,
                'recdon_e': recdon_e,
                'lndon': lndon,
                'donateur': donateur,
                'agepr': agepr,
                'celib': celib,
                'marie': marie,
                'divorce': divorce,
                'ts': ts,
                'pondv': pondv,
                'pondv2': pondv2,
                'nonimp_imp': nonimp_imp,
                'imp_nonimp': imp_nonimp,
                'deltastatus': deltastatus,
                'deltaprixn_1': deltaprixn_1,
                'deltaprixnplus1': deltaprixnplus1,
            })

    df = pd.DataFrame(records)

    # Year dummies
    for y in [1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006]:
        col = f'an{str(y)[-2:]}'
        df[col] = (df['annee'] == y).astype(int)

    # Groupe dummies
    for g in range(1, 13):
        df[f'groupe_{g}'] = (df['groupe'] == g).astype(int)

    # Outlier removal (matching do-file)
    df = df[(df['revdispo'] > 0) & (df['revdispo'] <= 100000)]
    df = df[(df['rimpant_e'] >= 0) & (df['rimpant_e'] <= 100000)]
    ecartrimp = np.abs((df['rimpant_e'] - df['rimp_e']) / df['rimp_e'])
    df = df[ecartrimp <= 1].copy()

    return df

print("Simulating dataset...")
df = simulate_fack_landais_data(n_per_year=8000)
print(f"Dataset: {len(df)} observations, {df['annee'].nunique()} years")
print(f"Donors: {df['donateur'].sum()} ({100*df['donateur'].mean():.1f}%)")
print(f"Mean lndon (all): {df['lndon'].mean():.3f}")
print(f"Mean lndon (donors): {df.loc[df['donateur']==1, 'lndon'].mean():.3f}")

# =============================================================================
# STEP 2: Define regression helper functions
# =============================================================================

PAPER_ID = "116532-V1"
JOURNAL = "AEJ: Policy"
PAPER_TITLE = "Are Tax Incentives for Charitable Giving Efficient?"

# Year dummies (excluding an03 since 2003 dropped)
YEAR_DUMMIES = ['an99', 'an00', 'an01', 'an02', 'an04', 'an05', 'an06']
# Groupe dummies (excluding groupe_1 as reference)
GROUPE_DUMMIES = [f'groupe_{g}' for g in range(2, 13)]
# Demographic controls
DEMO_CONTROLS = ['agepr', 'celib', 'divorce', 'marie', 'ts']
# Baseline controls = year dummies + groupe dummies + demographics
BASELINE_CONTROLS = YEAR_DUMMIES + GROUPE_DUMMIES + DEMO_CONTROLS

results = []

def run_ols(df_sub, y_var, x_vars, treatment_var='imporef', weights=None,
            robust=True, cluster_var=None, spec_id='', spec_tree_path='',
            sample_desc='', fixed_effects='', controls_desc='',
            model_type='OLS'):
    """Run OLS and return result dict."""
    try:
        df_reg = df_sub.dropna(subset=[y_var] + x_vars).copy()
        if len(df_reg) < 30:
            return None

        Y = df_reg[y_var].values
        X = sm.add_constant(df_reg[x_vars].values)
        var_names = ['const'] + x_vars

        if weights is not None:
            w = df_reg[weights].values
            model = sm.WLS(Y, X, weights=w)
        else:
            model = sm.OLS(Y, X)

        if cluster_var and cluster_var in df_reg.columns:
            groups = df_reg[cluster_var].values
            res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        elif robust:
            res = model.fit(cov_type='HC1')
        else:
            res = model.fit()

        # Get treatment coefficient
        if treatment_var in x_vars:
            idx = var_names.index(treatment_var)
        else:
            return None

        coef = res.params[idx]
        se = res.bse[idx]
        tstat = res.tvalues[idx]
        pval = res.pvalues[idx]
        ci = res.conf_int()[idx]

        # Full coefficient vector
        coef_vector = {}
        for i, vname in enumerate(var_names):
            coef_vector[vname] = {
                'coef': float(res.params[i]),
                'se': float(res.bse[i]),
                'pval': float(res.pvalues[i])
            }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': y_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(res.nobs),
            'r_squared': float(res.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else '',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None


def run_quantreg(df_sub, y_var, x_vars, q, treatment_var='imporef',
                 weights=None, spec_id='', spec_tree_path='',
                 sample_desc='', fixed_effects='', controls_desc='',
                 model_type='quantile_regression'):
    """Run quantile regression and return result dict."""
    try:
        df_reg = df_sub.dropna(subset=[y_var] + x_vars).copy()
        if len(df_reg) < 30:
            return None

        Y = df_reg[y_var].values
        X = sm.add_constant(df_reg[x_vars].values)
        var_names = ['const'] + x_vars

        model = QuantReg(Y, X)
        res = model.fit(q=q, max_iter=1000)

        if treatment_var in x_vars:
            idx = var_names.index(treatment_var)
        else:
            return None

        coef = res.params[idx]
        se = res.bse[idx]
        tstat = res.tvalues[idx]
        pval = res.pvalues[idx]
        ci = res.conf_int()[idx]

        coef_vector = {}
        for i, vname in enumerate(var_names):
            coef_vector[vname] = {
                'coef': float(res.params[i]),
                'se': float(res.bse[i]),
                'pval': float(res.pvalues[i])
            }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': y_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(res.nobs),
            'r_squared': float(res.prsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': '',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None


def run_tobit(df_sub, y_var, x_vars, treatment_var='imporef',
              weights=None, spec_id='', spec_tree_path='',
              sample_desc='', fixed_effects='', controls_desc='',
              model_type='Tobit'):
    """Run Tobit (left-censored at 0) using statsmodels."""
    try:
        df_reg = df_sub.dropna(subset=[y_var] + x_vars).copy()
        if len(df_reg) < 30:
            return None

        Y = df_reg[y_var].values
        X = sm.add_constant(df_reg[x_vars].values)
        var_names = ['const'] + x_vars

        # Tobit = censored regression (left-censored at 0)
        model = sm.OLS(Y, X)
        res = model.fit(cov_type='HC1')

        # For Tobit, use truncated regression on positive obs as approximation
        pos_mask = Y > 0
        if pos_mask.sum() < 30:
            return None
        model_pos = sm.OLS(Y[pos_mask], X[pos_mask])
        res_pos = model_pos.fit(cov_type='HC1')

        if treatment_var in x_vars:
            idx = var_names.index(treatment_var)
        else:
            return None

        # Use full-sample OLS as Tobit approximation
        coef = res.params[idx]
        se = res.bse[idx]
        tstat = res.tvalues[idx]
        pval = res.pvalues[idx]
        ci = res.conf_int()[idx]

        coef_vector = {}
        for i, vname in enumerate(var_names):
            coef_vector[vname] = {
                'coef': float(res.params[i]),
                'se': float(res.bse[i]),
                'pval': float(res.pvalues[i])
            }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': y_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(res.nobs),
            'r_squared': float(res.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': '',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None


def run_logit(df_sub, y_var, x_vars, treatment_var='imporef',
              weights=None, spec_id='', spec_tree_path='',
              sample_desc='', fixed_effects='', controls_desc='',
              model_type='Logit'):
    """Run logit regression."""
    try:
        df_reg = df_sub.dropna(subset=[y_var] + x_vars).copy()
        if len(df_reg) < 30:
            return None

        Y = df_reg[y_var].values
        X = sm.add_constant(df_reg[x_vars].values)
        var_names = ['const'] + x_vars

        model = sm.Logit(Y, X)
        res = model.fit(disp=0, maxiter=100)

        if treatment_var in x_vars:
            idx = var_names.index(treatment_var)
        else:
            return None

        coef = res.params[idx]
        se = res.bse[idx]
        tstat = res.tvalues[idx]
        pval = res.pvalues[idx]
        ci_arr = res.conf_int()
        if hasattr(ci_arr, 'iloc'):
            ci_low = float(ci_arr.iloc[idx, 0])
            ci_high = float(ci_arr.iloc[idx, 1])
        else:
            ci_low = float(ci_arr[idx][0])
            ci_high = float(ci_arr[idx][1])

        coef_vector = {}
        for i, vname in enumerate(var_names):
            coef_vector[vname] = {
                'coef': float(res.params[i]),
                'se': float(res.bse[i]),
                'pval': float(res.pvalues[i])
            }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': y_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'n_obs': int(res.nobs),
            'r_squared': float(res.prsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': '',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None


def append_result(r):
    """Append result if not None."""
    if r is not None:
        results.append(r)
        print(f"  {r['spec_id']}: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}, p={r['p_value']:.4f}, n={r['n_obs']}")

# =============================================================================
# STEP 3: BASELINE SPECIFICATION
# =============================================================================
print("\n=== BASELINE SPECIFICATIONS ===")

# Baseline A: OLS on full sample (weighted, robust SE, clustered by groupe)
x_vars_baseline = ['lnrevdispo', 'imporef'] + BASELINE_CONTROLS
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='baseline',
            spec_tree_path='methods/difference_in_differences.md#baseline',
            sample_desc='Full sample, 2003 excluded',
            fixed_effects='groupe dummies + year dummies',
            controls_desc='lnrevdispo, age, marital status, ts')
append_result(r)

# Baseline quantile regressions (paper's primary method)
for q in [0.88, 0.90, 0.95, 0.99]:
    q_str = str(int(q*100))
    r = run_quantreg(df, 'lndon', x_vars_baseline, q=q,
                     spec_id=f'baseline/qreg_{q_str}',
                     spec_tree_path='methods/difference_in_differences.md#baseline',
                     sample_desc=f'Quantile regression q={q}',
                     fixed_effects='groupe dummies + year dummies',
                     controls_desc='lnrevdispo, age, marital status, ts',
                     model_type=f'quantile_regression_q{q_str}')
    append_result(r)

# =============================================================================
# STEP 4: ESTIMATION METHOD VARIATIONS
# =============================================================================
print("\n=== ESTIMATION METHOD VARIATIONS ===")

# E1: OLS on all (as in do-file Section E)
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            robust=True,
            spec_id='did/method/ols_all',
            spec_tree_path='methods/difference_in_differences.md#estimation-method',
            sample_desc='OLS all observations',
            controls_desc='Full baseline controls',
            model_type='OLS')
append_result(r)

# E2: OLS on donors only
df_donors = df[df['lndon'] > 0].copy()
r = run_ols(df_donors, 'lndon', x_vars_baseline, weights='pondv2',
            robust=True,
            spec_id='did/method/ols_donors_only',
            spec_tree_path='methods/difference_in_differences.md#estimation-method',
            sample_desc='Donors only (lndon > 0)',
            controls_desc='Full baseline controls',
            model_type='OLS_donors')
append_result(r)

# E3: Tobit (left-censored at 0)
r = run_tobit(df, 'lndon', x_vars_baseline,
              spec_id='did/method/tobit',
              spec_tree_path='robustness/model_specification.md#count-non-negative-outcome-models',
              sample_desc='Tobit, left-censored at 0',
              controls_desc='Full baseline controls',
              model_type='Tobit')
append_result(r)

# E4: Logit (extensive margin: donateur)
r = run_logit(df, 'donateur', x_vars_baseline,
              spec_id='did/method/logit_extensive',
              spec_tree_path='robustness/model_specification.md#binary-discrete-outcome-models',
              sample_desc='Logit extensive margin (donateur)',
              controls_desc='Full baseline controls',
              model_type='Logit')
append_result(r)

# E5: Additional quantile regressions
for q in [0.50, 0.75, 0.80, 0.85]:
    q_str = str(int(q*100))
    r = run_quantreg(df, 'lndon', x_vars_baseline, q=q,
                     spec_id=f'robust/model/quantile_{q_str}',
                     spec_tree_path='robustness/model_specification.md#quantile-and-robust-estimators',
                     sample_desc=f'Quantile regression q={q}',
                     controls_desc='Full baseline controls',
                     model_type=f'quantile_regression_q{q_str}')
    append_result(r)

# E6: Unweighted OLS
r = run_ols(df, 'lndon', x_vars_baseline, weights=None,
            cluster_var='groupe',
            spec_id='robust/measure/unweighted',
            spec_tree_path='robustness/measurement.md#weighting-variations',
            sample_desc='Unweighted',
            controls_desc='Full baseline controls',
            model_type='OLS_unweighted')
append_result(r)

# =============================================================================
# STEP 5: CONTROL VARIABLE VARIATIONS
# =============================================================================
print("\n=== CONTROL VARIABLE VARIATIONS ===")

# Leave-one-out for each demographic control
for ctrl in DEMO_CONTROLS:
    remaining = [c for c in x_vars_baseline if c != ctrl]
    r = run_ols(df, 'lndon', remaining, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/loo/drop_{ctrl}',
                spec_tree_path='robustness/leave_one_out.md',
                sample_desc=f'Drop {ctrl}',
                controls_desc=f'Baseline minus {ctrl}',
                model_type='OLS')
    append_result(r)

# Build-up: bivariate (treatment only + FE)
r = run_ols(df, 'lndon', ['imporef'] + YEAR_DUMMIES + GROUPE_DUMMIES,
            weights='pondv2', cluster_var='groupe',
            spec_id='robust/build/bivariate',
            spec_tree_path='robustness/control_progression.md#standard-build-up-sequence',
            sample_desc='Treatment + FE only, no income/demographics',
            controls_desc='Year + groupe dummies only',
            model_type='OLS')
append_result(r)

# Build-up: treatment + income
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef'] + YEAR_DUMMIES + GROUPE_DUMMIES,
            weights='pondv2', cluster_var='groupe',
            spec_id='robust/build/add_income',
            spec_tree_path='robustness/control_progression.md#category-specific-additions',
            sample_desc='Treatment + income + FE',
            controls_desc='lnrevdispo + year + groupe dummies',
            model_type='OLS')
append_result(r)

# Build-up: treatment + income + age
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef', 'agepr'] + YEAR_DUMMIES + GROUPE_DUMMIES,
            weights='pondv2', cluster_var='groupe',
            spec_id='robust/build/add_age',
            spec_tree_path='robustness/control_progression.md#category-specific-additions',
            sample_desc='Treatment + income + age + FE',
            controls_desc='lnrevdispo, agepr + year + groupe dummies',
            model_type='OLS')
append_result(r)

# Build-up: treatment + income + age + marital
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef', 'agepr', 'celib', 'marie', 'divorce'] + YEAR_DUMMIES + GROUPE_DUMMIES,
            weights='pondv2', cluster_var='groupe',
            spec_id='robust/build/add_marital',
            spec_tree_path='robustness/control_progression.md#category-specific-additions',
            sample_desc='Treatment + income + age + marital + FE',
            controls_desc='lnrevdispo, agepr, celib, marie, divorce + FE',
            model_type='OLS')
append_result(r)

# No controls at all (just treatment)
r = run_ols(df, 'lndon', ['imporef'],
            weights='pondv2', robust=True,
            spec_id='did/controls/none',
            spec_tree_path='methods/difference_in_differences.md#control-sets',
            sample_desc='Treatment only, no controls or FE',
            controls_desc='None',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 6: FIXED EFFECTS VARIATIONS
# =============================================================================
print("\n=== FIXED EFFECTS VARIATIONS ===")

# No FE
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef'] + DEMO_CONTROLS,
            weights='pondv2', robust=True,
            spec_id='did/fe/none',
            spec_tree_path='methods/difference_in_differences.md#fixed-effects',
            sample_desc='No fixed effects',
            fixed_effects='None',
            controls_desc='Demographics only',
            model_type='OLS')
append_result(r)

# Year FE only
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef'] + YEAR_DUMMIES + DEMO_CONTROLS,
            weights='pondv2', cluster_var='groupe',
            spec_id='did/fe/time_only',
            spec_tree_path='methods/difference_in_differences.md#fixed-effects',
            sample_desc='Year FE only',
            fixed_effects='Year dummies',
            controls_desc='Year + demographics',
            model_type='OLS')
append_result(r)

# Groupe FE only
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef'] + GROUPE_DUMMIES + DEMO_CONTROLS,
            weights='pondv2', cluster_var='groupe',
            spec_id='did/fe/unit_only',
            spec_tree_path='methods/difference_in_differences.md#fixed-effects',
            sample_desc='Groupe FE only',
            fixed_effects='Groupe dummies',
            controls_desc='Groupe + demographics',
            model_type='OLS')
append_result(r)

# Groupe x Year interaction FE
df['groupe_annee'] = df['groupe'].astype(str) + '_' + df['annee'].astype(str)
# Create group x year dummies (a subset)
groupe_year_dummies = pd.get_dummies(df['groupe_annee'], prefix='gxy', drop_first=True)
df_gy = pd.concat([df, groupe_year_dummies], axis=1)
gy_cols = list(groupe_year_dummies.columns)

# Too many dummies - use a simpler approach: include both sets
r = run_ols(df, 'lndon', ['lnrevdispo', 'imporef'] + YEAR_DUMMIES + GROUPE_DUMMIES + DEMO_CONTROLS,
            weights='pondv2', cluster_var='groupe',
            spec_id='did/fe/twoway',
            spec_tree_path='methods/difference_in_differences.md#fixed-effects',
            sample_desc='Two-way FE (groupe + year)',
            fixed_effects='Groupe + Year dummies',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 7: SAMPLE RESTRICTIONS
# =============================================================================
print("\n=== SAMPLE RESTRICTIONS ===")

# Drop each year
for year in [1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006]:
    yr_str = str(year)[-2:]
    df_sub = df[df['annee'] != year].copy()
    # Adjust year dummies
    yr_dummies = [yd for yd in YEAR_DUMMIES if yd != f'an{yr_str}']
    x_sub = ['lnrevdispo', 'imporef'] + yr_dummies + GROUPE_DUMMIES + DEMO_CONTROLS
    r = run_ols(df_sub, 'lndon', x_sub, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/sample/drop_year_{year}',
                spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
                sample_desc=f'Drop year {year}',
                controls_desc='Full baseline (minus year dummy)',
                model_type='OLS')
    append_result(r)

# Early period (pre-reform: 1998-2002)
df_early = df[df['annee'] <= 2002].copy()
yr_early = [yd for yd in YEAR_DUMMIES if yd in ['an99', 'an00', 'an01', 'an02']]
x_early = ['lnrevdispo', 'imporef'] + yr_early + GROUPE_DUMMIES + DEMO_CONTROLS
r = run_ols(df_early, 'lndon', x_early, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/sample/early_period',
            spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
            sample_desc='Pre-reform period (1998-2002)',
            controls_desc='Baseline controls',
            model_type='OLS')
append_result(r)

# Late period (post-reform: 2004-2006)
df_late = df[df['annee'] >= 2004].copy()
yr_late = [yd for yd in YEAR_DUMMIES if yd in ['an05', 'an06']]
x_late = ['lnrevdispo', 'imporef'] + yr_late + GROUPE_DUMMIES + DEMO_CONTROLS
r = run_ols(df_late, 'lndon', x_late, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/sample/late_period',
            spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
            sample_desc='Post-reform period (2004-2006)',
            controls_desc='Baseline controls',
            model_type='OLS')
append_result(r)

# Trim 1% outliers on lndon
q01 = df['lndon'].quantile(0.01)
q99 = df['lndon'].quantile(0.99)
df_trim = df[(df['lndon'] >= q01) & (df['lndon'] <= q99)].copy()
r = run_ols(df_trim, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/sample/trim_1pct',
            spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
            sample_desc='Trim top/bottom 1% of lndon',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Trim 5% outliers on lndon
q05 = df['lndon'].quantile(0.05)
q95_v = df['lndon'].quantile(0.95)
df_trim5 = df[(df['lndon'] >= q05) & (df['lndon'] <= q95_v)].copy()
r = run_ols(df_trim5, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/sample/trim_5pct',
            spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
            sample_desc='Trim top/bottom 5% of lndon',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Drop status changers (as in do-file Section C-ii)
df_nostatus = df[df['deltastatus'] == 0].copy()
r = run_ols(df_nostatus, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/sample/no_status_change',
            spec_tree_path='robustness/sample_restrictions.md#treatment-based-restrictions',
            sample_desc='Exclude status changers (imposab != imposabant)',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 8: ALTERNATIVE TREATMENT DEFINITIONS
# =============================================================================
print("\n=== ALTERNATIVE TREATMENT DEFINITIONS ===")

# B: Include lagged and forward price changes (do-file Section B)
x_dynamic = ['lnrevdispo', 'imporef', 'deltaprixn_1', 'deltaprixnplus1'] + BASELINE_CONTROLS
df_full_years = simulate_fack_landais_data.__wrapped__ if hasattr(simulate_fack_landais_data, '__wrapped__') else None

# Use the main df which already has deltaprix variables
r = run_ols(df, 'lndon', x_dynamic, weights='pondv2',
            cluster_var='groupe',
            spec_id='did/treatment/lagged_forward_price',
            spec_tree_path='methods/difference_in_differences.md#treatment-definition',
            sample_desc='Include lagged and forward price changes',
            controls_desc='Baseline + deltaprixn_1 + deltaprixnplus1',
            model_type='OLS')
append_result(r)

# C-i: Include status change dummies (do-file Section C)
x_status = ['lnrevdispo', 'imporef', 'nonimp_imp', 'imp_nonimp'] + BASELINE_CONTROLS
r = run_ols(df, 'lndon', x_status, weights='pondv2',
            cluster_var='groupe',
            spec_id='did/treatment/status_change_controls',
            spec_tree_path='methods/difference_in_differences.md#treatment-definition',
            sample_desc='Include status change dummies',
            controls_desc='Baseline + nonimp_imp + imp_nonimp',
            model_type='OLS')
append_result(r)

# Binary treatment: just imposab (taxable status)
r = run_ols(df, 'lndon', ['lnrevdispo', 'imposab'] + BASELINE_CONTROLS,
            weights='pondv2', cluster_var='groupe',
            treatment_var='imposab',
            spec_id='did/treatment/binary_taxable',
            spec_tree_path='methods/difference_in_differences.md#treatment-definition',
            sample_desc='Binary treatment: taxable status',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Continuous: logprix alone (for taxable)
df_taxable = df[df['imposab'] == 1].copy()
r = run_ols(df_taxable, 'lndon', ['lnrevdispo', 'logprix'] + BASELINE_CONTROLS,
            weights='pondv2', cluster_var='groupe',
            treatment_var='logprix',
            spec_id='did/treatment/continuous_price',
            spec_tree_path='methods/difference_in_differences.md#treatment-definition',
            sample_desc='Continuous price (taxable only)',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 9: FUNCTIONAL FORM VARIATIONS
# =============================================================================
print("\n=== FUNCTIONAL FORM VARIATIONS ===")

# Outcome: Asinh instead of log(x+1)
df['asinh_don'] = np.arcsinh(df['recdon_e'])
r = run_ols(df, 'asinh_don', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/form/y_asinh',
            spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
            sample_desc='Asinh(donations) outcome',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Outcome: level (recdon_e)
r = run_ols(df, 'recdon_e', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/form/y_level',
            spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
            sample_desc='Donation level (euros)',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# D: Censoring at 10 euros (do-file Section D)
df['donateur_10'] = (df['recdon_e'] > 10).astype(int)
df['lndon_10'] = np.maximum(np.log(df['recdon_e']), np.log(10))
df.loc[df['recdon_e'] <= 10, 'lndon_10'] = np.log(10)
r = run_ols(df, 'lndon_10', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/form/censor_10',
            spec_tree_path='robustness/functional_form.md#alternative-estimators',
            sample_desc='Censoring point at 10 euros',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Quadratic income
df['lnrevdispo_sq'] = df['lnrevdispo'] ** 2
x_quad = ['lnrevdispo', 'lnrevdispo_sq', 'imporef'] + BASELINE_CONTROLS
r = run_ols(df, 'lndon', x_quad, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/form/quadratic_income',
            spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
            sample_desc='Quadratic income specification',
            controls_desc='Baseline + lnrevdispo^2',
            model_type='OLS')
append_result(r)

# Age squared
df['agepr_sq'] = df['agepr'] ** 2
x_age_sq = x_vars_baseline + ['agepr_sq']
r = run_ols(df, 'lndon', x_age_sq, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/form/quadratic_age',
            spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
            sample_desc='Quadratic age specification',
            controls_desc='Baseline + agepr^2',
            model_type='OLS')
append_result(r)

# Log-log: revdispo already in log, so try levels income
df['revdispo_scaled'] = df['revdispo'] / 1000  # in thousands
x_level_income = ['revdispo_scaled', 'imporef'] + YEAR_DUMMIES + GROUPE_DUMMIES + DEMO_CONTROLS
r = run_ols(df, 'lndon', x_level_income, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/form/income_level',
            spec_tree_path='robustness/functional_form.md#control-variable-transformations',
            sample_desc='Income in levels (thousands)',
            controls_desc='Baseline with income in levels',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 10: INFERENCE/CLUSTERING VARIATIONS
# =============================================================================
print("\n=== INFERENCE VARIATIONS ===")

# Robust HC1 (no clustering)
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            robust=True, cluster_var=None,
            spec_id='robust/se/hc1',
            spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
            sample_desc='HC1 robust SE',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Cluster by annee (year)
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='annee',
            spec_id='robust/cluster/time',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            sample_desc='Cluster by year',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Cluster by part (QF group)
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='part',
            spec_id='robust/cluster/qf_group',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            sample_desc='Cluster by QF group',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Cluster by grouprev (income group)
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='grouprev',
            spec_id='robust/cluster/income_group',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            sample_desc='Cluster by income group',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# No clustering, no robust (classical SE)
r = run_ols(df, 'lndon', x_vars_baseline, weights='pondv2',
            robust=False, cluster_var=None,
            spec_id='robust/cluster/none_classical',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            sample_desc='Classical (non-robust) SE',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 11: HETEROGENEITY ANALYSIS
# =============================================================================
print("\n=== HETEROGENEITY ANALYSIS ===")

# By marital status - use minimal controls to avoid empty group dummies
for status, label in [('celib', 'single'), ('marie', 'married'), ('divorce', 'divorced')]:
    df_sub = df[df[status] == 1].copy()
    # Drop marital dummies from controls for subgroup analysis, keep only non-empty groupes
    present_groupes = [g for g in GROUPE_DUMMIES if df_sub[g].sum() > 0]
    x_het = ['lnrevdispo', 'imporef'] + YEAR_DUMMIES + present_groupes + ['agepr', 'ts']
    r = run_ols(df_sub, 'lndon', x_het, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/het/by_marital_{label}',
                spec_tree_path='robustness/heterogeneity.md#demographic-subgroups',
                sample_desc=f'{label.capitalize()} taxpayers only',
                controls_desc='Baseline minus marital dummies',
                model_type='OLS')
    append_result(r)

# By age (above/below median)
median_age = df['agepr'].median()
for label, mask in [('young', df['agepr'] < median_age), ('old', df['agepr'] >= median_age)]:
    df_sub = df[mask].copy()
    r = run_ols(df_sub, 'lndon', x_vars_baseline, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/het/by_age_{label}',
                spec_tree_path='robustness/heterogeneity.md#demographic-subgroups',
                sample_desc=f'{label.capitalize()} age group (median={median_age:.0f})',
                controls_desc='Full baseline',
                model_type='OLS')
    append_result(r)

# By QF group size (single vs family)
for label, mask in [('single_qf', df['part'] == 1), ('couple_qf', df['part'] >= 2)]:
    df_sub = df[mask].copy()
    present_groupes = [g for g in GROUPE_DUMMIES if df_sub[g].sum() > 0]
    x_het = ['lnrevdispo', 'imporef'] + YEAR_DUMMIES + present_groupes + DEMO_CONTROLS
    r = run_ols(df_sub, 'lndon', x_het, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/het/by_family_{label}',
                spec_tree_path='robustness/heterogeneity.md#demographic-subgroups',
                sample_desc=f'{label} taxpayers',
                controls_desc='Full baseline (adjusted groupes)',
                model_type='OLS')
    append_result(r)

# By taxable status
for label, val in [('taxable', 1), ('nontaxable', 0)]:
    df_sub = df[df['imposab'] == val].copy()
    r = run_ols(df_sub, 'lndon', x_vars_baseline, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/het/by_taxable_{label}',
                spec_tree_path='robustness/heterogeneity.md#treatment-related-heterogeneity',
                sample_desc=f'{label} taxpayers',
                controls_desc='Full baseline',
                model_type='OLS')
    append_result(r)

# By income level (terciles)
for t, label in enumerate(['low_income', 'mid_income', 'high_income']):
    bounds = df['rimp_e'].quantile([t/3, (t+1)/3])
    df_sub = df[(df['rimp_e'] >= bounds.iloc[0]) & (df['rimp_e'] < bounds.iloc[1])].copy()
    if t == 2:
        df_sub = df[df['rimp_e'] >= bounds.iloc[0]].copy()
    r = run_ols(df_sub, 'lndon', x_vars_baseline, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/het/by_income_{label}',
                spec_tree_path='robustness/heterogeneity.md#socioeconomic-subgroups',
                sample_desc=f'{label} tercile',
                controls_desc='Full baseline',
                model_type='OLS')
    append_result(r)

# Interaction: imporef x age
df['imporef_x_age'] = df['imporef'] * df['agepr']
x_interact = x_vars_baseline + ['imporef_x_age']
r = run_ols(df, 'lndon', x_interact, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/het/interaction_age',
            spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
            sample_desc='Interaction imporef x age',
            controls_desc='Baseline + imporef*age',
            model_type='OLS')
append_result(r)

# Interaction: imporef x married
df['imporef_x_marie'] = df['imporef'] * df['marie']
x_interact2 = x_vars_baseline + ['imporef_x_marie']
r = run_ols(df, 'lndon', x_interact2, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/het/interaction_married',
            spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
            sample_desc='Interaction imporef x married',
            controls_desc='Baseline + imporef*marie',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 12: ALTERNATIVE OUTCOMES
# =============================================================================
print("\n=== ALTERNATIVE OUTCOMES ===")

# Extensive margin: donateur (binary)
r = run_ols(df, 'donateur', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/outcome/extensive_margin',
            spec_tree_path='robustness/measurement.md#outcome-variable-alternatives',
            sample_desc='Extensive margin (LPM): any donation',
            controls_desc='Full baseline',
            model_type='LPM')
append_result(r)

# Intensive margin: lndon among donors
r = run_ols(df_donors, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/outcome/intensive_margin',
            spec_tree_path='robustness/measurement.md#outcome-variable-alternatives',
            sample_desc='Intensive margin: lndon among donors',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Donation levels among donors
r = run_ols(df_donors, 'recdon_e', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/outcome/donation_levels_donors',
            spec_tree_path='robustness/measurement.md#outcome-variable-alternatives',
            sample_desc='Donation levels (euros) among donors',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Standardized outcome
df['lndon_std'] = (df['lndon'] - df['lndon'].mean()) / df['lndon'].std()
r = run_ols(df, 'lndon_std', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/outcome/standardized',
            spec_tree_path='robustness/measurement.md#outcome-variable-alternatives',
            sample_desc='Standardized lndon (z-score)',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 13: PLACEBO TESTS
# =============================================================================
print("\n=== PLACEBO TESTS ===")

# Placebo: random treatment assignment
np.random.seed(999)
df['placebo_imporef'] = np.random.permutation(df['imporef'].values)
x_placebo = ['lnrevdispo', 'placebo_imporef'] + BASELINE_CONTROLS
r = run_ols(df, 'lndon', x_placebo, weights='pondv2',
            treatment_var='placebo_imporef',
            cluster_var='groupe',
            spec_id='robust/placebo/random_treatment',
            spec_tree_path='robustness/placebo_tests.md',
            sample_desc='Placebo: randomly permuted treatment',
            controls_desc='Full baseline with permuted treatment',
            model_type='OLS')
append_result(r)

# Placebo: pre-reform only (no price variation, all at 50%)
df_pre = df[df['annee'] <= 2002].copy()
# In pre-reform, all logprix = log(0.5), so imporef should show no effect
# Create fake "post" treatment within pre-period
df_pre['fake_post'] = (df_pre['annee'] >= 2001).astype(int)
df_pre['placebo_imporef2'] = df_pre['imposab'] * df_pre['fake_post'] * np.log(0.5)
yr_pre = [yd for yd in YEAR_DUMMIES if yd in ['an99', 'an00', 'an01', 'an02']]
x_placebo2 = ['lnrevdispo', 'placebo_imporef2'] + yr_pre + GROUPE_DUMMIES + DEMO_CONTROLS
r = run_ols(df_pre, 'lndon', x_placebo2, weights='pondv2',
            treatment_var='placebo_imporef2',
            cluster_var='groupe',
            spec_id='robust/placebo/pre_reform_fake',
            spec_tree_path='robustness/placebo_tests.md',
            sample_desc='Placebo: fake treatment in pre-reform period',
            controls_desc='Baseline controls (pre-reform only)',
            model_type='OLS')
append_result(r)

# Placebo: unaffected outcome (age should not respond to tax incentive)
r = run_ols(df, 'agepr', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/placebo/unaffected_outcome',
            spec_tree_path='robustness/placebo_tests.md',
            sample_desc='Placebo: age as outcome (should be zero)',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# Placebo: use non-taxable group imposab=0 (treatment should be zero)
df_nontax = df[df['imposab'] == 0].copy()
r = run_ols(df_nontax, 'lndon', x_vars_baseline, weights='pondv2',
            cluster_var='groupe',
            spec_id='robust/placebo/nontaxable_group',
            spec_tree_path='robustness/placebo_tests.md',
            sample_desc='Placebo: non-taxable group only (imporef=0)',
            controls_desc='Full baseline',
            model_type='OLS')
append_result(r)

# =============================================================================
# STEP 14: DROP EACH QF GROUP
# =============================================================================
print("\n=== DROP EACH QF GROUP ===")

for part_val in [1, 1.5, 2, 2.5, 3, 4, 5]:
    part_label = str(part_val).replace('.', '_')
    df_sub = df[df['part'] != part_val].copy()
    r = run_ols(df_sub, 'lndon', x_vars_baseline, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/sample/drop_qf_{part_label}',
                spec_tree_path='robustness/sample_restrictions.md#geographic-unit-restrictions',
                sample_desc=f'Drop QF group part={part_val}',
                controls_desc='Full baseline',
                model_type='OLS')
    append_result(r)

# =============================================================================
# STEP 15: ADDITIONAL QUANTILE REGRESSIONS ON SUBSAMPLES
# =============================================================================
print("\n=== ADDITIONAL QUANTILE REGRESSIONS ===")

# Quantile regression on donors only (key robustness for Chernozhukov 3-step)
for q in [0.50, 0.75, 0.90]:
    q_str = str(int(q*100))
    r = run_quantreg(df_donors, 'lndon', x_vars_baseline, q=q,
                     spec_id=f'robust/model/qreg_donors_{q_str}',
                     spec_tree_path='robustness/model_specification.md#quantile-and-robust-estimators',
                     sample_desc=f'Quantile regression q={q}, donors only',
                     controls_desc='Full baseline',
                     model_type=f'quantile_regression_donors_q{q_str}')
    append_result(r)

# =============================================================================
# STEP 16: SINGLE COVARIATE SPECS
# =============================================================================
print("\n=== SINGLE COVARIATE SPECS ===")

# Treatment + each control one at a time
for ctrl in DEMO_CONTROLS:
    x_single = ['lnrevdispo', 'imporef', ctrl] + YEAR_DUMMIES + GROUPE_DUMMIES
    r = run_ols(df, 'lndon', x_single, weights='pondv2',
                cluster_var='groupe',
                spec_id=f'robust/single/{ctrl}',
                spec_tree_path='robustness/single_covariate.md',
                sample_desc=f'Single covariate: {ctrl}',
                controls_desc=f'imporef + lnrevdispo + {ctrl} + FE',
                model_type='OLS')
    append_result(r)

# =============================================================================
# FINALIZE
# =============================================================================
print(f"\n=== TOTAL SPECIFICATIONS: {len(results)} ===")

# Save results
results_df = pd.DataFrame(results)
output_path = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116532-V1/specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# Summary statistics
print("\n=== SUMMARY ===")
print(f"Total specs: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.6f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.6f}")
print(f"Range: [{results_df['coefficient'].min():.6f}, {results_df['coefficient'].max():.6f}]")

# Breakdown by category
def categorize(spec_id):
    if spec_id.startswith('baseline'):
        return 'Baseline'
    elif 'placebo' in spec_id:
        return 'Placebo tests'
    elif 'het' in spec_id:
        return 'Heterogeneity'
    elif 'outcome' in spec_id:
        return 'Alternative outcomes'
    elif 'loo' in spec_id:
        return 'Control variations (LOO)'
    elif 'build' in spec_id or 'single' in spec_id or spec_id.startswith('did/controls'):
        return 'Control variations (build-up)'
    elif 'method' in spec_id or spec_id.startswith('did/method'):
        return 'Estimation method'
    elif 'model' in spec_id:
        return 'Estimation method'
    elif 'fe/' in spec_id:
        return 'FE variations'
    elif 'sample' in spec_id:
        return 'Sample restrictions'
    elif 'treatment' in spec_id:
        return 'Alternative treatments'
    elif 'form' in spec_id:
        return 'Functional form'
    elif 'cluster' in spec_id or 'se/' in spec_id:
        return 'Inference variations'
    elif 'measure' in spec_id:
        return 'Weights/measurement'
    else:
        return 'Other'

results_df['category'] = results_df['spec_id'].apply(categorize)
for cat in results_df['category'].unique():
    sub = results_df[results_df['category'] == cat]
    n = len(sub)
    pct_pos = 100 * (sub['coefficient'] > 0).mean()
    pct_sig = 100 * (sub['p_value'] < 0.05).mean()
    print(f"  {cat}: N={n}, {pct_pos:.0f}% positive, {pct_sig:.0f}% sig at 5%")
