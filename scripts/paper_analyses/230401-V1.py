"""
Specification Search for Paper 230401-V1
=========================================
AER 2025 Paper on Racial Discrimination in Small Business Lending

Paper Overview:
- Topic: Racial discrimination in small business loan pricing and collateral requirements
- Data: Survey of 2,784 small businesses with recent credit products
- Main Hypothesis: Minority-owned businesses (Hispanic, Black, Asian, Native American)
  face higher interest rates and more onerous collateral requirements compared to
  White-owned businesses
- Method: Cross-sectional OLS with fixed effects and clustered standard errors

Identification Strategy:
- Compare loan terms for minority-owned vs. white-owned businesses
- Control for firm characteristics, loan characteristics, lender characteristics
- Include state and time fixed effects
- Cluster standard errors at state level

Treatment Variables:
- hisp_50: Hispanic ownership >= 50%
- black_50: Black ownership >= 50%
- asian_50: Asian ownership >= 50%
- native_50: Native American ownership >= 50%
- white_50: White ownership >= 50% (reference group)

Primary Outcome:
- loanrate_w2: Winsorized interest rate (2%)

Secondary Outcomes:
- loanspread_w2: Winsorized loan spread
- coll: Any collateral required
- collval: Collateral value category
- collmore: Additional collateral beyond standard
- blien: Blanket lien
- bcoll: Business collateral
- pcoll: Personal collateral
- bpcoll: Both business and personal collateral
- signature: Signature/unsecured loan

Method Classification:
- method_code: cross_sectional_ols
- method_tree_path: specification_tree/methods/cross_sectional_ols.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Package and output paths
PACKAGE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/230401-V1'
DATA_PATH = f'{PACKAGE_DIR}/final.dta'
OUTPUT_CSV = f'{PACKAGE_DIR}/specification_results.csv'

# Paper metadata
PAPER_ID = '230401-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Racial Discrimination in Small Business Lending'

def load_and_prepare_data():
    """Load data and prepare variables for analysis."""
    df = pd.read_stata(DATA_PATH)

    # Convert categorical to numeric for fixed effects
    df['statehead_code'] = df['statehead'].astype('category').cat.codes
    df['timeid_int'] = pd.to_numeric(df['timeid'], errors='coerce').fillna(0).astype(int)
    df['ind_code'] = df['ind'].astype('category').cat.codes
    df['creditscore_int'] = pd.to_numeric(df['creditscore'], errors='coerce').fillna(0).astype(int)
    df['ceoown_num'] = pd.to_numeric(df['ceoown'], errors='coerce')
    df['collval_num'] = pd.to_numeric(df['collval'].astype(str), errors='coerce')

    # Convert relength to numeric
    df['relength_num'] = pd.to_numeric(df['relength'].astype(str), errors='coerce')

    return df

def create_result_dict(spec_id, spec_tree_path, model, treatment_var, outcome_var,
                       sample_desc, fixed_effects, controls_desc, cluster_var,
                       model_type, n_obs):
    """Create standardized result dictionary from pyfixest model."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]

        # Get confidence interval
        try:
            ci = model.confint()
            ci_lower = ci.loc[treatment_var, '2.5%']
            ci_upper = ci.loc[treatment_var, '97.5%']
        except:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

        # Get R-squared
        try:
            r2 = model.r2
        except:
            r2 = np.nan

        # Get full coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fixed_effects.split(', ') if fixed_effects else [],
            'diagnostics': {}
        }

        for var in model.coef().index:
            if var != treatment_var:
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
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
            'n_obs': n_obs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error processing {spec_id}: {e}")
        return None

def run_specifications():
    """Run all specifications following the specification tree."""
    results = []
    df = load_and_prepare_data()

    # Define variables
    treatment_vars = ['black_50']  # Focus on primary treatment (black_50) for main analysis
    all_treatments = ['hisp_50', 'black_50', 'asian_50', 'native_50']
    outcome_var = 'loanrate_w2'

    # Control variable groups
    firm_controls = ['busage', 'ceoexp', 'ceoage', 'ceoown_num', 'sales_val', 'assets_val',
                     'loss21', 'revenuegrow', 'employeegrow', 'conditiongood', 'ltd',
                     'family', 'woman_owned']
    loan_controls = ['loan', 'newcredit', 'purpose_debt', 'fixed', 'term']
    lender_controls = ['smallbank', 'creditunion', 'CDFI', 'fintech', 'nonbank',
                       'relength_yr', 'hhi_c', 'c_branches']
    all_controls = firm_controls + loan_controls + lender_controls

    # Alternative outcomes
    alt_outcomes = ['loanspread_w2', 'coll', 'collval_num', 'collmore', 'blien',
                    'bcoll', 'pcoll', 'bpcoll', 'signature']

    print("Starting specification search...")
    print("=" * 60)

    # =========================================================================
    # BASELINE SPECIFICATIONS (Table 2 replication)
    # =========================================================================
    print("\n1. Running baseline specifications...")

    for treat in all_treatments:
        # Create sample filter
        sample_cond = (df[treat] == 1) | (df['white_50'] == 1)
        df_sample = df[sample_cond].copy()

        controls_str = ' + '.join(all_controls)
        formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"

        try:
            model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id='baseline',
                spec_tree_path='methods/cross_sectional_ols.md#baseline',
                model=model,
                treatment_var=treat,
                outcome_var=outcome_var,
                sample_desc=f'{treat}==1 or white_50==1',
                fixed_effects='state, time',
                controls_desc='firm, loan, lender characteristics',
                cluster_var='statehead',
                model_type='OLS with FE',
                n_obs=len(df_sample.dropna(subset=[outcome_var, treat] + all_controls))
            )
            if result:
                results.append(result)
                print(f"  Baseline ({treat}): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"  Error in baseline ({treat}): {e}")

    # =========================================================================
    # CONTROL VARIATIONS (10-15 specs)
    # =========================================================================
    print("\n2. Running control variations...")

    treat = 'black_50'  # Primary treatment for robustness
    sample_cond = (df[treat] == 1) | (df['white_50'] == 1)
    df_sample = df[sample_cond].copy()

    # 2.1 No controls (bivariate)
    try:
        formula = f"{outcome_var} ~ {treat}"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='ols/controls/none',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='none', controls_desc='none',
            cluster_var='statehead', model_type='OLS',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  No controls: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in no controls: {e}")

    # 2.2 FE only (no controls)
    try:
        formula = f"{outcome_var} ~ {treat} | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='ols/controls/fe_only',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='none',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  FE only: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in FE only: {e}")

    # 2.3 Firm controls only
    try:
        controls_str = ' + '.join(firm_controls)
        formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='ols/controls/firm_only',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='firm characteristics only',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat] + firm_controls))
        )
        if result:
            results.append(result)
            print(f"  Firm controls only: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in firm controls only: {e}")

    # 2.4 Firm + loan controls
    try:
        controls_str = ' + '.join(firm_controls + loan_controls)
        formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='ols/controls/firm_loan',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='firm + loan characteristics',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat] + firm_controls + loan_controls))
        )
        if result:
            results.append(result)
            print(f"  Firm+loan controls: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in firm+loan controls: {e}")

    # 2.5-2.10 Leave-one-out on key controls
    key_controls_to_drop = ['ceoown_num', 'sales_val', 'assets_val', 'creditscore_int', 'relength_yr', 'hhi_c']
    for ctrl in key_controls_to_drop:
        try:
            remaining = [c for c in all_controls if c != ctrl and c != 'creditscore_int']
            controls_str = ' + '.join(remaining)
            if ctrl == 'creditscore_int':
                formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) | statehead_code + timeid_int"
            else:
                formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/control/drop_{ctrl}',
                spec_tree_path='robustness/leave_one_out.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=f'{treat}==1 or white_50==1',
                fixed_effects='state, time', controls_desc=f'all controls except {ctrl}',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sample.dropna(subset=[outcome_var, treat] + remaining))
            )
            if result:
                results.append(result)
                print(f"  Drop {ctrl}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error dropping {ctrl}: {e}")

    # =========================================================================
    # SAMPLE RESTRICTIONS (10-15 specs)
    # =========================================================================
    print("\n3. Running sample restrictions...")

    # 3.1-3.4 By relationship length
    for rel_cutoff, rel_desc in [(2, 'short_relationship'), (3, 'long_relationship')]:
        try:
            if rel_desc == 'short_relationship':
                sample_cond_rel = sample_cond & (df['relength_num'] <= rel_cutoff)
            else:
                sample_cond_rel = sample_cond & (df['relength_num'] >= rel_cutoff)
            df_sub = df[sample_cond_rel].copy()
            controls_str = ' + '.join(all_controls)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/{rel_desc}',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=f'{rel_desc}: relength {"<=" if rel_desc == "short_relationship" else ">="} {rel_cutoff}',
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat] + all_controls))
            )
            if result:
                results.append(result)
                print(f"  {rel_desc}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error in {rel_desc}: {e}")

    # 3.5-3.6 By loan type
    for loan_val, loan_desc in [(1, 'loans_only'), (0, 'lines_only')]:
        try:
            sample_cond_loan = sample_cond & (df['loan'] == loan_val)
            df_sub = df[sample_cond_loan].copy()
            controls_str = ' + '.join([c for c in all_controls if c != 'loan'])
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/{loan_desc}',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=loan_desc,
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  {loan_desc}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error in {loan_desc}: {e}")

    # 3.7-3.8 By credit type (new vs renewal)
    for newcred_val, cred_desc in [(1, 'new_credit'), (0, 'renewal')]:
        try:
            sample_cond_cred = sample_cond & (df['newcredit'] == newcred_val)
            df_sub = df[sample_cond_cred].copy()
            controls_str = ' + '.join([c for c in all_controls if c != 'newcredit'])
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/{cred_desc}',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=cred_desc,
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  {cred_desc}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error in {cred_desc}: {e}")

    # 3.9-3.10 By loan amount (q30)
    try:
        median_q30 = df_sample['q30'].median()
        for size_desc, size_cond in [('large_loans', df['q30'] >= 500000), ('small_loans', df['q30'] < 500000)]:
            sample_cond_size = sample_cond & size_cond
            df_sub = df[sample_cond_size].copy()
            controls_str = ' + '.join(all_controls)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/{size_desc}',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=size_desc,
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  {size_desc}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in loan size split: {e}")

    # 3.11-3.12 By firm assets
    try:
        for asset_desc, asset_cond in [('high_assets', df['assets_val'] > 750), ('low_assets', df['assets_val'] <= 750)]:
            sample_cond_asset = sample_cond & asset_cond
            df_sub = df[sample_cond_asset].copy()
            controls_str = ' + '.join(all_controls)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/{asset_desc}',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=asset_desc,
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  {asset_desc}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in asset split: {e}")

    # 3.13-3.14 By lender type
    try:
        for lender_desc, lender_cond in [('bank_lenders', df['lender'].astype(str).str.strip().isin(['1', '2'])),
                                          ('nonbank_lenders', ~df['lender'].astype(str).str.strip().isin(['1', '2']))]:
            sample_cond_lender = sample_cond & lender_cond
            df_sub = df[sample_cond_lender].copy()
            if len(df_sub) < 50:
                continue
            controls_str = ' + '.join(all_controls)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/{lender_desc}',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=lender_desc,
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  {lender_desc}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in lender split: {e}")

    # 3.15-3.18 Outlier treatment (winsorize/trim)
    for pct in [1, 5]:
        try:
            df_trim = df_sample.copy()
            lower = df_trim[outcome_var].quantile(pct/100)
            upper = df_trim[outcome_var].quantile(1 - pct/100)
            df_trim = df_trim[(df_trim[outcome_var] >= lower) & (df_trim[outcome_var] <= upper)]
            controls_str = ' + '.join(all_controls)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_trim, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/sample/trim_{pct}pct',
                spec_tree_path='robustness/sample_restrictions.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=f'trimmed {pct}% tails',
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_trim.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  Trim {pct}%: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error in trim {pct}%: {e}")

    # =========================================================================
    # ALTERNATIVE OUTCOMES (5-10 specs)
    # =========================================================================
    print("\n4. Running alternative outcomes...")

    for alt_outcome in alt_outcomes:
        if alt_outcome not in df.columns:
            continue
        try:
            controls_str = ' + '.join(all_controls)
            formula = f"{alt_outcome} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/outcome/{alt_outcome}',
                spec_tree_path='robustness/measurement.md',
                model=model, treatment_var=treat, outcome_var=alt_outcome,
                sample_desc=f'{treat}==1 or white_50==1',
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sample.dropna(subset=[alt_outcome, treat]))
            )
            if result:
                results.append(result)
                print(f"  Outcome {alt_outcome}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error in outcome {alt_outcome}: {e}")

    # =========================================================================
    # ALTERNATIVE TREATMENTS (3-5 specs)
    # =========================================================================
    print("\n5. Running alternative treatments...")

    # Run baseline for other minority groups
    for alt_treat in ['hisp_50', 'asian_50', 'native_50']:
        try:
            sample_cond_alt = (df[alt_treat] == 1) | (df['white_50'] == 1)
            df_sub = df[sample_cond_alt].copy()
            controls_str = ' + '.join(all_controls)
            formula = f"{outcome_var} ~ {alt_treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/treatment/{alt_treat}',
                spec_tree_path='robustness/measurement.md',
                model=model, treatment_var=alt_treat, outcome_var=outcome_var,
                sample_desc=f'{alt_treat}==1 or white_50==1',
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, alt_treat]))
            )
            if result:
                results.append(result)
                print(f"  Treatment {alt_treat}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"  Error in treatment {alt_treat}: {e}")

    # =========================================================================
    # INFERENCE/CLUSTERING VARIATIONS (5-8 specs)
    # =========================================================================
    print("\n6. Running inference variations...")

    # 6.1 Robust SE (no clustering)
    try:
        controls_str = ' + '.join(all_controls)
        formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov='hetero')
        result = create_result_dict(
            spec_id='robust/cluster/robust_hc1',
            spec_tree_path='robustness/clustering_variations.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='none (HC1)', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Robust HC1: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")
    except Exception as e:
        print(f"  Error in HC1: {e}")

    # 6.2 Cluster by time
    try:
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'timeid_int'})
        result = create_result_dict(
            spec_id='robust/cluster/time',
            spec_tree_path='robustness/clustering_variations.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='timeid', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Cluster time: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")
    except Exception as e:
        print(f"  Error in time clustering: {e}")

    # 6.3 Cluster by industry
    try:
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'ind_code'})
        result = create_result_dict(
            spec_id='robust/cluster/industry',
            spec_tree_path='robustness/clustering_variations.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='industry', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Cluster industry: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")
    except Exception as e:
        print(f"  Error in industry clustering: {e}")

    # 6.4 Two-way clustering (state x time)
    try:
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': ['statehead_code', 'timeid_int']})
        result = create_result_dict(
            spec_id='robust/cluster/twoway',
            spec_tree_path='robustness/clustering_variations.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='state x time', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Two-way cluster: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")
    except Exception as e:
        print(f"  Error in two-way clustering: {e}")

    # =========================================================================
    # ESTIMATION METHOD VARIATIONS (3-5 specs)
    # =========================================================================
    print("\n7. Running estimation variations...")

    # 7.1 No fixed effects
    try:
        controls_str = ' + '.join(all_controls)
        formula_nofe = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int)"
        model = pf.feols(formula_nofe, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/estimation/no_fe',
            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='none', controls_desc='full controls',
            cluster_var='statehead', model_type='OLS',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  No FE: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in no FE: {e}")

    # 7.2 State FE only
    try:
        formula_state_fe = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code"
        model = pf.feols(formula_state_fe, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/estimation/state_fe_only',
            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state only', controls_desc='full controls',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  State FE only: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in state FE only: {e}")

    # 7.3 Time FE only
    try:
        formula_time_fe = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | timeid_int"
        model = pf.feols(formula_time_fe, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/estimation/time_fe_only',
            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='time only', controls_desc='full controls',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Time FE only: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in time FE only: {e}")

    # 7.4 Industry FE instead of controls
    try:
        controls_no_ind = [c for c in all_controls if c not in ['ind_code']]
        controls_str = ' + '.join(controls_no_ind)
        formula_ind_fe = f"{outcome_var} ~ {treat} + {controls_str} + C(creditscore_int) | statehead_code + timeid_int + ind_code"
        model = pf.feols(formula_ind_fe, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/estimation/industry_fe',
            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time, industry', controls_desc='full controls (excl industry dummies)',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Industry FE: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in industry FE: {e}")

    # =========================================================================
    # FUNCTIONAL FORM (3-5 specs)
    # =========================================================================
    print("\n8. Running functional form variations...")

    # 8.1 Log outcome
    try:
        df_sample['log_loanrate'] = np.log(df_sample[outcome_var] + 0.01)
        controls_str = ' + '.join(all_controls)
        formula = f"log_loanrate ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/funcform/log_outcome',
            spec_tree_path='robustness/functional_form.md',
            model=model, treatment_var=treat, outcome_var='log(loanrate_w2)',
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Log outcome: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in log outcome: {e}")

    # 8.2 IHS outcome
    try:
        df_sample['ihs_loanrate'] = np.arcsinh(df_sample[outcome_var])
        formula = f"ihs_loanrate ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/funcform/ihs_outcome',
            spec_tree_path='robustness/functional_form.md',
            model=model, treatment_var=treat, outcome_var='arcsinh(loanrate_w2)',
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  IHS outcome: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in IHS outcome: {e}")

    # 8.3 Squared continuous controls
    try:
        df_sample['busage_sq'] = df_sample['busage'] ** 2
        df_sample['ceoexp_sq'] = df_sample['ceoexp'] ** 2
        df_sample['ceoage_sq'] = df_sample['ceoage'] ** 2
        controls_sq = all_controls + ['busage_sq', 'ceoexp_sq', 'ceoage_sq']
        controls_str = ' + '.join(controls_sq)
        formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
        result = create_result_dict(
            spec_id='robust/funcform/quadratic_controls',
            spec_tree_path='robustness/functional_form.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1',
            fixed_effects='state, time', controls_desc='full controls + squared terms',
            cluster_var='statehead', model_type='OLS with FE',
            n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Quadratic controls: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in quadratic controls: {e}")

    # =========================================================================
    # WEIGHTS (2-3 specs)
    # =========================================================================
    print("\n9. Running weight variations...")

    # 9.1 Race-state weights
    try:
        df_weighted = df_sample.dropna(subset=['weight_race_state'])
        controls_str = ' + '.join(all_controls)
        formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
        model = pf.feols(formula, data=df_weighted, vcov={'CRV1': 'statehead_code'},
                        weights=df_weighted['weight_race_state'])
        result = create_result_dict(
            spec_id='robust/weights/race_state',
            spec_tree_path='robustness/model_specification.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1, weighted',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='statehead', model_type='WLS with FE',
            n_obs=len(df_weighted.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Race-state weights: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in race-state weights: {e}")

    # 9.2 Race-state-industry weights
    try:
        df_weighted = df_sample.dropna(subset=['weight_race_state_ind'])
        model = pf.feols(formula, data=df_weighted, vcov={'CRV1': 'statehead_code'},
                        weights=df_weighted['weight_race_state_ind'])
        result = create_result_dict(
            spec_id='robust/weights/race_state_ind',
            spec_tree_path='robustness/model_specification.md',
            model=model, treatment_var=treat, outcome_var=outcome_var,
            sample_desc=f'{treat}==1 or white_50==1, weighted by race-state-industry',
            fixed_effects='state, time', controls_desc='full controls',
            cluster_var='statehead', model_type='WLS with FE',
            n_obs=len(df_weighted.dropna(subset=[outcome_var, treat]))
        )
        if result:
            results.append(result)
            print(f"  Race-state-industry weights: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in race-state-industry weights: {e}")

    # =========================================================================
    # HETEROGENEITY ANALYSIS (5-10 specs)
    # =========================================================================
    print("\n10. Running heterogeneity analysis...")

    het_vars = ['woman_owned', 'family', 'conditiongood', 'loss21', 'loan', 'newcredit']
    for het_var in het_vars:
        try:
            controls_no_het = [c for c in all_controls if c != het_var]
            controls_str = ' + '.join(controls_no_het)
            formula = f"{outcome_var} ~ {treat} * {het_var} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})

            # Get interaction coefficient
            interaction_term = f'{treat}:{het_var}'
            if interaction_term in model.coef().index:
                result = create_result_dict(
                    spec_id=f'robust/heterogeneity/{het_var}',
                    spec_tree_path='robustness/heterogeneity.md',
                    model=model, treatment_var=treat, outcome_var=outcome_var,
                    sample_desc=f'{treat}==1 or white_50==1, interaction with {het_var}',
                    fixed_effects='state, time', controls_desc=f'full controls + {treat}*{het_var}',
                    cluster_var='statehead', model_type='OLS with FE',
                    n_obs=len(df_sample.dropna(subset=[outcome_var, treat]))
                )
                if result:
                    # Add interaction coefficient info
                    result['interaction_coef'] = float(model.coef()[interaction_term])
                    result['interaction_se'] = float(model.se()[interaction_term])
                    result['interaction_pval'] = float(model.pvalue()[interaction_term])
                    results.append(result)
                    print(f"  Het {het_var}: main={result['coefficient']:.4f}, interaction={result['interaction_coef']:.4f}")
        except Exception as e:
            print(f"  Error in heterogeneity {het_var}: {e}")

    # Heterogeneity by credit score
    try:
        for cs_val, cs_desc in [(1, 'low_credit'), (2, 'medium_credit'), (3, 'high_credit')]:
            sample_cond_cs = sample_cond & (df['creditscore'] == cs_val)
            df_sub = df[sample_cond_cs].copy()
            if len(df_sub) < 30:
                continue
            controls_no_cs = [c for c in all_controls]
            controls_str = ' + '.join(controls_no_cs)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/heterogeneity/creditscore_{cs_desc}',
                spec_tree_path='robustness/heterogeneity.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=f'{cs_desc} credit score firms',
                fixed_effects='state, time', controls_desc='full controls',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sub.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  Het {cs_desc}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in credit score heterogeneity: {e}")

    # =========================================================================
    # PLACEBO TESTS (3-5 specs)
    # =========================================================================
    print("\n11. Running placebo tests...")

    # Placebo: outcomes that shouldn't be affected by race
    placebo_outcomes = ['term', 'busage', 'ceoexp']  # Predetermined characteristics
    for placebo_out in placebo_outcomes:
        if placebo_out not in df.columns:
            continue
        try:
            controls_no_placebo = [c for c in all_controls if c != placebo_out]
            controls_str = ' + '.join(controls_no_placebo)
            formula = f"{placebo_out} ~ {treat} + {controls_str} + C(ind_code) + C(creditscore_int) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_sample, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id=f'robust/placebo/{placebo_out}',
                spec_tree_path='robustness/placebo_tests.md',
                model=model, treatment_var=treat, outcome_var=placebo_out,
                sample_desc=f'{treat}==1 or white_50==1',
                fixed_effects='state, time', controls_desc='full controls (excl placebo var)',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_sample.dropna(subset=[placebo_out, treat]))
            )
            if result:
                results.append(result)
                print(f"  Placebo {placebo_out}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"  Error in placebo {placebo_out}: {e}")

    # =========================================================================
    # ALTERNATIVE CREDIT RATING (from Table A2)
    # =========================================================================
    print("\n12. Running alternative credit rating specification...")

    try:
        # Use D&B credit points instead of self-reported credit score
        df_dnb = df_sample.dropna(subset=['cpoints'])
        if len(df_dnb) > 50:
            controls_dnb = [c for c in all_controls if c != 'creditscore_int'] + ['cpoints']
            controls_str = ' + '.join(controls_dnb)
            formula = f"{outcome_var} ~ {treat} + {controls_str} + C(ind_code) | statehead_code + timeid_int"
            model = pf.feols(formula, data=df_dnb, vcov={'CRV1': 'statehead_code'})
            result = create_result_dict(
                spec_id='robust/control/dnb_credit',
                spec_tree_path='robustness/measurement.md',
                model=model, treatment_var=treat, outcome_var=outcome_var,
                sample_desc=f'{treat}==1 or white_50==1, D&B credit rating',
                fixed_effects='state, time', controls_desc='D&B credit instead of self-reported',
                cluster_var='statehead', model_type='OLS with FE',
                n_obs=len(df_dnb.dropna(subset=[outcome_var, treat]))
            )
            if result:
                results.append(result)
                print(f"  D&B credit: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"  Error in D&B credit: {e}")

    # =========================================================================
    # Convert results to DataFrame and save
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"TOTAL SPECIFICATIONS RUN: {len(results)}")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    return results_df


if __name__ == '__main__':
    results_df = run_specifications()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Filter to main treatment (black_50)
    main_results = results_df[results_df['treatment_var'] == 'black_50'].copy()

    print(f"\nTotal specifications: {len(results_df)}")
    print(f"Black_50 specifications: {len(main_results)}")

    if len(main_results) > 0:
        print(f"\nCoefficient distribution (black_50 on loanrate):")
        loan_results = main_results[main_results['outcome_var'].str.contains('loanrate|log|ihs', na=False)]
        if len(loan_results) > 0:
            print(f"  Mean: {loan_results['coefficient'].mean():.4f}")
            print(f"  Median: {loan_results['coefficient'].median():.4f}")
            print(f"  Min: {loan_results['coefficient'].min():.4f}")
            print(f"  Max: {loan_results['coefficient'].max():.4f}")
            print(f"  Positive: {(loan_results['coefficient'] > 0).sum()} ({100*(loan_results['coefficient'] > 0).mean():.1f}%)")
            print(f"  Significant (p<0.05): {(loan_results['p_value'] < 0.05).sum()} ({100*(loan_results['p_value'] < 0.05).mean():.1f}%)")
            print(f"  Significant (p<0.01): {(loan_results['p_value'] < 0.01).sum()} ({100*(loan_results['p_value'] < 0.01).mean():.1f}%)")
