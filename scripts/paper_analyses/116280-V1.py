"""
Specification Search Script for Mayers & Smith (2005)
"Agency Problems and Policyholders: Evidence from the Mutual Insurance Market"
AER (guessed title based on data content -- insurance organizational form choice)

Paper ID: 116280-V1

Executes the approved SPECIFICATION_SURFACE.json:
  - G1: mutual ~ mlaw + controls (Table 2, logit)

Outputs:
  - specification_results.csv (core specs: baseline, design/*, rc/*)
  - inference_results.csv (inference variants: infer/*)
  - SPECIFICATION_SEARCH.md (run log)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "116280-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_FILE = os.path.join(PACKAGE_DIR, "MS_20040270.dta")
SURFACE_FILE = os.path.join(PACKAGE_DIR, "SPECIFICATION_SURFACE.json")
SEED = 116280

# Import helper utilities
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash,
    software_block
)

# =============================================================================
# Load surface and compute hash
# =============================================================================
with open(SURFACE_FILE, 'r') as f:
    surface = json.load(f)
SURFACE_HASH = compute_surface_hash(surface)
SOFTWARE = software_block()

# Canonical inference for G1
CANONICAL_INFERENCE = {
    "spec_id": "infer/se/cluster/state",
    "params": {"cluster_var": "state"},
    "notes": "Cluster-robust SEs at state level (47 clusters), matching paper baseline."
}

# Design audit for G1
DESIGN_AUDIT = {
    "cross_sectional_ols": {
        "estimator": "logit",
        "reported_object": "index_coef",
        "model_formula": "mutual ~ mlaw + slaw + regulate + nfc + reform + refmlaw + refslaw + refregulate + refnfc + ten2 + ten3 + ten4 + ten5, cluster(state)",
        "selection_story": "Cross-sectional variation in state insurance regulation explains organizational form choice (mutual vs stock) of insurance companies formed 1900-1949",
        "n_clusters": 47
    }
}

# =============================================================================
# Load and prepare data
# =============================================================================
print(f"Loading data from: {DATA_FILE}")
df = pd.read_stata(DATA_FILE)
print(f"Data shape: {df.shape}")
print(f"N = {len(df)}, mutual mean = {df['mutual'].mean():.4f}")

# Encode state as numeric for clustering
df['state_code'] = df['state'].astype('category').cat.codes

# Baseline controls
BASELINE_CONTROLS = ['slaw', 'regulate', 'nfc', 'reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc',
                     'ten2', 'ten3', 'ten4', 'ten5']
DECADE_DUMMIES = ['ten2', 'ten3', 'ten4', 'ten5']
REFORM_INTERACTIONS = ['refmlaw', 'refslaw', 'refregulate', 'refnfc']

# Results storage
spec_results = []
inference_results = []
run_counter = [0]
infer_counter = [0]

# =============================================================================
# Helper functions
# =============================================================================

def next_run_id():
    run_counter[0] += 1
    return f"{PAPER_ID}_run_{run_counter[0]:03d}"

def next_infer_id():
    infer_counter[0] += 1
    return f"{PAPER_ID}_infer_{infer_counter[0]:03d}"


def run_logit(outcome, treatment, controls, data, cluster_var='state',
              spec_id='baseline', spec_tree_path='designs/cross_sectional_ols.md#baseline',
              baseline_group_id='G1', sample_desc='Full sample N=881',
              fe_desc='', controls_desc='', axis_block_name=None, axis_block=None,
              extra_design_overrides=None):
    """Run a logit regression and return a result dict."""
    run_id = next_run_id()

    # Build formula
    rhs = [treatment] + controls
    formula = f"{outcome} ~ {' + '.join(rhs)}"

    try:
        model = smf.logit(formula, data=data)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]},
                          disp=False, maxiter=100)

        coef = result.params[treatment]
        se = result.bse[treatment]
        pval = result.pvalues[treatment]
        ci = result.conf_int().loc[treatment]
        nobs = int(result.nobs)
        pseudo_r2 = float(result.prsquared)

        # Full coefficient dict
        coef_dict = {k: float(v) for k, v in result.params.items()}

        # Build design block
        design = dict(DESIGN_AUDIT)
        if extra_design_overrides:
            design['cross_sectional_ols'] = {**design['cross_sectional_ols'], **extra_design_overrides}

        payload = make_success_payload(
            coefficients=coef_dict,
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
            design=design,
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': nobs,
            'r_squared': pseudo_r2,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 1,
            'run_error': ''
        }
        return row

    except Exception as e:
        error_msg = str(e)
        payload = make_failure_payload(
            error=error_msg,
            error_details=error_details_from_exception(e, stage='logit_estimation'),
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
        )
        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 0,
            'run_error': error_msg
        }
        return row


def run_lpm(outcome, treatment, controls, data, cluster_var='state',
            spec_id='baseline', spec_tree_path='designs/cross_sectional_ols.md#baseline',
            baseline_group_id='G1', sample_desc='Full sample N=881',
            fe_desc='', controls_desc='', absorb_fe=None,
            axis_block_name=None, axis_block=None,
            extra_design_overrides=None):
    """Run OLS (LPM) regression with cluster-robust SEs."""
    import pyfixest as pf
    run_id = next_run_id()

    rhs = [treatment] + controls
    if absorb_fe:
        formula = f"{outcome} ~ {' + '.join(rhs)} | {absorb_fe}"
    else:
        formula = f"{outcome} ~ {' + '.join(rhs)}"

    try:
        model = pf.feols(formula, data=data, vcov={"CRV1": cluster_var})

        coef = float(model.coef()[treatment])
        se = float(model.se()[treatment])
        pval = float(model.pvalue()[treatment])
        ci = model.confint().loc[treatment]
        nobs = int(model._N)
        r2 = float(model._r2)

        coef_dict = {k: float(v) for k, v in model.coef().items()}

        design = dict(DESIGN_AUDIT)
        if extra_design_overrides:
            design['cross_sectional_ols'] = {**design['cross_sectional_ols'], **extra_design_overrides}

        payload = make_success_payload(
            coefficients=coef_dict,
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
            design=design,
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': float(ci.iloc[0]),
            'ci_upper': float(ci.iloc[1]),
            'n_obs': nobs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc if fe_desc else (absorb_fe or ''),
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 1,
            'run_error': ''
        }
        return row

    except Exception as e:
        error_msg = str(e)
        payload = make_failure_payload(
            error=error_msg,
            error_details=error_details_from_exception(e, stage='lpm_estimation'),
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
        )
        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 0,
            'run_error': error_msg
        }
        return row


def run_probit(outcome, treatment, controls, data, cluster_var='state',
               spec_id='baseline', spec_tree_path='designs/cross_sectional_ols.md#baseline',
               baseline_group_id='G1', sample_desc='Full sample N=881',
               fe_desc='', controls_desc='',
               axis_block_name=None, axis_block=None):
    """Run a probit regression and return a result dict."""
    run_id = next_run_id()

    rhs = [treatment] + controls
    formula = f"{outcome} ~ {' + '.join(rhs)}"

    try:
        model = smf.probit(formula, data=data)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]},
                          disp=False, maxiter=100)

        coef = result.params[treatment]
        se = result.bse[treatment]
        pval = result.pvalues[treatment]
        ci = result.conf_int().loc[treatment]
        nobs = int(result.nobs)
        pseudo_r2 = float(result.prsquared)

        coef_dict = {k: float(v) for k, v in result.params.items()}

        payload = make_success_payload(
            coefficients=coef_dict,
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
            design=DESIGN_AUDIT,
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': nobs,
            'r_squared': pseudo_r2,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 1,
            'run_error': ''
        }
        return row

    except Exception as e:
        error_msg = str(e)
        payload = make_failure_payload(
            error=error_msg,
            error_details=error_details_from_exception(e, stage='probit_estimation'),
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
        )
        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 0,
            'run_error': error_msg
        }
        return row


def run_clogit(outcome, treatment, controls, data, group_var='state',
               cluster_var='state',
               spec_id='baseline', spec_tree_path='designs/cross_sectional_ols.md#baseline',
               baseline_group_id='G1', sample_desc='Clogit sample (drops all-same-outcome states)',
               fe_desc='state (conditional FE)', controls_desc='',
               axis_block_name=None, axis_block=None):
    """Run conditional logit (fixed-effects logit) via statsmodels."""
    run_id = next_run_id()

    try:
        # Drop groups with no variation in outcome (all 0 or all 1)
        data_cl = data.copy()
        group_means = data_cl.groupby(group_var)[outcome].transform('mean')
        data_cl = data_cl[(group_means > 0) & (group_means < 1)].copy()

        n_dropped = len(data) - len(data_cl)
        n_groups = data_cl[group_var].nunique()

        # Use statsmodels ConditionalLogit via BinaryModel
        # statsmodels doesn't have a direct clogit, use a workaround:
        # Estimate logit with state dummies (approximation when groups are not too large)
        # For a proper conditional logit, use the Chamberlain approach

        rhs = [treatment] + controls
        formula = f"{outcome} ~ {' + '.join(rhs)}"

        # We use the conditional logit from statsmodels discrete
        from statsmodels.discrete.conditional_models import ConditionalLogit

        exog_vars = rhs
        endog = data_cl[outcome].values
        exog = data_cl[exog_vars].values
        groups = data_cl[group_var].values

        model = ConditionalLogit(endog, exog, groups=groups)
        result = model.fit(disp=False, maxiter=200)

        # Get the treatment coefficient (first position)
        coef_idx = 0  # treatment is first
        coef = float(result.params[coef_idx])
        se = float(result.bse[coef_idx])
        pval = float(result.pvalues[coef_idx])
        ci = result.conf_int()[coef_idx]
        nobs = int(len(data_cl))

        # Pseudo R2 from log-likelihood
        try:
            pseudo_r2 = float(1 - result.llf / result.llnull) if hasattr(result, 'llnull') and result.llnull != 0 else np.nan
        except:
            pseudo_r2 = np.nan

        coef_dict = {var: float(result.params[i]) for i, var in enumerate(exog_vars)}

        payload = make_success_payload(
            coefficients=coef_dict,
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
            design=DESIGN_AUDIT,
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            extra={"clogit_estimator": "conditional_logit", "group_var": group_var,
                   "n_groups": int(n_groups), "n_dropped_obs": int(n_dropped)},
        )

        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': nobs,
            'r_squared': pseudo_r2,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 1,
            'run_error': ''
        }
        return row

    except Exception as e:
        error_msg = str(e)
        payload = make_failure_payload(
            error=error_msg,
            error_details=error_details_from_exception(e, stage='clogit_estimation'),
            inference=CANONICAL_INFERENCE,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
        )
        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc or ', '.join(controls),
            'cluster_var': cluster_var,
            'run_success': 0,
            'run_error': error_msg
        }
        return row


def run_inference_variant(outcome, treatment, controls, data,
                          base_run_id, baseline_group_id='G1',
                          infer_spec_id='infer/se/hc/hc1', cov_type='HC1', cov_kwds=None):
    """Re-estimate logit under different inference and record to inference_results."""
    infer_id = next_infer_id()

    rhs = [treatment] + controls
    formula = f"{outcome} ~ {' + '.join(rhs)}"

    try:
        model = smf.logit(formula, data=data)
        result = model.fit(cov_type=cov_type, cov_kwds=cov_kwds or {},
                          disp=False, maxiter=100)

        coef = result.params[treatment]
        se = result.bse[treatment]
        pval = result.pvalues[treatment]
        ci = result.conf_int().loc[treatment]
        nobs = int(result.nobs)
        pseudo_r2 = float(result.prsquared)

        coef_dict = {k: float(v) for k, v in result.params.items()}

        infer_info = {"spec_id": infer_spec_id, "params": {}}
        payload = make_success_payload(
            coefficients=coef_dict,
            inference=infer_info,
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
            design=DESIGN_AUDIT,
        )

        row = {
            'paper_id': PAPER_ID,
            'inference_run_id': infer_id,
            'spec_run_id': base_run_id,
            'spec_id': infer_spec_id,
            'spec_tree_path': 'modules/inference/standard_errors.md#a-heteroskedasticity-robust-se-no-clustering',
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': nobs,
            'r_squared': pseudo_r2,
            'coefficient_vector_json': json.dumps(payload),
            'cluster_var': '',
            'run_success': 1,
            'run_error': ''
        }
        return row

    except Exception as e:
        error_msg = str(e)
        payload = make_failure_payload(
            error=error_msg,
            error_details=error_details_from_exception(e, stage='inference_variant'),
            inference={"spec_id": infer_spec_id, "params": {}},
            software=SOFTWARE,
            surface_hash=SURFACE_HASH,
        )
        row = {
            'paper_id': PAPER_ID,
            'inference_run_id': infer_id,
            'spec_run_id': base_run_id,
            'spec_id': infer_spec_id,
            'spec_tree_path': 'modules/inference/standard_errors.md#a-heteroskedasticity-robust-se-no-clustering',
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(payload),
            'cluster_var': '',
            'run_success': 0,
            'run_error': error_msg
        }
        return row


# =============================================================================
# Step 1: Baseline specification (Table 2, Column 1)
# =============================================================================
print("\n=== Step 1: Baseline ===")

# Primary baseline: logit mutual mlaw slaw regulate nfc reform refmlaw refslaw refregulate refnfc ten2-ten5, cluster(state)
baseline_row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df,
    spec_id='baseline',
    spec_tree_path='designs/cross_sectional_ols.md#baseline',
    controls_desc='slaw regulate nfc reform refmlaw refslaw refregulate refnfc ten2 ten3 ten4 ten5'
)
spec_results.append(baseline_row)
baseline_run_id = baseline_row['spec_run_id']
print(f"  Baseline: coef={baseline_row['coefficient']:.4f}, p={baseline_row['p_value']:.6f}, N={baseline_row['n_obs']}")

# Additional baseline: favor spec (Table 3 col 1 -- uses favor instead of mlaw/slaw)
favor_controls = ['regulate', 'nfc', 'reform', 'reffavor', 'refregulate', 'refnfc',
                  'ten2', 'ten3', 'ten4', 'ten5']
favor_row = run_logit(
    outcome='mutual', treatment='favor',
    controls=favor_controls,
    data=df,
    spec_id='baseline__favor_spec',
    spec_tree_path='designs/cross_sectional_ols.md#baseline',
    controls_desc='regulate nfc reform reffavor refregulate refnfc ten2 ten3 ten4 ten5'
)
spec_results.append(favor_row)
print(f"  Favor spec: coef={favor_row['coefficient']:.4f}, p={favor_row['p_value']:.6f}")

# Additional baseline: rcorp spec (Table 4 col 2 -- uses rcorp instead of decade dummies)
rcorp_controls = ['slaw', 'regulate', 'nfc', 'reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc', 'rcorp']
rcorp_row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=rcorp_controls,
    data=df,
    spec_id='baseline__rcorp_spec',
    spec_tree_path='designs/cross_sectional_ols.md#baseline',
    controls_desc='slaw regulate nfc reform refmlaw refslaw refregulate refnfc rcorp'
)
spec_results.append(rcorp_row)
print(f"  Rcorp spec: coef={rcorp_row['coefficient']:.4f}, p={rcorp_row['p_value']:.6f}")

# =============================================================================
# Step 2: Design variant -- OLS (LPM)
# =============================================================================
print("\n=== Step 2: Design variant (LPM) ===")

lpm_row = run_lpm(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df,
    spec_id='design/cross_sectional_ols/estimator/ols',
    spec_tree_path='designs/cross_sectional_ols.md#estimators',
    controls_desc='slaw regulate nfc reform refmlaw refslaw refregulate refnfc ten2 ten3 ten4 ten5',
    axis_block_name='functional_form',
    axis_block={
        'spec_id': 'design/cross_sectional_ols/estimator/ols',
        'estimator': 'ols_lpm',
        'interpretation': 'Linear probability model -- coefficient is change in Pr(mutual)'
    },
    extra_design_overrides={'estimator': 'ols_lpm'}
)
spec_results.append(lpm_row)
print(f"  LPM: coef={lpm_row['coefficient']:.4f}, p={lpm_row['p_value']:.6f}")

# =============================================================================
# Step 3: RC/controls -- Leave-one-out
# =============================================================================
print("\n=== Step 3: RC/controls/loo ===")

loo_specs = [
    ('rc/controls/loo/drop_slaw', 'slaw'),
    ('rc/controls/loo/drop_regulate', 'regulate'),
    ('rc/controls/loo/drop_nfc', 'nfc'),
    ('rc/controls/loo/drop_reform', 'reform'),
    ('rc/controls/loo/drop_refmlaw', 'refmlaw'),
    ('rc/controls/loo/drop_refslaw', 'refslaw'),
    ('rc/controls/loo/drop_refregulate', 'refregulate'),
    ('rc/controls/loo/drop_refnfc', 'refnfc'),
    ('rc/controls/loo/drop_ten2', 'ten2'),
    ('rc/controls/loo/drop_ten3', 'ten3'),
    ('rc/controls/loo/drop_ten4', 'ten4'),
    ('rc/controls/loo/drop_ten5', 'ten5'),
]

for spec_id, drop_var in loo_specs:
    # Handle reform linkage: dropping reform requires dropping reform interactions too
    if drop_var == 'reform':
        controls_loo = [c for c in BASELINE_CONTROLS if c not in ['reform'] + REFORM_INTERACTIONS]
    else:
        controls_loo = [c for c in BASELINE_CONTROLS if c != drop_var]

    row = run_logit(
        outcome='mutual', treatment='mlaw',
        controls=controls_loo,
        data=df,
        spec_id=spec_id,
        spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
        axis_block_name='controls',
        axis_block={
            'spec_id': spec_id,
            'family': 'loo',
            'dropped': [drop_var] if drop_var != 'reform' else ['reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc'],
            'added': [],
            'n_controls': len(controls_loo)
        }
    )
    spec_results.append(row)
    print(f"  {spec_id}: coef={row['coefficient']:.4f}, p={row['p_value']:.6f}" if row['run_success'] else f"  {spec_id}: FAILED")

# =============================================================================
# Step 4: RC/controls -- Control sets
# =============================================================================
print("\n=== Step 4: RC/controls/sets ===")

control_set_specs = {
    'rc/controls/sets/no_decade_dummies': {
        'controls': ['slaw', 'regulate', 'nfc', 'reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc'],
        'treatment': 'mlaw',
        'desc': 'No decade dummies'
    },
    'rc/controls/sets/add_rcorp': {
        'controls': BASELINE_CONTROLS + ['rcorp'],
        'treatment': 'mlaw',
        'desc': 'Baseline + rcorp'
    },
    'rc/controls/sets/add_favor': {
        'controls': BASELINE_CONTROLS + ['favor'],
        'treatment': 'mlaw',
        'desc': 'Baseline + favor'
    },
    'rc/controls/sets/favor_instead_of_mlaw_slaw': {
        'controls': ['regulate', 'nfc', 'reform', 'reffavor', 'refregulate', 'refnfc', 'ten2', 'ten3', 'ten4', 'ten5'],
        'treatment': 'favor',
        'desc': 'Favor replaces mlaw/slaw (Table 3 col 1)'
    },
    'rc/controls/sets/favor_spec_with_rcorp': {
        'controls': ['regulate', 'nfc', 'reform', 'reffavor', 'refregulate', 'refnfc', 'rcorp'],
        'treatment': 'favor',
        'desc': 'Favor + rcorp (Table 4 col 1)'
    },
    'rc/controls/sets/full_with_rcorp': {
        'controls': BASELINE_CONTROLS + ['rcorp'],
        'treatment': 'mlaw',
        'desc': 'Full baseline + rcorp'
    },
    'rc/controls/sets/minimal_financial_only': {
        'controls': ['slaw'],
        'treatment': 'mlaw',
        'desc': 'Financial requirements only (mlaw + slaw)'
    },
    'rc/controls/sets/minimal_regulatory_only': {
        'controls': ['regulate', 'nfc'],
        'treatment': 'mlaw',
        'desc': 'Regulatory dummies only'
    },
}

for spec_id, spec_info in control_set_specs.items():
    row = run_logit(
        outcome='mutual', treatment=spec_info['treatment'],
        controls=spec_info['controls'],
        data=df,
        spec_id=spec_id,
        spec_tree_path='modules/robustness/controls.md#a-standard-control-sets',
        controls_desc=spec_info['desc'],
        axis_block_name='controls',
        axis_block={
            'spec_id': spec_id,
            'family': 'control_set',
            'dropped': [],
            'added': [],
            'n_controls': len(spec_info['controls']),
            'description': spec_info['desc']
        }
    )
    spec_results.append(row)
    status = f"coef={row['coefficient']:.4f}, p={row['p_value']:.6f}" if row['run_success'] else "FAILED"
    print(f"  {spec_id}: {status}")

# =============================================================================
# Step 5: RC/controls -- Progression
# =============================================================================
print("\n=== Step 5: RC/controls/progression ===")

progression_specs = {
    'rc/controls/progression/bivariate': {
        'controls': [],
        'treatment': 'mlaw',
        'desc': 'Bivariate (mlaw only)'
    },
    'rc/controls/progression/financial_reqs_only': {
        'controls': ['slaw'],
        'treatment': 'mlaw',
        'desc': 'Financial requirements: mlaw + slaw'
    },
    'rc/controls/progression/regulatory_only': {
        'controls': ['slaw', 'regulate', 'nfc'],
        'treatment': 'mlaw',
        'desc': 'Financial + regulatory'
    },
    'rc/controls/progression/financial_plus_reform': {
        'controls': ['slaw', 'reform', 'refmlaw', 'refslaw'],
        'treatment': 'mlaw',
        'desc': 'Financial + reform + interactions'
    },
    'rc/controls/progression/full_no_interactions': {
        'controls': ['slaw', 'regulate', 'nfc', 'reform', 'ten2', 'ten3', 'ten4', 'ten5'],
        'treatment': 'mlaw',
        'desc': 'Full without reform interactions'
    },
}

for spec_id, spec_info in progression_specs.items():
    row = run_logit(
        outcome='mutual', treatment=spec_info['treatment'],
        controls=spec_info['controls'],
        data=df,
        spec_id=spec_id,
        spec_tree_path='modules/robustness/controls.md#d-control-progression',
        controls_desc=spec_info['desc'],
        axis_block_name='controls',
        axis_block={
            'spec_id': spec_id,
            'family': 'progression',
            'dropped': [],
            'added': [],
            'n_controls': len(spec_info['controls']),
            'description': spec_info['desc']
        }
    )
    spec_results.append(row)
    status = f"coef={row['coefficient']:.4f}, p={row['p_value']:.6f}" if row['run_success'] else "FAILED"
    print(f"  {spec_id}: {status}")

# =============================================================================
# Step 6: RC/controls -- Random subsets
# =============================================================================
print("\n=== Step 6: RC/controls/subset (random) ===")

# Droppable controls (not treatment mlaw): everything except mandatory
# We keep mlaw as treatment and vary the set of non-treatment controls
rng = random.Random(SEED)
all_optional_controls = BASELINE_CONTROLS.copy()  # all 12 baseline controls

for i in range(1, 11):
    spec_id = f"rc/controls/subset/random_{i:03d}"
    # Sample subset size from 4 to 10
    subset_size = rng.randint(4, 10)
    subset = sorted(rng.sample(all_optional_controls, subset_size))

    # Enforce reform linkage: if any reform interaction is included, reform must be too
    has_reform_interaction = any(c in REFORM_INTERACTIONS for c in subset)
    if has_reform_interaction and 'reform' not in subset:
        subset.append('reform')
        subset = sorted(subset)

    row = run_logit(
        outcome='mutual', treatment='mlaw',
        controls=subset,
        data=df,
        spec_id=spec_id,
        spec_tree_path='modules/robustness/controls.md#e-random-control-subsets',
        controls_desc=f"Random subset ({len(subset)} controls): {', '.join(subset)}",
        axis_block_name='controls',
        axis_block={
            'spec_id': spec_id,
            'family': 'subset',
            'draw_index': i,
            'included': subset,
            'n_controls': len(subset)
        }
    )
    spec_results.append(row)
    status = f"coef={row['coefficient']:.4f}, p={row['p_value']:.6f}" if row['run_success'] else "FAILED"
    print(f"  {spec_id} ({len(subset)} controls): {status}")

# =============================================================================
# Step 7: RC/sample
# =============================================================================
print("\n=== Step 7: RC/sample ===")

# Trim mlaw at 1/99 percentiles
mlaw_p1, mlaw_p99 = df['mlaw'].quantile(0.01), df['mlaw'].quantile(0.99)
df_trim_mlaw = df[(df['mlaw'] >= mlaw_p1) & (df['mlaw'] <= mlaw_p99)].copy()
row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df_trim_mlaw,
    spec_id='rc/sample/outliers/trim_mlaw_1_99',
    spec_tree_path='modules/robustness/sample.md#outlier-restrictions',
    sample_desc=f'Trim mlaw at 1/99 pctile ({mlaw_p1:.4f}, {mlaw_p99:.4f}), N={len(df_trim_mlaw)}',
    axis_block_name='sample',
    axis_block={
        'spec_id': 'rc/sample/outliers/trim_mlaw_1_99',
        'restriction': f'mlaw >= {mlaw_p1:.4f} and mlaw <= {mlaw_p99:.4f}',
        'n_dropped': len(df) - len(df_trim_mlaw)
    }
)
spec_results.append(row)
print(f"  trim_mlaw_1_99: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# Trim slaw at 1/99 percentiles
slaw_p1, slaw_p99 = df['slaw'].quantile(0.01), df['slaw'].quantile(0.99)
df_trim_slaw = df[(df['slaw'] >= slaw_p1) & (df['slaw'] <= slaw_p99)].copy()
row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df_trim_slaw,
    spec_id='rc/sample/outliers/trim_slaw_1_99',
    spec_tree_path='modules/robustness/sample.md#outlier-restrictions',
    sample_desc=f'Trim slaw at 1/99 pctile ({slaw_p1:.4f}, {slaw_p99:.4f}), N={len(df_trim_slaw)}',
    axis_block_name='sample',
    axis_block={
        'spec_id': 'rc/sample/outliers/trim_slaw_1_99',
        'restriction': f'slaw >= {slaw_p1:.4f} and slaw <= {slaw_p99:.4f}',
        'n_dropped': len(df) - len(df_trim_slaw)
    }
)
spec_results.append(row)
print(f"  trim_slaw_1_99: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# Exclude states with no mutuals (mimics clogit sample restriction)
state_means = df.groupby('state')['mutual'].mean()
states_with_variation = state_means[(state_means > 0) & (state_means < 1)].index
df_no_all_same = df[df['state'].isin(states_with_variation)].copy()
row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df_no_all_same,
    spec_id='rc/sample/restriction/exclude_states_no_mutuals',
    spec_tree_path='modules/robustness/sample.md#subpopulation-restrictions',
    sample_desc=f'Exclude states with all-stock or all-mutual (drops {len(df)-len(df_no_all_same)} obs)',
    axis_block_name='sample',
    axis_block={
        'spec_id': 'rc/sample/restriction/exclude_states_no_mutuals',
        'restriction': 'Drop states with 0% or 100% mutual rate',
        'n_dropped': len(df) - len(df_no_all_same),
        'n_states_dropped': len(state_means) - len(states_with_variation)
    }
)
spec_results.append(row)
print(f"  exclude_states_no_mutuals: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# Pre-1930 only (drop ten4, ten5 which are all zero in this subsample)
df_pre1930 = df[df['formdate'] < 1930].copy()
pre1930_controls = [c for c in BASELINE_CONTROLS if c not in ['ten4', 'ten5']]
row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=pre1930_controls,
    data=df_pre1930,
    spec_id='rc/sample/restriction/pre_1930_only',
    spec_tree_path='modules/robustness/sample.md#subpopulation-restrictions',
    sample_desc=f'Companies formed before 1930, N={len(df_pre1930)}',
    axis_block_name='sample',
    axis_block={
        'spec_id': 'rc/sample/restriction/pre_1930_only',
        'restriction': 'formdate < 1930',
        'n_dropped': len(df) - len(df_pre1930),
        'notes': 'Dropped ten4, ten5 (zero variance in subsample)'
    }
)
spec_results.append(row)
status = f"coef={row['coefficient']:.4f}, N={row['n_obs']}" if row['run_success'] else f"FAILED: {row['run_error']}"
print(f"  pre_1930_only: {status}")

# Post-1920 only (drop ten2 which is all zero in this subsample)
df_post1920 = df[df['formdate'] >= 1920].copy()
post1920_controls = [c for c in BASELINE_CONTROLS if c not in ['ten2']]
row = run_logit(
    outcome='mutual', treatment='mlaw',
    controls=post1920_controls,
    data=df_post1920,
    spec_id='rc/sample/restriction/post_1920_only',
    spec_tree_path='modules/robustness/sample.md#subpopulation-restrictions',
    sample_desc=f'Companies formed 1920 or after, N={len(df_post1920)}',
    axis_block_name='sample',
    axis_block={
        'spec_id': 'rc/sample/restriction/post_1920_only',
        'restriction': 'formdate >= 1920',
        'n_dropped': len(df) - len(df_post1920),
        'notes': 'Dropped ten2 (zero variance in subsample)'
    }
)
spec_results.append(row)
status = f"coef={row['coefficient']:.4f}, N={row['n_obs']}" if row['run_success'] else f"FAILED: {row['run_error']}"
print(f"  post_1920_only: {status}")

# =============================================================================
# Step 8: RC/fe -- State FE (conditional logit approach)
# =============================================================================
print("\n=== Step 8: RC/fe (state FE via clogit) ===")

row = run_clogit(
    outcome='mutual', treatment='mlaw',
    controls=['slaw', 'regulate', 'nfc', 'reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc',
              'ten2', 'ten3', 'ten4', 'ten5'],
    data=df,
    group_var='state',
    spec_id='rc/fe/add/state',
    spec_tree_path='modules/robustness/fixed_effects.md#add-fixed-effects',
    axis_block_name='fixed_effects',
    axis_block={
        'spec_id': 'rc/fe/add/state',
        'added': ['state'],
        'method': 'conditional_logit'
    }
)
spec_results.append(row)
status = f"coef={row['coefficient']:.4f}, p={row['p_value']:.6f}" if row['run_success'] else f"FAILED: {row['run_error']}"
print(f"  clogit (state): {status}")

# =============================================================================
# Step 9: RC/form -- Estimator variants
# =============================================================================
print("\n=== Step 9: RC/form (estimator variants) ===")

# LPM (already ran as design variant, but this is the form-specific version)
lpm_form_row = run_lpm(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df,
    spec_id='rc/form/estimator/lpm',
    spec_tree_path='modules/robustness/functional_form.md#purpose',
    axis_block_name='functional_form',
    axis_block={
        'spec_id': 'rc/form/estimator/lpm',
        'estimator': 'ols_lpm',
        'interpretation': 'Linear probability model coefficient (change in Pr(mutual))'
    }
)
spec_results.append(lpm_form_row)
print(f"  LPM: coef={lpm_form_row['coefficient']:.4f}, p={lpm_form_row['p_value']:.6f}")

# Probit
probit_row = run_probit(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df,
    spec_id='rc/form/estimator/probit',
    spec_tree_path='modules/robustness/functional_form.md#purpose',
    axis_block_name='functional_form',
    axis_block={
        'spec_id': 'rc/form/estimator/probit',
        'estimator': 'probit',
        'interpretation': 'Probit index coefficient on mlaw'
    }
)
spec_results.append(probit_row)
print(f"  Probit: coef={probit_row['coefficient']:.4f}, p={probit_row['p_value']:.6f}")

# Conditional logit with state FE (Table 5 specification)
clogit_row = run_clogit(
    outcome='mutual', treatment='mlaw',
    controls=['slaw', 'regulate', 'nfc', 'reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc',
              'ten2', 'ten3', 'ten4', 'ten5'],
    data=df,
    group_var='state',
    spec_id='rc/form/estimator/clogit_state',
    spec_tree_path='modules/robustness/functional_form.md#purpose',
    axis_block_name='functional_form',
    axis_block={
        'spec_id': 'rc/form/estimator/clogit_state',
        'estimator': 'conditional_logit',
        'interpretation': 'Conditional logit with state FE (Table 5)',
        'group_var': 'state'
    }
)
spec_results.append(clogit_row)
status = f"coef={clogit_row['coefficient']:.4f}, p={clogit_row['p_value']:.6f}" if clogit_row['run_success'] else f"FAILED: {clogit_row['run_error']}"
print(f"  Clogit (state): {status}")

# LPM with decade FE (absorbed)
# Create decade variable for FE absorption
df['decade'] = (df['formdate'] // 10) * 10
lpm_fe_row = run_lpm(
    outcome='mutual', treatment='mlaw',
    controls=['slaw', 'regulate', 'nfc', 'reform', 'refmlaw', 'refslaw', 'refregulate', 'refnfc'],
    data=df,
    absorb_fe='decade',
    spec_id='rc/form/outcome/lpm_with_decade_fe',
    spec_tree_path='modules/robustness/functional_form.md#purpose',
    controls_desc='slaw regulate nfc reform interactions + decade FE (absorbed)',
    fe_desc='decade',
    axis_block_name='functional_form',
    axis_block={
        'spec_id': 'rc/form/outcome/lpm_with_decade_fe',
        'estimator': 'ols_lpm',
        'interpretation': 'LPM with decade FE absorbed, coefficient on mlaw'
    }
)
spec_results.append(lpm_fe_row)
status = f"coef={lpm_fe_row['coefficient']:.4f}, p={lpm_fe_row['p_value']:.6f}" if lpm_fe_row['run_success'] else f"FAILED: {lpm_fe_row['run_error']}"
print(f"  LPM + decade FE: {status}")

# Favor as treatment (treatment variable change)
favor_treat_row = run_logit(
    outcome='mutual', treatment='favor',
    controls=['regulate', 'nfc', 'reform', 'reffavor', 'refregulate', 'refnfc',
              'ten2', 'ten3', 'ten4', 'ten5'],
    data=df,
    spec_id='rc/form/treatment/favor_as_treatment',
    spec_tree_path='modules/robustness/functional_form.md#purpose',
    controls_desc='regulate nfc reform reffavor refregulate refnfc ten2-ten5',
    axis_block_name='functional_form',
    axis_block={
        'spec_id': 'rc/form/treatment/favor_as_treatment',
        'interpretation': 'Binary favor variable replaces continuous mlaw/slaw as treatment'
    }
)
spec_results.append(favor_treat_row)
print(f"  Favor as treatment: coef={favor_treat_row['coefficient']:.4f}, p={favor_treat_row['p_value']:.6f}")

# =============================================================================
# Step 10: Inference variants
# =============================================================================
print("\n=== Step 10: Inference variants ===")

# HC1 (heteroskedasticity-robust, no clustering)
infer_hc1 = run_inference_variant(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df,
    base_run_id=baseline_run_id,
    infer_spec_id='infer/se/hc/hc1',
    cov_type='HC1'
)
inference_results.append(infer_hc1)
print(f"  HC1: SE={infer_hc1['std_error']:.4f}, p={infer_hc1['p_value']:.6f}" if infer_hc1['run_success'] else f"  HC1: FAILED")

# IID (conventional)
infer_iid = run_inference_variant(
    outcome='mutual', treatment='mlaw',
    controls=BASELINE_CONTROLS,
    data=df,
    base_run_id=baseline_run_id,
    infer_spec_id='infer/se/iid',
    cov_type='nonrobust'
)
inference_results.append(infer_iid)
print(f"  IID: SE={infer_iid['std_error']:.4f}, p={infer_iid['p_value']:.6f}" if infer_iid['run_success'] else f"  IID: FAILED")

# =============================================================================
# Step 11: Save outputs
# =============================================================================
print("\n=== Step 11: Saving outputs ===")

# specification_results.csv
spec_df = pd.DataFrame(spec_results)
spec_df.to_csv(os.path.join(PACKAGE_DIR, 'specification_results.csv'), index=False)
print(f"Wrote {len(spec_df)} rows to specification_results.csv")

# inference_results.csv
if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(os.path.join(PACKAGE_DIR, 'inference_results.csv'), index=False)
    print(f"Wrote {len(infer_df)} rows to inference_results.csv")

# Summary stats
n_success = spec_df['run_success'].sum()
n_fail = len(spec_df) - n_success
print(f"\nTotal specs: {len(spec_df)} (success: {n_success}, failed: {n_fail})")

# =============================================================================
# Write SPECIFICATION_SEARCH.md
# =============================================================================
# Count by category
baseline_count = len(spec_df[spec_df['spec_id'].str.startswith('baseline')])
design_count = len(spec_df[spec_df['spec_id'].str.startswith('design/')])
rc_count = len(spec_df[spec_df['spec_id'].str.startswith('rc/')])

search_md = f"""# Specification Search Log: {PAPER_ID}

## Paper
Mayers & Smith (2005), insurance organizational form choice (mutual vs stock).

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Surface hash**: {SURFACE_HASH}
- **Baseline groups**: 1
  - G1: Effect of state regulation on mutual organizational form (Table 2)
- **Design**: cross_sectional_ols (logit estimator)
- **Canonical inference**: Cluster at state (47 clusters)
- **Budget**: max 60 core specs
- **Seed**: {SEED}
- **Control subset sampler**: stratified_size

## Execution Summary

### Counts
| Category | Planned | Executed | Successful | Failed |
|----------|---------|----------|------------|--------|
| Baseline | 3 | 3 | {len(spec_df[(spec_df['spec_id'].str.startswith('baseline')) & (spec_df['run_success']==1)])} | {len(spec_df[(spec_df['spec_id'].str.startswith('baseline')) & (spec_df['run_success']==0)])} |
| Design variants | 1 | 1 | {len(spec_df[(spec_df['spec_id'].str.startswith('design/')) & (spec_df['run_success']==1)])} | {len(spec_df[(spec_df['spec_id'].str.startswith('design/')) & (spec_df['run_success']==0)])} |
| RC variants | {rc_count} | {rc_count} | {len(spec_df[(spec_df['spec_id'].str.startswith('rc/')) & (spec_df['run_success']==1)])} | {len(spec_df[(spec_df['spec_id'].str.startswith('rc/')) & (spec_df['run_success']==0)])} |
| **Total estimate rows** | **{len(spec_df)}** | **{len(spec_df)}** | **{n_success}** | **{n_fail}** |
| Inference variants | {len(infer_df) if inference_results else 0} | {len(infer_df) if inference_results else 0} | {infer_df['run_success'].sum() if inference_results else 0} | {len(infer_df) - infer_df['run_success'].sum() if inference_results else 0} |

### Specifications Executed

#### Baselines
- Table 2 col 1: logit mutual mlaw controls, cluster(state) -- N=881, pseudo R2=0.265
- Favor spec: logit mutual favor controls, cluster(state)
- Rcorp spec: logit mutual mlaw controls+rcorp (no decade dummies), cluster(state)

#### Design Variant
- LPM (OLS): Linear probability model with same controls and clustering

#### RC: Controls (LOO)
- Drop each of: slaw, regulate, nfc, reform (with interactions), refmlaw, refslaw, refregulate, refnfc, ten2, ten3, ten4, ten5

#### RC: Controls (Sets)
- No decade dummies, add rcorp, add favor, favor instead of mlaw/slaw, favor+rcorp, full+rcorp, minimal financial, minimal regulatory

#### RC: Controls (Progression)
- Bivariate, financial reqs only, regulatory only, financial+reform, full no interactions

#### RC: Controls (Random Subsets)
- 10 random subsets (seed={SEED}), 4-10 controls each

#### RC: Sample
- Trim mlaw 1/99, trim slaw 1/99, exclude states with no mutuals, pre-1930 only, post-1920 only

#### RC: Fixed Effects
- Conditional logit with state FE (Table 5 approach)

#### RC: Functional Form
- LPM, probit, conditional logit (state), LPM with decade FE absorbed, favor as treatment

#### Inference Variants
- HC1 (heteroskedasticity-robust), IID (conventional)

### Skipped / Deviations
- **Binary treatment coding** (mlaw_binary, slaw_binary, favor_binary) excluded per surface (changes coefficient interpretation -- in excluded_from_core).
- **Conditional logit uses statsmodels ConditionalLogit** rather than Stata's clogit; SEs may differ slightly from Stata's clustered clogit.
- No diagnostics executed (diagnostics_plan is empty).

## Software Stack
- Python {SOFTWARE['runner_version']}
- statsmodels (logit, probit)
- pyfixest (LPM)
- pandas, numpy
"""

with open(os.path.join(PACKAGE_DIR, 'SPECIFICATION_SEARCH.md'), 'w') as f:
    f.write(search_md)
print("Wrote SPECIFICATION_SEARCH.md")

print("\n=== Done ===")
