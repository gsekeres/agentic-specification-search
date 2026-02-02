"""
Specification Search: 181581-V1
Paper: "When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality"
Author: Edward Okeke

This is a randomized controlled trial (RCT) studying the effect of adding doctors to
primary health care facilities in Nigeria on neonatal mortality.

Treatment Arms:
- MLP (mid-level provider) arm
- Doctor arm
- Control arm

Primary Outcome: 7-day neonatal mortality (mort7)
Clustering: Facility level (fid)
Fixed Effects: Strata, quarter

Method Classification: Cross-sectional OLS with fixed effects (RCT)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import json
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/181581-V1/data/analysis'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/181581-V1'

# Load main dataset
print("Loading data...")
child = pd.read_stata(f'{DATA_PATH}/child.dta', convert_categoricals=False)
print(f"Child data loaded: {child.shape[0]} observations")

# Paper metadata
PAPER_ID = "181581-V1"
JOURNAL = "AEJ-Applied"  # Based on typical AEA journal
PAPER_TITLE = "When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality"

# Define control variable sets following the original Stata code
# cont_ind: individual controls
cont_ind = ['cct', 'first', 'hausa', 'autonomy', 'car', 'last', 'gest']
# Note: magedum and mschool are categorical - we'll create dummies

# cont_base: baseline controls (individual + male)
cont_base_vars = cont_ind + ['male']

# cont_hc: health center controls
cont_hc = ['hc_deliveries', 'hc_cesarean', 'hc_transfusion']
# Note: hc_clean is categorical

# Extended controls
cont_extended = ['pastdeath', 'hc_workers', 'hc_open24hrs', 'hc_equipment',
                 'hc_beds', 'hc_lab', 'hc_drugs', 'hc_nopower', 'hc_vent']
# Note: hc_cond is categorical

# Prepare data
print("Preparing data...")

# Create dummy variables for categorical vars
child['magedum'] = child['magedum'].fillna(0).astype(int)
mage_dummies = pd.get_dummies(child['magedum'], prefix='magedum', drop_first=True)
child = pd.concat([child, mage_dummies], axis=1)
mage_cols = [c for c in mage_dummies.columns]

child['mschool'] = child['mschool'].fillna(0).astype(int)
mschool_dummies = pd.get_dummies(child['mschool'], prefix='mschool', drop_first=True)
child = pd.concat([child, mschool_dummies], axis=1)
mschool_cols = [c for c in mschool_dummies.columns]

child['hc_clean'] = child['hc_clean'].fillna(0).astype(int)
hc_clean_dummies = pd.get_dummies(child['hc_clean'], prefix='hc_clean', drop_first=True)
child = pd.concat([child, hc_clean_dummies], axis=1)
hc_clean_cols = [c for c in hc_clean_dummies.columns]

child['hc_cond'] = child['hc_cond'].fillna(0).astype(int)
hc_cond_dummies = pd.get_dummies(child['hc_cond'], prefix='hc_cond', drop_first=True)
child = pd.concat([child, hc_cond_dummies], axis=1)
hc_cond_cols = [c for c in hc_cond_dummies.columns]

# Create quarter numeric for FE
child['qtr_num'] = child['qtr'].factorize()[0]

# Strata as integer
child['strata'] = child['strata'].fillna(0).astype(int)

# Facility ID
child['fid'] = child['fid'].fillna(0).astype(int)

# Full control sets
cont_ind_full = cont_ind + mage_cols + mschool_cols
cont_base_full = cont_ind_full + ['male']
cont_hc_full = cont_hc + hc_clean_cols
cont_all_full = cont_base_full + cont_hc_full + ['pastdeath', 'hc_workers', 'hc_open24hrs',
                                                   'hc_equipment', 'hc_beds', 'hc_lab',
                                                   'hc_drugs', 'hc_nopower', 'hc_vent'] + hc_cond_cols

# Filter to valid control set
cont_ind_full = [c for c in cont_ind_full if c in child.columns]
cont_base_full = [c for c in cont_base_full if c in child.columns]
cont_hc_full = [c for c in cont_hc_full if c in child.columns]
cont_all_full = [c for c in cont_all_full if c in child.columns]

print(f"Control vars - Individual: {len(cont_ind_full)}, Base: {len(cont_base_full)}, HC: {len(cont_hc_full)}, All: {len(cont_all_full)}")

# Results storage
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    sample_desc, fixed_effects, controls_desc, cluster_var, model_type, df_used):
    """Extract results from pyfixest model"""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint().loc[treatment_var]
        n_obs = model.nobs
        r2 = model.r2

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(" + ") if fixed_effects else [],
            "diagnostics": {}
        }

        # Add control coefficients
        for var in model.coef().index:
            if var != treatment_var and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci.iloc[0]),
            'ci_upper': float(ci.iloc[1]),
            'n_obs': int(n_obs),
            'r_squared': float(r2) if r2 is not None else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error extracting results for {spec_id}: {e}")
        return None

def run_spec(formula, data, spec_id, spec_tree_path, outcome_var, treatment_var,
             sample_desc, fixed_effects, controls_desc, cluster_var='fid',
             model_type='OLS-FE', vcov_type=None):
    """Run a specification and store results"""
    try:
        if vcov_type is None:
            vcov_type = {'CRV1': cluster_var}
        model = pf.feols(formula, data=data, vcov=vcov_type)
        result = extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                                sample_desc, fixed_effects, controls_desc, cluster_var, model_type, data)
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.6f}, se={result['std_error']:.6f}, p={result['p_value']:.4f}, n={result['n_obs']}")
            return result
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
    return None

# ============================================================================
# BASELINE SPECIFICATIONS (replicating Table 4)
# ============================================================================
print("\n" + "="*80)
print("BASELINE SPECIFICATIONS (Table 4 replication)")
print("="*80)

# Baseline 1: No controls, strata FE only
print("\n--- Baseline: No controls ---")
run_spec("mort7 ~ mlp + doctor | strata", child,
         "baseline", "methods/cross_sectional_ols.md#baseline",
         "mort7", "doctor", "Full sample", "strata", "None", "fid")

# Add MLP coefficient separately
run_spec("mort7 ~ mlp + doctor | strata", child,
         "baseline_mlp", "methods/cross_sectional_ols.md#baseline",
         "mort7", "mlp", "Full sample", "strata", "None", "fid")

# Baseline 2: Basic controls (Table 4 Column 2)
print("\n--- Baseline: Basic controls ---")
controls_str = " + ".join(cont_base_full)
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", child,
         "baseline_basic_controls", "methods/cross_sectional_ols.md#baseline",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls", "fid")

# Baseline 3: Extended controls (Table 4 Column 3)
print("\n--- Baseline: Extended controls ---")
hc_controls_str = " + ".join(cont_hc_full)
run_spec(f"mort7 ~ mlp + doctor + {controls_str} + {hc_controls_str} | strata + qtr_num", child,
         "baseline_extended_controls", "methods/cross_sectional_ols.md#baseline",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Extended controls", "fid")

# ============================================================================
# CONTROL VARIABLE VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("CONTROL VARIABLE VARIATIONS")
print("="*80)

# No controls at all (bivariate)
print("\n--- No controls (bivariate) ---")
run_spec("mort7 ~ mlp + doctor | strata", child,
         "robust/control/none", "robustness/control_progression.md",
         "mort7", "doctor", "Full sample", "strata", "No controls", "fid")

# Add controls incrementally
print("\n--- Incremental control addition ---")
for i, ctrl in enumerate(cont_base_full[:8]):  # First 8 controls
    ctrl_list = cont_base_full[:i+1]
    ctrl_str = " + ".join(ctrl_list)
    run_spec(f"mort7 ~ mlp + doctor + {ctrl_str} | strata + qtr_num", child,
             f"robust/control/add_{ctrl}", "robustness/control_progression.md",
             "mort7", "doctor", "Full sample", "strata + qtr_num", f"Controls up to {ctrl}", "fid")

# Leave-one-out control variations
print("\n--- Leave-one-out controls ---")
full_controls = cont_base_full + cont_hc_full[:4]  # Use reasonable subset
full_ctrl_str = " + ".join(full_controls)
for ctrl in full_controls[:10]:  # First 10 controls
    remaining = [c for c in full_controls if c != ctrl]
    if remaining:
        remaining_str = " + ".join(remaining)
        run_spec(f"mort7 ~ mlp + doctor + {remaining_str} | strata + qtr_num", child,
                 f"robust/control/drop_{ctrl}", "robustness/leave_one_out.md",
                 "mort7", "doctor", "Full sample", "strata + qtr_num", f"Drop {ctrl}", "fid")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*80)
print("SAMPLE RESTRICTIONS")
print("="*80)

# By quarter
print("\n--- By quarter ---")
for q in child['qtr_num'].unique():
    if child[child['qtr_num'] != q].shape[0] > 100:
        run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata",
                 child[child['qtr_num'] != q],
                 f"robust/sample/drop_qtr_{int(q)}", "robustness/sample_restrictions.md",
                 "mort7", "doctor", f"Drop quarter {int(q)}", "strata", "Basic controls", "fid")

# By gender
print("\n--- By gender ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['male'] == 1],
         "robust/sample/male_only", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Male infants only", "strata + qtr_num", "Basic controls", "fid")

run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['male'] == 0],
         "robust/sample/female_only", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Female infants only", "strata + qtr_num", "Basic controls", "fid")

# By first birth
print("\n--- By birth order ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['first'] == 1],
         "robust/sample/first_birth", "robustness/sample_restrictions.md",
         "mort7", "doctor", "First births only", "strata + qtr_num", "Basic controls", "fid")

run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['first'] == 0],
         "robust/sample/not_first_birth", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Not first births", "strata + qtr_num", "Basic controls", "fid")

# By CCT (conditional cash transfer)
print("\n--- By CCT status ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['cct'] == 1],
         "robust/sample/cct_yes", "robustness/sample_restrictions.md",
         "mort7", "doctor", "CCT recipients only", "strata + qtr_num", "Basic controls", "fid")

run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['cct'] == 0],
         "robust/sample/cct_no", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Non-CCT only", "strata + qtr_num", "Basic controls", "fid")

# By ethnicity
print("\n--- By ethnicity ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['hausa'] == 1],
         "robust/sample/hausa", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Hausa ethnicity", "strata + qtr_num", "Basic controls", "fid")

run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['hausa'] == 0],
         "robust/sample/non_hausa", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Non-Hausa ethnicity", "strata + qtr_num", "Basic controls", "fid")

# By multiple birth
print("\n--- By multiple birth ---")
if 'multiple' in child.columns:
    run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
             child[child['multiple'] == 0],
             "robust/sample/singleton", "robustness/sample_restrictions.md",
             "mort7", "doctor", "Singleton births only", "strata + qtr_num", "Basic controls", "fid")

# By cesarean delivery
print("\n--- By cesarean status ---")
if 'csection' in child.columns:
    run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
             child[child['csection'] == 0],
             "robust/sample/no_csection", "robustness/sample_restrictions.md",
             "mort7", "doctor", "Vaginal births only", "strata + qtr_num", "Basic controls", "fid")

# ============================================================================
# ALTERNATIVE OUTCOMES
# ============================================================================
print("\n" + "="*80)
print("ALTERNATIVE OUTCOMES")
print("="*80)

# 30-day mortality
if 'mort30' in child.columns:
    print("\n--- 30-day mortality ---")
    run_spec(f"mort30 ~ mlp + doctor + {controls_str} | strata + qtr_num", child,
             "robust/outcome/mort30", "robustness/measurement.md",
             "mort30", "doctor", "Full sample", "strata + qtr_num", "Basic controls", "fid")

    run_spec(f"mort30 ~ mlp + doctor | strata", child,
             "robust/outcome/mort30_nocontrols", "robustness/measurement.md",
             "mort30", "doctor", "Full sample", "strata", "No controls", "fid")

# ============================================================================
# INFERENCE VARIATIONS (CLUSTERING)
# ============================================================================
print("\n" + "="*80)
print("INFERENCE VARIATIONS")
print("="*80)

# Different clustering levels
print("\n--- Clustering variations ---")

# Robust SE (no clustering)
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", child,
         "robust/cluster/robust_hc1", "robustness/clustering_variations.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls", "robust_se",
         model_type="OLS-FE", vcov_type='hetero')

# Cluster at strata level
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", child,
         "robust/cluster/strata", "robustness/clustering_variations.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls", "strata",
         vcov_type={'CRV1': 'strata'})

# Two-way clustering (facility and quarter)
try:
    run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", child,
             "robust/cluster/twoway_fid_qtr", "robustness/clustering_variations.md",
             "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls", "fid+qtr_num",
             vcov_type={'CRV1': ['fid', 'qtr_num']})
except:
    print("  Two-way clustering failed")

# ============================================================================
# ESTIMATION METHOD VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("ESTIMATION METHOD VARIATIONS")
print("="*80)

# No fixed effects
print("\n--- Fixed effects variations ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str}", child,
         "robust/estimation/no_fe", "robustness/model_specification.md",
         "mort7", "doctor", "Full sample", "None", "Basic controls", "fid")

# Only strata FE
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata", child,
         "robust/estimation/strata_fe_only", "robustness/model_specification.md",
         "mort7", "doctor", "Full sample", "strata", "Basic controls", "fid")

# Only quarter FE
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | qtr_num", child,
         "robust/estimation/qtr_fe_only", "robustness/model_specification.md",
         "mort7", "doctor", "Full sample", "qtr_num", "Basic controls", "fid")

# Strata + Quarter FE (standard)
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", child,
         "robust/estimation/strata_qtr_fe", "robustness/model_specification.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls", "fid")

# Facility FE
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | fid", child,
         "robust/estimation/fid_fe", "robustness/model_specification.md",
         "mort7", "doctor", "Full sample", "fid", "Basic controls", "fid")

# ============================================================================
# TREATMENT VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("TREATMENT VARIATIONS")
print("="*80)

# MLP only (vs control)
print("\n--- Treatment arm comparisons ---")
child_mlp_control = child[child['doctor'] == 0].copy()
run_spec(f"mort7 ~ mlp + {controls_str} | strata + qtr_num", child_mlp_control,
         "robust/treatment/mlp_vs_control", "robustness/measurement.md",
         "mort7", "mlp", "MLP vs Control only", "strata + qtr_num", "Basic controls", "fid")

# Doctor only (vs control)
child_doc_control = child[child['mlp'] == 0].copy()
run_spec(f"mort7 ~ doctor + {controls_str} | strata + qtr_num", child_doc_control,
         "robust/treatment/doctor_vs_control", "robustness/measurement.md",
         "mort7", "doctor", "Doctor vs Control only", "strata + qtr_num", "Basic controls", "fid")

# Combined treatment (any treatment)
child['any_treatment'] = (child['mlp'] + child['doctor']).clip(0, 1)
run_spec(f"mort7 ~ any_treatment + {controls_str} | strata + qtr_num", child,
         "robust/treatment/any_treatment", "robustness/measurement.md",
         "mort7", "any_treatment", "Any treatment vs Control", "strata + qtr_num", "Basic controls", "fid")

# By dosage (Table 5)
print("\n--- By dosage ---")
if 'dose' in child.columns:
    # Low dose
    child_low_dose = child[child['dose'] == 1].copy()
    if child_low_dose.shape[0] > 100:
        run_spec(f"mort7 ~ doctor + {controls_str} | strata + qtr_num", child_low_dose,
                 "robust/treatment/low_dose", "robustness/sample_restrictions.md",
                 "mort7", "doctor", "Low dose only", "strata + qtr_num", "Basic controls", "fid")

    # High dose
    child_high_dose = child[child['dose'] == 2].copy()
    if child_high_dose.shape[0] > 100:
        run_spec(f"mort7 ~ doctor + {controls_str} | strata + qtr_num", child_high_dose,
                 "robust/treatment/high_dose", "robustness/sample_restrictions.md",
                 "mort7", "doctor", "High dose only", "strata + qtr_num", "Basic controls", "fid")

# ============================================================================
# HETEROGENEITY ANALYSES
# ============================================================================
print("\n" + "="*80)
print("HETEROGENEITY ANALYSES")
print("="*80)

# Gender interaction
print("\n--- Heterogeneity: Gender ---")
run_spec(f"mort7 ~ doctor*male + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/male", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + male interaction", "fid")

# CCT interaction
print("\n--- Heterogeneity: CCT ---")
run_spec(f"mort7 ~ doctor*cct + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/cct", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + CCT interaction", "fid")

# Ethnicity interaction
print("\n--- Heterogeneity: Ethnicity ---")
run_spec(f"mort7 ~ doctor*hausa + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/hausa", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + Hausa interaction", "fid")

# First birth interaction
print("\n--- Heterogeneity: First birth ---")
run_spec(f"mort7 ~ doctor*first + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/first_birth", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + first birth interaction", "fid")

# Cesarean interaction
print("\n--- Heterogeneity: Cesarean ---")
if 'csection' in child.columns:
    run_spec(f"mort7 ~ doctor*csection + mlp + {controls_str} | strata + qtr_num", child,
             "robust/heterogeneity/csection", "robustness/heterogeneity.md",
             "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + cesarean interaction", "fid")

# Multiple birth interaction
if 'multiple' in child.columns:
    run_spec(f"mort7 ~ doctor*multiple + mlp + {controls_str} | strata + qtr_num", child,
             "robust/heterogeneity/multiple", "robustness/heterogeneity.md",
             "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + multiple birth interaction", "fid")

# Facility characteristics interactions
print("\n--- Heterogeneity: Facility characteristics ---")
run_spec(f"mort7 ~ doctor*hc_cesarean + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/hc_cesarean", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + HC cesarean interaction", "fid")

run_spec(f"mort7 ~ doctor*hc_open24hrs + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/hc_open24hrs", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + HC 24hr interaction", "fid")

run_spec(f"mort7 ~ doctor*hc_lab + mlp + {controls_str} | strata + qtr_num", child,
         "robust/heterogeneity/hc_lab", "robustness/heterogeneity.md",
         "mort7", "doctor", "Full sample", "strata + qtr_num", "Basic controls + HC lab interaction", "fid")

# ============================================================================
# PLACEBO AND FALSIFICATION TESTS
# ============================================================================
print("\n" + "="*80)
print("PLACEBO AND FALSIFICATION TESTS")
print("="*80)

# Placebo outcomes that shouldn't be affected
print("\n--- Placebo outcomes ---")
# Gender should not be affected by treatment
run_spec(f"male ~ mlp + doctor | strata", child,
         "robust/placebo/male_outcome", "robustness/placebo_tests.md",
         "male", "doctor", "Full sample - placebo", "strata", "None", "fid")

# Prior births shouldn't be affected
if 'priorb' in child.columns:
    run_spec(f"priorb ~ mlp + doctor | strata", child,
             "robust/placebo/priorb_outcome", "robustness/placebo_tests.md",
             "priorb", "doctor", "Full sample - placebo", "strata", "None", "fid")

# ============================================================================
# ADDITIONAL ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL ROBUSTNESS CHECKS")
print("="*80)

# By state
print("\n--- By state ---")
states = child['state'].unique()
for st in states[:5]:  # Limit to first 5 states
    df_state = child[child['state'] == st]
    if df_state.shape[0] > 100 and df_state['mort7'].sum() > 5:
        run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", df_state,
                 f"robust/sample/state_{int(st)}", "robustness/sample_restrictions.md",
                 "mort7", "doctor", f"State {int(st)} only", "strata + qtr_num", "Basic controls", "fid")

# By mother's education
print("\n--- By mother's education ---")
for edu in child['mschool'].unique():
    df_edu = child[child['mschool'] == edu]
    if df_edu.shape[0] > 200:
        run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", df_edu,
                 f"robust/sample/mschool_{int(edu)}", "robustness/sample_restrictions.md",
                 "mort7", "doctor", f"Mother education level {int(edu)}", "strata + qtr_num", "Basic controls", "fid")

# By mother's age
print("\n--- By mother's age ---")
for age_cat in child['magedum'].unique():
    df_age = child[child['magedum'] == age_cat]
    if df_age.shape[0] > 200:
        run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", df_age,
                 f"robust/sample/magedum_{int(age_cat)}", "robustness/sample_restrictions.md",
                 "mort7", "doctor", f"Mother age category {int(age_cat)}", "strata + qtr_num", "Basic controls", "fid")

# By gestational age
print("\n--- By gestational age ---")
for gest_cat in child['gest'].unique():
    df_gest = child[child['gest'] == gest_cat]
    if df_gest.shape[0] > 200:
        run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num", df_gest,
                 f"robust/sample/gest_{int(gest_cat)}", "robustness/sample_restrictions.md",
                 "mort7", "doctor", f"Gestational age category {int(gest_cat)}", "strata + qtr_num", "Basic controls", "fid")

# By autonomy
print("\n--- By autonomy ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['autonomy'] == 1],
         "robust/sample/high_autonomy", "robustness/sample_restrictions.md",
         "mort7", "doctor", "High autonomy mothers", "strata + qtr_num", "Basic controls", "fid")

run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['autonomy'] == 0],
         "robust/sample/low_autonomy", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Low autonomy mothers", "strata + qtr_num", "Basic controls", "fid")

# By past child death
print("\n--- By past child death ---")
run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['pastdeath'] == 1],
         "robust/sample/pastdeath_yes", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Mothers with past child death", "strata + qtr_num", "Basic controls", "fid")

run_spec(f"mort7 ~ mlp + doctor + {controls_str} | strata + qtr_num",
         child[child['pastdeath'] == 0],
         "robust/sample/pastdeath_no", "robustness/sample_restrictions.md",
         "mort7", "doctor", "Mothers without past child death", "strata + qtr_num", "Basic controls", "fid")

# ============================================================================
# LINEAR PROBABILITY MODEL VS PROBIT
# ============================================================================
print("\n" + "="*80)
print("LPM VS PROBIT COMPARISON")
print("="*80)

# Probit model (using statsmodels)
print("\n--- Probit estimation ---")
try:
    # Prepare data for probit
    probit_controls = [c for c in cont_base_full if c in child.columns]
    X_probit = child[['mlp', 'doctor'] + probit_controls].copy()
    X_probit = sm.add_constant(X_probit)
    y_probit = child['mort7'].copy()

    # Drop missing
    valid_idx = X_probit.notna().all(axis=1) & y_probit.notna()
    X_probit = X_probit[valid_idx]
    y_probit = y_probit[valid_idx]

    probit_model = sm.Probit(y_probit, X_probit).fit(disp=0)

    # Extract marginal effects
    mfx = probit_model.get_margeff()

    coef_doctor = mfx.margeff[list(X_probit.columns).index('doctor') - 1]  # -1 for const
    se_doctor = mfx.margeff_se[list(X_probit.columns).index('doctor') - 1]
    pval_doctor = mfx.pvalues[list(X_probit.columns).index('doctor') - 1]

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/estimation/probit_mfx',
        'spec_tree_path': 'robustness/model_specification.md',
        'outcome_var': 'mort7',
        'treatment_var': 'doctor',
        'coefficient': float(coef_doctor),
        'std_error': float(se_doctor),
        't_stat': float(coef_doctor / se_doctor),
        'p_value': float(pval_doctor),
        'ci_lower': float(coef_doctor - 1.96 * se_doctor),
        'ci_upper': float(coef_doctor + 1.96 * se_doctor),
        'n_obs': int(probit_model.nobs),
        'r_squared': float(probit_model.prsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "doctor", "coef": float(coef_doctor), "se": float(se_doctor)}}),
        'sample_desc': 'Full sample',
        'fixed_effects': 'None (controls for strata)',
        'controls_desc': 'Basic controls',
        'cluster_var': 'None',
        'model_type': 'Probit (marginal effects)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  robust/estimation/probit_mfx: coef={coef_doctor:.6f}, se={se_doctor:.6f}, p={pval_doctor:.4f}")
except Exception as e:
    print(f"  Probit model failed: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Summary statistics
print(f"\nTotal specifications run: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.6f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.6f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.6f}, {results_df['coefficient'].max():.6f}]")

# Save to CSV
output_file = f'{OUTPUT_PATH}/specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\nDone!")
