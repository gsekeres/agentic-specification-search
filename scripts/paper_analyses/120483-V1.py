#!/usr/bin/env python3
"""
Specification Search: 120483-V1
Paper: Malaria and Slavery in the Early United States (AEJ: Applied Economics)

This paper examines the effect of malaria ecology on the geographic distribution
of slavery in the early United States. The main hypothesis is that malaria ecology
affected the relative productivity of African vs. European labor, thereby influencing
the adoption of slavery.

Method: Cross-sectional OLS with state fixed effects (county-level analysis)
        + Panel data analysis (state-level over time)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/120483-V1/AEJApp-2019-0372/dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/120483-V1"

# Load datasets
df_1790 = pd.read_stata(f"{DATA_DIR}/county_1790.dta")
df_1860 = pd.read_stata(f"{DATA_DIR}/county_1860.dta")
df_panel = pd.read_stata(f"{DATA_DIR}/panel_states.dta")
df_votes = pd.read_stata(f"{DATA_DIR}/county_votes.dta")
df_conv = pd.read_stata(f"{DATA_DIR}/county_convention.dta")
df_americas = pd.read_stata(f"{DATA_DIR}/americas.dta")

# Results container
results = []

# Paper metadata
PAPER_ID = "120483-V1"
JOURNAL = "AEJ-Applied"
PAPER_TITLE = "Malaria Ecology and the Spread of Slavery in the Early United States"

# Control variable sets
CROP_1790 = ['cotton', 'rice', 'sugar', 'tea', 'tobacco', 'indigo']
GEO_1790 = ['DISTSEA', 'DISTRIV', 'prec', 'temp', 'ELEV', 'lat_deg', 'long_deg', 'lat_long']
CROP_1860 = ['cotton', 'coffee', 'rice', 'sugar', 'tea', 'tobacco', 'indigo']
GEO_1860 = ['ELEV', 'prec', 'temp', 'DISTRIV', 'DISTSEA', 'lat_deg', 'long_deg', 'lat_long']
GEO_1860_SHORT = ['ELEV', 'prec', 'temp', 'DISTRIV', 'DISTSEA']

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var, model_type, df):
    """Extract results from a pyfixest model."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        t_stat = model.tstat()[treatment_var]
        p_val = model.pvalue()[treatment_var]

        # Get confidence intervals from tidy output
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%'] if treatment_var in ci.index else coef - 1.96 * se
        ci_upper = ci.loc[treatment_var, '97.5%'] if treatment_var in ci.index else coef + 1.96 * se

        n_obs = model._N
        r_squared = model._r2

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(p_val)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(" + ") if fixed_effects else [],
            "diagnostics": {}
        }

        # Add control coefficients
        for var in model.coef().index:
            if var != treatment_var and not var.startswith('C('):
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
            'coefficient': coef,
            'std_error': se,
            't_stat': t_stat,
            'p_value': p_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r_squared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

def run_spec(df, formula, spec_id, spec_tree_path, outcome_var, treatment_var,
            sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
            vcov_type='CRV1', vcov_var=None):
    """Run a specification and extract results."""
    try:
        if vcov_var:
            model = pf.feols(formula, data=df, vcov={vcov_type: vcov_var})
        else:
            model = pf.feols(formula, data=df, vcov='hetero')

        result = extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                                sample_desc, fixed_effects, controls_desc, cluster_var, model_type, df)
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
        return model
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None

print("=" * 60)
print("SPECIFICATION SEARCH: 120483-V1")
print("Malaria Ecology and Slavery in the Early United States")
print("=" * 60)

# ============================================================================
# SECTION 1: BASELINE SPECIFICATIONS (Table 1 replications)
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 1: BASELINE SPECIFICATIONS")
print("=" * 60)

# Baseline 1: 1790 data, no controls
print("\n1.1: 1790 Data - Minimal to Full Controls")
run_spec(df_1790, "slaveratio ~ MAL | state_g",
         "baseline", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "No controls", "state_g", "OLS-FE", "CRV1", "state_g")

# Baseline 2: 1790 with crop controls
run_spec(df_1790, f"slaveratio ~ MAL + {' + '.join(CROP_1790)} | state_g",
         "baseline_crop", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "Crop suitability", "state_g", "OLS-FE", "CRV1", "state_g")

# Baseline 3: 1790 with full controls (main specification)
run_spec(df_1790, f"slaveratio ~ MAL + {' + '.join(CROP_1790)} + {' + '.join(GEO_1790)} | state_g",
         "baseline_full", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "Crop suitability + Geography", "state_g", "OLS-FE", "CRV1", "state_g")

# 1860 Data specifications
print("\n1.2: 1860 Data - Minimal to Full Controls")
run_spec(df_1860, "slaveratio ~ MAL | state_g",
         "baseline_1860", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "No controls", "state_g", "OLS-FE", "CRV1", "state_g")

run_spec(df_1860, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "baseline_1860_full", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Crop suitability + Geography", "state_g", "OLS-FE", "CRV1", "state_g")

# Slave states only
print("\n1.3: Slave States Only")
df_1790_slave = df_1790[df_1790['slave_state'] == 1].copy()
run_spec(df_1790_slave, f"slaveratio ~ MAL + {' + '.join(CROP_1790)} + {' + '.join(GEO_1790)} | state_g",
         "baseline_slave_states_1790", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "Slave states 1790", "state_g",
         "Crop suitability + Geography", "state_g", "OLS-FE", "CRV1", "state_g")

df_1860_slave = df_1860[df_1860['slave_state'] == 1].copy()
run_spec(df_1860_slave, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "baseline_slave_states_1860", "methods/cross_sectional_ols.md#baseline",
         "slaveratio", "MAL", "Slave states 1860", "state_g",
         "Crop suitability + Geography", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 2: CONTROL VARIATIONS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 2: CONTROL VARIATIONS")
print("=" * 60)

# 2.1: Leave-one-out controls (1790)
print("\n2.1: Leave-One-Out Controls (1790)")
all_controls_1790 = CROP_1790 + GEO_1790
for control in all_controls_1790:
    remaining = [c for c in all_controls_1790 if c != control]
    formula = f"slaveratio ~ MAL + {' + '.join(remaining)} | state_g"
    run_spec(df_1790, formula,
             f"robust/control/drop_{control}", "robustness/control_progression.md",
             "slaveratio", "MAL", "All counties 1790", "state_g",
             f"Full controls minus {control}", "state_g", "OLS-FE", "CRV1", "state_g")

# 2.2: Incrementally add controls
print("\n2.2: Incrementally Add Controls (1790)")
# No controls
run_spec(df_1790, "slaveratio ~ MAL | state_g",
         "robust/control/none", "robustness/control_progression.md",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "No controls", "state_g", "OLS-FE", "CRV1", "state_g")

# Add crops only
for i, crop in enumerate(CROP_1790):
    controls_so_far = CROP_1790[:i+1]
    formula = f"slaveratio ~ MAL + {' + '.join(controls_so_far)} | state_g"
    run_spec(df_1790, formula,
             f"robust/control/add_{crop}", "robustness/control_progression.md",
             "slaveratio", "MAL", "All counties 1790", "state_g",
             f"Adding {crop}", "state_g", "OLS-FE", "CRV1", "state_g")

# 2.3: Alternative control sets
print("\n2.3: Alternative Control Sets")
# Only geography
run_spec(df_1790, f"slaveratio ~ MAL + {' + '.join(GEO_1790)} | state_g",
         "robust/control/geo_only", "robustness/control_progression.md",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "Geography only", "state_g", "OLS-FE", "CRV1", "state_g")

# Only crops
run_spec(df_1790, f"slaveratio ~ MAL + {' + '.join(CROP_1790)} | state_g",
         "robust/control/crop_only", "robustness/control_progression.md",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "Crops only", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 3: SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 3: SAMPLE RESTRICTIONS")
print("=" * 60)

# 3.1: Outlier handling
print("\n3.1: Outlier Handling")
for pct in [1, 5, 10]:
    df_trim = df_1860[
        (df_1860['slaveratio'] >= df_1860['slaveratio'].quantile(pct/100)) &
        (df_1860['slaveratio'] <= df_1860['slaveratio'].quantile(1 - pct/100))
    ].copy()
    run_spec(df_trim, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
             f"robust/sample/trim_{pct}pct", "robustness/sample_restrictions.md",
             "slaveratio", "MAL", f"Trimmed {pct}% tails 1860", "state_g",
             "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# Winsorize
for pct in [1, 5]:
    df_wins = df_1860.copy()
    lower = df_wins['slaveratio'].quantile(pct/100)
    upper = df_wins['slaveratio'].quantile(1 - pct/100)
    df_wins['slaveratio'] = df_wins['slaveratio'].clip(lower=lower, upper=upper)
    run_spec(df_wins, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
             f"robust/sample/winsor_{pct}pct", "robustness/sample_restrictions.md",
             "slaveratio", "MAL", f"Winsorized {pct}% 1860", "state_g",
             "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# 3.2: Geographic restrictions - drop states one at a time
print("\n3.2: Leave-One-State-Out")
states = df_1860['state_g'].unique()
for state in states[:10]:  # Top 10 states to keep spec count manageable
    df_sub = df_1860[df_1860['state_g'] != state].copy()
    run_spec(df_sub, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
             f"robust/sample/drop_state_{int(state)}", "robustness/sample_restrictions.md",
             "slaveratio", "MAL", f"Drop state {int(state)}", "state_g",
             "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# 3.3: High vs low malaria regions
print("\n3.3: Subgroup by Malaria Level")
df_high_mal = df_1860[df_1860['MAL'] > df_1860['MAL'].median()].copy()
df_low_mal = df_1860[df_1860['MAL'] <= df_1860['MAL'].median()].copy()

run_spec(df_high_mal, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/sample/high_malaria", "robustness/sample_restrictions.md",
         "slaveratio", "MAL", "High malaria counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

run_spec(df_low_mal, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/sample/low_malaria", "robustness/sample_restrictions.md",
         "slaveratio", "MAL", "Low malaria counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 4: INFERENCE VARIATIONS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 4: INFERENCE VARIATIONS")
print("=" * 60)

# 4.1: Different clustering
print("\n4.1: Clustering Variations")

# Robust SE (no clustering)
run_spec(df_1790, f"slaveratio ~ MAL + {' + '.join(CROP_1790)} + {' + '.join(GEO_1790)} | state_g",
         "robust/cluster/robust_hc1", "robustness/clustering_variations.md",
         "slaveratio", "MAL", "All counties 1790", "state_g",
         "Full controls", "None (robust)", "OLS-FE")

run_spec(df_1860, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/cluster/robust_hc1_1860", "robustness/clustering_variations.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Full controls", "None (robust)", "OLS-FE")

# ============================================================================
# SECTION 5: ESTIMATION METHOD VARIATIONS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 5: ESTIMATION METHOD VARIATIONS")
print("=" * 60)

# 5.1: No fixed effects
print("\n5.1: Fixed Effects Variations")
run_spec(df_1790, f"slaveratio ~ MAL + {' + '.join(CROP_1790)} + {' + '.join(GEO_1790)}",
         "robust/estimation/no_fe_1790", "robustness/model_specification.md",
         "slaveratio", "MAL", "All counties 1790", "None",
         "Full controls", "None", "OLS")

run_spec(df_1860, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)}",
         "robust/estimation/no_fe_1860", "robustness/model_specification.md",
         "slaveratio", "MAL", "All counties 1860", "None",
         "Full controls", "None", "OLS")

# ============================================================================
# SECTION 6: FUNCTIONAL FORM VARIATIONS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 6: FUNCTIONAL FORM VARIATIONS")
print("=" * 60)

# 6.1: Log outcome (add small constant for zeros)
print("\n6.1: Functional Form Variations")
df_1860_log = df_1860.copy()
df_1860_log['log_slaveratio'] = np.log(df_1860_log['slaveratio'] + 0.001)
run_spec(df_1860_log, f"log_slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/funcform/log_outcome", "robustness/model_specification.md",
         "log_slaveratio", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# IHS transformation
df_1860_ihs = df_1860.copy()
df_1860_ihs['ihs_slaveratio'] = np.arcsinh(df_1860_ihs['slaveratio'])
run_spec(df_1860_ihs, f"ihs_slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/funcform/ihs_outcome", "robustness/model_specification.md",
         "ihs_slaveratio", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# Quadratic treatment
df_1860_sq = df_1860.copy()
df_1860_sq['MAL_sq'] = df_1860_sq['MAL'] ** 2
run_spec(df_1860_sq, f"slaveratio ~ MAL + MAL_sq + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/funcform/quadratic", "robustness/model_specification.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Full controls + MAL^2", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 7: ALTERNATIVE OUTCOMES
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 7: ALTERNATIVE OUTCOMES")
print("=" * 60)

# 7.1: Alternative slavery measures
print("\n7.1: Alternative Outcome Variables")

# Black ratio instead of slave ratio (if available)
if 'blackratio_1860' in df_1860.columns:
    run_spec(df_1860, f"blackratio_1860 ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
             "robust/outcome/blackratio", "robustness/measurement.md",
             "blackratio_1860", "MAL", "All counties 1860", "state_g",
             "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 8: HETEROGENEITY ANALYSES
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 8: HETEROGENEITY ANALYSES")
print("=" * 60)

# 8.1: Interaction with cotton suitability
print("\n8.1: Heterogeneity by Cotton Suitability")
if 'Ginnedcotton_std' in df_1860.columns:
    df_1860_het = df_1860.copy()
    df_1860_het['MAL_x_cotton'] = df_1860_het['MAL'] * df_1860_het['Ginnedcotton_std']
    run_spec(df_1860_het, f"slaveratio ~ MAL + Ginnedcotton_std + MAL_x_cotton + {' + '.join(GEO_1860)} | state_g",
             "robust/heterogeneity/cotton", "robustness/heterogeneity.md",
             "slaveratio", "MAL", "All counties 1860", "state_g",
             "Geography + Cotton interaction", "state_g", "OLS-FE", "CRV1", "state_g")

# 8.2: Slave vs non-slave states interaction
df_1860_het = df_1860.copy()
df_1860_het['MAL_x_slavestate'] = df_1860_het['MAL'] * df_1860_het['slave_state']
run_spec(df_1860_het, f"slaveratio ~ MAL + MAL_x_slavestate + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/heterogeneity/slave_state", "robustness/heterogeneity.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Full controls + slave state interaction", "state_g", "OLS-FE", "CRV1", "state_g")

# 8.3: High vs low elevation
df_1860_het = df_1860.copy()
df_1860_het['high_elev'] = (df_1860_het['ELEV'] > df_1860_het['ELEV'].median()).astype(int)
df_1860_het['MAL_x_highelev'] = df_1860_het['MAL'] * df_1860_het['high_elev']
run_spec(df_1860_het, f"slaveratio ~ MAL + high_elev + MAL_x_highelev + {' + '.join(CROP_1860)} | state_g",
         "robust/heterogeneity/elevation", "robustness/heterogeneity.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Crops + elevation interaction", "state_g", "OLS-FE", "CRV1", "state_g")

# 8.4: Coastal vs inland
df_1860_het = df_1860.copy()
df_1860_het['coastal'] = (df_1860_het['DISTSEA'] < df_1860_het['DISTSEA'].median()).astype(int)
df_1860_het['MAL_x_coastal'] = df_1860_het['MAL'] * df_1860_het['coastal']
run_spec(df_1860_het, f"slaveratio ~ MAL + coastal + MAL_x_coastal + {' + '.join(CROP_1860)} | state_g",
         "robust/heterogeneity/coastal", "robustness/heterogeneity.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Crops + coastal interaction", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 9: PLACEBO TESTS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 9: PLACEBO TESTS")
print("=" * 60)

# 9.1: White population as placebo outcome
print("\n9.1: Placebo Outcomes")
df_1860_placebo = df_1860.copy()
df_1860_placebo['white_ratio'] = df_1860_placebo['WhitePopulation1860'] / df_1860_placebo['TotalPopulation1860']
run_spec(df_1860_placebo, f"white_ratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "robust/placebo/white_ratio", "robustness/placebo_tests.md",
         "white_ratio", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 10: VOTING OUTCOMES (Table 3)
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 10: POLITICAL OUTCOMES")
print("=" * 60)

# 10.1: Presidential election 1860
print("\n10.1: Presidential Election Outcomes")
# Republican (Lincoln) vote share 1860
run_spec(df_votes, f"vote1860_rep_pres ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "outcome/vote_lincoln", "methods/cross_sectional_ols.md",
         "vote1860_rep_pres", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# Democrat (Breckinridge) vote share 1860
run_spec(df_votes, f"vote1860_demeq_pres ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "outcome/vote_breckinridge", "methods/cross_sectional_ols.md",
         "vote1860_demeq_pres", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# 1868 elections
run_spec(df_votes, f"vote1868_rep_pres ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "outcome/vote_grant", "methods/cross_sectional_ols.md",
         "vote1868_rep_pres", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

run_spec(df_votes, f"vote1868_dem_pres ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "outcome/vote_seymour", "methods/cross_sectional_ols.md",
         "vote1868_dem_pres", "MAL", "All counties 1860", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# 10.2: Constitutional convention votes
print("\n10.2: Constitutional Convention Pro-Slavery Votes")
run_spec(df_conv, f"proslavery ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
         "outcome/proslavery_vote", "methods/cross_sectional_ols.md",
         "proslavery", "MAL", "Convention delegates", "state_g",
         "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 11: CROSS-COUNTRY EVIDENCE
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 11: CROSS-COUNTRY EVIDENCE (Americas)")
print("=" * 60)

# Americas data
print("\n11.1: Americas Sample")
run_spec(df_americas, f"coloredratio ~ MAL",
         "cross_country/americas_minimal", "methods/cross_sectional_ols.md",
         "coloredratio", "MAL", "US, Brazil, Cuba states", "None",
         "No controls", "robust", "OLS")

run_spec(df_americas, f"coloredratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860_SHORT)} | country",
         "cross_country/americas_full", "methods/cross_sectional_ols.md",
         "coloredratio", "MAL", "US, Brazil, Cuba states", "country",
         "Full controls", "robust", "OLS-FE")

# ============================================================================
# SECTION 12: PANEL ANALYSIS (Table 5)
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 12: PANEL ANALYSIS (Historical States)")
print("=" * 60)

# Prepare panel data
df_panel_clean = df_panel[df_panel['year'] <= 1750].copy()
df_panel_clean = df_panel_clean.dropna(subset=['black_totalpop', 'mal1690_x_ME1790_std'])

print("\n12.1: Panel Specifications - Share of Blacks")
# Main panel specification
run_spec(df_panel_clean, "black_totalpop ~ mal1690_x_ME1790_std + C(year) | state_g",
         "panel/baseline", "methods/panel_fixed_effects.md",
         "black_totalpop", "mal1690_x_ME1790_std", "States 1630-1750", "state_g + year",
         "Year fixed effects", "state_g", "Panel-FE", "CRV1", "state_g")

# Different time periods
print("\n12.2: Panel Sample Restrictions by Time")
df_panel_early = df_panel[(df_panel['year'] >= 1650) & (df_panel['year'] <= 1720)].dropna(subset=['black_totalpop', 'mal1690_x_ME1790_std'])
run_spec(df_panel_early, "black_totalpop ~ mal1690_x_ME1790_std + C(year) | state_g",
         "panel/early_period", "robustness/sample_restrictions.md",
         "black_totalpop", "mal1690_x_ME1790_std", "States 1650-1720", "state_g + year",
         "Year fixed effects", "state_g", "Panel-FE", "CRV1", "state_g")

# ============================================================================
# SECTION 13: ADDITIONAL ROBUSTNESS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 13: ADDITIONAL ROBUSTNESS")
print("=" * 60)

# 13.1: Different MAL measures (if available)
print("\n13.1: Alternative Malaria Measures")
alt_mal_vars = ['MAL_temp', 'MAL_risk', 'MAL_clim']
for mal_var in alt_mal_vars:
    if mal_var in df_1860.columns:
        df_temp = df_1860.dropna(subset=[mal_var])
        if len(df_temp) > 50:
            run_spec(df_temp, f"slaveratio ~ {mal_var} + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
                     f"robust/treatment/{mal_var}", "robustness/measurement.md",
                     "slaveratio", mal_var, "All counties 1860", "state_g",
                     "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# 13.2: Population quartiles
print("\n13.2: Population Subgroups")
df_1860_q = df_1860.copy()
df_1860_q['pop_quartile'] = pd.qcut(df_1860_q['TotalPopulation1860'], 4, labels=False)

for q in [0, 3]:  # Bottom and top quartiles
    df_sub = df_1860_q[df_1860_q['pop_quartile'] == q].copy()
    label = "small_pop" if q == 0 else "large_pop"
    run_spec(df_sub, f"slaveratio ~ MAL + {' + '.join(CROP_1860)} + {' + '.join(GEO_1860)} | state_g",
             f"robust/sample/{label}", "robustness/sample_restrictions.md",
             "slaveratio", "MAL", f"Population Q{q+1} 1860", "state_g",
             "Full controls", "state_g", "OLS-FE", "CRV1", "state_g")

# 13.3: Temperature interaction
print("\n13.3: Climate Interactions")
df_1860_het = df_1860.copy()
df_1860_het['high_temp'] = (df_1860_het['temp'] > df_1860_het['temp'].median()).astype(int)
df_1860_het['MAL_x_hightemp'] = df_1860_het['MAL'] * df_1860_het['high_temp']
run_spec(df_1860_het, f"slaveratio ~ MAL + high_temp + MAL_x_hightemp + {' + '.join(CROP_1860)} | state_g",
         "robust/heterogeneity/temperature", "robustness/heterogeneity.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Crops + temperature interaction", "state_g", "OLS-FE", "CRV1", "state_g")

# 13.4: Precipitation interaction
df_1860_het = df_1860.copy()
df_1860_het['high_prec'] = (df_1860_het['prec'] > df_1860_het['prec'].median()).astype(int)
df_1860_het['MAL_x_highprec'] = df_1860_het['MAL'] * df_1860_het['high_prec']
run_spec(df_1860_het, f"slaveratio ~ MAL + high_prec + MAL_x_highprec + {' + '.join(CROP_1860)} | state_g",
         "robust/heterogeneity/precipitation", "robustness/heterogeneity.md",
         "slaveratio", "MAL", "All counties 1860", "state_g",
         "Crops + precipitation interaction", "state_g", "OLS-FE", "CRV1", "state_g")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications: {len(results_df)}")

# Save to CSV
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Summary statistics
if len(results_df) > 0:
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Min coefficient: {results_df['coefficient'].min():.4f}")
    print(f"Max coefficient: {results_df['coefficient'].max():.4f}")
else:
    print("No results generated!")
