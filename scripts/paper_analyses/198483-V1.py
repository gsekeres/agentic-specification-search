"""
Specification Search: Paper 198483-V1
Title: National Solidarity Programme Impact Evaluation (Afghanistan)
Journal: AEJ: Applied
Method: Paired randomized experiment with panel data

This script replicates and extends the main analyses from the replication package.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/198483-V1/Replication files AEJ/Replication files AEJ revised 2025"

# Paper metadata
PAPER_ID = "198483-V1"
JOURNAL = "AEJ: Applied"
PAPER_TITLE = "National Solidarity Programme Impact on Security and Development (Afghanistan)"

# Method classification
METHOD_CODE = "panel_fixed_effects"  # Paired randomization with panel structure
METHOD_TREE_PATH = "specification_tree/methods/panel_fixed_effects.md"

results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
                   coefficient_vector_json=None):
    """Extract results from pyfixest model into standard format."""

    coef = model.coef()[treatment_var]
    se = model.se()[treatment_var]
    tstat = model.tstat()[treatment_var]
    pval = model.pvalue()[treatment_var]

    # 95% CI
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    n_obs = model._N  # pyfixest uses _N for number of observations
    r2 = model._r2 if hasattr(model, '_r2') else None

    # Build coefficient vector JSON if not provided
    if coefficient_vector_json is None:
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(", ") if fixed_effects else [],
            "diagnostics": {}
        }
        # Add other coefficients
        for var in model.coef().index:
            if var != treatment_var and not var.startswith("C("):
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })
        coefficient_vector_json = json.dumps(coef_vector)

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
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r2) if r2 else None,
        'coefficient_vector_json': coefficient_vector_json,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


# =============================================================================
# PART 1: Survey-based outcomes (Table 5 style)
# =============================================================================

print("Loading survey data...")
df_survey = pd.read_stata(f"{DATA_PATH}/data/processed/Combined_data_H_M_Full_SIGACTS_new.dta")

# Key variables
# treatment: 1 = treatment village, 0 = control
# Survey: 1 = FU1 (midline), 2 = FU2 (endline)
# Pair: matched pair identifier
# Cluster: clustering unit

# Create treatment interactions for panel structure
# treatment_FU1: treatment * (Survey==1)
# treatment_FU2: treatment * (Survey==2)

# Ensure numeric types
df_survey['treatment'] = pd.to_numeric(df_survey['treatment'], errors='coerce')
df_survey['Cluster'] = pd.to_numeric(df_survey['Cluster'], errors='coerce')
df_survey['Pair_Survey'] = pd.Categorical(df_survey['Pair_Survey']).codes

# Main survey outcomes (Anderson indices as in Table 5)
survey_outcomes = {
    'index_Economic_Andr_M': 'Economic Index (Anderson)',
    'index_PublicGoods_Andr': 'Public Goods Index (Anderson)',
    'index_Economic_Andr_Subj': 'Subjective Economic Index (Anderson)',
    'index_Attitudes_Andr_M': 'Attitudes Index (Anderson)',
}

security_perception_outcomes = {
    'index_Security_perc_Andr_M': 'Male Security Perceptions (Anderson)',
    'index_Security_perc_Andr_F': 'Female Security Perceptions (Anderson)',
    'index_Security_exp_Andr_M': 'Security Experience Index (Anderson)',
}

# Treatment variables
treatment_vars = ['treatment_FU1', 'treatment_FU2']

print("\n=== PART 1: Survey Outcomes (Table 5 Style) ===")

# Baseline specifications for survey outcomes
for outcome, outcome_label in {**survey_outcomes, **security_perception_outcomes}.items():
    # Check if outcome exists and has variation
    if outcome not in df_survey.columns:
        print(f"  Skipping {outcome} - not found")
        continue

    df_temp = df_survey.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Pair_Survey', 'Cluster'])

    if len(df_temp) < 100:
        print(f"  Skipping {outcome} - insufficient observations")
        continue

    # Baseline: areg outcome treatment_FU1 treatment_FU2, a(Pair_Survey) cluster(Cluster)
    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair_Survey"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        # Record both treatment coefficients
        for tvar in ['treatment_FU1', 'treatment_FU2']:
            period = "Midline" if "FU1" in tvar else "Endline"
            results.append(extract_results(
                model,
                spec_id=f"baseline/survey/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#baseline",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc=f"Full survey sample, {period}",
                fixed_effects="Pair x Survey",
                controls_desc="None (paired randomization)",
                cluster_var="Cluster",
                model_type="Panel FE with absorbed pair-survey FE"
            ))
        print(f"  Completed baseline for {outcome}")
    except Exception as e:
        print(f"  Error on baseline for {outcome}: {e}")

# =============================================================================
# PART 2: Heterogeneity by East (Pakistan border) - Table 8/9 style
# =============================================================================

print("\n=== PART 2: Heterogeneity by East Region ===")

for outcome, outcome_label in {**survey_outcomes, **security_perception_outcomes}.items():
    if outcome not in df_survey.columns:
        continue

    df_temp = df_survey.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2',
                                       'EastTreat_FU1', 'EastTreat_FU2',
                                       'Pair_Survey', 'Cluster'])

    if len(df_temp) < 100:
        continue

    try:
        # Add East interaction terms
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 + EastTreat_FU1 + EastTreat_FU2 | Pair_Survey"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        # Record main treatment effect (non-East) and interaction
        for tvar in ['treatment_FU1', 'treatment_FU2']:
            period = "Midline" if "FU1" in tvar else "Endline"
            results.append(extract_results(
                model,
                spec_id=f"heterogeneity/east/{outcome}/{tvar}_noneast",
                spec_tree_path="methods/panel_fixed_effects.md#heterogeneity",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc=f"Full survey sample, {period}, non-East effect",
                fixed_effects="Pair x Survey",
                controls_desc="East interaction included",
                cluster_var="Cluster",
                model_type="Panel FE with East heterogeneity"
            ))

        # Record East interaction terms
        for tvar in ['EastTreat_FU1', 'EastTreat_FU2']:
            period = "Midline" if "FU1" in tvar else "Endline"
            results.append(extract_results(
                model,
                spec_id=f"heterogeneity/east/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#heterogeneity",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc=f"Full survey sample, {period}, East differential",
                fixed_effects="Pair x Survey",
                controls_desc="East interaction",
                cluster_var="Cluster",
                model_type="Panel FE with East heterogeneity"
            ))
        print(f"  Completed East heterogeneity for {outcome}")
    except Exception as e:
        print(f"  Error on East heterogeneity for {outcome}: {e}")


# =============================================================================
# PART 3: Security outcomes (SIGACTS data) - Table 4 style
# =============================================================================

print("\n=== PART 3: Security Panel Data ===")

# Load security panel data
df_security_raw = pd.read_stata(f"{DATA_PATH}/data/processed/SIGACTS_cleaned_panel_prepared.dta")

# Load treatment assignment for merging
df_treatment = pd.read_stata(f"{DATA_PATH}/data/raw/Treatment assignment00000.dta")
df_treatment = df_treatment[['Geocode', 'treatment', 'Geocode1', 'C5', 'C2']].drop_duplicates()

# Create security indices following the paper's methodology
# The paper aggregates incidents at different km radii and creates Anderson-weighted indices

def create_security_panel(df_raw, df_treatment):
    """
    Recreate the security panel data following Stata code logic.
    This creates a village x period panel with security indices.
    """
    # Aggregate by Geocode and FU (follow-up period)
    incident_cols_num = [f'Inc{i}km' for i in range(1, 16) if f'Inc{i}km' in df_raw.columns]

    # Sum of incidents (intensive margin)
    agg_num = df_raw.groupby(['Geocode', 'FU'])[incident_cols_num].sum().reset_index()

    # Binary any incident (extensive margin)
    agg_dum = df_raw.groupby(['Geocode', 'FU'])[incident_cols_num].apply(
        lambda x: (x > 0).any().astype(int)
    ).reset_index()

    # Rename columns
    for col in incident_cols_num:
        agg_num = agg_num.rename(columns={col: f'{col}_num'})

    # Merge with treatment assignment
    panel = agg_num.merge(df_treatment, on='Geocode', how='left')

    # Adjust treatment coding (paper uses 0/1)
    if panel['treatment'].min() == 1:
        panel['treatment'] = panel['treatment'] - 1

    # Create cluster variable
    panel['Cluster'] = panel.groupby(['Geocode1', 'C5']).ngroup()
    panel.loc[panel['C5'] == 0, 'Cluster'] = panel.loc[panel['C5'] == 0, 'Geocode']

    # Create pair variable
    panel['Pair'] = panel.groupby(['Geocode1', 'C2']).ngroup()

    # Create Pair x FU interaction for FE
    panel['Pair_FU'] = panel.groupby(['Pair', 'FU']).ngroup()

    return panel

try:
    df_security = create_security_panel(df_security_raw, df_treatment)

    # Create log-transformed winsorized incident counts
    for km in range(1, 16):
        col = f'Inc{km}km_num'
        if col in df_security.columns:
            # Winsorize at 95th percentile
            p95 = df_security[col].quantile(0.95)
            df_security[f'{col}_wins'] = df_security[col].clip(upper=p95)
            df_security[f'{col}_wins_ln'] = np.log1p(df_security[f'{col}_wins'])

    # Create Anderson-weighted index (simplified version)
    # The paper uses control group covariance matrix weighting
    # Here we create a simple average as approximation
    ln_cols = [f'Inc{km}km_num_wins_ln' for km in range(2, 16) if f'Inc{km}km_num_wins_ln' in df_security.columns]

    if ln_cols:
        # Normalize each column by control group mean/SD
        df_control = df_security[df_security['treatment'] == 0]
        for col in ln_cols:
            if col in df_security.columns:
                mean_ctrl = df_control[col].mean()
                std_ctrl = df_control[col].std()
                if std_ctrl > 0:
                    df_security[f'{col}_nrm'] = (df_security[col] - mean_ctrl) / std_ctrl
                else:
                    df_security[f'{col}_nrm'] = 0

        # Create simple Anderson index (average of normalized values)
        nrm_cols = [f'{col}_nrm' for col in ln_cols if f'{col}_nrm' in df_security.columns]
        df_security['Inc_wins_ln_Anderson'] = df_security[nrm_cols].mean(axis=1)

    # Create treatment x period interactions
    for fu in df_security['FU'].unique():
        df_security[f'treatment_FU{int(fu)}'] = (df_security['treatment'] == 1) & (df_security['FU'] == fu)
        df_security[f'treatment_FU{int(fu)}'] = df_security[f'treatment_FU{int(fu)}'].astype(int)

    # Filter to post-treatment periods (FU >= 1)
    df_security_post = df_security[df_security['FU'] >= 1].copy()

    # Main security outcome analysis
    security_outcomes_panel = ['Inc_wins_ln_Anderson']

    for outcome in security_outcomes_panel:
        if outcome not in df_security_post.columns:
            continue

        df_temp = df_security_post.dropna(subset=[outcome, 'Pair_FU', 'Cluster'])

        # Get treatment vars that exist
        treat_vars = [col for col in df_temp.columns if col.startswith('treatment_FU') and df_temp[col].sum() > 0]

        if len(treat_vars) == 0 or len(df_temp) < 50:
            continue

        # Build formula dynamically
        treat_formula = " + ".join(treat_vars)
        formula = f"{outcome} ~ {treat_formula} | Pair_FU"

        try:
            model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

            for tvar in treat_vars:
                results.append(extract_results(
                    model,
                    spec_id=f"baseline/security/{outcome}/{tvar}",
                    spec_tree_path="methods/panel_fixed_effects.md#baseline",
                    outcome_var=outcome,
                    treatment_var=tvar,
                    sample_desc="Post-treatment security panel",
                    fixed_effects="Pair x Period",
                    controls_desc="None",
                    cluster_var="Cluster",
                    model_type="Panel FE - Security outcomes"
                ))
            print(f"  Completed baseline for security index")
        except Exception as e:
            print(f"  Error on security baseline: {e}")

except Exception as e:
    print(f"Error creating security panel: {e}")


# =============================================================================
# PART 4: Robustness - Alternative clustering
# =============================================================================

print("\n=== PART 4: Clustering Variations ===")

# Use main survey outcomes for clustering robustness
main_outcomes = ['index_Economic_Andr_M', 'index_Attitudes_Andr_M']

for outcome in main_outcomes:
    if outcome not in df_survey.columns:
        continue

    df_temp = df_survey.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Pair_Survey'])

    if len(df_temp) < 100:
        continue

    # Robust (heteroskedasticity-robust) SE
    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair_Survey"
        model = pf.feols(formula, data=df_temp, vcov='hetero')

        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"robust/cluster/none/{outcome}/{tvar}",
                spec_tree_path="robustness/clustering_variations.md#none",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="Pair x Survey",
                controls_desc="None",
                cluster_var="None (robust SE)",
                model_type="Panel FE - Robust SE"
            ))
        print(f"  Completed robust SE for {outcome}")
    except Exception as e:
        print(f"  Error on robust SE for {outcome}: {e}")

    # Cluster by Geocode (village)
    if 'Geocode' in df_temp.columns:
        try:
            model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Geocode'})

            for tvar in ['treatment_FU1', 'treatment_FU2']:
                results.append(extract_results(
                    model,
                    spec_id=f"robust/cluster/village/{outcome}/{tvar}",
                    spec_tree_path="robustness/clustering_variations.md#unit",
                    outcome_var=outcome,
                    treatment_var=tvar,
                    sample_desc="Full survey sample",
                    fixed_effects="Pair x Survey",
                    controls_desc="None",
                    cluster_var="Geocode (village)",
                    model_type="Panel FE - Village clustered"
                ))
            print(f"  Completed village clustering for {outcome}")
        except Exception as e:
            print(f"  Error on village clustering for {outcome}: {e}")

    # Cluster by District (Geocode1)
    if 'Geocode1' in df_temp.columns:
        try:
            model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Geocode1'})

            for tvar in ['treatment_FU1', 'treatment_FU2']:
                results.append(extract_results(
                    model,
                    spec_id=f"robust/cluster/district/{outcome}/{tvar}",
                    spec_tree_path="robustness/clustering_variations.md#region",
                    outcome_var=outcome,
                    treatment_var=tvar,
                    sample_desc="Full survey sample",
                    fixed_effects="Pair x Survey",
                    controls_desc="None",
                    cluster_var="Geocode1 (district)",
                    model_type="Panel FE - District clustered"
                ))
            print(f"  Completed district clustering for {outcome}")
        except Exception as e:
            print(f"  Error on district clustering for {outcome}: {e}")


# =============================================================================
# PART 5: Alternative fixed effects structures
# =============================================================================

print("\n=== PART 5: Alternative FE Structures ===")

for outcome in main_outcomes:
    if outcome not in df_survey.columns:
        continue

    df_temp = df_survey.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Cluster'])
    df_temp['Survey'] = pd.to_numeric(df_temp['Survey'], errors='coerce')
    df_temp['Pair'] = pd.to_numeric(df_temp['Pair'], errors='coerce')

    if len(df_temp) < 100:
        continue

    # Pair FE only (no survey/time FE)
    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"panel/fe/pair_only/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#fe/unit",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="Pair only",
                controls_desc="None",
                cluster_var="Cluster",
                model_type="Panel FE - Pair only"
            ))
        print(f"  Completed pair-only FE for {outcome}")
    except Exception as e:
        print(f"  Error on pair-only FE for {outcome}: {e}")

    # Survey FE only
    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Survey"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"panel/fe/time_only/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#fe/time",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="Survey (time) only",
                controls_desc="None",
                cluster_var="Cluster",
                model_type="Panel FE - Survey only"
            ))
        print(f"  Completed survey-only FE for {outcome}")
    except Exception as e:
        print(f"  Error on survey-only FE for {outcome}: {e}")

    # Two-way FE (Pair + Survey separately)
    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair + Survey"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"panel/fe/twoway/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#fe/twoway",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="Pair + Survey (two-way)",
                controls_desc="None",
                cluster_var="Cluster",
                model_type="Panel FE - Two-way"
            ))
        print(f"  Completed two-way FE for {outcome}")
    except Exception as e:
        print(f"  Error on two-way FE for {outcome}: {e}")

    # No FE (pooled OLS)
    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"panel/fe/none/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#fe/none",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="None (pooled OLS)",
                controls_desc="None",
                cluster_var="Cluster",
                model_type="Pooled OLS"
            ))
        print(f"  Completed pooled OLS for {outcome}")
    except Exception as e:
        print(f"  Error on pooled OLS for {outcome}: {e}")


# =============================================================================
# PART 6: Sample restrictions
# =============================================================================

print("\n=== PART 6: Sample Restrictions ===")

# Sample restricted to East region only
if 'East' in df_survey.columns:
    df_east = df_survey[df_survey['East'] == 1].copy()

    for outcome in main_outcomes:
        if outcome not in df_east.columns:
            continue

        df_temp = df_east.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Pair_Survey', 'Cluster'])

        if len(df_temp) < 50:
            continue

        try:
            formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair_Survey"
            model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

            for tvar in ['treatment_FU1', 'treatment_FU2']:
                results.append(extract_results(
                    model,
                    spec_id=f"sample/east_only/{outcome}/{tvar}",
                    spec_tree_path="robustness/sample_restrictions.md#subsample",
                    outcome_var=outcome,
                    treatment_var=tvar,
                    sample_desc="East region only",
                    fixed_effects="Pair x Survey",
                    controls_desc="None",
                    cluster_var="Cluster",
                    model_type="Panel FE - East subsample"
                ))
            print(f"  Completed East-only for {outcome}")
        except Exception as e:
            print(f"  Error on East-only for {outcome}: {e}")

# Sample restricted to non-East
if 'East' in df_survey.columns:
    df_noneast = df_survey[df_survey['East'] == 0].copy()

    for outcome in main_outcomes:
        if outcome not in df_noneast.columns:
            continue

        df_temp = df_noneast.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Pair_Survey', 'Cluster'])

        if len(df_temp) < 50:
            continue

        try:
            formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair_Survey"
            model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

            for tvar in ['treatment_FU1', 'treatment_FU2']:
                results.append(extract_results(
                    model,
                    spec_id=f"sample/noneast_only/{outcome}/{tvar}",
                    spec_tree_path="robustness/sample_restrictions.md#subsample",
                    outcome_var=outcome,
                    treatment_var=tvar,
                    sample_desc="Non-East region only",
                    fixed_effects="Pair x Survey",
                    controls_desc="None",
                    cluster_var="Cluster",
                    model_type="Panel FE - Non-East subsample"
                ))
            print(f"  Completed Non-East for {outcome}")
        except Exception as e:
            print(f"  Error on Non-East for {outcome}: {e}")


# =============================================================================
# PART 7: Alternative index construction (Katz vs Anderson)
# =============================================================================

print("\n=== PART 7: Alternative Index Construction ===")

alt_indices = {
    'index_Economic_Katz_M': ('index_Economic_Andr_M', 'Economic Index (Katz)'),
    'index_Attitudes_Katz_M': ('index_Attitudes_Andr_M', 'Attitudes Index (Katz)'),
    'index_Economic_pca_M': ('index_Economic_Andr_M', 'Economic Index (PCA)'),
    'index_Attitudes_pca_M': ('index_Attitudes_Andr_M', 'Attitudes Index (PCA)'),
}

for outcome, (baseline_outcome, label) in alt_indices.items():
    if outcome not in df_survey.columns:
        continue

    df_temp = df_survey.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Pair_Survey', 'Cluster'])

    if len(df_temp) < 100:
        continue

    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair_Survey"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        method = "Katz" if "Katz" in outcome else "PCA"
        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"custom/index_method/{method}/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#custom",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="Pair x Survey",
                controls_desc=f"Index constructed using {method} method",
                cluster_var="Cluster",
                model_type=f"Panel FE - {method} index"
            ))
        print(f"  Completed {method} index for {outcome}")
    except Exception as e:
        print(f"  Error on {method} index for {outcome}: {e}")


# =============================================================================
# PART 8: Individual component outcomes (disaggregated)
# =============================================================================

print("\n=== PART 8: Individual Component Outcomes ===")

# Individual security perception outcomes
individual_security = ['M12_19z', 'M12_20z', 'M12_20y', 'M12_17a', 'M12_17B', 'M12_19X']

for outcome in individual_security:
    if outcome not in df_survey.columns:
        continue

    df_temp = df_survey.dropna(subset=[outcome, 'treatment_FU1', 'treatment_FU2', 'Pair_Survey', 'Cluster'])

    if len(df_temp) < 100:
        continue

    try:
        formula = f"{outcome} ~ treatment_FU1 + treatment_FU2 | Pair_Survey"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': 'Cluster'})

        for tvar in ['treatment_FU1', 'treatment_FU2']:
            results.append(extract_results(
                model,
                spec_id=f"custom/component/{outcome}/{tvar}",
                spec_tree_path="methods/panel_fixed_effects.md#custom",
                outcome_var=outcome,
                treatment_var=tvar,
                sample_desc="Full survey sample",
                fixed_effects="Pair x Survey",
                controls_desc="Individual security component",
                cluster_var="Cluster",
                model_type="Panel FE - Component outcome"
            ))
        print(f"  Completed component {outcome}")
    except Exception as e:
        print(f"  Error on component {outcome}: {e}")


# =============================================================================
# Save results
# =============================================================================

print("\n=== Saving Results ===")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to package directory
output_path = f"{DATA_PATH}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved {len(results_df)} specifications to {output_path}")

# Also save to scripts directory for backup
backup_path = f"{BASE_PATH}/scripts/paper_analyses/198483-V1_results.csv"
results_df.to_csv(backup_path, index=False)

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total specifications: {len(results_df)}")
print(f"Unique outcomes: {results_df['outcome_var'].nunique()}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Print by spec category
print("\n=== Breakdown by Specification Type ===")
results_df['spec_category'] = results_df['spec_id'].str.split('/').str[0]
for cat, group in results_df.groupby('spec_category'):
    n = len(group)
    sig_5 = (group['p_value'] < 0.05).sum()
    print(f"  {cat}: {n} specs, {sig_5} ({100*sig_5/n:.0f}%) significant at 5%")

print("\nDone!")
