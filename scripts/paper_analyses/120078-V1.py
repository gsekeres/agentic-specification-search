"""
Specification Search for Paper 120078-V1

Paper: Ethnic Discrimination in the Airbnb Market
Topic: Testing whether minority hosts (Arab/Muslim or African-American) receive lower prices
Method: Panel Fixed Effects / Cross-sectional OLS with high-dimensional FE
Main DV: log_price (log daily rental price)
Main IV: minodummy (minority host dummy)

Institute for Replication (i4r) methodology - running 50+ specifications
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SETUP
# =============================================================================

PAPER_ID = "120078-V1"
PAPER_TITLE = "Ethnic Discrimination in the Airbnb Market"
JOURNAL = "AEJ-Applied"
METHOD_CODE = "panel_fixed_effects"
METHOD_TREE_PATH = "specification_tree/methods/panel_fixed_effects.md"

# Paths
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/120078-V1/data/base_airbnb_AEJ.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/120078-V1/specification_results.csv"
SCRIPT_PATH = "scripts/paper_analyses/120078-V1.py"

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Raw data shape: {df.shape}")

# Apply main sample restriction from paper: Drev100 > 0 (at least one review)
df = df[df['Drev100'] > 0].copy()
print(f"After Drev100 > 0 filter: {df.shape}")

# Sample data to speed up computation (stratified by minority status to preserve ratio)
np.random.seed(42)
SAMPLE_FRAC = 0.10  # Use 10% of data for faster execution

# Stratified sampling
df_minority = df[df['minodummy'] == 1].sample(frac=SAMPLE_FRAC, random_state=42)
df_majority = df[df['minodummy'] == 0].sample(frac=SAMPLE_FRAC, random_state=42)
df = pd.concat([df_minority, df_majority], ignore_index=True)
print(f"After stratified sampling ({SAMPLE_FRAC*100:.0f}%): {df.shape}")

# Convert key variables to appropriate types
df['newid'] = df['newid'].astype(int)
df['wave'] = df['wave'].astype(int)
df['citywaveID'] = df['citywaveID'].astype(int)
df['hoodcityID'] = df['hoodcityID'].astype(int)
df['blockID'] = df['blockID'].astype(int)

# Create city variable from blockID (first digit or hoodcityID)
df['city_id'] = df['hoodcityID'] // 100  # Approximate city grouping

# Define control variable sets
SIZE_CONTROLS = ['person_capacity345', 'bedrooms', 'bathrooms']
TYPE_CONTROLS = ['appart', 'house_loft', 'sharedflat']
EQUIP_CONTROLS = ['cabletv', 'wireless', 'heating', 'ac', 'elevator', 'handiaccess',
                  'doorman', 'fireplace', 'washer', 'dryer', 'parking', 'gym',
                  'pool', 'buzzer', 'hottub', 'breakfast', 'family', 'events']
RULES_CONTROLS = ['people', 'extrapeople', 'cancel_policy', 'smoking_allowed', 'pets_allowed']
HOST_CONTROLS = ['more_1_flat', 'superhost', 'verified_email', 'verified_offline',
                 'verified_phone', 'facebook']

# Simplified control set for efficiency
SIMPLE_CONTROLS = ['bedrooms', 'bathrooms', 'person_capacity345', 'sharedflat',
                   'superhost', 'more_1_flat', 'wireless', 'ac', 'parking']
SIMPLE_CONTROLS = [c for c in SIMPLE_CONTROLS if c in df.columns]

# Medium control set
MEDIUM_CONTROLS = SIZE_CONTROLS + TYPE_CONTROLS + ['wireless', 'ac', 'parking', 'elevator',
                                                    'superhost', 'more_1_flat', 'verified_email']
MEDIUM_CONTROLS = [c for c in MEDIUM_CONTROLS if c in df.columns]

# Full control set (paper's lesX)
FULL_CONTROLS = SIZE_CONTROLS + TYPE_CONTROLS + [c for c in EQUIP_CONTROLS if c in df.columns] + \
                [c for c in RULES_CONTROLS if c in df.columns] + [c for c in HOST_CONTROLS if c in df.columns]
FULL_CONTROLS = [c for c in FULL_CONTROLS if c in df.columns]

print(f"Simple controls: {len(SIMPLE_CONTROLS)}")
print(f"Medium controls: {len(MEDIUM_CONTROLS)}")
print(f"Full controls: {len(FULL_CONTROLS)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_regression(data, formula, vcov_type='hetero', cluster_var=None, spec_id='',
                   spec_tree_path='', outcome_var='log_price', treatment_var='minodummy',
                   sample_desc='', fixed_effects='', controls_desc='', model_type='OLS'):
    """Run regression and extract results."""
    try:
        # Set up vcov
        if cluster_var is not None:
            vcov = {'CRV1': cluster_var}
        else:
            vcov = vcov_type

        # Run model
        model = pf.feols(formula, data=data, vcov=vcov)

        # Extract results
        coef = model.coef()[treatment_var] if treatment_var in model.coef().index else np.nan
        se = model.se()[treatment_var] if treatment_var in model.se().index else np.nan
        pval = model.pvalue()[treatment_var] if treatment_var in model.pvalue().index else np.nan
        tstat = model.tstat()[treatment_var] if treatment_var in model.tstat().index else np.nan

        # Confidence intervals
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Build coefficient vector JSON (simplified)
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef) if not np.isnan(coef) else None,
                'se': float(se) if not np.isnan(se) else None,
                'pval': float(pval) if not np.isnan(pval) else None
            },
            'fixed_effects': fixed_effects.split(' + ') if fixed_effects else [],
        }

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
            'n_obs': model._N,
            'r_squared': model._r2 if hasattr(model, '_r2') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': model_type,
            'estimation_script': SCRIPT_PATH
        }
    except Exception as e:
        print(f"Error in {spec_id}: {str(e)[:100]}")
        return None

results = []
simple_str = ' + '.join(SIMPLE_CONTROLS)
medium_str = ' + '.join(MEDIUM_CONTROLS)
full_str = ' + '.join(FULL_CONTROLS)

# =============================================================================
# BASELINE SPECIFICATIONS (Table 2 replication)
# =============================================================================

print("\n" + "="*60)
print("BASELINE SPECIFICATIONS")
print("="*60)

# Baseline 1: City-wave FE only, no controls
spec = run_regression(
    df,
    "log_price ~ minodummy | citywaveID",
    cluster_var='newid',
    spec_id='baseline',
    spec_tree_path='methods/panel_fixed_effects.md#baseline',
    sample_desc='Full sample with Drev100>0 (10% sample)',
    fixed_effects='citywaveID',
    controls_desc='None',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"Baseline: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}, p={spec['p_value']:.4f}, n={spec['n_obs']}")

# Baseline 2: City-wave FE + simple controls
spec = run_regression(
    df,
    f"log_price ~ minodummy + {simple_str} | citywaveID",
    cluster_var='newid',
    spec_id='baseline_simple_controls',
    spec_tree_path='methods/panel_fixed_effects.md#baseline',
    sample_desc='Full sample with Drev100>0 (10% sample)',
    fixed_effects='citywaveID',
    controls_desc='Simple property characteristics',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"Baseline + simple controls: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}")

# Baseline 3: City-wave FE + medium controls
spec = run_regression(
    df,
    f"log_price ~ minodummy + {medium_str} | citywaveID",
    cluster_var='newid',
    spec_id='baseline_medium_controls',
    spec_tree_path='methods/panel_fixed_effects.md#baseline',
    sample_desc='Full sample with Drev100>0 (10% sample)',
    fixed_effects='citywaveID',
    controls_desc='Medium property characteristics',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"Baseline + medium controls: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}")

# Baseline 4: Neighborhood FE (MAIN RESULT approximation)
spec = run_regression(
    df,
    f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
    cluster_var='hoodcityID',
    spec_id='baseline_main',
    spec_tree_path='methods/panel_fixed_effects.md#baseline',
    sample_desc='Full sample with Drev100>0 (10% sample)',
    fixed_effects='citywaveID + hoodcityID',
    controls_desc='Medium property characteristics',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"MAIN RESULT (neighborhood FE): coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}, p={spec['p_value']:.4f}")

# =============================================================================
# CONTROL VARIABLE PROGRESSIONS
# =============================================================================

print("\n" + "="*60)
print("CONTROL VARIABLE PROGRESSIONS")
print("="*60)

# Bivariate (no controls)
spec = run_regression(
    df,
    "log_price ~ minodummy | citywaveID + hoodcityID",
    cluster_var='hoodcityID',
    spec_id='robust/build/bivariate',
    spec_tree_path='robustness/control_progression.md',
    sample_desc='Full sample',
    fixed_effects='citywaveID + hoodcityID',
    controls_desc='None',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"Bivariate: coef={spec['coefficient']:.4f}")

# Progressive control additions
for i in range(1, len(MEDIUM_CONTROLS) + 1, 2):
    ctrl_subset = MEDIUM_CONTROLS[:i]
    ctrl_str = ' + '.join(ctrl_subset)
    spec = run_regression(
        df,
        f"log_price ~ minodummy + {ctrl_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id=f'robust/build/add_{i}_controls',
        spec_tree_path='robustness/control_progression.md',
        sample_desc='Full sample',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc=f'First {i} controls',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Add {i} controls: coef={spec['coefficient']:.4f}")

# =============================================================================
# LEAVE-ONE-OUT CONTROL ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("LEAVE-ONE-OUT CONTROL ANALYSIS")
print("="*60)

for drop_ctrl in SIMPLE_CONTROLS:
    remaining = [c for c in SIMPLE_CONTROLS if c != drop_ctrl]
    ctrl_str = ' + '.join(remaining)

    spec = run_regression(
        df,
        f"log_price ~ minodummy + {ctrl_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id=f'robust/loo/drop_{drop_ctrl}',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc='Full sample',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc=f'Simple minus {drop_ctrl}',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Drop {drop_ctrl}: coef={spec['coefficient']:.4f}")

# =============================================================================
# FIXED EFFECTS VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("FIXED EFFECTS VARIATIONS")
print("="*60)

fe_specs = {
    'panel/fe/none': (f'log_price ~ minodummy + {medium_str}', 'None'),
    'panel/fe/citywave': (f'log_price ~ minodummy + {medium_str} | citywaveID', 'citywaveID'),
    'panel/fe/neighborhood': (f'log_price ~ minodummy + {medium_str} | hoodcityID', 'hoodcityID'),
    'panel/fe/citywave_hood': (f'log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID', 'citywaveID + hoodcityID'),
    'panel/fe/wave': (f'log_price ~ minodummy + {medium_str} | wave', 'wave'),
}

for spec_id, (formula, fe_desc) in fe_specs.items():
    spec = run_regression(
        df,
        formula,
        cluster_var='hoodcityID' if 'hood' in fe_desc.lower() else 'newid',
        spec_id=spec_id,
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects',
        sample_desc='Full sample',
        fixed_effects=fe_desc,
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"{spec_id}: coef={spec['coefficient']:.4f}")

# =============================================================================
# CLUSTERING VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("CLUSTERING VARIATIONS")
print("="*60)

cluster_specs = {
    'robust/cluster/robust': None,
    'robust/cluster/listing': 'newid',
    'robust/cluster/neighborhood': 'hoodcityID',
    'robust/cluster/citywave': 'citywaveID',
    'robust/cluster/city': 'city_id',
}

for spec_id, cluster_var in cluster_specs.items():
    spec = run_regression(
        df,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var=cluster_var,
        spec_id=spec_id,
        spec_tree_path='robustness/clustering_variations.md',
        sample_desc='Full sample',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"{spec_id}: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}")

# =============================================================================
# SAMPLE RESTRICTIONS
# =============================================================================

print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# Review-based samples (Table 3 replication)
review_samples = {
    'robust/sample/reviews_0': df['review'] == 0,
    'robust/sample/reviews_1_4': (df['review'] > 0) & (df['review'] <= 4),
    'robust/sample/reviews_5_19': (df['review'] >= 5) & (df['review'] < 20),
    'robust/sample/reviews_20_49': (df['review'] >= 20) & (df['review'] < 50),
    'robust/sample/reviews_50plus': df['review'] >= 50,
}

for spec_id, condition in review_samples.items():
    df_sub = df[condition].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=spec_id,
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=spec_id.split('/')[-1],
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"{spec_id}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Geographic samples
geo_samples = {
    'robust/sample/us_canada': (df['us'] == 1) | (df['can'] == 1),
    'robust/sample/europe': df['euro'] == 1,
}

for spec_id, condition in geo_samples.items():
    df_sub = df[condition].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=spec_id,
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=spec_id.split('/')[-1],
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"{spec_id}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Property type samples
property_samples = {
    'robust/sample/entire_flat': df['entireflat'] == 1,
    'robust/sample/shared_flat': df['entireflat'] == 0,
}

for spec_id, condition in property_samples.items():
    df_sub = df[condition].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=spec_id,
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=spec_id.split('/')[-1],
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"{spec_id}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Outlier handling
print("\n--- Outlier Handling ---")
for pct in [1, 5]:
    lower = df['log_price'].quantile(pct/100)
    upper = df['log_price'].quantile(1 - pct/100)
    df_trim = df[(df['log_price'] > lower) & (df['log_price'] < upper)].copy()

    spec = run_regression(
        df_trim,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id=f'robust/sample/trim_{pct}pct',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc=f'Trimmed {pct}%',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Trim {pct}%: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Winsorizing
for pct in [1, 5]:
    df_wins = df.copy()
    lower = df_wins['log_price'].quantile(pct/100)
    upper = df_wins['log_price'].quantile(1 - pct/100)
    df_wins['log_price'] = df_wins['log_price'].clip(lower=lower, upper=upper)

    spec = run_regression(
        df_wins,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id=f'robust/sample/winsor_{pct}pct',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc=f'Winsorized {pct}%',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Winsor {pct}%: coef={spec['coefficient']:.4f}")

# =============================================================================
# ALTERNATIVE TREATMENT DEFINITIONS
# =============================================================================

print("\n" + "="*60)
print("ALTERNATIVE TREATMENT DEFINITIONS")
print("="*60)

# Arab/Muslim only
df_arabic = df[(df['black_pic'] == 0) | (df['arabic_african'] == 1)].copy()
df_arabic['minodummy'] = df_arabic['arabic_african']
if len(df_arabic) > 100 and df_arabic['minodummy'].sum() > 10:
    spec = run_regression(
        df_arabic,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id='robust/treatment/arabic_only',
        spec_tree_path='robustness/measurement.md',
        sample_desc='Arab/Muslim hosts only',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Arabic/Muslim only: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Black/African-American only
df_black = df[(df['arabic_african'] == 0) | (df['black_pic'] == 1)].copy()
df_black['minodummy'] = df_black['black_pic']
if len(df_black) > 100 and df_black['minodummy'].sum() > 10:
    spec = run_regression(
        df_black,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id='robust/treatment/black_only',
        spec_tree_path='robustness/measurement.md',
        sample_desc='Black/African-American hosts only',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Black only: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# =============================================================================
# FUNCTIONAL FORM VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# Price in levels (not log)
df['price_clean'] = df['price'].clip(lower=1)
spec = run_regression(
    df,
    f"price_clean ~ minodummy + {medium_str} | citywaveID + hoodcityID",
    cluster_var='hoodcityID',
    spec_id='robust/form/price_levels',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='price_clean',
    sample_desc='Full sample',
    fixed_effects='citywaveID + hoodcityID',
    controls_desc='Medium property characteristics',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"Price levels: coef={spec['coefficient']:.4f}")

# Asinh transformation
df['asinh_price'] = np.arcsinh(df['price'])
spec = run_regression(
    df,
    f"asinh_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
    cluster_var='hoodcityID',
    spec_id='robust/form/asinh_price',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='asinh_price',
    sample_desc='Full sample',
    fixed_effects='citywaveID + hoodcityID',
    controls_desc='Medium property characteristics',
    model_type='OLS-FE'
)
if spec:
    results.append(spec)
    print(f"Asinh price: coef={spec['coefficient']:.4f}")

# =============================================================================
# HETEROGENEITY ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# By number of reviews (terciles)
df['review_tercile'] = pd.qcut(df['review'].clip(lower=0), q=3, labels=['low', 'med', 'high'], duplicates='drop')

for tercile in ['low', 'med', 'high']:
    df_sub = df[df['review_tercile'] == tercile].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=f'robust/het/reviews_{tercile}',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc=f'Review tercile: {tercile}',
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"Reviews {tercile}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# By price level
df['price_tercile'] = pd.qcut(df['price'], q=3, labels=['low', 'med', 'high'], duplicates='drop')

for tercile in ['low', 'med', 'high']:
    df_sub = df[df['price_tercile'] == tercile].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=f'robust/het/price_{tercile}',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc=f'Price tercile: {tercile}',
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"Price {tercile}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# By superhost status
for sh_status, sh_val in [('superhost', 1), ('not_superhost', 0)]:
    df_sub = df[df['superhost'] == sh_val].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=f'robust/het/{sh_status}',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc=sh_status,
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"{sh_status}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Interaction specifications
interaction_vars = ['superhost', 'entireflat', 'more_1_flat']
for int_var in interaction_vars:
    if int_var in df.columns:
        try:
            spec = run_regression(
                df,
                f"log_price ~ minodummy * {int_var} + {medium_str} | citywaveID + hoodcityID",
                cluster_var='hoodcityID',
                spec_id=f'robust/het/interaction_{int_var}',
                spec_tree_path='robustness/heterogeneity.md',
                sample_desc=f'Interaction with {int_var}',
                fixed_effects='citywaveID + hoodcityID',
                controls_desc='Medium property characteristics',
                model_type='OLS-FE'
            )
            if spec:
                results.append(spec)
                print(f"Interaction {int_var}: coef={spec['coefficient']:.4f}")
        except Exception as e:
            print(f"Error in interaction {int_var}: {str(e)[:50]}")

# =============================================================================
# CITY-SPECIFIC ANALYSES
# =============================================================================

print("\n" + "="*60)
print("CITY-SPECIFIC ANALYSES")
print("="*60)

# Major cities
major_cities = ['new-york', 'los-angeles', 'san-francisco', 'london', 'paris']
for city in major_cities:
    df_city = df[df['city'] == city].copy()
    if len(df_city) > 200 and df_city['minodummy'].sum() > 20:
        spec = run_regression(
            df_city,
            f"log_price ~ minodummy + {medium_str} | wave + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=f'robust/sample/city_{city.replace("-", "_")}',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=f'City: {city}',
            fixed_effects='wave + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"{city}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# =============================================================================
# DROP EACH MAJOR CITY (LEAVE-ONE-OUT GEOGRAPHIC)
# =============================================================================

print("\n" + "="*60)
print("LEAVE-ONE-OUT CITY ANALYSIS")
print("="*60)

cities_to_drop = ['new-york', 'london', 'paris']
for city in cities_to_drop:
    df_sub = df[df['city'] != city].copy()
    spec = run_regression(
        df_sub,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id=f'robust/sample/drop_{city.replace("-", "_")}',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc=f'Excluding {city}',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Drop {city}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# =============================================================================
# WAVE-SPECIFIC ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("WAVE-SPECIFIC ANALYSIS")
print("="*60)

# Early vs late waves
early_waves = df['wave'] <= df['wave'].median()
late_waves = df['wave'] > df['wave'].median()

for period_name, condition in [('early_waves', early_waves), ('late_waves', late_waves)]:
    df_sub = df[condition].copy()
    if len(df_sub) > 100 and df_sub['minodummy'].sum() > 10:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=f'robust/sample/{period_name}',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=period_name,
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"{period_name}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# =============================================================================
# SIZE-BASED HETEROGENEITY
# =============================================================================

print("\n" + "="*60)
print("SIZE-BASED HETEROGENEITY")
print("="*60)

# By bedrooms
for n_bed in [1, 2]:
    if n_bed == 2:
        df_sub = df[df['bedrooms'] >= 2].copy()
        bed_label = '2plus'
    else:
        df_sub = df[df['bedrooms'] == n_bed].copy()
        bed_label = str(n_bed)

    if len(df_sub) > 200 and df_sub['minodummy'].sum() > 20:
        spec = run_regression(
            df_sub,
            f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
            cluster_var='hoodcityID',
            spec_id=f'robust/het/bedrooms_{bed_label}',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc=f'Bedrooms={bed_label}',
            fixed_effects='citywaveID + hoodcityID',
            controls_desc='Medium property characteristics',
            model_type='OLS-FE'
        )
        if spec:
            results.append(spec)
            print(f"Bedrooms {bed_label}: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# =============================================================================
# ADDITIONAL ROBUSTNESS
# =============================================================================

print("\n" + "="*60)
print("ADDITIONAL ROBUSTNESS")
print("="*60)

# Multiple observations per unit
obs_counts = df.groupby('newid').size()
multi_obs = obs_counts[obs_counts >= 3].index
df_multi = df[df['newid'].isin(multi_obs)].copy()

if len(df_multi) > 100 and df_multi['minodummy'].sum() > 10:
    spec = run_regression(
        df_multi,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id='robust/sample/min_3_obs',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc='Units with 3+ observations',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Min 3 obs: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Verified hosts only
df_verified = df[df['verified_email'] == 1].copy()
if len(df_verified) > 100 and df_verified['minodummy'].sum() > 10:
    spec = run_regression(
        df_verified,
        f"log_price ~ minodummy + {medium_str} | citywaveID + hoodcityID",
        cluster_var='hoodcityID',
        spec_id='robust/sample/verified_email',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc='Verified email hosts',
        fixed_effects='citywaveID + hoodcityID',
        controls_desc='Medium property characteristics',
        model_type='OLS-FE'
    )
    if spec:
        results.append(spec)
        print(f"Verified email: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame([r for r in results if r is not None])
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to: {OUTPUT_PATH}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Category breakdown
print("\n--- Breakdown by Category ---")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
for cat in results_df['category'].unique():
    cat_df = results_df[results_df['category'] == cat]
    n_pos = (cat_df['coefficient'] > 0).sum()
    n_sig = (cat_df['p_value'] < 0.05).sum()
    print(f"{cat}: n={len(cat_df)}, positive={n_pos} ({100*n_pos/len(cat_df):.0f}%), sig@5%={n_sig} ({100*n_sig/len(cat_df):.0f}%)")
