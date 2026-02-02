"""
Specification Search: 120078-V1
Paper: "Can Information Reduce Ethnic Discrimination? Evidence from Airbnb"

This script runs a systematic specification search following the i4r methodology,
testing robustness of the main finding that ethnic minorities charge lower prices
on Airbnb and that this gap diminishes as information (reviews) accumulates.

Method: Panel Fixed Effects with heterogeneous treatment effects by information
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/120078-V1/data/base_airbnb_AEJ.dta'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/120078-V1/specification_results.csv'

# Paper metadata
PAPER_ID = '120078-V1'
JOURNAL = 'AEJ-Applied'
PAPER_TITLE = 'Can Information Reduce Ethnic Discrimination? Evidence from Airbnb'

print("Loading data...")
df = pd.read_stata(DATA_PATH, convert_categoricals=False)
print(f"Loaded {len(df):,} observations")

# Apply main sample restriction (listings with reviews)
df_main = df[df['Drev100'] > 0].copy()
print(f"Main sample (Drev100 > 0): {len(df_main):,} observations")

# Set up panel structure
df_main['newid'] = df_main['newid'].astype(int)
df_main['wave'] = df_main['wave'].astype(int)
df_main['citywaveID'] = df_main['citywaveID'].astype(int)

# Define control variable groups from the original code
size_controls = ['person_capacity345', 'bedrooms', 'bathrooms']
descrip_gen_controls = ['appart', 'house_loft']
descrip_spe_controls = ['couch', 'airbed', 'sofa', 'futon']
equip_controls = ['cabletv', 'wireless', 'heating', 'ac', 'elevator', 'handiaccess',
                  'doorman', 'fireplace', 'washer', 'dryer', 'parking', 'gym',
                  'pool', 'buzzer', 'hottub', 'breakfast', 'family', 'events']
rules_controls = ['people', 'extrapeople', 'cancel_policy', 'smoking_allowed', 'pets_allowed']
missing_controls = ['missingyear', 'missingcabletv', 'missingwireless', 'missingheating',
                   'missingac', 'missingelevator', 'missinghandiaccess', 'missingdoorman',
                   'missingfireplace', 'missingwasher', 'missingdryer', 'missingparking',
                   'missinggym', 'missingpool', 'missingbuzzer', 'missinghottub',
                   'missingbreakfast', 'missingfamily', 'missingevents', 'missingcancel_policy',
                   'missingnoccur_pro_true', 'missingverified_email', 'missingverified_phone',
                   'missingfacebook', 'missingverified_offline', 'missingsuperhost']
loueur_controls = ['more_1_flat', 'year2009', 'year2010', 'year2011', 'year2012',
                   'year2013', 'year2014', 'year2015', 'superhost', 'verified_email',
                   'verified_offline', 'verified_phone', 'facebook']
count_controls = ['count_descrip', 'count_about', 'count_languages', 'count_rules',
                  'picture_count', 'noccur_pro_true', 'change_pics']

# Combine into full control set (lesX)
lesX = ['sharedflat'] + size_controls + descrip_gen_controls + descrip_spe_controls + \
       equip_controls + rules_controls + missing_controls + loueur_controls + count_controls

# Filter to controls that exist in data
lesX = [c for c in lesX if c in df_main.columns]
print(f"Using {len(lesX)} control variables")

# Rating controls
rating_controls = ['accuracy_rating', 'cleanliness_rating', 'checkin_rating',
                   'communication_rating', 'location_rating', 'value_rating', 'rating_visible',
                   'accuracy_ratingNA', 'cleanliness_ratingNA', 'checkin_ratingNA',
                   'communication_ratingNA', 'location_ratingNA', 'value_ratingNA', 'rating_visibleNA']
rating_controls = [c for c in rating_controls if c in df_main.columns]

# Function to run regression and extract results
def run_spec(formula, data, spec_id, spec_tree_path, treatment_var='minodummy',
             outcome_var='log_price', cluster_var='newid', fixed_effects=None,
             controls_desc=None, sample_desc=None, model_type='Panel FE'):
    """Run a specification and return standardized results."""
    try:
        # Handle clustering
        if cluster_var and cluster_var in data.columns:
            vcov = {'CRV1': cluster_var}
        else:
            vcov = 'hetero'

        model = pf.feols(formula, data=data, vcov=vcov)

        # Extract treatment coefficient
        coef_names = model.coef().index.tolist()

        # Find the treatment variable coefficient
        treat_coef = None
        treat_se = None
        treat_pval = None

        for name in coef_names:
            if treatment_var in name and 'KKrho' not in name:
                treat_coef = model.coef()[name]
                treat_se = model.se()[name]
                treat_pval = model.pvalue()[name]
                break

        if treat_coef is None:
            return None

        t_stat = treat_coef / treat_se if treat_se > 0 else np.nan
        ci_lower = treat_coef - 1.96 * treat_se
        ci_upper = treat_coef + 1.96 * treat_se

        # Get N observations and R-squared
        try:
            n_obs = model._N
        except:
            n_obs = len(data)

        try:
            r2 = model._r2
        except:
            r2 = np.nan

        # Build coefficient vector JSON
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(treat_coef),
                'se': float(treat_se),
                'pval': float(treat_pval)
            },
            'controls': [],
            'fixed_effects': fixed_effects.split(' + ') if fixed_effects else [],
            'diagnostics': {}
        }

        # Add other coefficients
        for name in coef_names[:20]:  # Limit to first 20 for space
            if name != treatment_var and treatment_var not in name:
                coef_vector['controls'].append({
                    'var': name,
                    'coef': float(model.coef()[name]),
                    'se': float(model.se()[name]),
                    'pval': float(model.pvalue()[name])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(treat_coef),
            'std_error': float(treat_se),
            't_stat': float(t_stat),
            'p_value': float(treat_pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(n_obs),
            'r_squared': float(r2) if not np.isnan(r2) else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc or 'Full sample with reviews',
            'fixed_effects': fixed_effects or 'None',
            'controls_desc': controls_desc or 'None',
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {str(e)[:100]}")
        return None

# Store results
results = []

# Create formula helpers
def make_formula(outcome, treatment, controls, fe):
    """Create pyfixest formula string."""
    if controls:
        control_str = ' + '.join(controls)
        if fe:
            return f"{outcome} ~ {treatment} + {control_str} | {fe}"
        else:
            return f"{outcome} ~ {treatment} + {control_str}"
    else:
        if fe:
            return f"{outcome} ~ {treatment} | {fe}"
        else:
            return f"{outcome} ~ {treatment}"

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# ============================================================
# 1. BASELINE SPECIFICATIONS (Table 2 replications)
# ============================================================
print("\n1. BASELINE SPECIFICATIONS")
print("-" * 40)

# Baseline 1: City-wave FE only, no controls
print("  Running baseline_table2_col1...")
formula = "log_price ~ minodummy | citywaveID"
res = run_spec(formula, df_main, 'baseline_table2_col1',
               'methods/panel_fixed_effects.md#baseline',
               fixed_effects='citywaveID', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Baseline 2: City-wave FE + controls
print("  Running baseline_table2_col2...")
controls_subset = [c for c in lesX if c in df_main.columns][:30]  # Limit for performance
formula = make_formula('log_price', 'minodummy', controls_subset, 'citywaveID')
res = run_spec(formula, df_main, 'baseline_table2_col2',
               'methods/panel_fixed_effects.md#baseline',
               fixed_effects='citywaveID', controls_desc='Property characteristics (lesX)',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Baseline 3: Block FE (as absorbing FE)
print("  Running baseline_table2_col3...")
formula = "log_price ~ minodummy | blockID + citywaveID"
res = run_spec(formula, df_main, 'baseline_table2_col3',
               'methods/panel_fixed_effects.md#baseline',
               fixed_effects='blockID + citywaveID', controls_desc='None',
               cluster_var='blockID',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Baseline 4: Block FE + controls
print("  Running baseline_table2_col4...")
formula = make_formula('log_price', 'minodummy', controls_subset, 'blockID + citywaveID')
res = run_spec(formula, df_main, 'baseline_table2_col4',
               'methods/panel_fixed_effects.md#baseline',
               fixed_effects='blockID + citywaveID', controls_desc='Property characteristics (lesX)',
               cluster_var='blockID',
               sample_desc='Listings with reviews')
if res: results.append(res)

# ============================================================
# 2. FIXED EFFECTS VARIATIONS
# ============================================================
print("\n2. FIXED EFFECTS VARIATIONS")
print("-" * 40)

# No FE (pooled OLS)
print("  Running panel/fe/none...")
formula = "log_price ~ minodummy"
res = run_spec(formula, df_main, 'panel/fe/none',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='None', controls_desc='None',
               sample_desc='Listings with reviews', model_type='Pooled OLS')
if res: results.append(res)

# Unit FE only
print("  Running panel/fe/unit...")
formula = "log_price ~ minodummy | newid"
res = run_spec(formula, df_main, 'panel/fe/unit',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='newid', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Time (wave) FE only
print("  Running panel/fe/time...")
formula = "log_price ~ minodummy | wave"
res = run_spec(formula, df_main, 'panel/fe/time',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='wave', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Two-way FE (unit + time)
print("  Running panel/fe/twoway...")
formula = "log_price ~ minodummy | newid + wave"
res = run_spec(formula, df_main, 'panel/fe/twoway',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='newid + wave', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Neighborhood FE
print("  Running panel/fe/neighborhood...")
formula = "log_price ~ minodummy | hoodcityID"
res = run_spec(formula, df_main, 'panel/fe/neighborhood',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='hoodcityID', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# City FE only
print("  Running panel/fe/city...")
formula = "log_price ~ minodummy | city"
res = run_spec(formula, df_main, 'panel/fe/city',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='city', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# High-dimensional: Block + City-wave + Neighborhood
print("  Running panel/fe/high_dimensional...")
formula = "log_price ~ minodummy | blockID + citywaveID + hoodcityID"
res = run_spec(formula, df_main, 'panel/fe/high_dimensional',
               'methods/panel_fixed_effects.md#fixed-effects',
               fixed_effects='blockID + citywaveID + hoodcityID', controls_desc='None',
               cluster_var='blockID',
               sample_desc='Listings with reviews')
if res: results.append(res)

# ============================================================
# 3. CONTROL VARIATIONS - Leave-One-Out
# ============================================================
print("\n3. CONTROL VARIATIONS (Leave-One-Out)")
print("-" * 40)

# Define key control groups for LOO
control_groups = {
    'size': size_controls,
    'descrip_gen': descrip_gen_controls,
    'descrip_spe': descrip_spe_controls,
    'equip': equip_controls[:10],  # First 10 equipment controls
    'rules': rules_controls,
    'loueur': loueur_controls[:7],  # First 7 owner controls
    'count': count_controls
}

base_controls = [c for c in lesX if c in df_main.columns][:25]  # Reduced for performance

for group_name, group_vars in control_groups.items():
    # Drop entire group
    print(f"  Running robust/loo/drop_{group_name}...")
    remaining = [c for c in base_controls if c not in group_vars]
    if remaining:
        formula = make_formula('log_price', 'minodummy', remaining, 'citywaveID')
        res = run_spec(formula, df_main, f'robust/loo/drop_{group_name}',
                       'robustness/leave_one_out.md',
                       fixed_effects='citywaveID',
                       controls_desc=f'All controls except {group_name}',
                       sample_desc='Listings with reviews')
        if res: results.append(res)

# Individual key controls LOO
key_controls = ['sharedflat', 'person_capacity345', 'bedrooms', 'bathrooms',
                'superhost', 'verified_email', 'more_1_flat', 'picture_count']

for ctrl in key_controls:
    if ctrl in base_controls:
        print(f"  Running robust/loo/drop_{ctrl}...")
        remaining = [c for c in base_controls if c != ctrl]
        formula = make_formula('log_price', 'minodummy', remaining, 'citywaveID')
        res = run_spec(formula, df_main, f'robust/loo/drop_{ctrl}',
                       'robustness/leave_one_out.md',
                       fixed_effects='citywaveID',
                       controls_desc=f'All controls except {ctrl}',
                       sample_desc='Listings with reviews')
        if res: results.append(res)

# ============================================================
# 4. CONTROL PROGRESSION (Build-up)
# ============================================================
print("\n4. CONTROL PROGRESSION (Build-up)")
print("-" * 40)

# No controls
print("  Running robust/control/none...")
formula = "log_price ~ minodummy | citywaveID"
res = run_spec(formula, df_main, 'robust/control/none',
               'robustness/control_progression.md',
               fixed_effects='citywaveID', controls_desc='None',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Add control groups incrementally
cumulative_controls = []
for group_name, group_vars in [('size', size_controls),
                                ('descrip', descrip_gen_controls + descrip_spe_controls),
                                ('equip', equip_controls[:8]),
                                ('rules', rules_controls),
                                ('loueur', loueur_controls[:5])]:
    available = [c for c in group_vars if c in df_main.columns]
    cumulative_controls.extend(available)
    print(f"  Running robust/control/add_{group_name}...")
    formula = make_formula('log_price', 'minodummy', cumulative_controls, 'citywaveID')
    res = run_spec(formula, df_main, f'robust/control/add_{group_name}',
                   'robustness/control_progression.md',
                   fixed_effects='citywaveID',
                   controls_desc=f'Controls up to {group_name}',
                   sample_desc='Listings with reviews')
    if res: results.append(res)

# ============================================================
# 5. SAMPLE RESTRICTIONS
# ============================================================
print("\n5. SAMPLE RESTRICTIONS")
print("-" * 40)

# By number of reviews (Table 3 style)
review_cuts = [(0, 0, 'no_reviews'), (1, 4, 'reviews_1_4'),
               (5, 19, 'reviews_5_19'), (20, 49, 'reviews_20_49'),
               (50, 1000, 'reviews_50plus')]

for low, high, name in review_cuts:
    print(f"  Running robust/sample/{name}...")
    if low == 0 and high == 0:
        df_sub = df[df['review'] == 0].copy()
    else:
        df_sub = df[(df['review'] >= low) & (df['review'] <= high)].copy()

    if len(df_sub) > 1000:
        formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
        res = run_spec(formula, df_sub, f'robust/sample/{name}',
                       'robustness/sample_restrictions.md',
                       fixed_effects='citywaveID',
                       controls_desc='Property characteristics',
                       sample_desc=f'Reviews: {low}-{high}')
        if res: results.append(res)

# By wave (time period)
for wave_cut in [5, 10, 15]:
    print(f"  Running robust/sample/wave_le_{wave_cut}...")
    df_sub = df_main[df_main['wave'] <= wave_cut].copy()
    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_sub, f'robust/sample/wave_le_{wave_cut}',
                   'robustness/sample_restrictions.md',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc=f'Waves <= {wave_cut}')
    if res: results.append(res)

    print(f"  Running robust/sample/wave_gt_{wave_cut}...")
    df_sub = df_main[df_main['wave'] > wave_cut].copy()
    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_sub, f'robust/sample/wave_gt_{wave_cut}',
                   'robustness/sample_restrictions.md',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc=f'Waves > {wave_cut}')
    if res: results.append(res)

# By city
major_cities = ['paris', 'new-york', 'london', 'los-angeles', 'barcelona', 'berlin']
for city in major_cities:
    print(f"  Running robust/sample/city_{city.replace('-','_')}...")
    df_sub = df_main[df_main['city'] == city].copy()
    if len(df_sub) > 5000:
        formula = make_formula('log_price', 'minodummy', base_controls[:15], 'wave')
        res = run_spec(formula, df_sub, f'robust/sample/city_{city.replace("-","_")}',
                       'robustness/sample_restrictions.md',
                       fixed_effects='wave',
                       controls_desc='Property characteristics',
                       sample_desc=f'City: {city}')
        if res: results.append(res)

# Drop each city
print("  Running drop city specifications...")
for city in major_cities[:4]:  # Limit to 4 major cities
    print(f"  Running robust/sample/drop_city_{city.replace('-','_')}...")
    df_sub = df_main[df_main['city'] != city].copy()
    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_sub, f'robust/sample/drop_city_{city.replace("-","_")}',
                   'robustness/sample_restrictions.md',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc=f'Excluding {city}')
    if res: results.append(res)

# By property type
print("  Running robust/sample/entire_apartment...")
df_sub = df_main[df_main['entireflat'] == 1].copy()
formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_sub, 'robust/sample/entire_apartment',
               'robustness/sample_restrictions.md',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Entire apartments only')
if res: results.append(res)

print("  Running robust/sample/shared_flat...")
df_sub = df_main[df_main['sharedflat'] == 1].copy()
if len(df_sub) > 1000:
    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_sub, 'robust/sample/shared_flat',
                   'robustness/sample_restrictions.md',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc='Shared flats only')
    if res: results.append(res)

# ============================================================
# 6. OUTLIER HANDLING
# ============================================================
print("\n6. OUTLIER HANDLING")
print("-" * 40)

# Winsorize price at different levels
for pct in [1, 5, 10]:
    print(f"  Running robust/sample/winsorize_{pct}pct...")
    df_wins = df_main.copy()
    lower = df_wins['log_price'].quantile(pct/100)
    upper = df_wins['log_price'].quantile(1 - pct/100)
    df_wins['log_price'] = df_wins['log_price'].clip(lower=lower, upper=upper)

    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_wins, f'robust/sample/winsorize_{pct}pct',
                   'robustness/sample_restrictions.md',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc=f'Price winsorized at {pct}%')
    if res: results.append(res)

# Trim extreme values
print("  Running robust/sample/trim_1pct...")
lower = df_main['log_price'].quantile(0.01)
upper = df_main['log_price'].quantile(0.99)
df_trim = df_main[(df_main['log_price'] > lower) & (df_main['log_price'] < upper)].copy()
formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_trim, 'robust/sample/trim_1pct',
               'robustness/sample_restrictions.md',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Trimmed top/bottom 1%')
if res: results.append(res)

# ============================================================
# 7. CLUSTERING VARIATIONS
# ============================================================
print("\n7. CLUSTERING VARIATIONS")
print("-" * 40)

clustering_vars = [
    ('newid', 'Listing'),
    ('blockID', 'Block'),
    ('hoodcityID', 'Neighborhood-city'),
    ('city', 'City')
]

for cluster, desc in clustering_vars:
    print(f"  Running robust/cluster/{cluster}...")
    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_main, f'robust/cluster/{cluster}',
                   'robustness/clustering_variations.md',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   cluster_var=cluster,
                   sample_desc='Listings with reviews')
    if res: results.append(res)

# Robust (heteroskedasticity-consistent) SEs
print("  Running robust/cluster/robust_hc...")
try:
    formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
    model = pf.feols(formula, data=df_main, vcov='hetero')
    treat_coef = model.coef()['minodummy']
    treat_se = model.se()['minodummy']
    treat_pval = model.pvalue()['minodummy']

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/cluster/robust_hc',
        'spec_tree_path': 'robustness/clustering_variations.md',
        'outcome_var': 'log_price',
        'treatment_var': 'minodummy',
        'coefficient': float(treat_coef),
        'std_error': float(treat_se),
        't_stat': float(treat_coef / treat_se),
        'p_value': float(treat_pval),
        'ci_lower': float(treat_coef - 1.96 * treat_se),
        'ci_upper': float(treat_coef + 1.96 * treat_se),
        'n_obs': int(model._N),
        'r_squared': float(model._r2),
        'coefficient_vector_json': json.dumps({'treatment': {'var': 'minodummy', 'coef': float(treat_coef)}}),
        'sample_desc': 'Listings with reviews',
        'fixed_effects': 'citywaveID',
        'controls_desc': 'Property characteristics',
        'cluster_var': 'Robust HC',
        'model_type': 'Panel FE',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
except Exception as e:
    print(f"  Error: {e}")

# ============================================================
# 8. ALTERNATIVE OUTCOMES
# ============================================================
print("\n8. ALTERNATIVE OUTCOMES")
print("-" * 40)

# Price in levels
print("  Running robust/outcome/price_levels...")
formula = make_formula('price', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_main, 'robust/outcome/price_levels',
               'robustness/measurement.md',
               outcome_var='price',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Standardized log price (z-score within city-wave)
print("  Running robust/outcome/log_price_std...")
df_main['log_price_std'] = df_main.groupby('citywaveID')['log_price'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)
formula = make_formula('log_price_std', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_main, 'robust/outcome/log_price_std',
               'robustness/measurement.md',
               outcome_var='log_price_std',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Listings with reviews')
if res: results.append(res)

# ============================================================
# 9. ALTERNATIVE TREATMENT DEFINITIONS
# ============================================================
print("\n9. ALTERNATIVE TREATMENT DEFINITIONS")
print("-" * 40)

# Black hosts only
print("  Running robust/treatment/black...")
formula = make_formula('log_price', 'black', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_main, 'robust/treatment/black',
               'robustness/measurement.md',
               treatment_var='black',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Arabic/African hosts only
print("  Running robust/treatment/arabic_african...")
formula = make_formula('log_price', 'arabic_african', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_main, 'robust/treatment/arabic_african',
               'robustness/measurement.md',
               treatment_var='arabic_african',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Continuous minority measure (number of minority names)
if 'nber_mino_names' in df_main.columns:
    print("  Running robust/treatment/continuous_minority...")
    formula = make_formula('log_price', 'nber_mino_names', base_controls[:15], 'citywaveID')
    res = run_spec(formula, df_main, 'robust/treatment/continuous_minority',
                   'robustness/measurement.md',
                   treatment_var='nber_mino_names',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc='Listings with reviews')
    if res: results.append(res)

# ============================================================
# 10. FUNCTIONAL FORM VARIATIONS
# ============================================================
print("\n10. FUNCTIONAL FORM VARIATIONS")
print("-" * 40)

# IHS transformation of price
print("  Running robust/funcform/ihs_price...")
df_main['ihs_price'] = np.arcsinh(df_main['price'])
formula = make_formula('ihs_price', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_main, 'robust/funcform/ihs_price',
               'robustness/functional_form.md',
               outcome_var='ihs_price',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Include polynomial in reviews
print("  Running robust/funcform/reviews_polynomial...")
df_main['rev100_sq'] = df_main['rev100'] ** 2
formula = "log_price ~ minodummy + rev100 + rev100_sq | citywaveID"
res = run_spec(formula, df_main, 'robust/funcform/reviews_polynomial',
               'robustness/functional_form.md',
               fixed_effects='citywaveID',
               controls_desc='Reviews + reviews squared',
               sample_desc='Listings with reviews')
if res: results.append(res)

# ============================================================
# 11. HETEROGENEITY ANALYSES
# ============================================================
print("\n11. HETEROGENEITY ANALYSES")
print("-" * 40)

# By superhost status
print("  Running robust/heterogeneity/superhost...")
df_main['mino_superhost'] = df_main['minodummy'] * df_main['superhost']
formula = "log_price ~ minodummy + mino_superhost + superhost | citywaveID"
res = run_spec(formula, df_main, 'robust/heterogeneity/superhost',
               'robustness/heterogeneity.md',
               fixed_effects='citywaveID',
               controls_desc='Superhost interaction',
               sample_desc='Listings with reviews')
if res: results.append(res)

# By shared vs entire apartment
print("  Running robust/heterogeneity/sharedflat...")
df_main['mino_shared'] = df_main['minodummy'] * df_main['sharedflat']
formula = "log_price ~ minodummy + mino_shared + sharedflat | citywaveID"
res = run_spec(formula, df_main, 'robust/heterogeneity/sharedflat',
               'robustness/heterogeneity.md',
               fixed_effects='citywaveID',
               controls_desc='Shared flat interaction',
               sample_desc='Listings with reviews')
if res: results.append(res)

# By multiple listings
print("  Running robust/heterogeneity/more_1_flat...")
df_main['mino_multiple'] = df_main['minodummy'] * df_main['more_1_flat']
formula = "log_price ~ minodummy + mino_multiple + more_1_flat | citywaveID"
res = run_spec(formula, df_main, 'robust/heterogeneity/more_1_flat',
               'robustness/heterogeneity.md',
               fixed_effects='citywaveID',
               controls_desc='Multiple listings interaction',
               sample_desc='Listings with reviews')
if res: results.append(res)

# By verification status
print("  Running robust/heterogeneity/verified...")
df_main['mino_verified'] = df_main['minodummy'] * df_main['verified_email']
formula = "log_price ~ minodummy + mino_verified + verified_email | citywaveID"
res = run_spec(formula, df_main, 'robust/heterogeneity/verified',
               'robustness/heterogeneity.md',
               fixed_effects='citywaveID',
               controls_desc='Verification interaction',
               sample_desc='Listings with reviews')
if res: results.append(res)

# By number of reviews (key mechanism)
print("  Running robust/heterogeneity/high_reviews...")
df_main['high_reviews'] = (df_main['review'] >= 20).astype(int)
df_main['mino_highrev'] = df_main['minodummy'] * df_main['high_reviews']
formula = "log_price ~ minodummy + mino_highrev + high_reviews | citywaveID"
res = run_spec(formula, df_main, 'robust/heterogeneity/high_reviews',
               'robustness/heterogeneity.md',
               fixed_effects='citywaveID',
               controls_desc='High reviews interaction (20+)',
               sample_desc='Listings with reviews')
if res: results.append(res)

# By European vs US/Canadian cities
print("  Running robust/heterogeneity/europe...")
df_main['europe'] = df_main['euro'].fillna(0)
df_main['mino_europe'] = df_main['minodummy'] * df_main['europe']
formula = "log_price ~ minodummy + mino_europe + europe | citywaveID"
res = run_spec(formula, df_main, 'robust/heterogeneity/europe',
               'robustness/heterogeneity.md',
               fixed_effects='citywaveID',
               controls_desc='European city interaction',
               sample_desc='Listings with reviews')
if res: results.append(res)

# ============================================================
# 12. PLACEBO TESTS
# ============================================================
print("\n12. PLACEBO TESTS")
print("-" * 40)

# Fake treatment: random assignment
print("  Running robust/placebo/random_treatment...")
np.random.seed(42)
df_main['fake_mino'] = np.random.binomial(1, df_main['minodummy'].mean(), len(df_main))
formula = make_formula('log_price', 'fake_mino', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_main, 'robust/placebo/random_treatment',
               'robustness/placebo_tests.md',
               treatment_var='fake_mino',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Listings with reviews')
if res: results.append(res)

# Placebo outcome: should not be affected
# Use picture_count change as placebo (shouldn't correlate with minority status)
if 'change_pics' in df_main.columns:
    print("  Running robust/placebo/picture_change...")
    formula = make_formula('change_pics', 'minodummy', base_controls[:10], 'citywaveID')
    res = run_spec(formula, df_main, 'robust/placebo/picture_change',
                   'robustness/placebo_tests.md',
                   outcome_var='change_pics',
                   fixed_effects='citywaveID',
                   controls_desc='Property characteristics',
                   sample_desc='Listings with reviews')
    if res: results.append(res)

# ============================================================
# 13. ADDITIONAL ROBUSTNESS - ESTIMATION METHODS
# ============================================================
print("\n13. ADDITIONAL ESTIMATION METHODS")
print("-" * 40)

# First differences
print("  Running panel/method/first_diff...")
df_sorted = df_main.sort_values(['newid', 'wave'])
df_sorted['log_price_diff'] = df_sorted.groupby('newid')['log_price'].diff()
df_sorted['minodummy_diff'] = df_sorted.groupby('newid')['minodummy'].diff()
df_fd = df_sorted.dropna(subset=['log_price_diff', 'minodummy_diff'])

if len(df_fd) > 10000:
    formula = "log_price_diff ~ minodummy_diff | wave"
    res = run_spec(formula, df_fd, 'panel/method/first_diff',
                   'methods/panel_fixed_effects.md#estimation-method',
                   treatment_var='minodummy_diff',
                   outcome_var='log_price_diff',
                   fixed_effects='wave',
                   controls_desc='First differences',
                   sample_desc='First differences',
                   model_type='First Differences')
    if res: results.append(res)

# ============================================================
# 14. STRUCTURAL MODEL APPROXIMATION
# ============================================================
print("\n14. STRUCTURAL MODEL APPROXIMATION")
print("-" * 40)

# Create KKrho proxy (reviews / (rho + reviews)) with different rho values
for rho in [0.1, 0.15, 0.2, 0.3]:
    print(f"  Running structural/rho_{rho}...")
    df_main[f'KKrho_{rho}'] = df_main['rev100'] / (rho + df_main['rev100'])
    df_main[f'mino_KKrho_{rho}'] = -df_main['minodummy'] * df_main[f'KKrho_{rho}']

    formula = f"log_price ~ minodummy + mino_KKrho_{rho} + KKrho_{rho} | newid + citywaveID"
    res = run_spec(formula, df_main, f'structural/rho_{rho}',
                   'methods/panel_fixed_effects.md',
                   fixed_effects='newid + citywaveID',
                   controls_desc=f'Learning parameter rho={rho}',
                   sample_desc='Listings with reviews')
    if res: results.append(res)

# ============================================================
# 15. ADDITIONAL SAMPLE SPLITS
# ============================================================
print("\n15. ADDITIONAL SAMPLE SPLITS")
print("-" * 40)

# High vs low price markets
print("  Running robust/sample/high_price_market...")
median_price = df_main.groupby('hoodcityID')['price'].transform('median')
df_main['high_price_market'] = (df_main['price'] >= median_price).astype(int)
df_high = df_main[df_main['high_price_market'] == 1].copy()
formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_high, 'robust/sample/high_price_market',
               'robustness/sample_restrictions.md',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Above median price markets')
if res: results.append(res)

print("  Running robust/sample/low_price_market...")
df_low = df_main[df_main['high_price_market'] == 0].copy()
formula = make_formula('log_price', 'minodummy', base_controls[:15], 'citywaveID')
res = run_spec(formula, df_low, 'robust/sample/low_price_market',
               'robustness/sample_restrictions.md',
               fixed_effects='citywaveID',
               controls_desc='Property characteristics',
               sample_desc='Below median price markets')
if res: results.append(res)

# By bedroom count
for beds in [1, 2, 3]:
    print(f"  Running robust/sample/{beds}_bedroom...")
    df_sub = df_main[df_main['bedrooms'] == beds].copy()
    if len(df_sub) > 10000:
        formula = make_formula('log_price', 'minodummy',
                              [c for c in base_controls if c != 'bedrooms'][:15],
                              'citywaveID')
        res = run_spec(formula, df_sub, f'robust/sample/{beds}_bedroom',
                       'robustness/sample_restrictions.md',
                       fixed_effects='citywaveID',
                       controls_desc='Property characteristics (excl bedrooms)',
                       sample_desc=f'{beds} bedroom listings')
        if res: results.append(res)

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to: {OUTPUT_PATH}")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total specifications: {len(results_df)}")
if len(results_df) > 0:
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"\nCoefficient statistics:")
    print(f"  Mean: {results_df['coefficient'].mean():.4f}")
    print(f"  Median: {results_df['coefficient'].median():.4f}")
    print(f"  Min: {results_df['coefficient'].min():.4f}")
    print(f"  Max: {results_df['coefficient'].max():.4f}")
    print(f"  Std: {results_df['coefficient'].std():.4f}")
else:
    print("No results collected - all specifications failed")

print("\nDone!")
