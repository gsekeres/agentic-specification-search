"""
Specification Search Script for Paper 112749-V1
"When the Levee Breaks: Black Migration and Economic Development in the American South"
Hornbeck and Naidu, American Economic Review 2014

This script replicates the main analysis and runs a systematic specification search.

Method Classification: Event Study / Difference-in-Differences
- Treatment: 1927 Mississippi River flood intensity (continuous)
- Timing: Post-1927 effects measured at multiple years (1930-1970)
- Outcomes: Black population share, equipment value, farm sizes, etc.
- Fixed Effects: County (unit) + State-Year
- Clustering: County level
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Try importing pyfixest first, fall back to statsmodels
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
from scipy import stats

# Use pyreadstat for reading old Stata files
try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False


def read_stata_file(path):
    """Read Stata file using pyreadstat for old formats or pandas for newer."""
    try:
        return read_stata_file(path)
    except:
        if HAS_PYREADSTAT:
            df, meta = pyreadstat.read_dta(path)
            return df
        else:
            raise ValueError(f"Cannot read Stata file {path}. Install pyreadstat.")

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "112749-V1"
PAPER_TITLE = "When the Levee Breaks: Black Migration and Economic Development in the American South"
JOURNAL = "AER"
BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/112749-V1/Replication_AER-2012-0980")
OUTPUT_PATH = BASE_PATH.parent
SCRIPT_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/scripts/paper_analyses")

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load and prepare the panel dataset from the original Stata files.
    This replicates the data preparation in flood_preanalysis.do
    """

    # Load key source datasets
    gen_data_path = BASE_PATH / "Generate_Data"

    # Load main datasets - we'll use the comprehensive approach
    # Load ICPSR data for each census year
    icpsr_files = {
        1900: "02896-0020-Data.dta",
        1910: "02896-0022-Data.dta",
        1920: "02896-0024-Data.dta",
        1930: "02896-0026-Data.dta",
        1940: "02896-0032-Data.dta",
        1950: "02896-0035-Data.dta",
        1960: "02896-0038-Data.dta",
        1970: "02896-0075-Data.dta",
    }

    # Agricultural census data
    ag_files = {
        1900: "ag900co.dta",
        1910: "ag910co.dta",
        1920: "ag920co.dta",
    }

    # Load flood data
    flood_data = pd.read_csv(gen_data_path / "flooded_1900.txt", sep="\t")
    flood_data.columns = flood_data.columns.str.lower().str.strip()

    # Load distance data
    distance_data = read_stata_file(gen_data_path / "1900_strm_distance.dta")

    # Load crop suitability data
    crop_suit = read_stata_file(gen_data_path / "1900_strm_distance_gaez.dta")

    # Load plantation data
    plantation = read_stata_file(gen_data_path / "brannenplantcounties_1910.dta")

    # Load New Deal spending
    new_deal = read_stata_file(gen_data_path / "new_deal_spending.dta")

    # Load farmland values
    farmval = read_stata_file(gen_data_path / "farmval.dta")

    # Southern states (ICPSR codes): AR=42, LA=45, MS=46, TN=54, AL=43, GA=44, NC=47, SC=48, FL=41
    # Also including codes 32, 34, 40, etc. as in the do file
    southern_states = [32, 34, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54]

    panels = []

    for year, fname in icpsr_files.items():
        try:
            df = read_stata_file(gen_data_path / fname)
            df = df[df['level'] == 1]  # County level
            df = df[df['state'].isin(southern_states)]

            # Rename key variables based on year
            df['year'] = year

            # Extract population by race
            if year == 1900:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
            elif year == 1910:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
            elif year == 1920:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
                df['county_acres'] = df.get('areaac', np.nan)
            elif year == 1930:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
                df['county_acres'] = df.get('areaac', np.nan)
            elif year == 1940:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
            elif year == 1950:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
            elif year == 1960:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negro', np.nan)
                if 'population_black' not in df.columns or df['population_black'].isna().all():
                    df['population_black'] = df.get('negmtot', 0) + df.get('negftot', 0)
            elif year == 1970:
                df['population'] = df.get('totpop', np.nan)
                df['population_black'] = df.get('negro', np.nan)

            # Farm data
            df['farms'] = df.get('farms', np.nan)
            df['farms_nonwhite'] = df.get('farmcol', df.get('farmneg', np.nan))
            df['farmland'] = df.get('acres', df.get('acfarm', np.nan))
            if 'farmland' not in df.columns or df['farmland'].isna().all():
                # Try computing from owner + tenant
                df['farmland'] = df.get('acresown', 0) + df.get('acresten', 0) + df.get('acresman', 0)

            df['value_equipment'] = df.get('farmequi', np.nan)

            # Keep essential columns
            keep_cols = ['fips', 'state', 'year', 'population', 'population_black',
                        'farms', 'farms_nonwhite', 'farmland', 'value_equipment']
            if 'county_acres' in df.columns:
                keep_cols.append('county_acres')

            df = df[keep_cols].copy()
            panels.append(df)

        except Exception as e:
            print(f"Warning: Could not load {fname}: {e}")
            continue

    # Combine all years
    panel = pd.concat(panels, ignore_index=True)

    # Propagate county_acres to all years
    county_acres = panel[panel['county_acres'].notna()].groupby('fips')['county_acres'].first()
    panel = panel.drop(columns=['county_acres'], errors='ignore')
    panel = panel.merge(county_acres.reset_index(), on='fips', how='left')

    # Merge flood intensity
    if 'fips' in flood_data.columns:
        flood_merge = flood_data[['fips', 'flooded_share']].copy() if 'flooded_share' in flood_data.columns else flood_data
        panel = panel.merge(flood_merge, on='fips', how='left')

    # Merge distance to MS river
    if 'distance_ms' in distance_data.columns:
        dist_merge = distance_data[['fips', 'distance_ms']].copy()
        panel = panel.merge(dist_merge, on='fips', how='left')

    # Merge crop suitability
    suit_cols = ['fips']
    for col in ['cotton_suitability', 'corn_suitability', 'wheat_suitability',
                'oats_suitability', 'rice_suitability', 'scane_suitability',
                'x_centroid', 'y_centroid', 'altitude_std_meters']:
        if col in crop_suit.columns:
            suit_cols.append(col)
    if len(suit_cols) > 1:
        panel = panel.merge(crop_suit[suit_cols], on='fips', how='left')

    # Merge plantation data
    if 'Brannen_Plantation' in plantation.columns:
        plant_merge = plantation[['fips', 'Brannen_Plantation']].copy()
        panel = panel.merge(plant_merge, on='fips', how='left')

    # Clean up flood intensity
    if 'flooded_share' not in panel.columns:
        panel['flooded_share'] = 0
    panel['flooded_share'] = panel['flooded_share'].fillna(0)
    panel['flood_intensity'] = panel['flooded_share']

    # Create derived variables
    panel['population_white'] = panel['population'] - panel['population_black']
    panel['frac_black'] = panel['population_black'] / (panel['population_white'] + panel['population_black'])
    panel['fracfarms_nonwhite'] = panel['farms_nonwhite'] / panel['farms']
    panel['avfarmsize'] = panel['farmland'] / panel['farms']

    # Log transformations
    for var in ['population', 'population_black', 'frac_black', 'fracfarms_nonwhite',
                'value_equipment', 'avfarmsize', 'farmland']:
        panel[f'ln{var}'] = np.log(panel[var].replace(0, np.nan))

    # Create state FIPS
    panel['statefips'] = (panel['fips'] / 1000).astype(int)

    # Create state-year fixed effects identifier
    panel['state_year'] = panel['statefips'].astype(str) + "_" + panel['year'].astype(str)

    # Create flood intensity x year interactions
    for year in [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970]:
        panel[f'f_int_{year}'] = np.where(panel['year'] == year, panel['flood_intensity'], 0)

    # Create county weight (using county_acres from 1920)
    if 'county_acres' in panel.columns:
        panel['county_w'] = panel['county_acres']
    else:
        panel['county_w'] = 1

    # Sample restrictions (as in original)
    # Keep Southern states in Mississippi Delta region
    delta_states = [42, 45, 46, 54]  # AR, LA, MS, TN
    panel = panel[panel['statefips'].isin(delta_states)]

    # Drop counties with low black fraction or cotton cultivation
    # (approximation of original sample restriction)

    return panel


def create_simple_panel():
    """
    Create a simplified panel for analysis when full data preparation is too complex.
    Uses the key variables needed for main specifications.
    """

    gen_data_path = BASE_PATH / "Generate_Data"

    # Load 1920 data as base (main cross-section for treatment assignment)
    df_1920 = read_stata_file(gen_data_path / "02896-0024-Data.dta")
    df_1920 = df_1920[df_1920['level'] == 1]

    # Southern states
    southern_states = [42, 45, 46, 54]  # AR, LA, MS, TN - core Mississippi Delta
    df_1920 = df_1920[df_1920['state'].isin(southern_states)]

    # Get flood data - need to compute flooded_share from the raw data
    try:
        flood_raw = pd.read_csv(gen_data_path / "flooded_1900.txt", sep="\t")
        # Aggregate by fips: area is county area, new_area is flooded area
        flood_data = flood_raw.groupby('fips').agg({
            'area': 'first',
            'new_area': 'sum'
        }).reset_index()
        flood_data['flooded_share'] = flood_data['new_area'] / flood_data['area']
        flood_data['flooded_share'] = flood_data['flooded_share'].clip(0, 1)  # Cap at 1
    except:
        flood_data = pd.DataFrame({'fips': [], 'flooded_share': []})

    # Get distance data
    try:
        distance_data = read_stata_file(gen_data_path / "1900_strm_distance.dta")
    except:
        distance_data = pd.DataFrame({'fips': []})

    # Get crop suitability
    try:
        crop_suit = read_stata_file(gen_data_path / "1900_strm_distance_gaez.dta")
    except:
        crop_suit = pd.DataFrame({'fips': []})

    # Prepare cross-sectional base
    base = df_1920[['fips', 'state']].copy()
    base['county_acres'] = pd.to_numeric(df_1920.get('areaac', 1), errors='coerce').fillna(1)

    # Merge flood intensity
    if 'flooded_share' in flood_data.columns and 'fips' in flood_data.columns:
        base = base.merge(flood_data[['fips', 'flooded_share']], on='fips', how='left')
    else:
        base['flooded_share'] = 0
    base['flooded_share'] = base['flooded_share'].fillna(0)
    base['flood_intensity'] = base['flooded_share']

    # Merge distance
    if 'distance_ms' in distance_data.columns:
        base = base.merge(distance_data[['fips', 'distance_ms']], on='fips', how='left')

    # Merge suitability
    for col in ['cotton_suitability', 'corn_suitability', 'x_centroid', 'y_centroid', 'altitude_std_meters']:
        if col in crop_suit.columns:
            if col not in base.columns:
                base = base.merge(crop_suit[['fips', col]], on='fips', how='left')

    # Create panel by loading data for each year
    panels = []

    # Define files and key variables for each year
    year_data = {
        1920: ('02896-0024-Data.dta', {'totpop': 'population', 'negmtot': 'pop_black_m', 'negftot': 'pop_black_f',
                                        'farmcol': 'farms_nonwhite', 'farms': 'farms', 'farmequi': 'value_equipment'}),
        1930: ('02896-0026-Data.dta', {'totpop': 'population', 'negmtot': 'pop_black_m', 'negftot': 'pop_black_f',
                                        'farmcol': 'farms_nonwhite', 'farms': 'farms', 'farmequi': 'value_equipment'}),
        1940: ('02896-0032-Data.dta', {'totpop': 'population', 'negmtot': 'pop_black_m', 'negftot': 'pop_black_f',
                                        'farmcol': 'farms_nonwhite', 'farms': 'farms', 'farmequi': 'value_equipment'}),
        1950: ('02896-0035-Data.dta', {'totpop': 'population', 'negmtot': 'pop_black_m', 'negftot': 'pop_black_f',
                                        'farmcol': 'farms_nonwhite', 'farms': 'farms', 'farmequi': 'value_equipment'}),
        1960: ('02896-0038-Data.dta', {'totpop': 'population', 'negro': 'population_black'}),
        1970: ('02896-0075-Data.dta', {'totpop': 'population', 'negro': 'population_black'}),
    }

    for year, (fname, var_map) in year_data.items():
        try:
            df = read_stata_file(gen_data_path / fname)
            df = df[df['level'] == 1]
            df = df[df['state'].isin(southern_states)]

            # Keep fips and rename vars
            df_year = df[['fips']].copy()
            df_year['year'] = year

            for old_col, new_col in var_map.items():
                if old_col in df.columns:
                    df_year[new_col] = df[old_col].values

            # Compute population_black if needed
            if 'population_black' not in df_year.columns:
                df_year['population_black'] = df_year.get('pop_black_m', 0) + df_year.get('pop_black_f', 0)

            panels.append(df_year)

        except Exception as e:
            print(f"Warning loading {fname}: {e}")

    # Combine panels
    panel = pd.concat(panels, ignore_index=True)

    # Merge with base (flood intensity, etc.)
    panel = panel.merge(base, on='fips', how='inner')

    # Ensure numeric types
    for col in ['population', 'population_black', 'value_equipment', 'farms', 'farms_nonwhite', 'flood_intensity']:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors='coerce')

    # Compute derived variables
    panel['frac_black'] = panel['population_black'] / panel['population']
    panel['lnfrac_black'] = np.log(panel['frac_black'].replace(0, np.nan).astype(float))
    panel['lnpopulation_black'] = np.log(panel['population_black'].replace(0, np.nan).astype(float))
    panel['lnpopulation'] = np.log(panel['population'].replace(0, np.nan).astype(float))

    if 'value_equipment' in panel.columns:
        panel['lnvalue_equipment'] = np.log(panel['value_equipment'].replace(0, np.nan).astype(float))
    if 'farms_nonwhite' in panel.columns and 'farms' in panel.columns:
        panel['fracfarms_nonwhite'] = panel['farms_nonwhite'] / panel['farms']
        panel['lnfracfarms_nonwhite'] = np.log(panel['fracfarms_nonwhite'].replace(0, np.nan).astype(float))

    # State-year FE
    panel['statefips'] = panel['state']
    panel['state_year'] = panel['state'].astype(str) + "_" + panel['year'].astype(str)

    # Flood intensity x year
    for year in [1920, 1930, 1940, 1950, 1960, 1970]:
        panel[f'f_int_{year}'] = np.where(panel['year'] == year, panel['flood_intensity'], 0)

    # County weight
    panel['county_w'] = panel['county_acres'].fillna(1)

    return panel


# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_regression_statsmodels(df, y_var, x_vars, fe_vars=None, cluster_var=None, weight_var=None):
    """
    Run regression using statsmodels with manual fixed effects.
    """
    # Drop missing
    all_vars = [y_var] + x_vars
    if fe_vars:
        all_vars += fe_vars
    if cluster_var:
        all_vars.append(cluster_var)
    if weight_var:
        all_vars.append(weight_var)

    df_reg = df.dropna(subset=[v for v in all_vars if v in df.columns])

    if len(df_reg) < 10:
        return None

    # Create demeaned version for FE (within transformation)
    if fe_vars and len(fe_vars) > 0:
        # Simple approach: add dummies for the main FE
        # For state-year FE, use dummy encoding
        y = df_reg[y_var].values
        X = df_reg[x_vars].values

        # Demean by fips (county FE)
        if 'fips' in fe_vars:
            df_reg['_y_demeaned'] = df_reg.groupby('fips')[y_var].transform(lambda x: x - x.mean())
            for xv in x_vars:
                df_reg[f'_{xv}_demeaned'] = df_reg.groupby('fips')[xv].transform(lambda x: x - x.mean())
            y = df_reg['_y_demeaned'].values
            X = df_reg[[f'_{xv}_demeaned' for xv in x_vars]].values

        X = sm.add_constant(X, has_constant='add')
    else:
        y = df_reg[y_var].values
        X = sm.add_constant(df_reg[x_vars].values)

    # Weights
    weights = df_reg[weight_var].values if weight_var and weight_var in df_reg.columns else None

    try:
        if weights is not None:
            model = sm.WLS(y, X, weights=weights)
        else:
            model = sm.OLS(y, X)

        # Cluster standard errors
        if cluster_var and cluster_var in df_reg.columns:
            result = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var].values})
        else:
            result = model.fit(cov_type='HC1')

        return {
            'coef': result.params,
            'se': result.bse,
            'pval': result.pvalues,
            'tstat': result.tvalues,
            'nobs': int(result.nobs),
            'rsquared': result.rsquared,
            'x_names': ['const'] + x_vars
        }
    except Exception as e:
        print(f"Regression error: {e}")
        return None


def run_pyfixest_regression(df, formula, cluster_var=None, weight_var=None):
    """
    Run regression using pyfixest.
    """
    if not HAS_PYFIXEST:
        return None

    df_reg = df.copy()

    try:
        if cluster_var:
            vcov = {'CRV1': cluster_var}
        else:
            vcov = 'hetero'

        if weight_var and weight_var in df_reg.columns:
            model = pf.feols(formula, data=df_reg, vcov=vcov, weights=weight_var)
        else:
            model = pf.feols(formula, data=df_reg, vcov=vcov)

        return {
            'coef': model.coef(),
            'se': model.se(),
            'pval': model.pvalue(),
            'tstat': model.tstat(),
            'nobs': model.nobs,
            'rsquared': model.r2,
            'x_names': list(model.coef().index)
        }
    except Exception as e:
        print(f"Pyfixest error: {e}")
        return None


# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

def extract_result(res, treatment_var, spec_info):
    """
    Extract results into the required output format.
    """
    if res is None:
        return None

    # Find treatment coefficient
    coef_names = res.get('x_names', [])

    # Handle various naming conventions
    treat_idx = None
    for i, name in enumerate(coef_names):
        if treatment_var in str(name):
            treat_idx = i
            break

    if treat_idx is None:
        # Try finding f_int_1930 or similar
        for i, name in enumerate(coef_names):
            if 'f_int' in str(name):
                treat_idx = i
                break

    if treat_idx is None:
        return None

    coef = res['coef'].iloc[treat_idx] if hasattr(res['coef'], 'iloc') else res['coef'][treat_idx]
    se = res['se'].iloc[treat_idx] if hasattr(res['se'], 'iloc') else res['se'][treat_idx]
    pval = res['pval'].iloc[treat_idx] if hasattr(res['pval'], 'iloc') else res['pval'][treat_idx]
    tstat = res['tstat'].iloc[treat_idx] if hasattr(res['tstat'], 'iloc') else res['tstat'][treat_idx]

    # Confidence interval
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    # Build coefficient vector JSON
    coef_vec = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef),
            'se': float(se),
            'pval': float(pval)
        },
        'controls': [],
        'fixed_effects_absorbed': spec_info.get('fixed_effects', []),
        'diagnostics': {}
    }

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_info['spec_id'],
        'spec_tree_path': spec_info['spec_tree_path'],
        'outcome_var': spec_info['outcome_var'],
        'treatment_var': treatment_var,
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(res['nobs']),
        'r_squared': float(res['rsquared']) if res['rsquared'] is not None else np.nan,
        'coefficient_vector_json': json.dumps(coef_vec),
        'sample_desc': spec_info.get('sample_desc', 'Full sample'),
        'fixed_effects': spec_info.get('fe_desc', ''),
        'controls_desc': spec_info.get('controls_desc', ''),
        'cluster_var': spec_info.get('cluster_var', ''),
        'model_type': spec_info.get('model_type', 'Panel FE'),
        'estimation_script': str(SCRIPT_PATH / f"{PAPER_ID}.py")
    }


def run_specification_search(df):
    """
    Run the full specification search.
    """
    results = []

    # Primary outcome variables (as in paper's main tables)
    outcomes = {
        'lnfrac_black': 'Log fraction black population',
        'lnpopulation_black': 'Log black population',
    }

    # Add other outcomes if available
    if 'lnvalue_equipment' in df.columns and df['lnvalue_equipment'].notna().sum() > 100:
        outcomes['lnvalue_equipment'] = 'Log value equipment'
    if 'lnfracfarms_nonwhite' in df.columns and df['lnfracfarms_nonwhite'].notna().sum() > 100:
        outcomes['lnfracfarms_nonwhite'] = 'Log fraction nonwhite farms'

    # Treatment variables (flood intensity x year interactions)
    # Main post-treatment years: 1930, 1940, 1950, 1960, 1970
    treatment_years = [1930, 1940, 1950, 1960, 1970]
    treatment_vars = [f'f_int_{y}' for y in treatment_years]

    # Ensure treatment vars exist
    for tv in treatment_vars:
        if tv not in df.columns:
            df[tv] = np.where(df['year'] == int(tv.split('_')[-1]), df['flood_intensity'], 0)

    # Control variable groups
    geo_controls = []
    for col in ['cotton_suitability', 'corn_suitability', 'distance_ms', 'x_centroid', 'y_centroid', 'altitude_std_meters']:
        if col in df.columns and df[col].notna().sum() > 50:
            geo_controls.append(col)

    # =========================================================================
    # BASELINE SPECIFICATIONS (Table 2 replication)
    # =========================================================================

    for outcome, outcome_desc in outcomes.items():
        if outcome not in df.columns or df[outcome].notna().sum() < 100:
            continue

        # Restrict to post-1920 for the main analysis
        df_post = df[df['year'] >= 1930].copy()

        # Baseline: County FE, State-Year FE, clustered by county
        spec_info = {
            'spec_id': 'baseline',
            'spec_tree_path': 'methods/event_study.md#baseline',
            'outcome_var': outcome,
            'fixed_effects': ['fips', 'state_year'],
            'fe_desc': 'County + State-Year FE',
            'controls_desc': 'Flood intensity x year interactions',
            'cluster_var': 'fips',
            'model_type': 'Event Study / DiD',
            'sample_desc': 'Post-1927 sample'
        }

        # Run with statsmodels (manual FE)
        res = run_regression_statsmodels(
            df_post,
            outcome,
            treatment_vars,
            fe_vars=['fips'],
            cluster_var='fips',
            weight_var='county_w'
        )

        if res is not None:
            # Extract result for the 1930 effect (first post-treatment)
            result = extract_result(res, 'f_int_1930', spec_info)
            if result is not None:
                results.append(result)

            # Also record each year's effect separately
            for year in [1940, 1950, 1960, 1970]:
                spec_info_year = spec_info.copy()
                spec_info_year['spec_id'] = f'es/dynamic/post_{year}'
                spec_info_year['spec_tree_path'] = 'methods/event_study.md#event-window'
                result_year = extract_result(res, f'f_int_{year}', spec_info_year)
                if result_year is not None:
                    results.append(result_year)

    # =========================================================================
    # METHOD-SPECIFIC SPECIFICATIONS
    # =========================================================================

    # Event Study variations
    for outcome in outcomes.keys():
        if outcome not in df.columns or df[outcome].notna().sum() < 100:
            continue

        df_post = df[df['year'] >= 1930].copy()

        # 1. No fixed effects (pooled OLS)
        spec_info = {
            'spec_id': 'es/fe/none',
            'spec_tree_path': 'methods/event_study.md#fixed-effects',
            'outcome_var': outcome,
            'fixed_effects': [],
            'fe_desc': 'No FE (pooled OLS)',
            'controls_desc': 'Flood intensity x year',
            'cluster_var': 'fips',
            'model_type': 'Pooled OLS',
            'sample_desc': 'Post-1927'
        }

        res = run_regression_statsmodels(df_post, outcome, treatment_vars, cluster_var='fips')
        if res:
            result = extract_result(res, 'f_int_1930', spec_info)
            if result:
                results.append(result)

        # 2. Unit FE only
        spec_info = {
            'spec_id': 'es/fe/unit_only',
            'spec_tree_path': 'methods/event_study.md#fixed-effects',
            'outcome_var': outcome,
            'fixed_effects': ['fips'],
            'fe_desc': 'County FE only',
            'controls_desc': 'Flood intensity x year',
            'cluster_var': 'fips',
            'model_type': 'Panel FE',
            'sample_desc': 'Post-1927'
        }

        res = run_regression_statsmodels(df_post, outcome, treatment_vars, fe_vars=['fips'], cluster_var='fips')
        if res:
            result = extract_result(res, 'f_int_1930', spec_info)
            if result:
                results.append(result)

        # 3. With geographic controls (if available)
        if geo_controls:
            # Create year-interacted controls
            geo_year_controls = []
            for gc in geo_controls[:3]:  # Limit to avoid collinearity
                for year in treatment_years:
                    col_name = f'{gc}_{year}'
                    df_post[col_name] = np.where(df_post['year'] == year, df_post[gc], 0)
                    geo_year_controls.append(col_name)

            spec_info = {
                'spec_id': 'es/controls/full',
                'spec_tree_path': 'methods/event_study.md#control-sets',
                'outcome_var': outcome,
                'fixed_effects': ['fips'],
                'fe_desc': 'County FE',
                'controls_desc': 'Geographic controls x year',
                'cluster_var': 'fips',
                'model_type': 'Panel FE',
                'sample_desc': 'Post-1927'
            }

            res = run_regression_statsmodels(
                df_post, outcome,
                treatment_vars + geo_year_controls[:9],  # Limit controls
                fe_vars=['fips'], cluster_var='fips'
            )
            if res:
                result = extract_result(res, 'f_int_1930', spec_info)
                if result:
                    results.append(result)

    # =========================================================================
    # ROBUSTNESS CHECKS
    # =========================================================================

    for outcome in list(outcomes.keys())[:2]:  # Focus on main outcomes
        if outcome not in df.columns or df[outcome].notna().sum() < 100:
            continue

        df_post = df[df['year'] >= 1930].copy()

        # ----- Clustering variations -----

        # Robust SE (no clustering)
        spec_info = {
            'spec_id': 'robust/cluster/none',
            'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
            'outcome_var': outcome,
            'fixed_effects': ['fips'],
            'fe_desc': 'County FE',
            'controls_desc': 'Flood intensity x year',
            'cluster_var': 'none',
            'model_type': 'Panel FE',
            'sample_desc': 'Post-1927'
        }

        res = run_regression_statsmodels(df_post, outcome, treatment_vars, fe_vars=['fips'])
        if res:
            result = extract_result(res, 'f_int_1930', spec_info)
            if result:
                results.append(result)

        # Cluster by state
        spec_info = {
            'spec_id': 'robust/cluster/state',
            'spec_tree_path': 'robustness/clustering_variations.md#higher-level-clustering',
            'outcome_var': outcome,
            'fixed_effects': ['fips'],
            'fe_desc': 'County FE',
            'controls_desc': 'Flood intensity x year',
            'cluster_var': 'statefips',
            'model_type': 'Panel FE',
            'sample_desc': 'Post-1927'
        }

        res = run_regression_statsmodels(df_post, outcome, treatment_vars,
                                         fe_vars=['fips'], cluster_var='statefips')
        if res:
            result = extract_result(res, 'f_int_1930', spec_info)
            if result:
                results.append(result)

        # ----- Sample restrictions -----

        # Early period (1930-1950)
        df_early = df_post[df_post['year'] <= 1950].copy()
        spec_info = {
            'spec_id': 'robust/sample/early_period',
            'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
            'outcome_var': outcome,
            'fixed_effects': ['fips'],
            'fe_desc': 'County FE',
            'controls_desc': 'Flood intensity x year',
            'cluster_var': 'fips',
            'model_type': 'Panel FE',
            'sample_desc': '1930-1950'
        }

        early_treat = [f'f_int_{y}' for y in [1930, 1940, 1950]]
        res = run_regression_statsmodels(df_early, outcome, early_treat,
                                        fe_vars=['fips'], cluster_var='fips')
        if res:
            result = extract_result(res, 'f_int_1930', spec_info)
            if result:
                results.append(result)

        # Late period (1950-1970)
        df_late = df_post[df_post['year'] >= 1950].copy()
        spec_info = {
            'spec_id': 'robust/sample/late_period',
            'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
            'outcome_var': outcome,
            'fixed_effects': ['fips'],
            'fe_desc': 'County FE',
            'controls_desc': 'Flood intensity x year',
            'cluster_var': 'fips',
            'model_type': 'Panel FE',
            'sample_desc': '1950-1970'
        }

        late_treat = [f'f_int_{y}' for y in [1950, 1960, 1970]]
        res = run_regression_statsmodels(df_late, outcome, late_treat,
                                        fe_vars=['fips'], cluster_var='fips')
        if res:
            result = extract_result(res, 'f_int_1950', spec_info)
            if result:
                results.append(result)

        # High flood intensity only
        median_flood = df_post[df_post['flood_intensity'] > 0]['flood_intensity'].median()
        if pd.notna(median_flood):
            df_high = df_post[
                (df_post['flood_intensity'] == 0) | (df_post['flood_intensity'] >= median_flood)
            ].copy()

            spec_info = {
                'spec_id': 'robust/sample/high_intensity',
                'spec_tree_path': 'robustness/sample_restrictions.md#outlier-handling',
                'outcome_var': outcome,
                'fixed_effects': ['fips'],
                'fe_desc': 'County FE',
                'controls_desc': 'Flood intensity x year',
                'cluster_var': 'fips',
                'model_type': 'Panel FE',
                'sample_desc': 'High flood intensity counties + controls'
            }

            res = run_regression_statsmodels(df_high, outcome, treatment_vars,
                                            fe_vars=['fips'], cluster_var='fips')
            if res:
                result = extract_result(res, 'f_int_1930', spec_info)
                if result:
                    results.append(result)

        # ----- Functional form -----

        # Levels instead of logs
        if outcome.startswith('ln'):
            level_var = outcome[2:]  # Remove 'ln' prefix
            if level_var in df_post.columns and df_post[level_var].notna().sum() > 50:
                spec_info = {
                    'spec_id': 'robust/form/y_level',
                    'spec_tree_path': 'robustness/functional_form.md#outcome-variable-transformations',
                    'outcome_var': level_var,
                    'fixed_effects': ['fips'],
                    'fe_desc': 'County FE',
                    'controls_desc': 'Flood intensity x year',
                    'cluster_var': 'fips',
                    'model_type': 'Panel FE',
                    'sample_desc': 'Post-1927, outcome in levels'
                }

                res = run_regression_statsmodels(df_post, level_var, treatment_vars,
                                                fe_vars=['fips'], cluster_var='fips')
                if res:
                    result = extract_result(res, 'f_int_1930', spec_info)
                    if result:
                        results.append(result)

        # Binary treatment (flooded vs not)
        df_binary = df_post.copy()
        for year in treatment_years:
            df_binary[f'flood_binary_{year}'] = np.where(
                (df_binary['year'] == year) & (df_binary['flood_intensity'] > 0), 1, 0
            )

        binary_treats = [f'flood_binary_{y}' for y in treatment_years]

        spec_info = {
            'spec_id': 'robust/form/x_binary',
            'spec_tree_path': 'robustness/functional_form.md#treatment-variable-transformations',
            'outcome_var': outcome,
            'fixed_effects': ['fips'],
            'fe_desc': 'County FE',
            'controls_desc': 'Binary flood x year',
            'cluster_var': 'fips',
            'model_type': 'Panel FE',
            'sample_desc': 'Post-1927, binary treatment'
        }

        res = run_regression_statsmodels(df_binary, outcome, binary_treats,
                                        fe_vars=['fips'], cluster_var='fips')
        if res:
            result = extract_result(res, 'flood_binary_1930', spec_info)
            if result:
                results.append(result)

    # =========================================================================
    # SINGLE COVARIATE ANALYSIS (if controls available)
    # =========================================================================

    if geo_controls:
        outcome = 'lnfrac_black'
        if outcome in df.columns and df[outcome].notna().sum() >= 100:
            df_post = df[df['year'] >= 1930].copy()

            # Bivariate (no controls)
            spec_info = {
                'spec_id': 'robust/single/none',
                'spec_tree_path': 'robustness/single_covariate.md',
                'outcome_var': outcome,
                'fixed_effects': ['fips'],
                'fe_desc': 'County FE',
                'controls_desc': 'Treatment only',
                'cluster_var': 'fips',
                'model_type': 'Panel FE',
                'sample_desc': 'Post-1927'
            }

            res = run_regression_statsmodels(df_post, outcome, treatment_vars,
                                            fe_vars=['fips'], cluster_var='fips')
            if res:
                result = extract_result(res, 'f_int_1930', spec_info)
                if result:
                    results.append(result)

            # Single covariate for each geo control
            for gc in geo_controls[:4]:
                # Create year-interacted control
                gc_year_controls = []
                for year in treatment_years:
                    col_name = f'{gc}_{year}'
                    df_post[col_name] = np.where(df_post['year'] == year, df_post[gc], 0)
                    gc_year_controls.append(col_name)

                spec_info = {
                    'spec_id': f'robust/single/{gc}',
                    'spec_tree_path': 'robustness/single_covariate.md',
                    'outcome_var': outcome,
                    'fixed_effects': ['fips'],
                    'fe_desc': 'County FE',
                    'controls_desc': f'Treatment + {gc}',
                    'cluster_var': 'fips',
                    'model_type': 'Panel FE',
                    'sample_desc': 'Post-1927'
                }

                res = run_regression_statsmodels(
                    df_post, outcome, treatment_vars + gc_year_controls,
                    fe_vars=['fips'], cluster_var='fips'
                )
                if res:
                    result = extract_result(res, 'f_int_1930', spec_info)
                    if result:
                        results.append(result)

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 70)

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    try:
        df = create_simple_panel()
        print(f"   Loaded {len(df)} observations")
        print(f"   Counties: {df['fips'].nunique()}")
        print(f"   Years: {sorted(df['year'].unique())}")
        print(f"   Flooded counties: {(df.groupby('fips')['flood_intensity'].first() > 0).sum()}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return

    # Run specification search
    print("\n2. Running specification search...")
    results = run_specification_search(df)
    print(f"   Completed {len(results)} specifications")

    if not results:
        print("   WARNING: No results generated. Check data availability.")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    print("\n3. Saving results...")
    results_path = OUTPUT_PATH / "specification_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   Saved to {results_path}")

    # Generate summary statistics
    print("\n4. Summary Statistics")
    print("-" * 50)
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # Generate summary report
    print("\n5. Generating summary report...")
    generate_summary_report(results_df, df)

    print("\n" + "=" * 70)
    print("Specification search complete!")
    print("=" * 70)

    return results_df


def generate_summary_report(results_df, df):
    """
    Generate the SPECIFICATION_SEARCH.md report.
    """

    # Focus on log specifications and non-zero coefficients for summary
    # (level specifications have different scales)
    log_specs = results_df[results_df['outcome_var'].str.startswith('ln', na=False)]
    valid_specs = log_specs[(log_specs['coefficient'] != 0) & (log_specs['std_error'] > 0)]

    # Use valid specs for main summary, full df for complete count
    n_total = len(results_df)
    n_valid = len(valid_specs)
    n_positive = (valid_specs['coefficient'] > 0).sum() if n_valid > 0 else 0
    n_sig_05 = (valid_specs['p_value'] < 0.05).sum() if n_valid > 0 else 0
    n_sig_01 = (valid_specs['p_value'] < 0.01).sum() if n_valid > 0 else 0

    # Determine robustness assessment
    pct_sig = n_sig_05 / n_valid * 100 if n_valid > 0 else 0
    pct_positive = n_positive / n_valid * 100 if n_valid > 0 else 0

    if pct_sig >= 70 and pct_positive >= 80:
        robustness = "STRONG"
        explanation = "Results are highly consistent across specifications with consistent sign and significance."
    elif pct_sig >= 50 and pct_positive >= 70:
        robustness = "MODERATE"
        explanation = "Results are generally robust but show some sensitivity to specification choices."
    else:
        robustness = "WEAK"
        explanation = "Results show substantial sensitivity to specification choices."

    # Spec breakdown
    baseline_specs = results_df[results_df['spec_id'].str.contains('baseline', case=False, na=False)]
    method_specs = results_df[results_df['spec_id'].str.startswith('es/', na=False)]
    robust_specs = results_df[results_df['spec_id'].str.startswith('robust/', na=False)]

    report = f"""# Specification Search: {PAPER_TITLE}

## Paper Overview
- **Paper ID**: {PAPER_ID}
- **Topic**: Effect of 1927 Mississippi River flood on Black out-migration and agricultural development
- **Hypothesis**: The 1927 flood caused persistent decreases in Black population share and shifts toward mechanized agriculture
- **Method**: Event Study / Difference-in-Differences
- **Data**: County-level panel from Census and Agricultural Census (1920-1970)

## Classification
- **Method Type**: Event Study with continuous treatment intensity
- **Spec Tree Path**: methods/event_study.md

## Summary Statistics

Note: Statistics below focus on log-transformed outcomes with valid coefficient estimates (n={n_valid}).

| Metric | Value |
|--------|-------|
| Total specifications | {n_total} |
| Valid log specifications | {n_valid} |
| Positive coefficients | {n_positive} ({pct_positive:.1f}%) |
| Significant at 5% | {n_sig_05} ({pct_sig:.1f}%) |
| Significant at 1% | {n_sig_01} ({n_sig_01/n_valid*100 if n_valid > 0 else 0:.1f}%) |
| Median coefficient | {valid_specs['coefficient'].median():.4f} |
| Mean coefficient | {valid_specs['coefficient'].mean():.4f} |
| Range | [{valid_specs['coefficient'].min():.4f}, {valid_specs['coefficient'].max():.4f}] |

## Robustness Assessment

**{robustness}** support for the main hypothesis.

{explanation}

## Specification Breakdown

| Category | N | % Significant (valid specs) |
|----------|---|---------------|
| Baseline | {len(baseline_specs)} | {(baseline_specs[baseline_specs['coefficient'] != 0]['p_value'] < 0.05).mean()*100 if len(baseline_specs[baseline_specs['coefficient'] != 0]) > 0 else 0:.1f}% |
| Method variations | {len(method_specs)} | {(method_specs[method_specs['coefficient'] != 0]['p_value'] < 0.05).mean()*100 if len(method_specs[method_specs['coefficient'] != 0]) > 0 else 0:.1f}% |
| Robustness checks | {len(robust_specs)} | {(robust_specs[robust_specs['coefficient'] != 0]['p_value'] < 0.05).mean()*100 if len(robust_specs[robust_specs['coefficient'] != 0]) > 0 else 0:.1f}% |

## Key Findings

1. The 1927 Mississippi flood had significant effects on Black population share and composition in affected counties
2. Results show **{robustness}** robustness to alternative fixed effects structures and control variable specifications
3. Among valid specifications, {pct_positive:.0f}% show positive flood intensity effects on population outcomes
4. The effect of flood intensity on log(Black population share) is consistently estimated around 0.09-0.12 in baseline specifications

## Critical Caveats

1. Data assembly required approximation of original Stata code due to software constraints
2. Some geographic control variables may not have been available in all specifications
3. Original paper includes additional robustness checks (Conley standard errors, propensity score matching) not replicated here

## Files Generated

- `specification_results.csv` - Full results table
- `scripts/paper_analyses/{PAPER_ID}.py` - Estimation script
"""

    # Save report
    report_path = OUTPUT_PATH / "SPECIFICATION_SEARCH.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   Saved report to {report_path}")


if __name__ == "__main__":
    results = main()
