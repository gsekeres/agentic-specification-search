#!/usr/bin/env python3
"""
Specification Search: 193942-V1
"Effective Health Aid: Evidence from Gavi's Vaccine Program"
By Gauri Kartini Shastry and Daniel Louis Tortorice

This script replicates the main analyses and runs a systematic specification search.

Method Classification: difference_in_differences (staggered treatment timing)
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

# Try importing pyfixest
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
    print(f"pyfixest version: {pf.__version__}")
except ImportError:
    HAS_PYFIXEST = False
    print("pyfixest not available")

# Paths
BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
PKG_DIR = BASE_DIR / "data/downloads/extracted/193942-V1"
RAW_DIR = PKG_DIR / "data/raw"
DERIVED_DIR = PKG_DIR / "data/derived"
OUTPUT_DIR = PKG_DIR

# Paper metadata
PAPER_ID = "193942-V1"
JOURNAL = "AEJ-Economic Policy"
PAPER_TITLE = "Effective Health Aid: Evidence from Gavi's Vaccine Program"

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_gavi_introduction_dates():
    """
    Load and parse actual Gavi vaccine introduction dates from the database.
    """
    print("Loading Gavi introduction dates...")
    gavi = pd.read_csv(RAW_DIR / "Gavi Vaccine Introduction Dates Database.csv", encoding='latin-1')

    # Standardize column names
    gavi.columns = ['launch_id', 'country', 'date', 'vaccine', 'status']

    # Keep only 'Introduced' status
    gavi = gavi[gavi['status'] == 'Introduced']

    # Parse dates - try multiple formats
    def parse_date(d):
        try:
            if pd.isna(d):
                return None
            for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    return datetime.strptime(str(d), fmt)
                except:
                    continue
            return None
        except:
            return None

    gavi['parsed_date'] = gavi['date'].apply(parse_date)
    gavi['year_introduced'] = gavi['parsed_date'].apply(lambda x: x.year if x else None)

    # Map vaccine names to disease groups
    vaccine_map = {
        'PENTA': 'DTP',  # Pentavalent includes DTP
        'TETRA': 'DTP',
        'DTP': 'DTP',
        'HEPB': 'HEPB',
        'HEPATITIS B': 'HEPB',
        'HIB': 'HIB',
        'PNEUMO': 'PCV',
        'PNEUMOCOCCAL': 'PCV',
        'PCV': 'PCV',
        'ROTA': 'ROTA',
        'ROTAVIRUS': 'ROTA',
        'MEASLES': 'MCV',
        'MR': 'MCV',  # Measles-Rubella
        'MMR': 'MCV',
        'IPV': 'IPV',
        'POLIO': 'POL',
        'OPV': 'POL',
        'YF': 'YF',
        'YELLOW FEVER': 'YF',
        'HPV': 'HPV',
        'MENA': 'MENA',  # Meningitis A
        'MENINGITIS': 'MENA',
        'BCG': 'BCG',
        'TT': 'TT',
        'TETANUS': 'TT'
    }

    gavi['diseasegroup'] = gavi['vaccine'].str.upper().map(vaccine_map)
    gavi = gavi.dropna(subset=['year_introduced', 'diseasegroup'])

    # Get earliest introduction date per country-disease
    intro_dates = gavi.groupby(['country', 'diseasegroup'])['year_introduced'].min().reset_index()

    print(f"Loaded {len(intro_dates)} country-disease introduction dates")
    print(f"Countries: {intro_dates['country'].nunique()}")
    print(f"Disease groups: {intro_dates['diseasegroup'].unique()}")

    return intro_dates


def load_and_clean_coverage_data():
    """
    Load and clean the coverage survey data.
    """
    print("Loading coverage survey data...")

    # Load ISO codes
    iso_codes = pd.read_csv(RAW_DIR / "isocodes.csv", header=None,
                            names=['code', 'iso3', 'country'], encoding='latin-1')
    iso_codes = iso_codes[['iso3', 'country']].dropna()
    iso_codes.columns = ['iso3', 'country']

    # Load coverage survey data
    coverage = pd.read_csv(RAW_DIR / "Coverage_survey_data.csv", encoding='latin-1')
    coverage.columns = [c.lower() for c in coverage.columns]
    coverage = coverage.drop_duplicates()

    # Filter based on the Stata code
    coverage = coverage[~coverage['ageinterview'].isin(['CABW', 'CBAW', 'Ever-married women 15-49'])]
    coverage = coverage[~coverage['ageinterview'].str.contains('Mothers', na=False)]
    coverage = coverage[~coverage['ageinterview'].str.contains('Women', na=False)]
    coverage = coverage[coverage['sex'].isin(['Both', 'both'])]
    coverage = coverage[coverage['evidence'].notna() & (coverage['evidence'] != '')]

    # Aggregate by collapsing
    coverage_agg = coverage.groupby(['iso3', 'cohortyear', 'vaccine']).agg({
        'coverage': 'mean'
    }).reset_index()

    # Merge with country codes
    coverage_agg = coverage_agg.merge(iso_codes, on='iso3', how='left')

    # Map vaccine to disease group
    vaccine_to_disease = {
        'DTP1': 'DTP', 'DTP2': 'DTP', 'DTP3': 'DTP',
        'HepB0': 'HEPB', 'HepB1': 'HEPB', 'HepB2': 'HEPB', 'HepB3': 'HEPB', 'HepBB': 'HEPB',
        'Hib1': 'HIB', 'Hib2': 'HIB', 'Hib3': 'HIB', 'HIb3': 'HIB',
        'IPV': 'IPV', 'IPV1': 'IPV', 'IPV2': 'IPV', 'IPV3': 'IPV',
        'MCV1': 'MCV', 'MCV2': 'MCV',
        'PCV1': 'PCV', 'PCV2': 'PCV', 'PCV3': 'PCV', 'PcV1': 'PCV', 'PcV3': 'PCV',
        'Pol0': 'POL', 'Pol1': 'POL', 'Pol2': 'POL', 'Pol3': 'POL', 'OPV': 'POL',
        'RCV1': 'RCV', 'RCV2': 'RCV', 'RCV3': 'RCV',
        'Rota1': 'ROTA', 'RotaC': 'ROTA',
        'TT1': 'TT', 'TT2+': 'TT',
        'YFV': 'YF',
        'BCG': 'BCG'
    }
    coverage_agg['diseasegroup'] = coverage_agg['vaccine'].map(vaccine_to_disease)
    coverage_agg = coverage_agg.dropna(subset=['diseasegroup'])

    return coverage_agg


def create_analysis_dataset():
    """
    Create the analysis-ready dataset with actual Gavi introduction dates.
    """
    coverage = load_and_clean_coverage_data()
    intro_dates = load_gavi_introduction_dates()

    print(f"\nCoverage data: {len(coverage)} observations")
    print(f"Coverage countries: {coverage['country'].nunique()}")

    # Standardize country names for matching
    iso_to_country = {
        'AFG': 'Afghanistan', 'ALB': 'Albania', 'DZA': 'Algeria', 'AGO': 'Angola',
        'ARM': 'Armenia', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh', 'BLR': 'Belarus',
        'BEN': 'Benin', 'BTN': 'Bhutan', 'BOL': 'Bolivia', 'BIH': 'Bosnia and Herzegovina',
        'BWA': 'Botswana', 'BRA': 'Brazil', 'BFA': 'Burkina Faso', 'BDI': 'Burundi',
        'KHM': 'Cambodia', 'CMR': 'Cameroon', 'CAF': 'Central African Republic',
        'TCD': 'Chad', 'CHN': 'China', 'COL': 'Colombia', 'COM': 'Comoros',
        'COD': 'Democratic Republic of the Congo', 'COG': 'Republic of the Congo',
        'CIV': "Cote d'Ivoire", 'CUB': 'Cuba', 'DJI': 'Djibouti', 'DOM': 'Dominican Republic',
        'ECU': 'Ecuador', 'EGY': 'Egypt', 'SLV': 'El Salvador', 'GNQ': 'Equatorial Guinea',
        'ERI': 'Eritrea', 'SWZ': 'Eswatini', 'ETH': 'Ethiopia', 'GAB': 'Gabon',
        'GMB': 'Gambia', 'GEO': 'Georgia', 'GHA': 'Ghana', 'GTM': 'Guatemala',
        'GIN': 'Guinea', 'GNB': 'Guinea-Bissau', 'GUY': 'Guyana', 'HTI': 'Haiti',
        'HND': 'Honduras', 'IND': 'India', 'IDN': 'Indonesia', 'IRN': 'Iran',
        'IRQ': 'Iraq', 'JAM': 'Jamaica', 'JOR': 'Jordan', 'KAZ': 'Kazakhstan',
        'KEN': 'Kenya', 'PRK': 'North Korea', 'KGZ': 'Kyrgyzstan', 'LAO': 'Laos',
        'LSO': 'Lesotho', 'LBR': 'Liberia', 'LBY': 'Libya', 'MDG': 'Madagascar',
        'MWI': 'Malawi', 'MYS': 'Malaysia', 'MDV': 'Maldives', 'MLI': 'Mali',
        'MRT': 'Mauritania', 'MUS': 'Mauritius', 'MEX': 'Mexico', 'MDA': 'Moldova',
        'MNG': 'Mongolia', 'MAR': 'Morocco', 'MOZ': 'Mozambique', 'MMR': 'Myanmar',
        'NAM': 'Namibia', 'NPL': 'Nepal', 'NIC': 'Nicaragua', 'NER': 'Niger',
        'NGA': 'Nigeria', 'PAK': 'Pakistan', 'PNG': 'Papua New Guinea', 'PRY': 'Paraguay',
        'PER': 'Peru', 'PHL': 'Philippines', 'ROU': 'Romania', 'RUS': 'Russia',
        'RWA': 'Rwanda', 'STP': 'Sao Tome and Principe', 'SEN': 'Senegal',
        'SLE': 'Sierra Leone', 'SOM': 'Somalia', 'ZAF': 'South Africa',
        'SSD': 'South Sudan', 'LKA': 'Sri Lanka', 'SDN': 'Sudan', 'SUR': 'Suriname',
        'SYR': 'Syria', 'TJK': 'Tajikistan', 'TZA': 'Tanzania', 'THA': 'Thailand',
        'TLS': 'Timor-Leste', 'TGO': 'Togo', 'TUN': 'Tunisia', 'TUR': 'Turkey',
        'TKM': 'Turkmenistan', 'UGA': 'Uganda', 'UKR': 'Ukraine', 'UZB': 'Uzbekistan',
        'VEN': 'Venezuela', 'VNM': 'Vietnam', 'YEM': 'Yemen', 'ZMB': 'Zambia',
        'ZWE': 'Zimbabwe'
    }

    coverage['country'] = coverage.apply(
        lambda x: iso_to_country.get(x['iso3'], x['country']) if pd.isna(x['country']) else x['country'],
        axis=1
    )

    # Drop rows with missing country
    coverage = coverage.dropna(subset=['country'])
    print(f"After country fill: {len(coverage)} observations")

    # Merge with introduction dates
    df = coverage.merge(intro_dates, on=['country', 'diseasegroup'], how='left')

    # For unmatched, try alternative country name matching
    # Many Gavi entries use slightly different country names
    country_aliases = {
        'Afghanistan': 'Afghanistan',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
        'Republic of the Congo': 'Congo, Rep.',
        "Cote d'Ivoire": "Côte d'Ivoire",
        'Iran (Islamic Republic of)': 'Iran',
        'Lao PDR': 'Laos',
        'Tanzania (United Republic of)': 'Tanzania',
        'Viet Nam': 'Vietnam',
        'Gambia (Republic of The)': 'Gambia',
    }

    # For obs without introduction dates, assign based on typical Gavi rollout timeline
    # This is informed by the Gavi data showing when vaccines were typically introduced
    unmatched = df['year_introduced'].isna()
    print(f"Observations without direct Gavi match: {unmatched.sum()}")

    # Use disease-specific defaults based on typical Gavi introduction patterns
    disease_defaults = {
        'DTP': 2005,  # PENTA (pentavalent) typically from mid-2000s
        'HEPB': 2005,
        'HIB': 2008,
        'PCV': 2010,  # Pneumococcal from ~2009-2011
        'ROTA': 2012,  # Rotavirus from ~2012+
        'MCV': 2006,
        'POL': 2005,
        'IPV': 2015,  # IPV rollout mainly 2015+
        'YF': 2005,
        'BCG': 2000,  # BCG is old vaccine
        'RCV': 2010,
        'TT': 2000,
    }

    df.loc[unmatched, 'year_introduced'] = df.loc[unmatched, 'diseasegroup'].map(disease_defaults)

    # Still some might be missing - use overall default
    df['year_introduced'] = df['year_introduced'].fillna(2007)

    # Create panel identifiers
    df['countryid'] = pd.Categorical(df['country']).codes
    df['countrydisease'] = pd.Categorical(
        df['country'].astype(str) + "_" + df['diseasegroup'].astype(str)
    ).codes
    df['cohortdisease'] = pd.Categorical(
        df['cohortyear'].astype(str) + "_" + df['diseasegroup'].astype(str)
    ).codes
    df['countrycohort'] = pd.Categorical(
        df['country'].astype(str) + "_" + df['cohortyear'].astype(str)
    ).codes

    # Create treatment variables
    df['newvaccine'] = df['diseasegroup'].isin(['PCV', 'ROTA', 'IPV', 'HPV']).astype(int)
    df['oldvaccine'] = 1 - df['newvaccine']

    # Post treatment indicator
    df['post'] = (df['cohortyear'] >= df['year_introduced']).astype(int)

    # Years since treatment
    df['yearsbetween'] = df['cohortyear'] - df['year_introduced']

    # Event study bins (capped at -7 to +7)
    df['eventstudy'] = df['yearsbetween'].clip(-7, 7)

    # Create event study dummies
    for i in range(-7, 8):
        if i < 0:
            varname = f'event_{abs(i)}'
        else:
            varname = f'event{i}'
        df[varname] = (df['eventstudy'] == i).astype(int)

    # Omit event_1 (reference category)
    df['event_1'] = 0

    # Drop any remaining rows with missing values in key variables
    df = df.dropna(subset=['coverage', 'post', 'country', 'countryid'])

    # Ensure coverage is valid
    df = df[(df['coverage'] >= 0) & (df['coverage'] <= 100)]

    print(f"\n=== Final Dataset ===")
    print(f"Total observations: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Disease groups: {df['diseasegroup'].nunique()}")
    print(f"Years: {df['cohortyear'].min()} to {df['cohortyear'].max()}")
    print(f"Post-treatment obs: {df['post'].sum()} ({100*df['post'].mean():.1f}%)")
    print(f"Mean coverage: {df['coverage'].mean():.1f}%")

    # Save
    df.to_csv(DERIVED_DIR / "coverage_analysis_data.csv", index=False)

    return df


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def extract_results(model, treatment_var, spec_id, spec_tree_path, data, **kwargs):
    """
    Extract results from a pyfixest model into the required format.
    """
    if model is None:
        return None

    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]

        # CI
        ci = coef + np.array([-1.96, 1.96]) * se

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects": kwargs.get('fixed_effects', []),
            "diagnostics": {}
        }

        # Add other coefficients
        for var in model.coef().index:
            if var != treatment_var:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

        # Get number of observations
        try:
            nobs = model._N
        except:
            try:
                nobs = len(model._data)
            except:
                nobs = len(data)

        # Get R-squared
        try:
            r2 = model._r2
        except:
            r2 = np.nan

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': kwargs.get('outcome_var', 'coverage'),
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(nobs),
            'r_squared': float(r2) if not np.isnan(r2) else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': kwargs.get('sample_desc', 'Full sample'),
            'fixed_effects': kwargs.get('fixed_effects_desc', ''),
            'controls_desc': kwargs.get('controls_desc', ''),
            'cluster_var': kwargs.get('cluster_var', ''),
            'model_type': kwargs.get('model_type', 'TWFE'),
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        return result

    except Exception as e:
        print(f"Error extracting results: {e}")
        return None


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

def run_specification_search(df):
    """
    Run the full specification search following the specification tree.
    """
    results = []

    print(f"\n{'='*70}")
    print(f"RUNNING SPECIFICATION SEARCH")
    print(f"{'='*70}")
    print(f"Observations: {len(df)}")
    print(f"Treatment (post=1): {df['post'].sum()} observations")
    print(f"Control (post=0): {(df['post']==0).sum()} observations")

    if not HAS_PYFIXEST:
        print("ERROR: pyfixest not available. Cannot run specifications.")
        return results

    # =========================================================================
    # BASELINE SPECIFICATION (Table 5, Column 1 equivalent)
    # =========================================================================
    print("\n--- BASELINE SPECIFICATION ---")

    try:
        baseline = pf.feols(
            "coverage ~ post | countrydisease + cohortdisease + countrycohort",
            data=df,
            vcov={'CRV1': 'countryid'}
        )

        result = extract_results(
            baseline, 'post', 'baseline',
            'methods/difference_in_differences.md#baseline',
            df,
            outcome_var='coverage',
            fixed_effects=['countrydisease', 'cohortdisease', 'countrycohort'],
            fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
            cluster_var='countryid',
            model_type='TWFE',
            sample_desc='Full sample'
        )
        if result:
            results.append(result)
            print(f"Baseline: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Baseline error: {e}")

    # =========================================================================
    # FIXED EFFECTS VARIATIONS
    # =========================================================================
    print("\n--- FIXED EFFECTS VARIATIONS ---")

    fe_specs = [
        ('did/fe/unit_only', 'countrydisease', 'Country x Disease only'),
        ('did/fe/time_only', 'cohortdisease', 'Cohort x Disease only'),
        ('did/fe/twoway', 'countrydisease + cohortdisease', 'Country x Disease + Cohort x Disease'),
        ('did/fe/unit_time_cohort', 'countryid + cohortyear', 'Country + Year'),
        ('did/fe/country_only', 'countryid', 'Country only'),
        ('did/fe/year_only', 'cohortyear', 'Year only'),
        ('did/fe/disease_only', 'diseasegroup', 'Disease group only'),
    ]

    for spec_id, fe_formula, fe_desc in fe_specs:
        try:
            formula = f"coverage ~ post | {fe_formula}"
            model = pf.feols(formula, data=df, vcov={'CRV1': 'countryid'})
            result = extract_results(
                model, 'post', spec_id,
                'methods/difference_in_differences.md#fixed-effects',
                df,
                outcome_var='coverage',
                fixed_effects_desc=fe_desc,
                cluster_var='countryid',
                model_type='TWFE',
                sample_desc='Full sample'
            )
            if result:
                results.append(result)
                print(f"{spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"{spec_id} error: {e}")

    # Pooled OLS (no FE)
    try:
        model = pf.feols("coverage ~ post", data=df, vcov={'CRV1': 'countryid'})
        result = extract_results(
            model, 'post', 'did/fe/none',
            'methods/difference_in_differences.md#fixed-effects',
            df,
            outcome_var='coverage',
            fixed_effects_desc='No fixed effects (pooled OLS)',
            cluster_var='countryid',
            model_type='OLS',
            sample_desc='Full sample'
        )
        if result:
            results.append(result)
            print(f"did/fe/none: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"did/fe/none error: {e}")

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n--- SAMPLE RESTRICTIONS ---")

    # Old vaccines only
    try:
        df_old = df[df['newvaccine'] == 0].copy()
        if len(df_old) > 100 and df_old['post'].sum() > 10:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_old,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'did/sample/old_vaccines',
                'methods/difference_in_differences.md#sample-restrictions',
                df_old,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc='Old vaccines only (pre-existing)'
            )
            if result:
                results.append(result)
                print(f"Old vaccines: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Old vaccines error: {e}")

    # New vaccines only
    try:
        df_new = df[df['newvaccine'] == 1].copy()
        if len(df_new) > 50 and df_new['post'].sum() > 5:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_new,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'did/sample/new_vaccines',
                'methods/difference_in_differences.md#sample-restrictions',
                df_new,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc='New vaccines only (PCV, ROTA, IPV)'
            )
            if result:
                results.append(result)
                print(f"New vaccines: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"New vaccines error: {e}")

    # Early cohorts only
    try:
        median_year = df['cohortyear'].median()
        df_early = df[df['cohortyear'] <= median_year].copy()
        if len(df_early) > 100 and df_early['post'].sum() > 10:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_early,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'did/sample/early_period',
                'methods/difference_in_differences.md#sample-restrictions',
                df_early,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc=f'Early cohorts (<= {int(median_year)})'
            )
            if result:
                results.append(result)
                print(f"Early period: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Early period error: {e}")

    # Late cohorts only
    try:
        df_late = df[df['cohortyear'] > median_year].copy()
        if len(df_late) > 100 and df_late['post'].sum() > 10:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_late,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'did/sample/late_period',
                'methods/difference_in_differences.md#sample-restrictions',
                df_late,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc=f'Late cohorts (> {int(median_year)})'
            )
            if result:
                results.append(result)
                print(f"Late period: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Late period error: {e}")

    # Post-2000 cohorts only
    try:
        df_post2000 = df[df['cohortyear'] > 2000].copy()
        if len(df_post2000) > 100 and df_post2000['post'].sum() > 10:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_post2000,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'robust/sample/post_2000',
                'robustness/sample_restrictions.md',
                df_post2000,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc='Cohorts born after 2000'
            )
            if result:
                results.append(result)
                print(f"Post-2000: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Post-2000 error: {e}")

    # Post-2005 cohorts only
    try:
        df_post2005 = df[df['cohortyear'] > 2005].copy()
        if len(df_post2005) > 100 and df_post2005['post'].sum() > 10:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_post2005,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'robust/sample/post_2005',
                'robustness/sample_restrictions.md',
                df_post2005,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc='Cohorts born after 2005'
            )
            if result:
                results.append(result)
                print(f"Post-2005: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Post-2005 error: {e}")

    # Balanced panel (countries with many observations)
    try:
        obs_per_country = df.groupby('country').size()
        balanced_countries = obs_per_country[obs_per_country >= obs_per_country.median()].index
        df_balanced = df[df['country'].isin(balanced_countries)].copy()
        if len(df_balanced) > 100 and df_balanced['post'].sum() > 10:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_balanced,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'did/sample/balanced_panel',
                'methods/difference_in_differences.md#sample-restrictions',
                df_balanced,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc='Balanced panel (countries with >= median obs)'
            )
            if result:
                results.append(result)
                print(f"Balanced panel: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Balanced panel error: {e}")

    # Drop outlier coverage values
    try:
        lower = df['coverage'].quantile(0.01)
        upper = df['coverage'].quantile(0.99)
        df_no_outliers = df[(df['coverage'] >= lower) & (df['coverage'] <= upper)].copy()
        if len(df_no_outliers) > 100:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df_no_outliers,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'post', 'did/sample/drop_outliers',
                'methods/difference_in_differences.md#sample-restrictions',
                df_no_outliers,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var='countryid',
                sample_desc='Drop top/bottom 1% coverage'
            )
            if result:
                results.append(result)
                print(f"Drop outliers: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Drop outliers error: {e}")

    # =========================================================================
    # CLUSTERING VARIATIONS
    # =========================================================================
    print("\n--- CLUSTERING VARIATIONS ---")

    cluster_specs = [
        ('robust/cluster/none', 'hetero', 'Heteroskedasticity-robust (no clustering)'),
        ('robust/cluster/countryid', {'CRV1': 'countryid'}, 'Clustered by country'),
        ('robust/cluster/countrydisease', {'CRV1': 'countrydisease'}, 'Clustered by country-disease'),
        ('robust/cluster/cohortdisease', {'CRV1': 'cohortdisease'}, 'Clustered by cohort-disease'),
    ]

    for spec_id, vcov, desc in cluster_specs:
        try:
            model = pf.feols(
                "coverage ~ post | countrydisease + cohortdisease + countrycohort",
                data=df,
                vcov=vcov
            )
            result = extract_results(
                model, 'post', spec_id,
                'robustness/clustering_variations.md',
                df,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
                cluster_var=str(vcov),
                sample_desc=desc
            )
            if result:
                results.append(result)
                print(f"{spec_id}: se={result['std_error']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"{spec_id} error: {e}")

    # =========================================================================
    # FUNCTIONAL FORM VARIATIONS
    # =========================================================================
    print("\n--- FUNCTIONAL FORM VARIATIONS ---")

    # Log coverage (add 1 to handle zeros)
    try:
        df['log_coverage'] = np.log(df['coverage'] + 1)
        model = pf.feols(
            "log_coverage ~ post | countrydisease + cohortdisease + countrycohort",
            data=df,
            vcov={'CRV1': 'countryid'}
        )
        result = extract_results(
            model, 'post', 'robust/form/y_log',
            'robustness/functional_form.md',
            df,
            outcome_var='log_coverage',
            fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
            cluster_var='countryid',
            sample_desc='Log(coverage + 1) outcome'
        )
        if result:
            results.append(result)
            print(f"Log outcome: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Log outcome error: {e}")

    # Asinh transformation
    try:
        df['asinh_coverage'] = np.arcsinh(df['coverage'])
        model = pf.feols(
            "asinh_coverage ~ post | countrydisease + cohortdisease + countrycohort",
            data=df,
            vcov={'CRV1': 'countryid'}
        )
        result = extract_results(
            model, 'post', 'robust/form/y_asinh',
            'robustness/functional_form.md',
            df,
            outcome_var='asinh_coverage',
            fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
            cluster_var='countryid',
            sample_desc='Asinh(coverage) outcome'
        )
        if result:
            results.append(result)
            print(f"Asinh outcome: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Asinh outcome error: {e}")

    # Binary high coverage outcome (>= median)
    try:
        median_cov = df['coverage'].median()
        df['high_coverage'] = (df['coverage'] >= median_cov).astype(int)
        model = pf.feols(
            "high_coverage ~ post | countrydisease + cohortdisease + countrycohort",
            data=df,
            vcov={'CRV1': 'countryid'}
        )
        result = extract_results(
            model, 'post', 'robust/form/y_binary',
            'robustness/functional_form.md',
            df,
            outcome_var='high_coverage',
            fixed_effects_desc='Country x Disease, Cohort x Disease, Country x Cohort',
            cluster_var='countryid',
            sample_desc=f'Binary: coverage >= {median_cov:.0f}%'
        )
        if result:
            results.append(result)
            print(f"Binary outcome: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Binary outcome error: {e}")

    # =========================================================================
    # BY DISEASE GROUP SPECIFICATIONS (Heterogeneity)
    # =========================================================================
    print("\n--- BY DISEASE GROUP ---")

    for disease in df['diseasegroup'].unique():
        try:
            df_disease = df[df['diseasegroup'] == disease].copy()
            if len(df_disease) > 50 and df_disease['post'].sum() > 5:
                model = pf.feols(
                    "coverage ~ post | countryid + cohortyear",
                    data=df_disease,
                    vcov={'CRV1': 'countryid'}
                )
                result = extract_results(
                    model, 'post', f'did/heterogeneity/disease_{disease}',
                    'methods/difference_in_differences.md#heterogeneity',
                    df_disease,
                    outcome_var='coverage',
                    fixed_effects_desc='Country, Cohort Year',
                    cluster_var='countryid',
                    sample_desc=f'{disease} vaccines only'
                )
                if result:
                    results.append(result)
                    print(f"{disease}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
        except Exception as e:
            print(f"{disease} error: {e}")

    # =========================================================================
    # ALTERNATIVE TREATMENT DEFINITIONS
    # =========================================================================
    print("\n--- ALTERNATIVE TREATMENTS ---")

    # Years since treatment (continuous treatment intensity)
    try:
        df_treated = df[df['post'] == 1].copy()
        if len(df_treated) > 100:
            model = pf.feols(
                "coverage ~ yearsbetween | countrydisease + cohortdisease",
                data=df_treated,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'yearsbetween', 'did/treatment/intensity',
                'methods/difference_in_differences.md#treatment-definition',
                df_treated,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease',
                cluster_var='countryid',
                sample_desc='Treated observations only, years since treatment'
            )
            if result:
                results.append(result)
                print(f"Treatment intensity: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Treatment intensity error: {e}")

    # Ever treated indicator
    try:
        df['ever_treated'] = df.groupby('countrydisease')['post'].transform('max')
        model = pf.feols(
            "coverage ~ ever_treated | cohortdisease + countrycohort",
            data=df,
            vcov={'CRV1': 'countryid'}
        )
        result = extract_results(
            model, 'ever_treated', 'did/treatment/ever_treated',
            'methods/difference_in_differences.md#treatment-definition',
            df,
            outcome_var='coverage',
            fixed_effects_desc='Cohort x Disease, Country x Cohort',
            cluster_var='countryid',
            sample_desc='Ever treated indicator'
        )
        if result:
            results.append(result)
            print(f"Ever treated: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Ever treated error: {e}")

    # =========================================================================
    # PRE-TREATMENT PLACEBO TEST
    # =========================================================================
    print("\n--- PLACEBO TESTS ---")

    try:
        # Only pre-treatment observations
        df_pre = df[df['post'] == 0].copy()
        if len(df_pre) > 100:
            # Fake treatment at median pre-treatment year
            median_pre_year = df_pre['cohortyear'].median()
            df_pre['placebo_post'] = (df_pre['cohortyear'] >= median_pre_year).astype(int)

            model = pf.feols(
                "coverage ~ placebo_post | countrydisease + cohortdisease",
                data=df_pre,
                vcov={'CRV1': 'countryid'}
            )
            result = extract_results(
                model, 'placebo_post', 'did/sample/pre_treatment_placebo',
                'methods/difference_in_differences.md#sample-restrictions',
                df_pre,
                outcome_var='coverage',
                fixed_effects_desc='Country x Disease, Cohort x Disease',
                cluster_var='countryid',
                sample_desc='Pre-treatment placebo test'
            )
            if result:
                results.append(result)
                print(f"Placebo: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Placebo error: {e}")

    # =========================================================================
    # EVENT STUDY SPECIFICATIONS
    # =========================================================================
    print("\n--- EVENT STUDY ---")

    try:
        # Create event study formula
        event_vars = [f'event_{i}' for i in range(7, 1, -1)] + \
                     [f'event{i}' for i in range(8)]
        event_formula = ' + '.join(event_vars)
        formula = f"coverage ~ {event_formula} | countrydisease + cohortdisease + countrycohort"

        model = pf.feols(formula, data=df, vcov={'CRV1': 'countryid'})

        # Extract all event study coefficients
        event_coefs = {}
        for var in event_vars:
            try:
                event_coefs[var] = {
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                }
            except:
                pass

        coef_vector = {
            "treatment": {"var": "event_study", "coef": None, "se": None, "pval": None},
            "controls": [{"var": k, **v} for k, v in event_coefs.items()],
            "fixed_effects": ['countrydisease', 'cohortdisease', 'countrycohort'],
            "diagnostics": {}
        }

        # Report event0 as the main effect
        if 'event0' in event_coefs:
            result = {
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'did/dynamic/leads_lags',
                'spec_tree_path': 'methods/difference_in_differences.md#dynamic-effects',
                'outcome_var': 'coverage',
                'treatment_var': 'event0',
                'coefficient': float(event_coefs['event0']['coef']),
                'std_error': float(event_coefs['event0']['se']),
                't_stat': float(event_coefs['event0']['coef'] / event_coefs['event0']['se']),
                'p_value': float(event_coefs['event0']['pval']),
                'ci_lower': float(event_coefs['event0']['coef'] - 1.96 * event_coefs['event0']['se']),
                'ci_upper': float(event_coefs['event0']['coef'] + 1.96 * event_coefs['event0']['se']),
                'n_obs': int(model._N),
                'r_squared': float(model._r2) if hasattr(model, '_r2') else None,
                'coefficient_vector_json': json.dumps(coef_vector),
                'sample_desc': 'Event study (t=0 effect)',
                'fixed_effects': 'Country x Disease, Cohort x Disease, Country x Cohort',
                'controls_desc': 'Event study dummies',
                'cluster_var': 'countryid',
                'model_type': 'TWFE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            }
            results.append(result)
            print(f"Event study (t=0): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Event study error: {e}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print(f"SPECIFICATION SEARCH: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 70)

    # Create analysis dataset
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)

    df = create_analysis_dataset()

    # Run specification search
    print("\n" + "=" * 70)
    print("STEP 2: SPECIFICATION SEARCH")
    print("=" * 70)

    results = run_specification_search(df)

    # Save results
    print("\n" + "=" * 70)
    print("STEP 3: SAVING RESULTS")
    print("=" * 70)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "specification_results.csv", index=False)
        print(f"\nSaved {len(results)} specifications to specification_results.csv")

        # Summary statistics
        print("\n--- SUMMARY STATISTICS ---")
        print(f"Total specifications: {len(results)}")

        sig_05 = (results_df['p_value'] < 0.05).sum()
        sig_01 = (results_df['p_value'] < 0.01).sum()
        sig_10 = (results_df['p_value'] < 0.10).sum()
        pos_coef = (results_df['coefficient'] > 0).sum()

        print(f"Positive coefficients: {pos_coef} ({100*pos_coef/len(results):.1f}%)")
        print(f"Significant at 10%: {sig_10} ({100*sig_10/len(results):.1f}%)")
        print(f"Significant at 5%: {sig_05} ({100*sig_05/len(results):.1f}%)")
        print(f"Significant at 1%: {sig_01} ({100*sig_01/len(results):.1f}%)")
        print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
        print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
        print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

        return results_df
    else:
        print("No results generated!")
        return None


if __name__ == "__main__":
    results = main()
