"""
Specification Search: 112870-V2
Paper: "Optimal Life Cycle Unemployment Insurance" by Claudio Michelacci and Hernan Ruffo
American Economic Review

Main methodology: Cox proportional hazards models for unemployment duration elasticity
Key hypothesis: Unemployment duration elasticity to UI benefits varies by age (higher for older workers)

Treatment variable: log UI benefits (l_wba)
Outcome: Unemployment duration (dur)
Event: Job finding (censrd==0 means found job)

ORIGINAL STATA RESULTS (Table 1, Panel A):
- All workers: l_wba = -0.362 (SE=0.108), duractwba = -0.011 (SE=0.004)
- Young (20-40): l_wba = -0.228 (SE=0.162), duractwba = -0.019 (SE=0.006)
- Old (41-60): l_wba = -0.859 (SE=0.191), duractwba = 0.011 (SE=0.008)

NOTE: The original analysis uses Stata's stcox with episode-split data.
The data is split at each failure time, creating person-period observations.
The model includes l_wba and duractwba = l_wba * dur as time-varying covariate.

INTERPRETATION:
- Negative l_wba coefficient means higher benefits -> lower job finding hazard -> longer unemployment
- The coefficient represents the elasticity of the hazard with respect to benefits
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "112870-V2"
PAPER_TITLE = "Optimal Life Cycle Unemployment Insurance"
JOURNAL = "AER"
BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/112870-V2/DataCodes"

# Load data
print("Loading SIPP durations data...")
df = pd.read_stata(DATA_PATH / "SIPP_durations.dta")
print(f"Original data shape: {df.shape}")

# Apply baseline sample restrictions (from original code)
# ui_yn==1 (yes) and templayoff==0
df_analysis = df[(df['ui_yn'] == 'yes') & (df['templayoff'] == 0)].copy()
print(f"After sample restrictions (UI eligible, not temp layoff): {len(df_analysis)}")

# Convert categorical age to numeric
df_analysis['age'] = df_analysis['age'].astype(float)

# Censor at 50 weeks (as in original code)
df_analysis['dur_cens'] = df_analysis['dur'].clip(upper=50)
df_analysis['event'] = (df_analysis['censrd'] == 0).astype(int)
df_analysis.loc[df_analysis['dur'] > 50, 'event'] = 0

# Create onseam variable
df_analysis['onseam'] = ((df_analysis['dur'] == df_analysis['seam1wks']) |
                          (df_analysis['dur'] == df_analysis['seam2wks'])).fillna(0).astype(int)

# Create duration interaction with benefits (time-varying covariate)
df_analysis['duractwba'] = df_analysis['l_wba'] * df_analysis['dur_cens']

# Get valid rows
required_cols = ['l_wba', 'duractwba', 'dur_cens', 'event', 'age', 'mardum', 'ed', 'onseam']
df_analysis = df_analysis.dropna(subset=required_cols)
print(f"After dropping NA: {len(df_analysis)}")
print(f"Events (job found): {df_analysis['event'].sum()}")
print(f"Censored: {(df_analysis['event'] == 0).sum()}")

# Results storage
results = []


def create_fe_dummies(data, col_name, prefix):
    """Create fixed effect dummies, dropping first category."""
    if col_name not in data.columns:
        return data, []

    try:
        dummies = pd.get_dummies(data[col_name].astype(str), prefix=prefix, drop_first=True)
        fe_cols = list(dummies.columns)
        for col in fe_cols:
            data[col] = dummies[col].values
        return data, fe_cols
    except Exception as e:
        print(f"  Warning: Could not create dummies for {col_name}: {e}")
        return data, []


def run_cox_model(data, treatment_vars, controls, use_fe=True, cluster_var=None,
                  spec_id='baseline', spec_tree_path='methods/duration_survival.md',
                  sample_desc='Full sample', outcome_var='dur_cens'):
    """
    Run Cox proportional hazards model.

    treatment_vars: list of main treatment variables
    controls: list of control variables
    use_fe: whether to include state/year/occupation/industry fixed effects
    """
    data = data.copy()

    # Add FE if requested
    fe_vars = []
    if use_fe:
        data, state_fe = create_fe_dummies(data, 'sippst', 'state')
        fe_vars.extend(state_fe)

        data, year_fe = create_fe_dummies(data, 'year', 'year')
        fe_vars.extend(year_fe)

        data, occ_fe = create_fe_dummies(data, 'occ', 'occ')
        fe_vars.extend(occ_fe)

        data, ind_fe = create_fe_dummies(data, 'gind', 'ind')
        fe_vars.extend(ind_fe)

    # Wage splines
    wage_splines = [f'l_annwg_spl{i}' for i in range(1, 11)]
    valid_wage_splines = [ws for ws in wage_splines if ws in data.columns and data[ws].notna().sum() > 100]

    # All model variables
    model_vars = ['dur_cens', 'event'] + treatment_vars + controls + fe_vars + valid_wage_splines
    model_vars = [v for v in model_vars if v in data.columns]

    model_df = data[model_vars].dropna()

    if len(model_df) < 100:
        print(f"  Warning: Only {len(model_df)} observations for {spec_id}")
        return None

    # Fit Cox model
    cph = CoxPHFitter()

    try:
        cph.fit(model_df, duration_col='dur_cens', event_col='event',
                robust=True, show_progress=False)
    except Exception as e:
        print(f"  Error fitting model {spec_id}: {e}")
        return None

    # Extract main treatment coefficient (first in treatment_vars)
    main_treat = treatment_vars[0]
    if main_treat not in cph.summary.index:
        print(f"  Warning: Treatment {main_treat} not in results")
        return None

    coef = cph.summary.loc[main_treat, 'coef']
    se = cph.summary.loc[main_treat, 'se(coef)']
    pval = cph.summary.loc[main_treat, 'p']
    ci_lower = cph.summary.loc[main_treat, 'coef lower 95%']
    ci_upper = cph.summary.loc[main_treat, 'coef upper 95%']

    # Build coefficient vector
    coef_vector = {
        "treatment": {
            "var": main_treat,
            "coef": float(coef),
            "se": float(se),
            "hazard_ratio": float(np.exp(coef)),
            "pval": float(pval),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        },
        "controls": [],
        "diagnostics": {
            "concordance_index": float(cph.concordance_index_),
            "log_likelihood": float(cph.log_likelihood_),
            "n_subjects": int(len(model_df)),
            "n_events": int(model_df['event'].sum()),
            "n_censored": int((model_df['event'] == 0).sum())
        },
        "fixed_effects_absorbed": ['ind', 'state', 'year', 'occ'] if use_fe else []
    }

    # Add time-varying effect if duractwba is in treatment_vars
    if 'duractwba' in treatment_vars and 'duractwba' in cph.summary.index:
        coef_vector["time_varying_effect"] = {
            "var": "dur_x_lwba",
            "coef": float(cph.summary.loc['duractwba', 'coef']),
            "se": float(cph.summary.loc['duractwba', 'se(coef)']),
            "pval": float(cph.summary.loc['duractwba', 'p'])
        }

    # Add control coefficients
    for ctrl in controls:
        if ctrl in cph.summary.index:
            coef_vector["controls"].append({
                "var": ctrl,
                "coef": float(cph.summary.loc[ctrl, 'coef']),
                "se": float(cph.summary.loc[ctrl, 'se(coef)']),
                "pval": float(cph.summary.loc[ctrl, 'p'])
            })

    # Add wage spline coefficients
    for ws in valid_wage_splines:
        if ws in cph.summary.index:
            coef_vector["controls"].append({
                "var": ws,
                "coef": float(cph.summary.loc[ws, 'coef']),
                "se": float(cph.summary.loc[ws, 'se(coef)']),
                "pval": float(cph.summary.loc[ws, 'p'])
            })

    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': f'{outcome_var} (unemployment duration, weeks)',
        'treatment_var': main_treat,
        'coefficient': coef,
        'std_error': se,
        't_stat': coef / se if se > 0 else np.nan,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': len(model_df),
        'r_squared': np.nan,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': 'ind, state, year, occ' if use_fe else 'None',
        'controls_desc': ', '.join(controls),
        'cluster_var': 'robust SE',
        'model_type': 'Cox PH',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

    return result


# ============================================================
# RUN SPECIFICATION SEARCH
# ============================================================

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# Base controls (matching original Stata: onseam age mardum ed)
base_controls = ['onseam', 'age', 'mardum', 'ed']

# -----------------------------------------------------------
# 1. BASELINE - WITH TIME-VARYING COEFFICIENT (as in paper)
# -----------------------------------------------------------
print("\n1. Baseline specifications (with time-varying covariate)...")

# Main spec: l_wba + duractwba (l_wba * dur)
print("  Running: baseline (all workers, TVC)")
result = run_cox_model(
    df_analysis.copy(),
    treatment_vars=['l_wba', 'duractwba'],
    controls=base_controls,
    use_fe=True,
    spec_id='baseline',
    spec_tree_path='methods/duration_survival.md#baseline',
    sample_desc='All workers, UI eligible, not temp layoff'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (SE={result['std_error']:.4f}, p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 2. AGE SUBGROUP ANALYSIS (Core finding of the paper)
# -----------------------------------------------------------
print("\n2. Age subgroup analysis (CORE FINDING)...")

# Young workers (20-40)
df_young = df_analysis[(df_analysis['age'] >= 20) & (df_analysis['age'] <= 40)].copy()
print(f"  Running: Young workers (20-40), n={len(df_young)}")
result = run_cox_model(
    df_young,
    treatment_vars=['l_wba', 'duractwba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/sample/young',
    spec_tree_path='methods/duration_survival.md#sample-restrictions',
    sample_desc='Young workers (age 20-40)'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (SE={result['std_error']:.4f}, p={result['p_value']:.4f})")

# Old workers (41-60)
df_old = df_analysis[(df_analysis['age'] >= 41) & (df_analysis['age'] <= 60)].copy()
print(f"  Running: Old workers (41-60), n={len(df_old)}")
result = run_cox_model(
    df_old,
    treatment_vars=['l_wba', 'duractwba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/sample/old',
    spec_tree_path='methods/duration_survival.md#sample-restrictions',
    sample_desc='Older workers (age 41-60)'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (SE={result['std_error']:.4f}, p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 3. TIME-VARYING COVARIATE VARIATIONS
# -----------------------------------------------------------
print("\n3. Time-varying covariate variations...")

# Without TVC (just l_wba, no duration interaction)
print("  Running: No TVC (all workers)")
result = run_cox_model(
    df_analysis.copy(),
    treatment_vars=['l_wba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/tvc/none',
    spec_tree_path='methods/duration_survival.md#time-varying-covariates',
    sample_desc='All workers, no TVC'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Without TVC - Young workers
print("  Running: No TVC (young workers)")
result = run_cox_model(
    df_young.copy(),
    treatment_vars=['l_wba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/tvc/none_young',
    spec_tree_path='methods/duration_survival.md#time-varying-covariates',
    sample_desc='Young workers (20-40), no TVC'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Without TVC - Old workers
print("  Running: No TVC (old workers)")
result = run_cox_model(
    df_old.copy(),
    treatment_vars=['l_wba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/tvc/none_old',
    spec_tree_path='methods/duration_survival.md#time-varying-covariates',
    sample_desc='Older workers (41-60), no TVC'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 4. FIXED EFFECTS VARIATIONS
# -----------------------------------------------------------
print("\n4. Fixed effects variations...")

# No FE (pooled)
print("  Running: No fixed effects (pooled)")
result = run_cox_model(
    df_analysis.copy(),
    treatment_vars=['l_wba', 'duractwba'],
    controls=base_controls,
    use_fe=False,
    spec_id='duration/fe/none',
    spec_tree_path='methods/duration_survival.md#fixed-effects',
    sample_desc='All workers, no FE'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# No FE, no TVC
print("  Running: No FE, no TVC")
result = run_cox_model(
    df_analysis.copy(),
    treatment_vars=['l_wba'],
    controls=base_controls,
    use_fe=False,
    spec_id='duration/fe/none_no_tvc',
    spec_tree_path='methods/duration_survival.md#fixed-effects',
    sample_desc='All workers, no FE, no TVC'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 5. LEAVE-ONE-OUT ROBUSTNESS
# -----------------------------------------------------------
print("\n5. Leave-one-out robustness...")

for drop_var in base_controls:
    remaining = [c for c in base_controls if c != drop_var]
    print(f"  Running: Drop {drop_var}")
    result = run_cox_model(
        df_analysis.copy(),
        treatment_vars=['l_wba', 'duractwba'],
        controls=remaining,
        use_fe=True,
        spec_id=f'robust/loo/drop_{drop_var}',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc=f'All workers, dropping {drop_var}'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 6. SINGLE COVARIATE ANALYSIS
# -----------------------------------------------------------
print("\n6. Single covariate analysis...")

# Bivariate (with TVC)
print("  Running: Bivariate (TVC only)")
result = run_cox_model(
    df_analysis.copy(),
    treatment_vars=['l_wba', 'duractwba'],
    controls=[],
    use_fe=False,
    spec_id='robust/single/none',
    spec_tree_path='robustness/single_covariate.md',
    sample_desc='All workers, bivariate with TVC'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Single controls (with TVC)
for single_var in base_controls:
    print(f"  Running: Treatment + {single_var}")
    result = run_cox_model(
        df_analysis.copy(),
        treatment_vars=['l_wba', 'duractwba'],
        controls=[single_var],
        use_fe=False,
        spec_id=f'robust/single/{single_var}',
        spec_tree_path='robustness/single_covariate.md',
        sample_desc=f'All workers, treatment + {single_var}'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 7. AGE-SPECIFIC LEAVE-ONE-OUT
# -----------------------------------------------------------
print("\n7. Age-specific leave-one-out...")

for drop_var in ['onseam', 'mardum', 'ed']:
    remaining = [c for c in base_controls if c != drop_var]

    # Young
    print(f"  Running: Young, drop {drop_var}")
    result = run_cox_model(
        df_young.copy(),
        treatment_vars=['l_wba', 'duractwba'],
        controls=remaining,
        use_fe=True,
        spec_id=f'duration/sample/young/loo/drop_{drop_var}',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc=f'Young workers (20-40), dropping {drop_var}'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

    # Old
    print(f"  Running: Old, drop {drop_var}")
    result = run_cox_model(
        df_old.copy(),
        treatment_vars=['l_wba', 'duractwba'],
        controls=remaining,
        use_fe=True,
        spec_id=f'duration/sample/old/loo/drop_{drop_var}',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc=f'Older workers (41-60), dropping {drop_var}'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 8. CENSORING VARIATIONS
# -----------------------------------------------------------
print("\n8. Censoring variations...")

# Censor at 26 weeks
df_cens26 = df_analysis.copy()
df_cens26['dur_cens'] = df_cens26['dur'].clip(upper=26)
df_cens26.loc[df_cens26['dur'] > 26, 'event'] = 0
df_cens26['duractwba'] = df_cens26['l_wba'] * df_cens26['dur_cens']
print("  Running: Censor at 26 weeks")
result = run_cox_model(
    df_cens26,
    treatment_vars=['l_wba', 'duractwba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/censor/right_26',
    spec_tree_path='methods/duration_survival.md#censoring-treatment',
    sample_desc='All workers, censor at 26 weeks'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Censor at 99 weeks (effectively no artificial censoring for most)
df_cens99 = df_analysis.copy()
df_cens99['dur_cens'] = df_cens99['dur'].clip(upper=99)
df_cens99.loc[df_cens99['dur'] > 99, 'event'] = 0
df_cens99['duractwba'] = df_cens99['l_wba'] * df_cens99['dur_cens']
print("  Running: Censor at 99 weeks")
result = run_cox_model(
    df_cens99,
    treatment_vars=['l_wba', 'duractwba'],
    controls=base_controls,
    use_fe=True,
    spec_id='duration/censor/right_99',
    spec_tree_path='methods/duration_survival.md#censoring-treatment',
    sample_desc='All workers, censor at 99 weeks'
)
if result:
    results.append(result)
    print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 9. DEMOGRAPHIC SPLITS
# -----------------------------------------------------------
print("\n9. Demographic splits...")

# Married only
df_married = df_analysis[df_analysis['mardum'] == 1].copy()
if len(df_married) > 500:
    print(f"  Running: Married only, n={len(df_married)}")
    result = run_cox_model(
        df_married,
        treatment_vars=['l_wba', 'duractwba'],
        controls=['onseam', 'age', 'ed'],
        use_fe=True,
        spec_id='duration/sample/married',
        spec_tree_path='methods/duration_survival.md#sample-restrictions',
        sample_desc='Married workers only'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Not married
df_single = df_analysis[df_analysis['mardum'] == 0].copy()
if len(df_single) > 500:
    print(f"  Running: Not married, n={len(df_single)}")
    result = run_cox_model(
        df_single,
        treatment_vars=['l_wba', 'duractwba'],
        controls=['onseam', 'age', 'ed'],
        use_fe=True,
        spec_id='duration/sample/single',
        spec_tree_path='methods/duration_survival.md#sample-restrictions',
        sample_desc='Non-married workers'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# High education (HS or more)
df_high_ed = df_analysis[df_analysis['ed'] >= 12].copy()
if len(df_high_ed) > 500:
    print(f"  Running: High school or more, n={len(df_high_ed)}")
    result = run_cox_model(
        df_high_ed,
        treatment_vars=['l_wba', 'duractwba'],
        controls=['onseam', 'age', 'mardum'],
        use_fe=True,
        spec_id='duration/sample/high_education',
        spec_tree_path='methods/duration_survival.md#sample-restrictions',
        sample_desc='High school diploma or more'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Low education
df_low_ed = df_analysis[df_analysis['ed'] < 12].copy()
if len(df_low_ed) > 500:
    print(f"  Running: Less than high school, n={len(df_low_ed)}")
    result = run_cox_model(
        df_low_ed,
        treatment_vars=['l_wba', 'duractwba'],
        controls=['onseam', 'age', 'mardum'],
        use_fe=True,
        spec_id='duration/sample/low_education',
        spec_tree_path='methods/duration_survival.md#sample-restrictions',
        sample_desc='Less than high school'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 10. FINER AGE BANDS (as in Figure 1)
# -----------------------------------------------------------
print("\n10. Finer age bands...")

age_bands = [(15, 25), (20, 30), (25, 35), (30, 40), (35, 45), (40, 50), (45, 55), (50, 60)]
for age_min, age_max in age_bands:
    df_band = df_analysis[(df_analysis['age'] >= age_min) & (df_analysis['age'] <= age_max)].copy()
    if len(df_band) > 200:
        print(f"  Running: Age {age_min}-{age_max}, n={len(df_band)}")
        result = run_cox_model(
            df_band,
            treatment_vars=['l_wba', 'duractwba'],
            controls=['onseam', 'mardum', 'ed'],  # Exclude age since restricted
            use_fe=True,
            spec_id=f'duration/sample/age_{age_min}_{age_max}',
            spec_tree_path='methods/duration_survival.md#sample-restrictions',
            sample_desc=f'Workers age {age_min}-{age_max}'
        )
        if result:
            results.append(result)
            print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 11. ALTERNATIVE TREATMENT MEASURES
# -----------------------------------------------------------
print("\n11. Alternative treatment measures...")

# Average WBA
if 'l_avgwba' in df_analysis.columns and df_analysis['l_avgwba'].notna().sum() > 1000:
    df_avg = df_analysis.copy()
    df_avg['duravgwba'] = df_avg['l_avgwba'] * df_avg['dur_cens']
    print("  Running: Average WBA")
    result = run_cox_model(
        df_avg,
        treatment_vars=['l_avgwba', 'duravgwba'],
        controls=base_controls,
        use_fe=True,
        spec_id='duration/treatment/avg_wba',
        spec_tree_path='methods/duration_survival.md',
        sample_desc='All workers, average WBA'
    )
    if result:
        results.append(result)
        print(f"    l_avgwba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Max WBA
if 'l_maxwba' in df_analysis.columns and df_analysis['l_maxwba'].notna().sum() > 1000:
    df_max = df_analysis.copy()
    df_max['durmaxwba'] = df_max['l_maxwba'] * df_max['dur_cens']
    print("  Running: Max WBA")
    result = run_cox_model(
        df_max,
        treatment_vars=['l_maxwba', 'durmaxwba'],
        controls=base_controls,
        use_fe=True,
        spec_id='duration/treatment/max_wba',
        spec_tree_path='methods/duration_survival.md',
        sample_desc='All workers, max WBA'
    )
    if result:
        results.append(result)
        print(f"    l_maxwba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 12. TIME PERIOD SPLITS
# -----------------------------------------------------------
print("\n12. Time period splits...")

# Get median year
median_year = df_analysis['year'].median()

# Early period
df_early = df_analysis[df_analysis['year'] < median_year].copy()
if len(df_early) > 500:
    print(f"  Running: Early period (before {median_year}), n={len(df_early)}")
    result = run_cox_model(
        df_early,
        treatment_vars=['l_wba', 'duractwba'],
        controls=base_controls,
        use_fe=True,
        spec_id='robust/sample/early_period',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc=f'Early period (before {median_year})'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# Late period
df_late = df_analysis[df_analysis['year'] >= median_year].copy()
if len(df_late) > 500:
    print(f"  Running: Late period (from {median_year}), n={len(df_late)}")
    result = run_cox_model(
        df_late,
        treatment_vars=['l_wba', 'duractwba'],
        controls=base_controls,
        use_fe=True,
        spec_id='robust/sample/late_period',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc=f'Late period (from {median_year})'
    )
    if result:
        results.append(result)
        print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# 13. INTERACTION WITH AGE SUBGROUPS (no TVC)
# -----------------------------------------------------------
print("\n13. Age subgroups without TVC...")

for (age_min, age_max, label) in [(20, 40, 'young'), (41, 60, 'old')]:
    df_sub = df_analysis[(df_analysis['age'] >= age_min) & (df_analysis['age'] <= age_max)].copy()
    if len(df_sub) > 200:
        print(f"  Running: {label} workers, no TVC")
        result = run_cox_model(
            df_sub,
            treatment_vars=['l_wba'],
            controls=base_controls,
            use_fe=True,
            spec_id=f'duration/sample/{label}_no_tvc',
            spec_tree_path='methods/duration_survival.md#time-varying-covariates',
            sample_desc=f'{label.title()} workers ({age_min}-{age_max}), no TVC'
        )
        if result:
            results.append(result)
            print(f"    l_wba coef: {result['coefficient']:.4f} (p={result['p_value']:.4f})")

# -----------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
output_path = DATA_PATH / "specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved {len(results_df)} specifications to: {output_path}")

# Print summary
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total specifications: {len(results_df)}")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")

print("\n--- By Specification Category ---")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else x)
for cat, group in results_df.groupby('category'):
    sig_rate = 100 * (group['p_value'] < 0.05).mean()
    neg_rate = 100 * (group['coefficient'] < 0).mean()
    print(f"  {cat}: {len(group)} specs, {sig_rate:.0f}% significant, {neg_rate:.0f}% negative, median coef = {group['coefficient'].median():.4f}")

print("\n--- Key Findings ---")
baseline_result = results_df[results_df['spec_id'] == 'baseline']
young_result = results_df[results_df['spec_id'] == 'duration/sample/young']
old_result = results_df[results_df['spec_id'] == 'duration/sample/old']

if len(baseline_result) > 0:
    print(f"  Baseline (all workers): {baseline_result['coefficient'].values[0]:.4f} (SE={baseline_result['std_error'].values[0]:.4f})")
if len(young_result) > 0:
    print(f"  Young workers (20-40): {young_result['coefficient'].values[0]:.4f} (SE={young_result['std_error'].values[0]:.4f})")
if len(old_result) > 0:
    print(f"  Old workers (41-60): {old_result['coefficient'].values[0]:.4f} (SE={old_result['std_error'].values[0]:.4f})")

if len(young_result) > 0 and len(old_result) > 0:
    young_coef = young_result['coefficient'].values[0]
    old_coef = old_result['coefficient'].values[0]
    if young_coef != 0:
        ratio = old_coef / young_coef
        print(f"  Ratio (old/young): {ratio:.2f}x")
    print(f"  -> Paper finding: Older workers have {'larger' if abs(old_coef) > abs(young_coef) else 'smaller'} elasticity (in magnitude)")

print("\nDone!")
