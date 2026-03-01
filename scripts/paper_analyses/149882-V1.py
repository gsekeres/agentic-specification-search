#!/usr/bin/env python3
"""
Specification search runner for paper 149882-V1.
Dhar, Jain, Jayachandran (2022) "Reshaping Adolescents' Gender Attitudes:
Evidence from a School-Based Experiment in India"

Design: randomized_experiment (school-level)
Baseline groups:
  G1: E_Sgender_index2 ~ B_treat (Gender attitude index, EL1)
  G2: E_Sbehavior_index2 ~ B_treat (Self-reported behavior index, EL1)
"""

import sys, os, json, warnings, traceback
import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

# ----- Paths -----
REPO = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PKG = os.path.join(REPO, "data/downloads/extracted/149882-V1")
DATA = os.path.join(PKG, "data")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from agent_output_utils import (
    make_success_payload, make_failure_payload, error_details_from_exception,
    surface_hash as compute_surface_hash, software_block, safe_single_line
)

PAPER_ID = "149882-V1"

with open(os.path.join(PKG, "SPECIFICATION_SURFACE.json")) as f:
    SURFACE = json.load(f)
SHASH = compute_surface_hash(SURFACE)
SWBLOCK = software_block()

def load_dta(name):
    return pd.read_stata(os.path.join(DATA, name), convert_categoricals=False)

def safe_get(df, col, default=np.nan):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

def gen_neg_dummy(s):
    """Disagree/strongly disagree (4,5) = progressive = 1"""
    return s.apply(lambda x: 1.0 if pd.notna(x) and x >= 4 else (0.0 if pd.notna(x) and x > 0 else np.nan))

def gen_pos_dummy(s):
    """Agree/strongly agree (1,2) = progressive = 1"""
    return s.apply(lambda x: 1.0 if pd.notna(x) and x in [1,2] else (0.0 if pd.notna(x) and x > 0 else np.nan))

def icw_index(df_in, items, treat_col='B_treat', impute_groups=None):
    """
    Inverse-Covariance-Weighted (ICW) index following Anderson (2008).
    Returns (index_ni, flags_dict, weights).
    index_ni: non-imputed re-weighted version (the paper's final choice).
    """
    flags = {}
    work = df_in.copy()

    for item in items:
        flag_name = f'{item}_flag'
        flags[flag_name] = work[item].isna().astype(float)
        if impute_groups is not None:
            gm = work.groupby(impute_groups)[item].transform('mean')
            work.loc[work[item].isna(), item] = gm[work[item].isna()]

    # Standardize by overall mean and control-group SD
    temps = []
    for item in items:
        t = f'_t_{item}'
        mu = work[item].mean()
        sd_ctrl = work.loc[work[treat_col] == 0, item].std()
        if pd.isna(sd_ctrl) or sd_ctrl == 0:
            sd_ctrl = 1.0
        work[t] = (work[item] - mu) / sd_ctrl
        temps.append(t)

    # ICW weights from covariance matrix
    valid = work[temps].dropna()
    if len(valid) < len(temps) + 1:
        weights = np.ones(len(temps))
    else:
        C = np.cov(valid.values, rowvar=False)
        try:
            Cinv = np.linalg.inv(C)
            weights = Cinv.sum(axis=1)
        except np.linalg.LinAlgError:
            weights = np.ones(len(temps))

    W = weights.sum()

    # Non-imputed index (re-weighted for missingness)
    idx_ni = np.zeros(len(df_in))
    wsum = np.zeros(len(df_in))
    for j, item in enumerate(items):
        not_miss = (~df_in[item].isna()).astype(float).values
        kw = not_miss * weights[j]
        idx_ni += kw * work[temps[j]].fillna(0).values
        wsum += kw

    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(wsum > 0, W / wsum, np.nan)
    idx_ni *= scale

    work.drop(columns=temps, inplace=True, errors='ignore')
    return idx_ni, flags, weights


# ============================================================================
# DATA CONSTRUCTION
# ============================================================================
print("Loading raw data...")
bl_student = load_dta("baseline_student_raw.dta")
bl_school = load_dta("baseline_school_cleaned.dta")
bl_census = load_dta("baseline_census_cleaned.dta")
el1_raw = load_dta("endline1_student_raw.dta")
el2_raw = load_dta("endline2_student_raw.dta")

# ---- BASELINE STUDENT ----
print("Processing baseline student data...")
bl = bl_student.copy()
bl['school_id'] = pd.to_numeric(bl['school_id'], errors='coerce')
bl = bl[bl['child_disability'] != 4].copy()

# Gender corrections
wrong_ids = [4403014,4219046,4501026,4211125,4503045,4503042,2615060,3201019,
             1506041,1506028,1308038,1206035,1117086,1316101,3315043,3316025,
             3401011,3507104,3203050,3426090,3418048,2515021,2609037,2407047,
             2604054,2604052]
bl.loc[bl['child_id'].isin(wrong_ids), 'student_gender'] = 2
bl['girl'] = (bl['student_gender'] == 2).astype(float)
bl.loc[bl['student_gender'].isna(), 'girl'] = np.nan
bl['grade6'] = (bl['enrolled'] == 2).astype(float)
bl['class_num'] = np.where(bl['enrolled'] == 2, 6, np.where(bl['enrolled'] == 3, 7, np.nan))
bl['age'] = bl['child_age']
bl['caste_sc'] = bl['caste'].apply(lambda x: 1.0 if x == 16 else (0.0 if pd.notna(x) else np.nan))
bl['caste_st'] = bl['caste'].apply(lambda x: 1.0 if x == 17 else (0.0 if pd.notna(x) else np.nan))
bl['muslim'] = (bl['religion'] == 2).astype(float)
bl['house_pukka_y'] = bl['house_pukka'].apply(lambda x: 1.0 if x == 2 else (0.0 if x in [1,3] else np.nan))
bl['flush_toilet'] = bl['house_toilet'].apply(lambda x: 1.0 if x in [1,2] else (0.0 if x in [3,4,5] else np.nan))
bl['nonflush_toilet'] = bl['house_toilet'].apply(lambda x: 1.0 if x in [3,4] else (0.0 if x in [1,2,5] else np.nan))
bl['tap_water'] = bl['source_water'].apply(lambda x: 1.0 if x == 1 else (0.0 if x in [2,3,4] else np.nan))

# Gender attitude items (9 items for BL index)
for v in ['child_woman_role', 'child_man_final_deci', 'child_woman_tol_viol',
          'child_wives_less_edu', 'child_boy_more_opps']:
    bl[f'{v}_n'] = gen_neg_dummy(bl[v])
for v in ['child_equal_opps', 'child_girl_allow_study', 'child_similar_right', 'child_elect_woman']:
    bl[f'{v}_y'] = gen_pos_dummy(bl[v])

# BL behavior: talk_opp_gender (comfortable talking = progressive)
bl['talk_opp_gender_comm'] = bl['talk_opp_gender'].apply(
    lambda x: 1.0 if pd.notna(x) and x in [1,2] else (0.0 if pd.notna(x) and x > 0 else np.nan))
# cook_clean: help_cook == 1 means they cook => progressive for boys
bl['cook_clean_comm'] = safe_get(bl, 'help_cook').apply(
    lambda x: 1.0 if pd.notna(x) and x == 1 else (0.0 if pd.notna(x) and x > 0 else np.nan))

# Rename BL student with B_S prefix
bl_rn = {c: f'B_S{c}' for c in bl.columns if c not in ['school_id', 'child_id']}
bl = bl.rename(columns=bl_rn).rename(columns={'school_id': 'Sschool_id'})

# ---- BASELINE SCHOOL ----
print("Processing school data...")
sch = bl_school.copy()
sch = sch[sch['School_ID'] != 2704].copy()
sch = sch.rename(columns={'School_ID': 'Sschool_id', 'Coed': 'B_coed'})
if 'pct_male_teacher' in sch.columns:
    sch['pct_female_teacher'] = 1 - sch['pct_male_teacher']
sch_keep = ['Sschool_id', 'treat', 'B_coed'] + \
    [c for c in ['urban', 'q10_guest_teachr', 'fulltime_teacher', 'pct_female_teacher',
                 'q13_counselor', 'q18_pta_meet', 'q22_library', 'q22_toilets',
                 'q22_electricity', 'q22_avail_computers', 'q22_avail_internet',
                 'q22_sports_field', 'q22_mid_meal', 'q22_auditorium',
                 'q22_avail_edusat', 'q21_week1', 'q21_week6',
                 'coed_status', 'distance_hq'] if c in sch.columns]
sch = sch[[c for c in sch_keep if c in sch.columns]].copy()
sch_rn = {c: f'B_{c}' for c in sch.columns if c not in ['Sschool_id', 'treat', 'B_coed']}
sch = sch.rename(columns=sch_rn)

# ---- CENSUS ----
print("Processing census data...")
cen = bl_census.copy().rename(columns={'school_id': 'Sschool_id'})
cen.loc[cen['Sschool_id'] == 2704, 'Sschool_id'] = 2711
cen['Cfem_lab_part'] = cen['Ctot_work_f'] / cen['Ctot_f']
cen['Cfem_lit_rate'] = cen['Cf_lit'] / cen['Ctot_f']
cen['Cmale_lit_rate'] = cen['Cm_lit'] / cen['Ctot_m']
cen = cen[['Sschool_id', 'Cfem_lab_part', 'Cfem_lit_rate', 'Cmale_lit_rate']].copy()

# ---- ENDLINE 1 ----
print("Processing endline 1 data...")
el = el1_raw.copy()
# Drops
for cond in [('student_consent', [0]), ('parent_consent', [0]),
             ('student_present', [0])]:
    col, vals = cond
    if col in el.columns:
        el = el[~el[col].isin(vals)].copy()
if 'disability' in el.columns:
    el = el[~el['disability'].isin([3,4,5,6])].copy()
el = el[el['child_id'] != 3205037].copy()

# Gender attitude items (17 items)
for v in ['wives_less_edu', 'man_final_deci', 'woman_viol', 'control_daughters',
          'woman_role_home', 'men_better_suited', 'marriage_more_imp', 'teacher_suitable']:
    if v in el.columns:
        el[f'{v}_n'] = gen_neg_dummy(el[v])
for v in ['elect_woman', 'similar_right', 'allow_work']:
    if v in el.columns:
        el[f'{v}_y'] = gen_pos_dummy(el[v])
if 'boy_more_oppo' in el.columns:
    el['boy_more_oppo_n'] = gen_neg_dummy(el['boy_more_oppo'])
if 'town_studies' in el.columns:
    el['town_studies_y'] = el['town_studies'].apply(
        lambda x: 1.0 if pd.notna(x) and x in [2,3] else (0.0 if pd.notna(x) and x > 0 else np.nan))
# girl_marriage_age_19
if 'girl_marriage_age_numb' in el.columns:
    el['girl_marriage_age_19'] = el['girl_marriage_age_numb'].apply(
        lambda x: 1.0 if pd.notna(x) and x > 19 else (0.0 if pd.notna(x) and x <= 19 else np.nan))
# marriage_age_diff_m
if all(c in el.columns for c in ['boy_marriage_age_numb', 'girl_marriage_age_numb']):
    el['marriage_age_diff'] = el['boy_marriage_age_numb'] - el['girl_marriage_age_numb']
    med = el.loc[el['treatment'] == 0, 'marriage_age_diff'].median()
    el['marriage_age_diff_m'] = el['marriage_age_diff'].apply(
        lambda x: 1.0 if pd.notna(x) and x <= med else (0.0 if pd.notna(x) else np.nan))
# study_marry
if all(c in el.columns for c in ['girl_study_marry', 'boy_study_marry']):
    gsm = gen_pos_dummy(el['girl_study_marry'])
    bsm = gen_pos_dummy(el['boy_study_marry'])
    el['study_marry'] = np.where(pd.notna(gsm) & pd.notna(bsm), (gsm == bsm).astype(float), np.nan)
# fertility
if all(c in el.columns for c in ['two_girls', 'two_boys']):
    def _fert(r):
        tg, tb = r['two_girls'], r['two_boys']
        if pd.isna(tg) or pd.isna(tb): return np.nan
        if tb == 1 and tg in [2,3]: return -1.0
        if tg == 1 and tb in [2,3]: return 1.0
        return 0.0
    el['fertility'] = el.apply(_fert, axis=1)

# Behavior items
if all(c in el.columns for c in ['comf_opp_gender_girl', 'comf_opp_gender_boy']):
    el['talk_opp_gender'] = el['comf_opp_gender_girl'].fillna(0) + el['comf_opp_gender_boy'].fillna(0)
    el.loc[el['comf_opp_gender_girl'].isna() & el['comf_opp_gender_boy'].isna(), 'talk_opp_gender'] = np.nan
    el['talk_opp_gender_comm'] = el['talk_opp_gender'].apply(
        lambda x: 1.0 if pd.notna(x) and x in [1,2] else (0.0 if pd.notna(x) else np.nan))
if all(c in el.columns for c in ['sit_opp_gender_girl', 'sit_opp_gender_boy']):
    el['sit_opp_gender'] = el['sit_opp_gender_girl'].fillna(0) + el['sit_opp_gender_boy'].fillna(0)
    el.loc[el['sit_opp_gender_girl'].isna() & el['sit_opp_gender_boy'].isna(), 'sit_opp_gender'] = np.nan
    el['sit_opp_gender_comm'] = el['sit_opp_gender'].apply(
        lambda x: 1.0 if pd.notna(x) and x == 1 else (0.0 if pd.notna(x) else np.nan))
if 'cook_clean' in el.columns:
    el['cook_clean_comm'] = el['cook_clean'].apply(
        lambda x: 1.0 if pd.notna(x) and x == 1 else (0.0 if pd.notna(x) else np.nan))
if 'absent_sch_reason_hhwork' in el.columns:
    el['absent_sch_hhwork_comm'] = el['absent_sch_reason_hhwork'].apply(
        lambda x: 0.0 if pd.notna(x) and x == 1 else (1.0 if pd.notna(x) and x > 0 else np.nan))
elif 'absent_sch' in el.columns:
    el['absent_sch_hhwork_comm'] = np.nan  # not available at item level
if 'discourage_college' in el.columns:
    el['discourage_college_comm'] = el['discourage_college'].apply(
        lambda x: 0.0 if pd.notna(x) and x == 1 else (1.0 if pd.notna(x) and x > 0 else np.nan))
if 'discourage_work' in el.columns:
    el['discourage_work_comm'] = el['discourage_work'].apply(
        lambda x: 0.0 if pd.notna(x) and x == 1 else (1.0 if pd.notna(x) and x > 0 else np.nan))

# Rename EL1 with E_S prefix
el_rn = {c: f'E_S{c}' for c in el.columns if c not in ['school_id', 'child_id']}
el = el.rename(columns=el_rn).rename(columns={'school_id': 'Sschool_id'})

# ---- MERGE ----
print("Merging datasets...")

# EL2 uses school_id not Sschool_id; rename
el2_raw = el2_raw.rename(columns={'school_id': 'Sschool_id'})

# Start with el2 for merge structure (master dataset)
df = el2_raw[['Sschool_id', 'child_id']].copy()

# Merge BL student
df = df.merge(bl, on=['Sschool_id', 'child_id'], how='left', suffixes=('', '_bldup'))

# Merge EL1
df = df.merge(el, on=['Sschool_id', 'child_id'], how='left', suffixes=('', '_eldup'))

# Attrition
df['attrition_el'] = df['E_Streatment'].isna().astype(int)
df.loc[df['child_id'] == 1503100, 'attrition_el'] = 1

# Treatment from school
df = df.merge(sch[['Sschool_id', 'treat']], on='Sschool_id', how='left')
df['B_treat'] = df['treat']
df.loc[(df['Sschool_id'] == 2711) & df['B_treat'].isna(), 'B_treat'] = \
    df.loc[(df['Sschool_id'] == 2711) & df['B_treat'].isna(), 'E_Streatment']

# Fix school 2711 (not in BL)
df.loc[df['Sschool_id'] == 2711, 'B_Sgirl'] = 1.0
df.loc[df['Sschool_id'] == 2711, 'B_Sdistrict'] = df.loc[df['Sschool_id'] == 2711, 'E_Sdistrict']
df.loc[(df['Sschool_id'] == 2711) & (df.get('E_Sclass', pd.Series()) == 4), 'B_Sgrade6'] = 1.0
df.loc[(df['Sschool_id'] == 2711) & (df.get('E_Sclass', pd.Series()) == 5), 'B_Sgrade6'] = 0.0

# Merge school characteristics & census
df = df.merge(sch.drop(columns=['treat'], errors='ignore'), on='Sschool_id', how='left', suffixes=('', '_sch2'))
df = df.merge(cen, on='Sschool_id', how='left')

# E_Steam_id for sample restriction
# The do-file uses E_Steam_id (team_id from endline 1)
if 'E_Steam_id' not in df.columns:
    if 'E_Steam_id' in df.columns:
        pass
    elif 'E_Ssteam_id' in df.columns:
        df['E_Steam_id'] = df['E_Ssteam_id']
    else:
        # Use any non-null EL1 variable as proxy for "was surveyed in EL1"
        df['E_Steam_id'] = np.where(df['attrition_el'] == 0, 1.0, np.nan)

print(f"Merged dataset: {df.shape[0]} obs, {df['Sschool_id'].nunique()} schools")

# ---- STRATA FE ----
print("Creating strata FE...")
dg = df['B_Sdistrict'].astype(str) + '_' + df['B_Sgirl'].astype(str)
dg_dum = pd.get_dummies(dg, prefix='district_gender', dtype=float)
for i, c in enumerate(sorted(dg_dum.columns)):
    df[f'district_gender_{i+1}'] = dg_dum[c]
n_dg = len([c for c in df.columns if c.startswith('district_gender_')])

gg = df['B_Sgirl'].astype(str) + '_' + df['B_Sgrade6'].astype(str)
gg_dum = pd.get_dummies(gg, prefix='gender_grade', dtype=float)
for i, c in enumerate(sorted(gg_dum.columns)):
    df[f'gender_grade_{i+1}'] = gg_dum[c]
n_gg = len([c for c in df.columns if c.startswith('gender_grade_')])

dg_vars = [f'district_gender_{i+1}' for i in range(n_dg)]
gg_vars = [f'gender_grade_{i+1}' for i in range(n_gg)]
strata_fe = dg_vars + gg_vars
print(f"  {n_dg} district-gender dummies, {n_gg} gender-grade dummies")

# ---- ICW INDICES ----
print("Constructing ICW indices...")

# Endline gender attitude index items
el1_gender_items = []
for v in ['E_Swives_less_edu_n', 'E_Select_woman_y', 'E_Sboy_more_oppo_n', 'E_Stown_studies_y',
          'E_Sman_final_deci_n', 'E_Swoman_viol_n', 'E_Scontrol_daughters_n', 'E_Swoman_role_home_n',
          'E_Smen_better_suited_n', 'E_Ssimilar_right_y', 'E_Smarriage_more_imp_n',
          'E_Steacher_suitable_n', 'E_Sgirl_marriage_age_19', 'E_Smarriage_age_diff_m',
          'E_Sstudy_marry', 'E_Sallow_work_y', 'E_Sfertility']:
    if v in df.columns:
        el1_gender_items.append(v)
print(f"  Gender index: {len(el1_gender_items)} items")

el1_behav_items = []
for v in ['E_Stalk_opp_gender_comm', 'E_Ssit_opp_gender_comm', 'E_Scook_clean_comm',
          'E_Sabsent_sch_hhwork_comm', 'E_Sdiscourage_college_comm', 'E_Sdiscourage_work_comm']:
    if v in df.columns:
        el1_behav_items.append(v)
print(f"  Behavior index: {len(el1_behav_items)} items")

# Build indices
impute_groups = ['B_Sgirl', 'B_Sdistrict', 'B_treat']
valid_groups = [g for g in impute_groups if g in df.columns]

if len(el1_gender_items) >= 3:
    gi, gi_flags, gi_wts = icw_index(df, el1_gender_items, 'B_treat', valid_groups)
    df['E_Sgender_index2'] = gi
    for fn, fv in gi_flags.items():
        df[fn] = fv
else:
    df['E_Sgender_index2'] = np.nan

if len(el1_behav_items) >= 2:
    bi, bi_flags, bi_wts = icw_index(df, el1_behav_items, 'B_treat', valid_groups)
    df['E_Sbehavior_index2'] = bi
    for fn, fv in bi_flags.items():
        df[fn] = fv
else:
    df['E_Sbehavior_index2'] = np.nan

# Baseline indices
bl_gender_items = [v for v in ['B_Schild_woman_role_n', 'B_Schild_man_final_deci_n',
    'B_Schild_woman_tol_viol_n', 'B_Schild_wives_less_edu_n', 'B_Schild_boy_more_opps_n',
    'B_Schild_equal_opps_y', 'B_Schild_girl_allow_study_y',
    'B_Schild_similar_right_y', 'B_Schild_elect_woman_y'] if v in df.columns]

bl_behav_items = [v for v in ['B_Scook_clean_comm', 'B_Stalk_opp_gender_comm'] if v in df.columns]

if len(bl_gender_items) >= 3:
    bgi, bgi_flags, _ = icw_index(df, bl_gender_items, 'B_treat', valid_groups)
    df['B_Sgender_index2'] = bgi
    df['B_Sgender_index2_flag'] = df['B_Sgender_index2'].isna().astype(float)
    gm = df.groupby(['B_Sdistrict', 'B_Sgirl'])['B_Sgender_index2'].transform('mean')
    df.loc[df['B_Sgender_index2'].isna(), 'B_Sgender_index2'] = gm
else:
    df['B_Sgender_index2'] = 0.0
    df['B_Sgender_index2_flag'] = 1.0

if len(bl_behav_items) >= 2:
    bbi, bbi_flags, _ = icw_index(df, bl_behav_items, 'B_treat', valid_groups)
    df['B_Sbehavior_index2'] = bbi
    df['B_Sbehavior_index2_flag'] = df['B_Sbehavior_index2'].isna().astype(float)
    gm = df.groupby(['B_Sdistrict', 'B_Sgirl'])['B_Sbehavior_index2'].transform('mean')
    df.loc[df['B_Sbehavior_index2'].isna(), 'B_Sbehavior_index2'] = gm
else:
    df['B_Sbehavior_index2'] = 0.0
    df['B_Sbehavior_index2_flag'] = 1.0

# Standardize
for v in ['E_Sgender_index2', 'E_Sbehavior_index2', 'B_Sgender_index2', 'B_Sbehavior_index2']:
    ctrl = df.loc[df['B_treat'] == 0, v]
    m, s = ctrl.mean(), ctrl.std()
    if pd.notna(s) and s > 0:
        df[v] = (df[v] - m) / s

# Identify flag columns
el_gender_flags = sorted([c for c in df.columns if c.endswith('_flag') and
    any(x in c for x in ['wives_less_edu', 'elect_woman', 'boy_more_oppo', 'town_studies',
        'man_final_deci', 'woman_viol', 'control_daughters', 'woman_role_home',
        'men_better_suited', 'similar_right', 'marriage_more_imp', 'teacher_suitable',
        'girl_marriage_age', 'marriage_age_diff', 'study_marry', 'allow_work', 'fertility'])
    and c.startswith('E_S')])

el_behav_flags = sorted([c for c in df.columns if c.endswith('_flag') and
    any(x in c for x in ['talk_opp_gender_comm', 'sit_opp_gender_comm', 'cook_clean_comm',
        'absent_sch_hhwork_comm', 'discourage_college_comm', 'discourage_work_comm'])
    and c.startswith('E_S')])

print(f"Gender flags: {len(el_gender_flags)}, Behavior flags: {len(el_behav_flags)}")

# Sample
df['in_sample'] = ((df['attrition_el'] == 0) & df['E_Steam_id'].notna()).astype(int)
# Fallback: if E_Steam_id is all NaN, use attrition_el only
if df['in_sample'].sum() == 0:
    df['in_sample'] = (df['attrition_el'] == 0).astype(int)
print(f"Analysis sample: {df['in_sample'].sum()} obs")

# ============================================================================
# SPECIFICATION RUNNER
# ============================================================================

results = []
infer_results = []
rc = [0]  # mutable counter

def make_design():
    """Copy design_audit verbatim from surface for baseline/rc rows."""
    return {"randomized_experiment": {
        "estimator": "ols_with_covariates",
        "randomization_unit": "school",
        "strata_or_blocks": "district x gender x grade (district_gender_*, gender_grade_*)",
        "estimand": "ITT",
        "treatment_arms": ["B_treat (Breakthrough gender-equality curriculum)"],
        "n_treatment_arms_focal": 1,
        "cluster_var": "Sschool_id",
        "weights": "none",
        "control_selection": "LASSO-selected extended controls (double LASSO, seed=5212021)",
        "n_clusters_approx": 314,
        "notes": "~314 schools (half treatment, half control). 4 districts."
    }}

def run_one(spec_id, spec_run_id, gid, yvar, xvar, ctrls, mask, tree_path,
            sdesc='', fedesc='', cdesc='', extra=None, vcov=None):
    rc[0] += 1
    if vcov is None:
        vcov = {"CRV1": "Sschool_id"}
    row = dict(paper_id=PAPER_ID, spec_run_id=spec_run_id, spec_id=spec_id,
               spec_tree_path=tree_path, baseline_group_id=gid,
               outcome_var=yvar, treatment_var=xvar,
               sample_desc=sdesc, fixed_effects=fedesc,
               controls_desc=cdesc, cluster_var='Sschool_id')
    try:
        adf = df[mask].copy()
        rhs = [v for v in [xvar] + ctrls if v in adf.columns]
        reg_vars = [yvar] + rhs + ['Sschool_id']
        reg_vars = list(dict.fromkeys([v for v in reg_vars if v in adf.columns]))
        adf = adf.dropna(subset=reg_vars)
        if len(adf) < 20:
            raise ValueError(f"Only {len(adf)} obs after dropna")
        fml = f"{yvar} ~ " + " + ".join(rhs)
        m = pf.feols(fml, data=adf, vcov=vcov)
        coef = float(m.coef()[xvar])
        se = float(m.se()[xvar])
        pv = float(m.pvalue()[xvar])
        ci = m.confint()
        cil = float(ci.loc[xvar, '2.5%']) if xvar in ci.index else np.nan
        ciu = float(ci.loc[xvar, '97.5%']) if xvar in ci.index else np.nan
        nobs = int(m._N)
        r2 = float(m._r2)
        all_c = {k: float(v) for k, v in m.coef().items()}
        payload = make_success_payload(
            coefficients=all_c,
            inference={"spec_id": "infer/se/cluster/school", "params": {"cluster_var": "Sschool_id"}},
            software=SWBLOCK, surface_hash=SHASH, design=make_design(), extra=extra)
        row.update(coefficient=coef, std_error=se, p_value=pv,
                   ci_lower=cil, ci_upper=ciu, n_obs=nobs, r_squared=r2,
                   coefficient_vector_json=json.dumps(payload), run_success=1, run_error='')
    except Exception as e:
        em = safe_single_line(str(e))
        ed = error_details_from_exception(e, stage='estimation')
        payload = make_failure_payload(error=em, error_details=ed, software=SWBLOCK, surface_hash=SHASH)
        row.update(coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                   ci_lower=np.nan, ci_upper=np.nan, n_obs=np.nan, r_squared=np.nan,
                   coefficient_vector_json=json.dumps(payload), run_success=0, run_error=em)
    return row

def run_infer(base, isid, vcov, iparams):
    rc[0] += 1
    irow = dict(paper_id=PAPER_ID, inference_run_id=f'{PAPER_ID}_infer_{rc[0]}',
                spec_run_id=base['spec_run_id'], spec_id=isid,
                spec_tree_path='specification_tree/modules/inference/standard_errors.md',
                baseline_group_id=base['baseline_group_id'],
                outcome_var=base.get('outcome_var', ''),
                treatment_var=base.get('treatment_var', ''),
                cluster_var=base.get('cluster_var', 'Sschool_id'))
    if base['run_success'] == 0:
        payload = make_failure_payload(error='base failed',
            error_details={'stage':'inference','exception_type':'BaseSpecFailed','exception_message':'base failed'},
            software=SWBLOCK, surface_hash=SHASH)
        irow.update(coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                    ci_lower=np.nan, ci_upper=np.nan, n_obs=np.nan, r_squared=np.nan,
                    coefficient_vector_json=json.dumps(payload), run_success=0, run_error='base failed')
        return irow
    try:
        bp = json.loads(base['coefficient_vector_json'])
        yv = base['outcome_var']
        xv = base['treatment_var']
        rhs = [k for k in bp.get('coefficients', {}).keys() if k != 'Intercept']
        mask = df['in_sample'] == 1
        adf = df[mask].copy()
        rhs_ok = [v for v in rhs if v in adf.columns]
        reg_vars = list(dict.fromkeys([yv] + rhs_ok + ['Sschool_id'] +
            ([list(vcov.values())[0]] if isinstance(vcov, dict) else [])))
        reg_vars = [v for v in reg_vars if v in adf.columns]
        adf = adf.dropna(subset=reg_vars)
        fml = f"{yv} ~ " + " + ".join(rhs_ok)
        m = pf.feols(fml, data=adf, vcov=vcov)
        coef = float(m.coef()[xv])
        se = float(m.se()[xv])
        pv = float(m.pvalue()[xv])
        ci = m.confint()
        payload = {'coefficients': {k: float(v) for k, v in m.coef().items()},
                   'inference': {'spec_id': isid, 'params': iparams},
                   'software': SWBLOCK, 'surface_hash': SHASH}
        irow.update(coefficient=coef, std_error=se, p_value=pv,
                    ci_lower=float(ci.loc[xv,'2.5%']), ci_upper=float(ci.loc[xv,'97.5%']),
                    n_obs=int(m._N), r_squared=float(m._r2),
                    coefficient_vector_json=json.dumps(payload), run_success=1, run_error='')
    except Exception as e:
        em = safe_single_line(str(e))
        ed = error_details_from_exception(e, stage='inference')
        payload = make_failure_payload(error=em, error_details=ed, software=SWBLOCK, surface_hash=SHASH)
        irow.update(coefficient=np.nan, std_error=np.nan, p_value=np.nan,
                    ci_lower=np.nan, ci_upper=np.nan, n_obs=np.nan, r_squared=np.nan,
                    coefficient_vector_json=json.dumps(payload), run_success=0, run_error=em)
    return irow


# ============================================================================
# RUN ALL SPECS
# ============================================================================
print("\n" + "="*80)
print("EXECUTING SPECIFICATIONS")
print("="*80)

mask = df['in_sample'] == 1

groups = {
    'G1': dict(outcome='E_Sgender_index2', bl_idx='B_Sgender_index2', flags=el_gender_flags, label='gender_attitudes'),
    'G2': dict(outcome='E_Sbehavior_index2', bl_idx='B_Sbehavior_index2', flags=el_behav_flags, label='behavior'),
}

# Extended control candidates
ext_vars = [v for v in [
    'B_Sage', 'B_Sgrade6', 'B_Surban',
    'B_Scaste_sc', 'B_Scaste_st', 'B_Smuslim',
    'B_Shouse_pukka_y', 'B_Shouse_elec', 'B_Sflush_toilet', 'B_Snonflush_toilet',
    'B_Sown_house', 'B_Snewspaper_house', 'B_Stap_water',
    'Cfem_lit_rate', 'Cmale_lit_rate', 'Cfem_lab_part',
    'B_coed', 'B_fulltime_teacher', 'B_pct_female_teacher',
    'B_q13_counselor', 'B_q18_pta_meet',
    'B_q22_library', 'B_q22_toilets', 'B_q22_electricity',
    'B_q22_avail_computers', 'B_q22_avail_internet', 'B_q22_sports_field',
    'B_q22_mid_meal', 'B_q22_auditorium', 'B_q22_avail_edusat',
] if v in df.columns]
demo_vars = [v for v in ['B_Sage', 'B_Sgrade6', 'B_Surban', 'B_Scaste_sc', 'B_Scaste_st', 'B_Smuslim'] if v in df.columns]

OLS_PATH = 'specification_tree/methods/cross_sectional_ols.md'
RCT_PATH = 'specification_tree/designs/randomized_experiment.md'
CTRL_PATH = 'specification_tree/modules/robustness/controls.md'
SAMP_PATH = 'specification_tree/modules/robustness/sample.md'
DATA_PATH = 'specification_tree/modules/robustness/data_construction.md'

for gid, g in groups.items():
    yv = g['outcome']
    bi = g['bl_idx']
    fl = g['flags']
    lbl = g['label']
    base_ctrls = [bi] + strata_fe + fl
    base_ctrls = [v for v in base_ctrls if v in df.columns]

    print(f"\n--- {gid}: {lbl} ---")

    # 1. BASELINE
    r = run_one('baseline', f'{PAPER_ID}_{gid}_baseline', gid, yv, 'B_treat', base_ctrls, mask,
                OLS_PATH, 'EL1 non-attrited', 'strata FE dummies', f'BL {lbl} index + strata FE + EL flags')
    results.append(r); bl_row = r
    print(f"  baseline: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}" if r['run_success'] else f"  baseline: FAILED - {r['run_error']}")

    # 2. DESIGN VARIANTS
    r = run_one('design/randomized_experiment/estimator/diff_in_means',
                f'{PAPER_ID}_{gid}_d_dim', gid, yv, 'B_treat', [], mask,
                f'{RCT_PATH}#diff_in_means', 'EL1', 'none', 'No controls')
    results.append(r)

    r = run_one('design/randomized_experiment/estimator/strata_fe',
                f'{PAPER_ID}_{gid}_d_strata', gid, yv, 'B_treat', strata_fe, mask,
                f'{RCT_PATH}#strata_fe', 'EL1', 'strata FE', 'Strata FE only')
    results.append(r)

    # 3. RC/CONTROLS/PROGRESSION
    for sid, rid, clist, cdesc in [
        ('rc/controls/progression/bivariate', f'{PAPER_ID}_{gid}_rc_bivar', [], 'Bivariate'),
        ('rc/controls/progression/strata_fe_only', f'{PAPER_ID}_{gid}_rc_strata', strata_fe, 'Strata FE'),
        ('rc/controls/progression/strata_fe_plus_bl_index', f'{PAPER_ID}_{gid}_rc_strata_bl', [bi]+strata_fe, f'Strata FE + BL {lbl} idx'),
        ('rc/controls/progression/strata_fe_plus_bl_index_plus_demographics', f'{PAPER_ID}_{gid}_rc_strata_demo',
         [bi]+strata_fe+demo_vars, f'Strata FE + BL idx + demographics'),
        ('rc/controls/progression/strata_fe_plus_lasso_extended', f'{PAPER_ID}_{gid}_rc_lasso',
         [bi]+strata_fe+fl+ext_vars, f'Strata FE + BL idx + flags + LASSO extended'),
    ]:
        clist_ok = [v for v in clist if v in df.columns]
        r = run_one(sid, rid, gid, yv, 'B_treat', clist_ok, mask,
                    f'{CTRL_PATH}#progression', 'EL1', 'strata FE', cdesc)
        results.append(r)

    # 4. RC/CONTROLS/SETS
    for sid, rid, clist, cdesc in [
        ('rc/controls/sets/strata_plus_lasso', f'{PAPER_ID}_{gid}_rc_s_lasso',
         [bi]+strata_fe+ext_vars, 'Strata FE + BL idx + LASSO (no EL flags)'),
        ('rc/controls/sets/no_missing_flags', f'{PAPER_ID}_{gid}_rc_s_noflags',
         [bi]+strata_fe, 'Strata FE + BL idx (no flags)'),
    ]:
        clist_ok = [v for v in clist if v in df.columns]
        r = run_one(sid, rid, gid, yv, 'B_treat', clist_ok, mask,
                    f'{CTRL_PATH}#sets', 'EL1', 'strata FE', cdesc)
        results.append(r)

    # 5. RC/CONTROLS/LOO
    for sid, rid, drop_fn, cdesc in [
        ('rc/controls/loo/drop_baseline_index', f'{PAPER_ID}_{gid}_rc_l_bli',
         lambda v: v == bi, f'Baseline ctrls minus BL {lbl} index'),
        ('rc/controls/loo/drop_district_gender_fe', f'{PAPER_ID}_{gid}_rc_l_dg',
         lambda v: v.startswith('district_gender_'), 'Baseline ctrls minus district-gender FE'),
        ('rc/controls/loo/drop_gender_grade_fe', f'{PAPER_ID}_{gid}_rc_l_gg',
         lambda v: v.startswith('gender_grade_'), 'Baseline ctrls minus gender-grade FE'),
        ('rc/controls/loo/drop_missing_flags', f'{PAPER_ID}_{gid}_rc_l_fl',
         lambda v: v in fl, 'Baseline ctrls minus EL flags'),
    ]:
        loo = [v for v in base_ctrls if not drop_fn(v)]
        r = run_one(sid, rid, gid, yv, 'B_treat', loo, mask,
                    f'{CTRL_PATH}#loo', 'EL1', 'strata FE', cdesc)
        results.append(r)

    # 6. RC/SAMPLE/OUTLIERS
    p1 = df.loc[mask, yv].quantile(0.01)
    p99 = df.loc[mask, yv].quantile(0.99)
    tmask = mask & (df[yv] >= p1) & (df[yv] <= p99)
    r = run_one('rc/sample/outliers/trim_y_1_99', f'{PAPER_ID}_{gid}_rc_trim',
                gid, yv, 'B_treat', base_ctrls, tmask,
                f'{SAMP_PATH}#outliers', 'EL1 trimmed 1-99 pctile', 'strata FE', f'Baseline ctrls')
    results.append(r)

    # 7. RC/DATA/INDEX_CONSTRUCTION/EQUAL_WEIGHT
    items = el1_gender_items if gid == 'G1' else el1_behav_items
    if len(items) >= 2:
        eq = df[items].mean(axis=1, skipna=True)
        cm = eq[df['B_treat']==0].mean()
        cs = eq[df['B_treat']==0].std()
        df[f'{yv}_eqwt'] = (eq - cm) / cs if cs > 0 else eq
    else:
        df[f'{yv}_eqwt'] = np.nan
    r = run_one('rc/data/index_construction/equal_weight_index', f'{PAPER_ID}_{gid}_rc_eqwt',
                gid, f'{yv}_eqwt', 'B_treat', base_ctrls, mask,
                DATA_PATH, 'EL1', 'strata FE', 'Equal-weight index, same controls',
                extra={'data_construction': {'description': 'Equal-weight average instead of ICW'}})
    results.append(r)

    # 8. INFERENCE VARIANTS
    print(f"  Inference variants...")
    infer_results.append(run_infer(bl_row, 'infer/se/hc/hc1', 'hetero', {}))
    infer_results.append(run_infer(bl_row, 'infer/se/hc/hc3', 'HC3', {}))
    if 'B_Sdistrict' in df.columns:
        infer_results.append(run_infer(bl_row, 'infer/se/cluster/district',
                                       {"CRV1": "B_Sdistrict"}, {"cluster_var": "B_Sdistrict"}))

# ============================================================================
# EXPLORATION SPECS (subgroup analyses - separate table)
# ============================================================================
print("\n--- Exploration specs (subgroups) ---")
explore_results = []

for gid, g in groups.items():
    yv = g['outcome']
    bi = g['bl_idx']
    fl = g['flags']
    lbl = g['label']
    base_ctrls = [bi] + strata_fe + fl
    base_ctrls = [v for v in base_ctrls if v in df.columns]

    # Girls only: use district dummies + grade6 instead of strata FE
    girls_ctrls = [bi] + [f'district_{d}' if f'district_{d}' in df.columns else f'B_Sdistrict_{d}' for d in range(1,5)] + ['B_Sgrade6'] + fl
    girls_ctrls = [v for v in girls_ctrls if v in df.columns]
    girls_mask = mask & (df['B_Sgirl'] == 1)
    r = run_one('explore/population/girls_only', f'{PAPER_ID}_{gid}_ex_girls',
                gid, yv, 'B_treat', girls_ctrls, girls_mask,
                f'{RCT_PATH}#subgroup', 'EL1, girls only', 'district FE + grade',
                f'BL {lbl} idx + district FE + grade + EL flags')
    explore_results.append(r)

    # Boys only
    boys_mask = mask & (df['B_Sgirl'] == 0)
    r = run_one('explore/population/boys_only', f'{PAPER_ID}_{gid}_ex_boys',
                gid, yv, 'B_treat', girls_ctrls, boys_mask,
                f'{RCT_PATH}#subgroup', 'EL1, boys only', 'district FE + grade',
                f'BL {lbl} idx + district FE + grade + EL flags')
    explore_results.append(r)

    # Grade 6 only
    g6_mask = mask & (df['B_Sgrade6'] == 1)
    r = run_one('explore/population/grade6_only', f'{PAPER_ID}_{gid}_ex_g6',
                gid, yv, 'B_treat', base_ctrls, g6_mask,
                f'{RCT_PATH}#subgroup', 'EL1, grade 6 at baseline', 'strata FE',
                'Baseline controls')
    explore_results.append(r)

    # Grade 7 only
    g7_mask = mask & (df['B_Sgrade6'] == 0)
    r = run_one('explore/population/grade7_only', f'{PAPER_ID}_{gid}_ex_g7',
                gid, yv, 'B_treat', base_ctrls, g7_mask,
                f'{RCT_PATH}#subgroup', 'EL1, grade 7 at baseline', 'strata FE',
                'Baseline controls')
    explore_results.append(r)

    # Coed schools only
    if 'B_coed_status' in df.columns:
        coed_mask = mask & df['B_coed_status'].isin([1, 4])
    elif 'B_coed' in df.columns:
        coed_mask = mask & (df['B_coed'] == 1)
    else:
        coed_mask = mask  # fallback
    r = run_one('explore/population/coed_schools_only', f'{PAPER_ID}_{gid}_ex_coed',
                gid, yv, 'B_treat', base_ctrls, coed_mask,
                f'{RCT_PATH}#subgroup', 'EL1, coed schools', 'strata FE',
                'Baseline controls')
    explore_results.append(r)

    # Outcome subindices
    if gid == 'G1':
        # Education subindex
        educ_items = [v for v in ['E_Swives_less_edu_n', 'E_Sboy_more_oppo_n', 'E_Stown_studies_y'] if v in df.columns]
        if len(educ_items) >= 2:
            eq = df[educ_items].mean(axis=1, skipna=True)
            cm, cs = eq[df['B_treat']==0].mean(), eq[df['B_treat']==0].std()
            df['E_Sgender_educ_subidx'] = (eq - cm) / cs if cs > 0 else eq
        r = run_one('explore/outcome/gender_subindex_education', f'{PAPER_ID}_{gid}_ex_educ',
                    gid, 'E_Sgender_educ_subidx', 'B_treat', base_ctrls, mask,
                    f'{RCT_PATH}#subgroup', 'EL1', 'strata FE', 'Education attitudes subindex')
        explore_results.append(r)

        # Employment subindex
        emp_items = [v for v in ['E_Swoman_role_home_n', 'E_Smen_better_suited_n',
                     'E_Smarriage_more_imp_n', 'E_Steacher_suitable_n', 'E_Sallow_work_y'] if v in df.columns]
        if len(emp_items) >= 2:
            eq = df[emp_items].mean(axis=1, skipna=True)
            cm, cs = eq[df['B_treat']==0].mean(), eq[df['B_treat']==0].std()
            df['E_Sgender_emp_subidx'] = (eq - cm) / cs if cs > 0 else eq
        r = run_one('explore/outcome/gender_subindex_employment', f'{PAPER_ID}_{gid}_ex_emp',
                    gid, 'E_Sgender_emp_subidx', 'B_treat', base_ctrls, mask,
                    f'{RCT_PATH}#subgroup', 'EL1', 'strata FE', 'Employment attitudes subindex')
        explore_results.append(r)

        # Subjugation subindex
        sub_items = [v for v in ['E_Select_woman_y', 'E_Sman_final_deci_n', 'E_Swoman_viol_n',
                     'E_Scontrol_daughters_n', 'E_Ssimilar_right_y', 'E_Sgirl_marriage_age_19',
                     'E_Smarriage_age_diff_m', 'E_Sstudy_marry'] if v in df.columns]
        if len(sub_items) >= 2:
            eq = df[sub_items].mean(axis=1, skipna=True)
            cm, cs = eq[df['B_treat']==0].mean(), eq[df['B_treat']==0].std()
            df['E_Sgender_sub_subidx'] = (eq - cm) / cs if cs > 0 else eq
        r = run_one('explore/outcome/gender_subindex_subjugation', f'{PAPER_ID}_{gid}_ex_sub',
                    gid, 'E_Sgender_sub_subidx', 'B_treat', base_ctrls, mask,
                    f'{RCT_PATH}#subgroup', 'EL1', 'strata FE', 'Female subjugation attitudes subindex')
        explore_results.append(r)

    elif gid == 'G2':
        # Behavior subindices: opposite sex interaction
        oppsex_items = [v for v in ['E_Stalk_opp_gender_comm', 'E_Ssit_opp_gender_comm'] if v in df.columns]
        if len(oppsex_items) >= 2:
            eq = df[oppsex_items].mean(axis=1, skipna=True)
            cm, cs = eq[df['B_treat']==0].mean(), eq[df['B_treat']==0].std()
            df['E_Sbehav_oppsex_subidx'] = (eq - cm) / cs if cs > 0 else eq
        r = run_one('explore/outcome/behavior_subindex_oppsex', f'{PAPER_ID}_{gid}_ex_oppsex',
                    gid, 'E_Sbehav_oppsex_subidx', 'B_treat', base_ctrls, mask,
                    f'{RCT_PATH}#subgroup', 'EL1', 'strata FE', 'Opposite sex interaction subindex')
        explore_results.append(r)

        # HH chores subindex
        hh_items = [v for v in ['E_Scook_clean_comm', 'E_Sabsent_sch_hhwork_comm'] if v in df.columns]
        if len(hh_items) >= 1:
            eq = df[hh_items].mean(axis=1, skipna=True)
            cm, cs = eq[df['B_treat']==0].mean(), eq[df['B_treat']==0].std()
            df['E_Sbehav_hh_subidx'] = (eq - cm) / cs if cs > 0 else eq
        r = run_one('explore/outcome/behavior_subindex_hhchores', f'{PAPER_ID}_{gid}_ex_hh',
                    gid, 'E_Sbehav_hh_subidx', 'B_treat', base_ctrls, mask,
                    f'{RCT_PATH}#subgroup', 'EL1', 'strata FE', 'HH chores subindex')
        explore_results.append(r)

        # Relatives support subindex
        rel_items = [v for v in ['E_Sdiscourage_college_comm', 'E_Sdiscourage_work_comm'] if v in df.columns]
        if len(rel_items) >= 1:
            eq = df[rel_items].mean(axis=1, skipna=True)
            cm, cs = eq[df['B_treat']==0].mean(), eq[df['B_treat']==0].std()
            df['E_Sbehav_rel_subidx'] = (eq - cm) / cs if cs > 0 else eq
        r = run_one('explore/outcome/behavior_subindex_relatives', f'{PAPER_ID}_{gid}_ex_rel',
                    gid, 'E_Sbehav_rel_subidx', 'B_treat', base_ctrls, mask,
                    f'{RCT_PATH}#subgroup', 'EL1', 'strata FE', 'Female relatives support subindex')
        explore_results.append(r)

explore_df = pd.DataFrame(explore_results)
print(f"Exploration specs: {len(explore_df)} ({explore_df['run_success'].sum()} OK)")

# ============================================================================
# WRITE OUTPUTS
# ============================================================================
spec_df = pd.DataFrame(results)
infer_df = pd.DataFrame(infer_results)

print(f"\n{'='*80}")
print(f"RESULTS: {len(spec_df)} estimate rows ({spec_df['run_success'].sum()} OK), "
      f"{len(infer_df)} inference rows ({infer_df['run_success'].sum() if len(infer_df) else 0} OK)")

cols_spec = ['paper_id','spec_run_id','spec_id','spec_tree_path','baseline_group_id',
    'outcome_var','treatment_var','coefficient','std_error','p_value',
    'ci_lower','ci_upper','n_obs','r_squared','coefficient_vector_json',
    'sample_desc','fixed_effects','controls_desc','cluster_var','run_success','run_error']
spec_df[[c for c in cols_spec if c in spec_df.columns]].to_csv(
    os.path.join(PKG, 'specification_results.csv'), index=False)

cols_infer = ['paper_id','inference_run_id','spec_run_id','spec_id','spec_tree_path',
    'baseline_group_id','outcome_var','treatment_var','coefficient','std_error','p_value',
    'ci_lower','ci_upper','n_obs','r_squared','coefficient_vector_json',
    'cluster_var','run_success','run_error']
if len(infer_df) > 0:
    infer_df[[c for c in cols_infer if c in infer_df.columns]].to_csv(
        os.path.join(PKG, 'inference_results.csv'), index=False)

# Exploration results
if len(explore_df) > 0:
    explore_df[[c for c in cols_spec if c in explore_df.columns]].to_csv(
        os.path.join(PKG, 'exploration_results.csv'), index=False)

# Print summary
for gid in ['G1','G2']:
    gdf = spec_df[spec_df['baseline_group_id']==gid]
    print(f"\n{gid}:")
    for _, r in gdf.iterrows():
        s = 'OK' if r['run_success'] else 'FAIL'
        c = f"{r['coefficient']:.4f}" if pd.notna(r['coefficient']) else 'NA'
        p = f"{r['p_value']:.4f}" if pd.notna(r['p_value']) else 'NA'
        n = f"{r['n_obs']:.0f}" if pd.notna(r['n_obs']) else 'NA'
        print(f"  [{s}] {r['spec_id']}: coef={c}, p={p}, N={n}")

print(f"\nWrote specification_results.csv and inference_results.csv to {PKG}")
print("Done!")
