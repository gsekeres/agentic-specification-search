"""
Specification Search: 158401-V1
Bold, Ghisolfi, Nsonzi & Svensson - Market Access and Quality Upgrading
Cluster-RCT, ANCOVA + season FE, village-clustered SE
"""
import pandas as pd, numpy as np, pyfixest as pf, json, sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent_output_utils import (make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash, software_block)

PAPER_ID = "158401-V1"
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE, "data/downloads/extracted/158401-V1/data/raw")
OUTPUT_DIR = os.path.join(BASE, "data/downloads/extracted/158401-V1")

with open(os.path.join(OUTPUT_DIR, "SPECIFICATION_SURFACE.json")) as f:
    surface_obj = json.load(f)
SURFACE_HASH = compute_surface_hash(surface_obj)
SW_BLOCK = software_block()
XR = {1:3575, 2:3601, 3:3626, 4:3758, 5:3670, 6:3685, 7:3681, 8:3685}

def _sr(df, cols):
    """Safe row total (min_count=1)."""
    ex = [c for c in cols if c in df.columns]
    return df[ex].sum(axis=1, min_count=1) if ex else pd.Series(np.nan, index=df.index)

def construct_data():
    """Build analysis dataset from panel_g1.dta raw data."""
    df = pd.read_stata(os.path.join(RAW_DIR, "panel_g1.dta"), convert_categoricals=False)
    ss = df['survey_season']  # int8 1-7
    for c in df.select_dtypes(include=[np.float32]).columns:
        df[c] = df[c].astype(np.float64)
    phone7 = (ss == 7) & (df.get('form_used', pd.Series('', index=df.index)) == 'Phone_Market_Survey_spring2020')
    mkt7 = (ss == 7) & (df.get('form_used', pd.Series('', index=df.index)) == 'Market_Survey_spring2020')

    # ACREAGE
    pc = sorted([c for c in df.columns if c.startswith('plot_size_') and c[10:].isdigit()])
    df['acreage'] = _sr(df, pc)
    df.loc[df['acreage'].isna() & (df.get('maize_tot_plotn', pd.Series(np.nan, index=df.index)) == 0), 'acreage'] = 0
    if 'acres_tot' in df.columns:
        df.loc[phone7, 'acreage'] = df.loc[phone7, 'acres_tot']

    # CHEMICALS
    df['tot_chem_costs'] = _sr(df, [c for c in df.columns if c.startswith('tot_chem_costs_') and c[-1].isdigit()])

    # SEEDS
    for n in range(1, 13):
        st = f'seeds_types_{n}'
        if st not in df.columns: continue
        for t in ['opv', 'hybrid', 'rechyb']:
            d = f'{t}_seeds_{n}_dummy'
            df[d] = np.nan
            m = df[st].notna() & (df[st] != '') & (df[st] != 'dk')
            df.loc[m, d] = df.loc[m, st].astype(str).str.contains(t, na=False).astype(float)
        cols3 = [f'{t}_seeds_{n}_dummy' for t in ['opv', 'hybrid', 'rechyb'] if f'{t}_seeds_{n}_dummy' in df.columns]
        if cols3:
            an = df[cols3].notna().any(axis=1)
            df[f'modern_seeds_{n}_dummy'] = np.nan
            df.loc[an, f'modern_seeds_{n}_dummy'] = df.loc[an, cols3].max(axis=1)

    msd = [c for c in df.columns if c.startswith('modern_seeds_') and c.endswith('_dummy')]
    df['modern_seeds_any'] = df[msd].max(axis=1) if msd else np.nan
    if 'seeds_type' in df.columns:
        df.loc[phone7 & df['seeds_type'].isin(['hybrid', 'rechyb', 'opv']), 'modern_seeds_any'] = 1
        df.loc[phone7 & (df['seeds_type'] == 'own'), 'modern_seeds_any'] = 0

    # Modern seed costs
    scc = sorted([c for c in df.columns if c.startswith('seeds_cost_') and c[11:].isdigit()])
    for c in scc:
        n = c.split('_')[-1]
        md = f'modern_seeds_{n}_dummy'
        if md in df.columns:
            df[f'modern_seeds_{n}_cost'] = df[c] * df[md]
    msc = [c for c in df.columns if c.startswith('modern_seeds_') and c.endswith('_cost')]
    df['modern_seeds_cost_tot'] = _sr(df, msc)
    if 'seeds_cost' in df.columns:
        df.loc[phone7 & (df['modern_seeds_any'] == 1), 'modern_seeds_cost_tot'] = df.loc[phone7 & (df['modern_seeds_any'] == 1), 'seeds_cost']
        df.loc[phone7 & (df['modern_seeds_any'] == 0), 'modern_seeds_cost_tot'] = 0

    df['seeds_cost_tot'] = np.nan
    df.loc[ss >= 2, 'seeds_cost_tot'] = _sr(df.loc[ss >= 2], scc)
    if 'seeds_cost' in df.columns:
        df.loc[phone7, 'seeds_cost_tot'] = df.loc[phone7, 'seeds_cost']

    # FERTILIZER
    fcc = sorted([c for c in df.columns if c.startswith('fert_cost_') and c[10:].isdigit()])
    df['fert_cost_tot'] = np.nan
    df.loc[ss >= 2, 'fert_cost_tot'] = _sr(df.loc[ss >= 2], fcc)
    if 'fert_cost' in df.columns:
        df.loc[phone7, 'fert_cost_tot'] = df.loc[phone7, 'fert_cost']

    # BOOSTER, MANURE
    for inp in ['booster', 'manure']:
        cc = sorted([c for c in df.columns if c.startswith(f'{inp}_cost_') and c.split('_')[-1].isdigit()])
        df[f'{inp}_cost_tot'] = np.nan
        df.loc[ss >= 2, f'{inp}_cost_tot'] = _sr(df.loc[ss >= 2], cc)
    if 'booster_cost' in df.columns:
        df.loc[phone7, 'booster_cost_tot'] = df.loc[phone7, 'booster_cost']
    df.loc[df['manure_cost_tot'].isna() & df['acreage'].notna() & (df['acreage'] != 0) & ss.isin([4, 6]), 'manure_cost_tot'] = 0
    df.loc[df['manure_cost_tot'].isna() & df['acreage'].notna() & (df['acreage'] != 0) & mkt7, 'manure_cost_tot'] = 0

    df['expenses_fert_seeds_ugx'] = _sr(df, ['fert_cost_tot', 'modern_seeds_cost_tot'])
    df['expenses_inputs_ugx'] = _sr(df, ['tot_chem_costs', 'booster_cost_tot', 'fert_cost_tot', 'seeds_cost_tot', 'manure_cost_tot'])

    # HARVEST
    hkc = sorted([c for c in df.columns if c.startswith('harvest_kg_') and c[11:].isdigit()])
    df['harvest_kg_tot'] = _sr(df, hkc)
    if 'harvest_kg' in df.columns:
        df.loc[phone7, 'harvest_kg_tot'] = df.loc[phone7, 'harvest_kg']
    df['yield'] = df['harvest_kg_tot'] / df['acreage']

    # TARPAULIN
    df['tarpaulin_d'] = np.nan
    if 'dry_how' in df.columns:
        mt = (ss >= 2) & df['dry_how'].notna() & (df['dry_how'] != '')
        df.loc[mt, 'tarpaulin_d'] = df.loc[mt, 'dry_how'].astype(str).str.contains('tarpaulin', na=False).astype(float)

    # SORT, WINNOW
    for act in ['sort', 'winnow']:
        df[f'{act}_d'] = np.nan
        if act in df.columns:
            ma = (ss >= 5) & df[act].notna() & (df[act] != '')
            df.loc[ma, f'{act}_d'] = (df.loc[ma, act] == 'yes').astype(float)

    # LABOR COSTS
    for act in ['prep', 'plant', 'weed', 'spray', 'harv']:
        lcc = sorted([c for c in df.columns if c.startswith(f'lc_{act}_costs_season') and c.split('season')[-1].isdigit()])
        df[f'labor_cost_{act}_tot'] = np.nan
        if lcc:
            df.loc[ss != 1, f'labor_cost_{act}_tot'] = _sr(df.loc[ss != 1], lcc)
        df.loc[df[f'labor_cost_{act}_tot'].isna() & df['acreage'].notna() & (df['acreage'] != 0) & (ss == 5), f'labor_cost_{act}_tot'] = 0

    df['expenses_labor_preharvest_ugx'] = np.nan
    m_pl = (ss != 1) & (ss != 4) & (~phone7)
    df.loc[m_pl, 'expenses_labor_preharvest_ugx'] = _sr(df.loc[m_pl],
        ['labor_cost_prep_tot', 'labor_cost_plant_tot', 'labor_cost_spray_tot', 'labor_cost_weed_tot'])

    # Post-harvest activity costs
    for act in ['decob', 'winnow', 'sort', 'bag']:
        for ct in [f'lc_{act}_labour_costs', f'lc_{act}_other_costs']:
            if ct in df.columns:
                df.loc[df[ct].isna() & (ss >= 5), ct] = 0
        lm = f'lc_{act}_lmtot_costs'
        lb, pr = f'lc_{act}_labmaize_costs', f'lc_{act}_price'
        if lb in df.columns and pr in df.columns:
            df[lm] = np.nan
            df.loc[ss >= 5, lm] = df.loc[ss >= 5, lb].fillna(0) * df.loc[ss >= 5, pr].fillna(0)
            df.loc[df[lm].isna() & (ss >= 5), lm] = 0
        parts = [c for c in [f'lc_{act}_labour_costs', f'lc_{act}_lmtot_costs', f'lc_{act}_other_costs'] if c in df.columns]
        df[f'{act}_cost_tot'] = np.nan
        if parts:
            df.loc[ss >= 5, f'{act}_cost_tot'] = _sr(df.loc[ss >= 5], parts)

    # Decobbing season 2
    dcc = sorted([c for c in df.columns if c.startswith('decobbing_cost_') and c.split('_')[-1].isdigit()])
    df['decobbing_cost_tot'] = np.nan
    if dcc:
        df.loc[ss == 2, 'decobbing_cost_tot'] = _sr(df.loc[ss == 2], dcc)
        df.loc[df['acreage'] == 0, 'decobbing_cost_tot'] = np.nan

    for c in ['storage_cost', 'drying_cost', 'otherph_cost', 'labor_cost_ph']:
        if c in df.columns:
            df.loc[df['acreage'] == 0, c] = np.nan
    if 'labor_cost_store' in df.columns and 'storage' in df.columns:
        df.loc[df['labor_cost_store'].isna() & df['storage'].notna() & (df['storage'] != '') & (ss >= 5), 'labor_cost_store'] = 0

    # Post-harvest labor
    df['expenses_labor_postharvest_ugx'] = np.nan
    ph5 = ['labor_cost_harv_tot', 'labor_cost_store', 'lc_decob_labour_costs', 'lc_sort_labour_costs', 'lc_winnow_labour_costs', 'lc_bag_labour_costs']
    m_s5 = (ss >= 5) & (~phone7)
    df.loc[m_s5, 'expenses_labor_postharvest_ugx'] = _sr(df.loc[m_s5], ph5)
    php = ['labor_cost_harv_tot', 'labor_cost_ph', 'labor_cost_dry', 'labor_cost_store']
    df.loc[ss.isin([2, 3]), 'expenses_labor_postharvest_ugx'] = _sr(df.loc[ss.isin([2, 3])], php)

    # Post-harvest expenses total
    df['expenses_postharvest_ugx'] = np.nan
    ph5t = ['labor_cost_harv_tot', 'decob_cost_tot', 'winnow_cost_tot', 'sort_cost_tot', 'bag_cost_tot', 'drying_cost', 'storage_cost', 'labor_cost_store']
    df.loc[m_s5, 'expenses_postharvest_ugx'] = _sr(df.loc[m_s5], ph5t)
    hcc = sorted([c for c in df.columns if c.startswith('harvest_cost_') and c.split('_')[-1].isdigit()])
    df['harvest_cost_tot'] = np.nan
    if hcc:
        df.loc[ss.isin([2, 3]), 'harvest_cost_tot'] = _sr(df.loc[ss.isin([2, 3])], hcc)
    php2 = ['labor_cost_harv_tot', 'harvest_cost_tot', 'storage_cost', 'drying_cost', 'otherph_cost', 'labor_cost_ph', 'labor_cost_dry', 'labor_cost_store', 'decobbing_cost_tot']
    df.loc[ss.isin([2, 3]), 'expenses_postharvest_ugx'] = _sr(df.loc[ss.isin([2, 3])], php2)

    # SELLING
    skc = sorted([c for c in df.columns if c.startswith('sold_kg_p_') and c.split('_')[-1].isdigit()])
    df['sold_kg_tot'] = _sr(df, skc)
    if 'sold_kg' in df.columns:
        df.loc[phone7, 'sold_kg_tot'] = df.loc[phone7, 'sold_kg']
    for hid in [10249047, 10328006]:
        df.loc[(df['hhh_id'] == hid) & (ss == 1), 'sold_kg_tot'] = 0
    for s in [2, 3]:
        df.loc[df['harvest_kg_tot'].notna() & df['sold_kg_tot'].isna() & (ss == s), 'sold_kg_tot'] = 0
    if 'sold_crop' in df.columns:
        for s in [5, 6]:
            df.loc[df['harvest_kg_tot'].notna() & df['sold_kg_tot'].isna() & (ss == s) & (df['sold_crop'] == 'no'), 'sold_kg_tot'] = 0
        df.loc[df['harvest_kg_tot'].notna() & df['sold_kg_tot'].isna() & mkt7 & (df['sold_crop'] == 'no'), 'sold_kg_tot'] = 0
    f18 = [10078042, 10181008, 10181009, 10231001, 10247021, 10326064, 10328005, 10328014, 10328034, 10350040]
    df.loc[df['hhh_id'].isin(f18) & (ss == 4), 'sold_kg_tot'] = 0
    df.loc[df['hhh_id'].isin([10219013, 10231002, 10354079]) & (ss == 5), 'sold_kg_tot'] = 0

    # Revenue
    rvc = [c for c in df.columns if c.startswith('tot_rev_combined')]
    df['rev_tot'] = _sr(df, rvc)
    if 'tot_rev_told_tot' in df.columns:
        df.loc[phone7, 'rev_tot'] = df.loc[phone7, 'tot_rev_told_tot']
    df.loc[df['harvest_kg_tot'].notna() & df['rev_tot'].isna() & (df['sold_kg_tot'] == 0), 'rev_tot'] = 0
    df['price_ugx'] = df['rev_tot'] / df['sold_kg_tot']
    df['price_vm'] = df.groupby(['ea_code', 'survey_season'])['price_ugx'].transform('mean')

    df['remainder'] = df['harvest_kg_tot'] - df['sold_kg_tot']
    df['harvest_value_ugx'] = np.where(df['remainder'] >= 0,
        df['rev_tot'] + df['remainder'] * df['price_vm'],
        df['harvest_kg_tot'] * df['price_ugx'])
    df.loc[df['harvest_kg_tot'].isna(), 'harvest_value_ugx'] = np.nan
    df.loc[(df['hhh_id'] == 10249026) & (ss == 3), 'harvest_value_ugx'] = (
        df.loc[(df['hhh_id'] == 10249026) & (ss == 3), 'harvest_kg_tot'] *
        df.loc[(df['hhh_id'] == 10249026) & (ss == 3), 'price_vm'])

    # TRANSPORT
    for tc in ['cost_transportpre', 'cost_transportpost', 'tot_transport_cost']:
        if tc not in df.columns:
            df[tc] = np.nan
    stc = sorted([c for c in df.columns if c.startswith('sale_transport_cost_') and c.split('_')[-1].isdigit()])
    df['sale_transport_cost_tot'] = np.nan
    if stc:
        df.loc[ss >= 3, 'sale_transport_cost_tot'] = _sr(df.loc[ss >= 3], stc)

    tlc = sorted([c for c in df.columns if c.startswith('tot_lab_costs_') and c[14:].isdigit()])
    df['tot_lab_costs_preh_aggh_tot'] = np.nan
    if tlc:
        df.loc[ss >= 4, 'tot_lab_costs_preh_aggh_tot'] = _sr(df.loc[ss >= 4], tlc)
    df.loc[ss.isin([2, 3]), 'tot_lab_costs_preh_aggh_tot'] = _sr(df.loc[ss.isin([2, 3])], ['expenses_labor_preharvest_ugx', 'labor_cost_harv_tot'])

    eqc = sorted([c for c in df.columns if c.startswith('equipment_cost_') and c.split('_')[-1].isdigit()])
    df['equipment_cost_tot'] = np.nan
    if eqc:
        df.loc[ss == 3, 'equipment_cost_tot'] = _sr(df.loc[ss == 3], eqc)

    # TOTAL EXPENSES
    def _se(cols, mask):
        return _sr(df.loc[mask], cols)

    df['expenses_ugx'] = np.nan
    s2 = ['seeds_cost_tot', 'fert_cost_tot', 'booster_cost_tot', 'manure_cost_tot', 'tot_chem_costs', 'expenses_labor_preharvest_ugx', 'labor_cost_harv_tot', 'harvest_cost_tot', 'decobbing_cost_tot', 'storage_cost', 'drying_cost', 'otherph_cost', 'labor_cost_ph', 'cost_transportpre', 'cost_transportpost']
    df.loc[ss == 2, 'expenses_ugx'] = _se(s2, ss == 2)
    s3 = ['seeds_cost_tot', 'fert_cost_tot', 'manure_cost_tot', 'booster_cost_tot', 'equipment_cost_tot', 'expenses_labor_preharvest_ugx', 'labor_cost_harv_tot', 'tot_chem_costs', 'harvest_cost_tot', 'storage_cost', 'drying_cost', 'otherph_cost', 'labor_cost_ph', 'labor_cost_dry', 'labor_cost_store', 'cost_transportpre', 'cost_transportpost', 'sale_transport_cost_tot']
    df.loc[ss == 3, 'expenses_ugx'] = _se(s3, ss == 3)
    s4 = ['fert_cost_tot', 'tot_chem_costs', 'seeds_cost_tot', 'booster_cost_tot', 'manure_cost_tot', 'tot_lab_costs_preh_aggh_tot', 'drying_cost', 'storage_cost', 'labor_cost_store', 'otherph_cost', 'labor_cost_ph', 'tot_transport_cost', 'sale_transport_cost_tot']
    df.loc[ss == 4, 'expenses_ugx'] = _se(s4, ss == 4)
    s56 = ['fert_cost_tot', 'seeds_cost_tot', 'booster_cost_tot', 'manure_cost_tot', 'tot_chem_costs', 'expenses_labor_preharvest_ugx', 'labor_cost_harv_tot', 'drying_cost', 'storage_cost', 'labor_cost_store', 'cost_transportpre', 'cost_transportpost', 'sale_transport_cost_tot', 'decob_cost_tot', 'sort_cost_tot', 'winnow_cost_tot', 'bag_cost_tot']
    for s in [5, 6]:
        df.loc[ss == s, 'expenses_ugx'] = _se(s56, ss == s)
    df.loc[mkt7, 'expenses_ugx'] = _se(s56, mkt7)
    df.loc[phone7, 'expenses_ugx'] = np.nan

    # OWN HOURS
    hw = 833
    for act in ['prep', 'plant', 'spray', 'weed', 'harv']:
        hrc = sorted([c for c in df.columns if c.startswith(f'lh_{act}_hours_season') and c.split('season')[-1].isdigit()])
        df[f'{act}_unpaid_lab_tothrs'] = np.nan
        if hrc:
            df.loc[ss >= 5, f'{act}_unpaid_lab_tothrs'] = _sr(df.loc[ss >= 5], hrc)
    df.loc[df['spray_unpaid_lab_tothrs'].isna() & df['acreage'].notna() & (df['acreage'] != 0) & (ss == 5), 'spray_unpaid_lab_tothrs'] = 0
    ohc = [f'{a}_unpaid_lab_tothrs' for a in ['prep', 'plant', 'weed', 'spray', 'harv']]
    df['tot_lab_hours_preh_aggh_tot'] = np.nan
    df.loc[ss >= 5, 'tot_lab_hours_preh_aggh_tot'] = _sr(df.loc[ss >= 5], ohc)
    df['preh_unpaid_lab_aggh_sh'] = df['tot_lab_hours_preh_aggh_tot'] * hw
    for act in ['decob', 'sort', 'winnow', 'bag']:
        if f'lh_{act}_hours' in df.columns:
            df[f'lh_{act}_hours_sh'] = np.nan
            df.loc[ss >= 5, f'lh_{act}_hours_sh'] = df.loc[ss >= 5, f'lh_{act}_hours'] * hw

    df['expenses_hours_ugx'] = np.nan
    ehc = ['fert_cost_tot', 'tot_chem_costs', 'seeds_cost_tot', 'booster_cost_tot', 'manure_cost_tot', 'expenses_labor_preharvest_ugx', 'labor_cost_harv_tot', 'preh_unpaid_lab_aggh_sh', 'drying_cost', 'storage_cost', 'labor_cost_store', 'cost_transportpre', 'cost_transportpost', 'sale_transport_cost_tot', 'lh_decob_hours_sh', 'lh_sort_hours_sh', 'lh_winnow_hours_sh', 'lh_bag_hours_sh', 'decob_cost_tot', 'sort_cost_tot', 'winnow_cost_tot', 'bag_cost_tot']
    for s in [5, 6]:
        df.loc[ss == s, 'expenses_hours_ugx'] = _se(ehc, ss == s)
    df.loc[mkt7, 'expenses_hours_ugx'] = _se(ehc, mkt7)
    for s in [2, 3]:
        df.loc[ss == s, 'expenses_hours_ugx'] = df.loc[ss == s, 'expenses_ugx']

    df['surplus_ugx'] = df['harvest_value_ugx'] - df['expenses_ugx']
    df['surplus_hrs_ugx'] = df['harvest_value_ugx'] - df['expenses_hours_ugx']

    # UGX -> USD
    for v in ['price', 'harvest_value', 'expenses', 'surplus', 'surplus_hrs', 'expenses_fert_seeds', 'expenses_inputs', 'expenses_labor_preharvest', 'expenses_postharvest', 'expenses_labor_postharvest']:
        u = f'{v}_ugx'
        if u in df.columns:
            df[v] = np.nan
            for s, r in XR.items():
                df.loc[ss == s, v] = df.loc[ss == s, u] / r

    # ANCOVA controls
    all_ov = ['expenses_fert_seeds', 'expenses_inputs', 'tarpaulin_d', 'expenses_labor_preharvest', 'expenses_postharvest', 'expenses_labor_postharvest', 'price', 'acreage', 'harvest_kg_tot', 'yield', 'harvest_value', 'expenses', 'surplus', 'surplus_hrs']
    for v in all_ov:
        if v not in df.columns:
            continue
        p3m = df.loc[df['season_spring2018'] == 1].groupby('hhh_id')[v].mean()
        pm = df.loc[df['season_ante'] == 1].groupby('hhh_id')[v].mean()
        df[f'{v}_p3'] = df['hhh_id'].map(p3m)
        df.loc[df[f'{v}_p3'].isna(), f'{v}_p3'] = df.loc[df[f'{v}_p3'].isna(), 'hhh_id'].map(pm)

    # Forward-fill HH chars
    if 'main_road_min' in df.columns:
        df['main_road_min'] = pd.to_numeric(df['main_road_min'], errors='coerce')
    for c in ['mdm_female', 'mdm_primary', 'hhr_n', 'distance_kakumiro', 'main_road_min']:
        if c in df.columns:
            df[c] = df.groupby('hhh_id')[c].transform(lambda x: x.ffill().bfill())

    return df

# ── BUILD DATA ──
print("Building data...")
df_all = construct_data()
df_post = df_all[df_all['season_post'] == 1].copy()
print(f"Post-treatment: {len(df_post)} obs, {df_post['hhh_id'].nunique()} HHs, {df_post['ea_code'].nunique()} villages")

# Quick check
for v in ['expenses_fert_seeds', 'tarpaulin_d', 'sort_d', 'surplus', 'harvest_value', 'yield']:
    n = df_post[v].notna().sum() if v in df_post.columns else 0
    mn = df_post[v].mean() if v in df_post.columns and n > 0 else np.nan
    print(f"  {v}: N={n}, mean={mn:.3f}")

# ── SPEC RUNNER ──
da_g1 = surface_obj["baseline_groups"][0]["design_audit"]
da_g2 = surface_obj["baseline_groups"][1]["design_audit"]
inf_can = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
hh_chars = surface_obj["available_covariates"]["household_characteristics"]
NO_ANCOVA = ['sort_d', 'winnow_d']

results, inference_results = [], []
spc, irc = [0], [0]

def get_anc(ov):
    return [] if ov in NO_ANCOVA else [f'{ov}_p3']

def run_spec(sid, stp, gid, ov, tv, ctrls, fe, data, sd, cd, fe_str="survey_season", abn=None, ab=None):
    spc[0] += 1
    rid = f"{PAPER_ID}_run_{spc[0]:03d}"
    da = da_g1 if gid == "G1" else da_g2
    try:
        rhs = " + ".join([tv] + ctrls) if ctrls else tv
        fml = f"{ov} ~ {rhs} | {fe}" if fe else f"{ov} ~ {rhs}"
        avs = list(set([v for v in [ov, tv] + ctrls if v in data.columns]))
        dc = data.dropna(subset=avs).copy()
        if len(dc) == 0:
            raise ValueError(f"No obs for {ov}")
        m = pf.feols(fml, data=dc, vcov={"CRV1": "ea_code"})
        co = float(m.coef().get(tv, np.nan))
        se = float(m.se().get(tv, np.nan))
        pv = float(m.pvalue().get(tv, np.nan))
        try:
            ci = m.confint(); cil = float(ci.loc[tv, ci.columns[0]]); ciu = float(ci.loc[tv, ci.columns[1]])
        except Exception:
            cil = ciu = np.nan
        ac = {k: float(v) for k, v in m.coef().items()}
        pay = make_success_payload(coefficients=ac, inference={"spec_id": inf_can["spec_id"], "params": inf_can["params"]},
            software=SW_BLOCK, surface_hash=SURFACE_HASH, design={"randomized_experiment": da},
            axis_block_name=abn, axis_block=ab)
        results.append(dict(paper_id=PAPER_ID, spec_run_id=rid, spec_id=sid, spec_tree_path=stp, baseline_group_id=gid,
            outcome_var=ov, treatment_var=tv, coefficient=co, std_error=se, p_value=pv, ci_lower=cil, ci_upper=ciu,
            n_obs=int(m._N), r_squared=float(m._r2), coefficient_vector_json=json.dumps(pay),
            sample_desc=sd, fixed_effects=fe_str, controls_desc=cd, cluster_var="ea_code", run_success=1, run_error=""))
        return rid
    except Exception as e:
        em = str(e)[:240]
        pay = make_failure_payload(error=em, error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        results.append(dict(paper_id=PAPER_ID, spec_run_id=rid, spec_id=sid, spec_tree_path=stp, baseline_group_id=gid,
            outcome_var=ov, treatment_var=tv, coefficient=np.nan, std_error=np.nan, p_value=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, n_obs=np.nan, r_squared=np.nan,
            coefficient_vector_json=json.dumps(pay), sample_desc=sd, fixed_effects=fe_str, controls_desc=cd,
            cluster_var="ea_code", run_success=0, run_error=em))
        return rid

# ── BASELINES ──
print("\n=== BASELINES ===")
bl_rids = {}
for gi, gid in [(0, "G1"), (1, "G2")]:
    bspecs = surface_obj["baseline_groups"][gi]["baseline_specs"]
    bids = surface_obj["baseline_groups"][gi]["core_universe"]["baseline_spec_ids"]
    for bs, bid in zip(bspecs, bids):
        rid = run_spec(bid, "specification_tree/methods/randomized_experiment.md#ols-ancova", gid,
            bs["outcome_var"], "buy_treatment", bs["controls"], "survey_season", df_post,
            "group==1 & season_post==1", f"ANCOVA: {', '.join(bs['controls'])}" if bs['controls'] else "No ANCOVA")
        bl_rids[bid] = rid
        r = results[-1]
        st = f"coef={r['coefficient']:.4f}, p={r['p_value']:.3f}, N={r['n_obs']}" if r['run_success'] else f"FAIL: {r['run_error']}"
        print(f"  {bid}: {st}")

# ── DESIGN & RC ──
print("\n=== DESIGN & RC ===")
G1_FOCAL = ["expenses_fert_seeds", "tarpaulin_d", "expenses_postharvest"]
G2_FOCAL = ["surplus", "harvest_value", "yield"]
G1_MON = ["expenses_fert_seeds", "expenses_postharvest"]
G2_MON = ["surplus", "harvest_value"]

for gid, focal, mon in [("G1", G1_FOCAL, G1_MON), ("G2", G2_FOCAL, G2_MON)]:
    print(f"\n--- {gid} ---")
    for ov in focal:
        anc = get_anc(ov)
        # diff-in-means
        run_spec("design/randomized_experiment/estimator/diff_in_means",
            "specification_tree/methods/randomized_experiment.md#diff-in-means",
            gid, ov, "buy_treatment", [], "", df_post, "group==1 & season_post==1", "None", fe_str="None")
        print(f"  dim/{ov}: {results[-1]['coefficient']:.4f}")

        # with covariates
        ext = anc + hh_chars
        run_spec("design/randomized_experiment/estimator/with_covariates",
            "specification_tree/methods/randomized_experiment.md#with-covariates",
            gid, ov, "buy_treatment", ext, "survey_season", df_post,
            "group==1 & season_post==1", f"ANCOVA+HH: {','.join(ext)}")
        print(f"  wcov/{ov}: {results[-1]['coefficient']:.4f}")

        # controls/none
        run_spec("rc/controls/sets/none", "specification_tree/modules/robustness/controls.md#sets-none",
            gid, ov, "buy_treatment", [], "survey_season", df_post,
            "group==1 & season_post==1", "No controls",
            abn="controls", ab={"spec_id": "rc/controls/sets/none", "family": "sets", "set": "none", "n_controls": 0})
        print(f"  none/{ov}: {results[-1]['coefficient']:.4f}")

        # controls/extended
        ext2 = anc + hh_chars
        run_spec("rc/controls/sets/extended_hh_chars", "specification_tree/modules/robustness/controls.md#sets-extended",
            gid, ov, "buy_treatment", ext2, "survey_season", df_post,
            "group==1 & season_post==1", f"Ext: {','.join(ext2)}",
            abn="controls", ab={"spec_id": "rc/controls/sets/extended_hh_chars", "family": "sets", "included": ext2, "n_controls": len(ext2)})
        print(f"  ext/{ov}: {results[-1]['coefficient']:.4f}")

        # trim (monetary only)
        if ov in mon:
            td = df_post.copy()
            for s in td['survey_season'].unique():
                ms = td['survey_season'] == s
                vals = td.loc[ms, ov].dropna()
                if len(vals) < 10:
                    continue
                p99 = vals.quantile(0.99)
                td.loc[ms & (td[ov] >= p99), ov] = np.nan
                if ov in ['surplus', 'surplus_hrs']:
                    p1 = vals.quantile(0.01)
                    td.loc[ms & (td[ov] <= p1), ov] = np.nan
            run_spec("rc/sample/outliers/trim_y_1_99", "specification_tree/modules/robustness/sample.md#outliers-trim",
                gid, ov, "buy_treatment", anc, "survey_season", td,
                "trimmed 1/99 per season", f"ANCOVA: {','.join(anc)}" if anc else "No ANCOVA",
                abn="sample", ab={"spec_id": "rc/sample/outliers/trim_y_1_99", "family": "outliers", "upper_pctile": 99})
            print(f"  trim/{ov}: {results[-1]['coefficient']:.4f}")

        # drop first post season
        run_spec("rc/sample/time/drop_first_post_season", "specification_tree/modules/robustness/sample.md#time-drop-period",
            gid, ov, "buy_treatment", anc, "survey_season", df_post[df_post['survey_season'] != 4].copy(),
            "drop fall2018", f"ANCOVA: {','.join(anc)}" if anc else "No ANCOVA",
            abn="sample", ab={"spec_id": "rc/sample/time/drop_first_post_season", "dropped": [4]})
        print(f"  drop1/{ov}: {results[-1]['coefficient']:.4f}")

        # drop last post season
        run_spec("rc/sample/time/drop_last_post_season", "specification_tree/modules/robustness/sample.md#time-drop-period",
            gid, ov, "buy_treatment", anc, "survey_season", df_post[df_post['survey_season'] != 7].copy(),
            "drop spring2020/COVID", f"ANCOVA: {','.join(anc)}" if anc else "No ANCOVA",
            abn="sample", ab={"spec_id": "rc/sample/time/drop_last_post_season", "dropped": [7]})
        print(f"  dropL/{ov}: {results[-1]['coefficient']:.4f}")

        # balanced panel
        sc = df_post.dropna(subset=[ov]).groupby('hhh_id')['survey_season'].nunique()
        bh = sc[sc == sc.max()].index
        run_spec("rc/sample/panel/balanced_only", "specification_tree/modules/robustness/sample.md#panel-balanced",
            gid, ov, "buy_treatment", anc, "survey_season", df_post[df_post['hhh_id'].isin(bh)].copy(),
            f"balanced ({len(bh)} HH)", f"ANCOVA: {','.join(anc)}" if anc else "No ANCOVA",
            abn="sample", ab={"spec_id": "rc/sample/panel/balanced_only", "n_hh": int(len(bh))})
        print(f"  bal/{ov}: {results[-1]['coefficient']:.4f}")

        # asinh (monetary only)
        if ov in mon:
            fd = df_post.copy()
            fd[f'{ov}_asinh'] = np.arcsinh(fd[ov])
            anc_as = []
            if anc:
                fd[f'{anc[0]}_asinh'] = np.arcsinh(fd[anc[0]])
                anc_as = [f'{anc[0]}_asinh']
            run_spec("rc/form/outcome/asinh", "specification_tree/modules/robustness/functional_form.md#outcome-transform",
                gid, f'{ov}_asinh', "buy_treatment", anc_as, "survey_season", fd,
                "group==1 & season_post==1", f"ANCOVA(asinh): {','.join(anc_as)}" if anc_as else "No ANCOVA",
                abn="functional_form", ab={"spec_id": "rc/form/outcome/asinh", "transform": "asinh",
                    "original_outcome": ov, "interpretation": "Semi-elasticity"})
            print(f"  asinh/{ov}: {results[-1]['coefficient']:.4f}")

            # log1p
            fd2 = df_post.copy()
            if ov in ['surplus', 'surplus_hrs']:
                fd2.loc[fd2[ov] < 0, ov] = np.nan
            fd2[f'{ov}_log1p'] = np.log1p(np.maximum(fd2[ov].fillna(np.nan), 0))
            anc_lp = []
            if anc:
                av = fd2[anc[0]].copy()
                if ov in ['surplus', 'surplus_hrs']:
                    av[av < 0] = np.nan
                fd2[f'{anc[0]}_log1p'] = np.log1p(np.maximum(av.fillna(np.nan), 0))
                anc_lp = [f'{anc[0]}_log1p']
            run_spec("rc/form/outcome/log1p", "specification_tree/modules/robustness/functional_form.md#outcome-transform",
                gid, f'{ov}_log1p', "buy_treatment", anc_lp, "survey_season", fd2,
                "group==1 & season_post==1", f"ANCOVA(log1p): {','.join(anc_lp)}" if anc_lp else "No ANCOVA",
                abn="functional_form", ab={"spec_id": "rc/form/outcome/log1p", "transform": "log1p",
                    "original_outcome": ov, "interpretation": "Semi-elasticity"})
            print(f"  log1p/{ov}: {results[-1]['coefficient']:.4f}")

# ── INFERENCE ──
print("\n=== INFERENCE ===")
for r in results:
    if not r['spec_id'].startswith('baseline') or r['run_success'] == 0:
        continue
    ov = r['outcome_var']
    gid = r['baseline_group_id']
    anc = get_anc(ov)
    irc[0] += 1
    iid = f"{PAPER_ID}_infer_{irc[0]:03d}"
    try:
        rhs = " + ".join(["buy_treatment"] + anc) if anc else "buy_treatment"
        fml = f"{ov} ~ {rhs} | survey_season"
        avs = list(set([v for v in [ov, "buy_treatment"] + anc if v in df_post.columns]))
        dc = df_post.dropna(subset=avs).copy()
        m = pf.feols(fml, data=dc, vcov="hetero")
        co = float(m.coef().get("buy_treatment", np.nan))
        se = float(m.se().get("buy_treatment", np.nan))
        pv = float(m.pvalue().get("buy_treatment", np.nan))
        try:
            ci = m.confint(); cil = float(ci.loc["buy_treatment", ci.columns[0]]); ciu = float(ci.loc["buy_treatment", ci.columns[1]])
        except Exception:
            cil = ciu = np.nan
        ac = {k: float(v) for k, v in m.coef().items()}
        pay = make_success_payload(coefficients=ac, inference={"spec_id": "infer/se/hc/hc1", "params": {}},
            software=SW_BLOCK, surface_hash=SURFACE_HASH, design={"randomized_experiment": da_g1 if gid == "G1" else da_g2})
        inference_results.append(dict(paper_id=PAPER_ID, inference_run_id=iid, spec_run_id=r['spec_run_id'],
            spec_id="infer/se/hc/hc1", spec_tree_path="specification_tree/modules/inference/se.md#hc1",
            baseline_group_id=gid, coefficient=co, std_error=se, p_value=pv, ci_lower=cil, ci_upper=ciu,
            n_obs=int(m._N), r_squared=float(m._r2), coefficient_vector_json=json.dumps(pay),
            run_success=1, run_error=""))
        print(f"  HC1 {ov}({gid}): se={se:.4f}, p={pv:.3f}")
    except Exception as e:
        em = str(e)[:240]
        pay = make_failure_payload(error=em, error_details=error_details_from_exception(e, stage="hc1"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        inference_results.append(dict(paper_id=PAPER_ID, inference_run_id=iid, spec_run_id=r['spec_run_id'],
            spec_id="infer/se/hc/hc1", spec_tree_path="specification_tree/modules/inference/se.md#hc1",
            baseline_group_id=gid, coefficient=np.nan, std_error=np.nan, p_value=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, n_obs=np.nan, r_squared=np.nan,
            coefficient_vector_json=json.dumps(pay), run_success=0, run_error=em))
        print(f"  HC1 FAIL {ov}({gid}): {em}")

# Fisher permutation inference
print("\nRunning Fisher permutation (10,000 perms per baseline)...")
np.random.seed(760130)
village_treat = df_post.groupby('ea_code')['buy_treatment'].first()
villages = village_treat.index.tolist()
n_treated = int(village_treat.sum())

for r in results:
    if not r['spec_id'].startswith('baseline') or r['run_success'] == 0:
        continue
    ov = r['outcome_var']
    gid = r['baseline_group_id']
    anc = get_anc(ov)
    irc[0] += 1
    iid = f"{PAPER_ID}_infer_{irc[0]:03d}"
    try:
        rhs = " + ".join(["buy_treatment"] + anc) if anc else "buy_treatment"
        fml = f"{ov} ~ {rhs} | survey_season"
        avs = list(set([v for v in [ov, "buy_treatment"] + anc + ["survey_season", "ea_code"] if v in df_post.columns]))
        dc = df_post.dropna(subset=avs).copy()
        m_base = pf.feols(fml, data=dc, vcov={"CRV1": "ea_code"})
        actual_t = abs(float(m_base.tstat().get("buy_treatment", 0)))
        reject = 0
        for _ in range(10000):
            pt = np.zeros(len(villages))
            pt[np.random.choice(len(villages), n_treated, replace=False)] = 1
            pm = dict(zip(villages, pt))
            dc2 = dc.copy()
            dc2['buy_treatment'] = dc2['ea_code'].map(pm)
            try:
                mp = pf.feols(fml, data=dc2, vcov={"CRV1": "ea_code"})
                if abs(float(mp.tstat().get("buy_treatment", 0))) > actual_t:
                    reject += 1
            except Exception:
                pass
        fp = reject / 10000
        co = float(m_base.coef().get("buy_treatment", np.nan))
        se = float(m_base.se().get("buy_treatment", np.nan))
        try:
            ci = m_base.confint(); cil = float(ci.loc["buy_treatment", ci.columns[0]]); ciu = float(ci.loc["buy_treatment", ci.columns[1]])
        except Exception:
            cil = ciu = np.nan
        ac = {k: float(v) for k, v in m_base.coef().items()}
        pay = make_success_payload(coefficients=ac, inference={"spec_id": "infer/ri/fisher/permutation",
            "params": {"n_permutations": 10000, "seed": 760130, "fisher_p_value": fp, "actual_t_stat": actual_t}},
            software=SW_BLOCK, surface_hash=SURFACE_HASH, design={"randomized_experiment": da_g1 if gid == "G1" else da_g2})
        inference_results.append(dict(paper_id=PAPER_ID, inference_run_id=iid, spec_run_id=r['spec_run_id'],
            spec_id="infer/ri/fisher/permutation", spec_tree_path="specification_tree/modules/inference/ri.md#fisher-permutation",
            baseline_group_id=gid, coefficient=co, std_error=se, p_value=fp, ci_lower=cil, ci_upper=ciu,
            n_obs=int(m_base._N), r_squared=float(m_base._r2), coefficient_vector_json=json.dumps(pay),
            run_success=1, run_error=""))
        print(f"  Fisher {ov}({gid}): p={fp:.4f}")
    except Exception as e:
        em = str(e)[:240]
        pay = make_failure_payload(error=em, error_details=error_details_from_exception(e, stage="fisher"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        inference_results.append(dict(paper_id=PAPER_ID, inference_run_id=iid, spec_run_id=r['spec_run_id'],
            spec_id="infer/ri/fisher/permutation", spec_tree_path="specification_tree/modules/inference/ri.md#fisher-permutation",
            baseline_group_id=gid, coefficient=np.nan, std_error=np.nan, p_value=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, n_obs=np.nan, r_squared=np.nan,
            coefficient_vector_json=json.dumps(pay), run_success=0, run_error=em))
        print(f"  Fisher FAIL {ov}({gid}): {em}")

# ── WRITE OUTPUTS ──
print("\n=== WRITING ===")
rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(OUTPUT_DIR, "specification_results.csv"), index=False)
nt = len(rdf); ns = int(rdf['run_success'].sum()); nf = nt - ns
nb = int(rdf['spec_id'].str.startswith('baseline').sum())
nd = int(rdf['spec_id'].str.startswith('design/').sum())
nr = int(rdf['spec_id'].str.startswith('rc/').sum())
print(f"specification_results.csv: {nt} rows ({ns} OK, {nf} fail)")

idf = pd.DataFrame(inference_results)
idf.to_csv(os.path.join(OUTPUT_DIR, "inference_results.csv"), index=False)
ni = len(idf); nis = int(idf['run_success'].sum()) if len(idf) > 0 else 0
print(f"inference_results.csv: {ni} rows ({nis} OK)")

md = f"""# Specification Search: {PAPER_ID}

**Paper**: Bold, Ghisolfi, Nsonzi & Svensson - "Market Access and Quality Upgrading"
**Design**: Cluster-RCT (village-level), ITT with ANCOVA

## Surface Summary
- **Groups**: G1 (Investment, Table 5), G2 (Productivity/Income, Table 6)
- **Treatment**: buy_treatment (12 of 20 villages)
- **Clustering**: ea_code (20 villages)
- **Surface Hash**: {SURFACE_HASH}

## Execution Summary

| Category | Count | Success | Fail |
|----------|-------|---------|------|
| Baseline | {nb} | {int(rdf[rdf['spec_id'].str.startswith('baseline')]['run_success'].sum())} | {nb - int(rdf[rdf['spec_id'].str.startswith('baseline')]['run_success'].sum())} |
| Design | {nd} | {int(rdf[rdf['spec_id'].str.startswith('design/')]['run_success'].sum())} | {nd - int(rdf[rdf['spec_id'].str.startswith('design/')]['run_success'].sum())} |
| RC | {nr} | {int(rdf[rdf['spec_id'].str.startswith('rc/')]['run_success'].sum())} | {nr - int(rdf[rdf['spec_id'].str.startswith('rc/')]['run_success'].sum())} |
| **Total** | **{nt}** | **{ns}** | **{nf}** |

### Inference (separate table)
| Variant | Count | Success |
|---------|-------|---------|
| infer/se/hc/hc1 | {len([r for r in inference_results if r['spec_id']=='infer/se/hc/hc1'])} | {len([r for r in inference_results if r['spec_id']=='infer/se/hc/hc1' and r['run_success']==1])} |
| infer/ri/fisher/permutation | {len([r for r in inference_results if r['spec_id']=='infer/ri/fisher/permutation'])} | {len([r for r in inference_results if r['spec_id']=='infer/ri/fisher/permutation' and r['run_success']==1])} |
| **Total** | **{ni}** | **{nis}** |

## Specs Executed
- **Baseline**: 16 (8 G1 + 8 G2), ANCOVA + season FE + village-clustered SE
- **design/diff_in_means**: 6 (3 focal per group, no controls/FE)
- **design/with_covariates**: 6 (ANCOVA + 5 HH characteristics)
- **rc/controls/none**: 6 (drop ANCOVA)
- **rc/controls/extended**: 6 (ANCOVA + HH chars)
- **rc/sample/trim**: 4 (monetary focal only, per-season 1/99)
- **rc/sample/drop_first_season**: 6 (drop fall 2018)
- **rc/sample/drop_last_season**: 6 (drop spring 2020/COVID)
- **rc/sample/balanced**: 6 (balanced panel HHs)
- **rc/form/asinh**: 4 (monetary focal only)
- **rc/form/log1p**: 4 (monetary focal only)
- **infer/hc1**: 16 (all baselines, HC1 robust SE)
- **infer/fisher**: 16 (all baselines, 10K permutations, seed=760130)

## Deviations
- Data constructed from raw panel_g1.dta replicating Stata do-files in Python
- Fisher inference uses seed 760130, permuting village-level treatment (12/20)

## Software
- Python {sys.version.split()[0]}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

with open(os.path.join(OUTPUT_DIR, "SPECIFICATION_SEARCH.md"), "w") as f:
    f.write(md)
print("SPECIFICATION_SEARCH.md written")
print(f"\nDONE: {nt} specs + {ni} inference rows")
for r in results:
    if r['spec_id'].startswith('baseline') and r['run_success'] == 1:
        print(f"  {r['spec_id']}: b={r['coefficient']:.4f}, se={r['std_error']:.4f}, p={r['p_value']:.3f}, N={r['n_obs']}")
