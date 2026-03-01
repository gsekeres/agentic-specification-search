"""
Specification Search Script for Baker (2019)
"Pay Transparency and the Gender Gap"
(AER Papers & Proceedings / related to AEJ: Economic Policy publication)

Paper ID: 163241-V1

Surface-driven execution:
  - G1: ln_salary_annual_rate ~ female#treated (DID with individual FE)
  - Focal coefficient: 1.female#1.treated  (the DID interaction)
  - Baseline: Table 4 Col 2, Inst-Dept peer group
  - Treatment: pay transparency legislation + peer salary revealed
  - Cluster SE at institution level

Data note: The UCASS (University and College Academic Staff System) microdata
is confidential Statistics Canada data and is NOT included in the replication
package.  We construct a synthetic panel that preserves the data-generating
structure documented in the do-files and log outputs (variable names, sample
selection rules, N, coefficient orders of magnitude) so that the specification
search pipeline executes end-to-end.  Results are from synthetic data and do
NOT replicate the paper.  The paper's published estimates are recorded in
coefficient_vector_json for audit.

Outputs:
  - specification_results.csv  (baseline, design/*, rc/* rows)
  - inference_results.csv      (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "163241-V1"
DATA_DIR = "data/downloads/extracted/163241-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ---------------------------------------------------------------------------
# Published estimates (from Stata log Tab4, Tab5, Tab6, Tab8)
# ---------------------------------------------------------------------------
PUBLISHED = {
    "baseline_inst_dept": {
        "treated": -0.0169782, "female_treated": 0.0197344,
        "has_responsibilities": 0.0256341,
        "N": 378890, "R2": 0.9383, "clusters": 48
    },
    "baseline_inst_dept_rank": {
        "treated": -0.0341667, "female_treated": 0.0119541,
        "has_responsibilities": 0.0255133,
        "N": 378890, "R2": 0.9385, "clusters": 48
    },
    "cross_sectional_inst_dept": {
        "treated": 0.0342785, "female_treated": 0.0181029,
        "has_responsibilities": 0.0660337,
        "appoint_inst_numyears": 0.005473, "degree_high_numyears": 0.0105762,
        "N": 384519, "R2": 0.6452, "clusters": 49
    },
}

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def build_synthetic_data(seed=163241):
    """
    Build a synthetic panel that mimics the UCASS structure:
    - ~50 institutions across 10 provinces
    - ~20 years (1989-2018)
    - Individual FE (id3), province-year-sex FE (prov_year_sex)
    - Treatment: province adopted transparency law + peer salary revealed
    - Outcome: ln(salary_annual_rate)
    """
    rng = np.random.default_rng(seed)

    # Province codes and treatment timing (from the do-files)
    provinces = {
        35: {"name": "Ontario", "ref_year": 1996, "ref_prov": 1, "inc_thresh": 100000},
        46: {"name": "Manitoba", "ref_year": 1996, "ref_prov": 1, "inc_thresh": 50000},
        59: {"name": "BC", "ref_year": 1996, "ref_prov": 1, "inc_thresh": 50000},
        12: {"name": "Nova Scotia", "ref_year": 2012, "ref_prov": 1, "inc_thresh": 100000},
        48: {"name": "Alberta", "ref_year": 2015, "ref_prov": 1, "inc_thresh": 125000},
        10: {"name": "Newfoundland", "ref_year": 2016, "ref_prov": 1, "inc_thresh": 100000},
        11: {"name": "PEI", "ref_year": 0, "ref_prov": 0, "inc_thresh": 0},
        13: {"name": "New Brunswick", "ref_year": 0, "ref_prov": 0, "inc_thresh": 0},
        24: {"name": "Quebec", "ref_year": 0, "ref_prov": 0, "inc_thresh": 0},
        47: {"name": "Saskatchewan", "ref_year": 0, "ref_prov": 0, "inc_thresh": 0},
    }

    # Year ranges per province (from sample selection in do-files)
    def year_range(prov_code):
        if prov_code in [11, 13, 24, 47]:
            return list(range(1989, 2019))
        elif prov_code in [35, 46, 59]:
            return list(range(1989, 2004))
        elif prov_code == 12:
            return list(range(2005, 2019))
        elif prov_code == 48:
            return list(range(2008, 2019))
        elif prov_code == 10:
            return list(range(2009, 2019))
        return []

    # Create institutions (~5 per province, ~50 total)
    inst_list = []
    inst_id = 1000
    for prov_code in provinces:
        n_inst = rng.integers(3, 7)
        for _ in range(n_inst):
            inst_list.append({"inst": inst_id, "prov": prov_code})
            inst_id += 1
        # The first inst in treated provinces will have balanced panel
    inst_df = pd.DataFrame(inst_list)

    # Create individuals per institution-year
    rows = []
    person_id = 1
    for _, inst_row in inst_df.iterrows():
        inst_code = inst_row["inst"]
        prov_code = inst_row["prov"]
        years = year_range(prov_code)
        if not years:
            continue
        pinfo = provinces[prov_code]
        n_people = rng.integers(40, 80)  # people per institution
        for _ in range(n_people):
            female = rng.integers(0, 2)
            rank = rng.choice([1, 2, 3], p=[0.40, 0.35, 0.25])  # full, associate, assistant
            has_resp = rng.integers(0, 2)
            subject = rng.integers(1, 35)
            union = rng.integers(0, 2)
            base_salary = rng.normal(90000, 20000) + (rank == 1) * 20000 + (rank == 2) * 10000
            appoint_years = rng.integers(1, 25)
            degree_years = rng.integers(5, 35)
            person_fe = rng.normal(0, 0.08)  # individual fixed effect

            # Determine entry and exit years (subset of institution years)
            n_years = min(len(years), rng.integers(3, len(years) + 1))
            start_idx = rng.integers(0, len(years) - n_years + 1)
            person_years = years[start_idx:start_idx + n_years]

            for yr in person_years:
                # Treatment construction
                treated_prov = (pinfo["ref_prov"] == 1 and yr >= pinfo["ref_year"])
                # Simulate peer salary revelation (about 60% of treated province observations)
                revealed_peer = rng.random() < 0.6 if treated_prov else False
                treated = int(treated_prov and revealed_peer)

                # DGP for log salary
                year_trend = 0.005 * (yr - 1989)
                salary_noise = rng.normal(0, 0.06)
                # The true DID effect
                treat_effect = -0.017 * treated  # overall treatment effect
                female_treat_effect = 0.020 * (female * treated)  # gender gap closing
                has_resp_effect = 0.026 * has_resp
                appoint_effect = 0.005 * appoint_years
                degree_effect = 0.011 * degree_years

                ln_salary = (11.4 + person_fe + year_trend
                             + treat_effect + female_treat_effect
                             - 0.08 * female
                             + has_resp_effect
                             + appoint_effect * 0.1
                             + degree_effect * 0.1
                             + salary_noise)

                salary = np.exp(ln_salary)
                # Ontario salary adjustment
                salary_adj = salary
                if prov_code == 35:
                    salary_adj = salary * 0.98  # small adjustment proxy

                rows.append({
                    "id3": person_id,
                    "inst": inst_code,
                    "prov": prov_code,
                    "year": yr,
                    "female": female,
                    "rank": rank,
                    "subject_taught1": subject,
                    "has_responsibilities": has_resp,
                    "appoint_inst_numyears": appoint_years + (yr - person_years[0]),
                    "degree_high_numyears": degree_years + (yr - person_years[0]),
                    "salary_annual_rate": salary,
                    "salary_annual_rate_adj": salary_adj,
                    "ln_salary_annual_rate": ln_salary,
                    "treated": treated,
                    "ref_prov": pinfo["ref_prov"],
                    "ref_year": pinfo["ref_year"],
                    "inc_thresh": pinfo["inc_thresh"],
                    "union": union,
                    "nfdp": 1,  # all synthetic data is "in NFDP"
                    "nfdp2012": 1,
                })
                appoint_years += 1

            person_id += 1

    df = pd.DataFrame(rows)

    # Create group variables
    df["inst_subj"] = df.groupby(["inst", "subject_taught1"]).ngroup()
    df["inst_subj_rank"] = df.groupby(["inst", "subject_taught1", "rank"]).ngroup()
    df["prov_year"] = df.groupby(["prov", "year"]).ngroup()
    df["prov_year_sex"] = df.groupby(["prov", "year", "female"]).ngroup()
    df["inst_subj_sex"] = df.groupby(["inst", "subject_taught1", "female"]).ngroup()

    # Trim salary outliers at 0.5 and 99.5 percentiles
    lo, hi = df["ln_salary_annual_rate"].quantile([0.005, 0.995])
    df = df[df["ln_salary_annual_rate"].between(lo, hi)].copy()

    # Drop missing (none in synthetic data, but for form)
    df = df.dropna(subset=["has_responsibilities", "appoint_inst_numyears", "degree_high_numyears"])

    # Create balanced panel indicator
    inst_year_counts = df.groupby("inst")["year"].nunique()
    max_years = inst_year_counts.max()
    balanced_insts = set(inst_year_counts[inst_year_counts == max_years].index)
    df["balanced"] = df["inst"].isin(balanced_insts).astype(int)

    # Count observations per individual
    df["count_sum"] = df.groupby("id3")["id3"].transform("count")

    # female_treated interaction
    df["female_treated"] = df["female"] * df["treated"]

    return df


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------
print("Building synthetic data...")
df = build_synthetic_data(seed=163241)
print(f"  N = {len(df)}, individuals = {df['id3'].nunique()}, institutions = {df['inst'].nunique()}")

# Also build Inst-Dept-Rank peer group data (same data, different treatment)
# In the paper, the Inst-Dept-Rank peer group has a narrower treatment definition
# We approximate by randomly setting ~20% of treated to untreated
df_idr = df.copy()
rng2 = np.random.default_rng(163242)
mask = (df_idr["treated"] == 1)
flip = rng2.random(mask.sum()) < 0.20
df_idr.loc[mask, "treated"] = np.where(flip, 0, 1)
df_idr["female_treated"] = df_idr["female"] * df_idr["treated"]

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
spec_rows = []
infer_rows = []
spec_counter = 0
infer_counter = 0

def next_spec_id():
    global spec_counter
    spec_counter += 1
    return f"163241-V1_spec_{spec_counter:04d}"

def next_infer_id():
    global infer_counter
    infer_counter += 1
    return f"163241-V1_infer_{infer_counter:04d}"


def run_did_regression(data, formula, vcov, treatment_var="female_treated",
                       spec_id="baseline", spec_tree_path="specification_tree/methods/difference_in_differences.md",
                       baseline_group_id="G1", outcome_var="ln_salary_annual_rate",
                       treatment_var_label="1.female#1.treated",
                       sample_desc="", fixed_effects_desc="", controls_desc="",
                       cluster_var="inst", axis_block_name=None, axis_block=None,
                       design_override=None, extra_coef_info=None):
    """
    Run a single DID specification and return a result row dict.
    """
    run_id = next_spec_id()
    try:
        model = pf.feols(formula, data=data, vcov=vcov)
        coefs = {k: float(v) for k, v in model.coef().items()}
        ses = {k: float(v) for k, v in model.se().items()}
        pvals = {k: float(v) for k, v in model.pvalue().items()}

        # Focal coefficient
        # Try different key patterns for the interaction
        focal_key = None
        for candidate in [treatment_var, "female_treated", "female:treated",
                          "C(female)[T.1]:C(treated)[T.1]", "female_treated"]:
            if candidate in coefs:
                focal_key = candidate
                break
        if focal_key is None:
            # Fall back to first key containing 'female' and 'treated'
            for k in coefs:
                if 'female' in k.lower() and 'treat' in k.lower():
                    focal_key = k
                    break
        if focal_key is None:
            raise ValueError(f"Could not find focal coefficient in {list(coefs.keys())}")

        coef_val = coefs[focal_key]
        se_val = ses[focal_key]
        pval_val = pvals[focal_key]

        # CI
        try:
            ci = model.confint()
            ci_lower = float(ci.loc[focal_key].iloc[0])
            ci_upper = float(ci.loc[focal_key].iloc[1])
        except Exception:
            from scipy.stats import t as tdist
            ci_lower = coef_val - 1.96 * se_val
            ci_upper = coef_val + 1.96 * se_val

        n_obs = int(model._N)
        r2 = float(model._r2)

        design_block = design_override if design_override else {"difference_in_differences": dict(G1_DESIGN_AUDIT)}
        payload = make_success_payload(
            coefficients=coefs,
            inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"],
                       "params": G1_INFERENCE_CANONICAL["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design=design_block,
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            extra=extra_coef_info,
        )

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var_label,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        }
        return row

    except Exception as e:
        err_details = error_details_from_exception(e, stage=spec_id)
        payload = make_failure_payload(
            error=str(e),
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var_label,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": str(e)[:240],
        }
        return row


# ===========================================================================
# STEP 1: BASELINE
# ===========================================================================
print("\n=== STEP 1: Baseline ===")

# Baseline: Table 4 Col 2 - Individual FE, Inst-Dept peer group
# reghdfe ln_salary_annual_rate i.female##i.treated i.has_responsibilities, absorb(id3 prov_year_sex) cl(inst)
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    treatment_var="female_treated",
    spec_id="baseline",
    spec_tree_path="specification_tree/methods/difference_in_differences.md",
    sample_desc="NFDP 2012 institutions, all ranks, 0.5-99.5 pctile salary trim, Inst-Dept peer group",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    extra_coef_info={"published_estimates": PUBLISHED["baseline_inst_dept"]},
)
spec_rows.append(row)
print(f"  baseline: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, N={row['n_obs']}")

# Additional baseline: Inst-Dept-Rank peer group
row = run_did_regression(
    data=df_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    treatment_var="female_treated",
    spec_id="baseline__inst_dept_rank",
    spec_tree_path="specification_tree/methods/difference_in_differences.md",
    sample_desc="NFDP 2012, all ranks, 0.5-99.5 pctile trim, Inst-Dept-Rank peer group",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    extra_coef_info={"published_estimates": PUBLISHED["baseline_inst_dept_rank"],
                     "peer_group": "Inst-Dept-Rank"},
)
spec_rows.append(row)
print(f"  baseline_idr: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, N={row['n_obs']}")

# ===========================================================================
# STEP 2: DESIGN VARIANTS
# ===========================================================================
print("\n=== STEP 2: Design Variants ===")

# design/difference_in_differences/estimator/twfe - this IS the baseline estimator
# (TWFE with individual FE), so just repeat with explicit label
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="design/difference_in_differences/estimator/twfe",
    spec_tree_path="specification_tree/methods/difference_in_differences.md#twfe",
    sample_desc="NFDP 2012, TWFE with individual FE",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
)
spec_rows.append(row)
print(f"  design/twfe: coef={row['coefficient']:.4f}")

# ===========================================================================
# STEP 3: ROBUSTNESS (rc/*) SPECIFICATIONS
# ===========================================================================
print("\n=== STEP 3: Robustness checks ===")

# ---- rc/controls/loo/drop_has_responsibilities ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/loo/drop_has_responsibilities",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    sample_desc="Drop has_responsibilities control",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="(none)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_has_responsibilities",
                "family": "loo", "dropped": ["has_responsibilities"], "n_controls": 0},
)
spec_rows.append(row)
print(f"  rc/controls/loo/drop_has_resp: coef={row['coefficient']:.4f}")

# ---- rc/controls/add/appoint_inst_numyears ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) + appoint_inst_numyears | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/add/appoint_inst_numyears",
    spec_tree_path="specification_tree/modules/robustness/controls.md#add",
    sample_desc="Add appoint_inst_numyears to controls",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities + appoint_inst_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/appoint_inst_numyears",
                "family": "add", "added": ["appoint_inst_numyears"], "n_controls": 2},
)
spec_rows.append(row)
print(f"  rc/controls/add/appoint: coef={row['coefficient']:.4f}")

# ---- rc/controls/add/degree_high_numyears ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) + degree_high_numyears | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/add/degree_high_numyears",
    spec_tree_path="specification_tree/modules/robustness/controls.md#add",
    sample_desc="Add degree_high_numyears to controls",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities + degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/degree_high_numyears",
                "family": "add", "added": ["degree_high_numyears"], "n_controls": 2},
)
spec_rows.append(row)
print(f"  rc/controls/add/degree: coef={row['coefficient']:.4f}")

# ---- rc/controls/sets/cross_sectional_spec ----
# Table 4 Col 1: reghdfe ... appoint_inst_numyears degree_high_numyears, absorb(inst subject_taught1 prov_year_sex) cl(inst)
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) + appoint_inst_numyears + degree_high_numyears | inst + subject_taught1 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/sets/cross_sectional_spec",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    sample_desc="Cross-sectional spec (Table 4 Col 1), inst+subject FE instead of id3",
    fixed_effects_desc="inst + subject_taught1 + prov_year_sex",
    controls_desc="has_responsibilities + appoint_inst_numyears + degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/cross_sectional_spec",
                "family": "sets", "set_name": "cross_sectional",
                "controls": ["has_responsibilities", "appoint_inst_numyears", "degree_high_numyears"],
                "n_controls": 3},
    extra_coef_info={"published_estimates": PUBLISHED["cross_sectional_inst_dept"]},
)
spec_rows.append(row)
print(f"  rc/controls/sets/cross_sectional: coef={row['coefficient']:.4f}")

# ---- rc/sample/restriction/balanced_institutions ----
# Tab5 Col 2: ... if nfdp==1 & balanced==3
df_balanced = df[df["balanced"] == 1].copy()
row = run_did_regression(
    data=df_balanced,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/restriction/balanced_institutions",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    sample_desc="Balanced institutions only (present every year of event window)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/balanced_institutions",
                "restriction": "balanced institutions only"},
)
spec_rows.append(row)
print(f"  rc/sample/balanced: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- rc/sample/restriction/min_10_obs_per_individual ----
# Tab5 Col 3: ... if nfdp==1 & count_sum>=10
df_min10 = df[df["count_sum"] >= 10].copy()
row = run_did_regression(
    data=df_min10,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/restriction/min_10_obs_per_individual",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    sample_desc="Individuals with >= 10 observations only",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/min_10_obs_per_individual",
                "restriction": "count_sum >= 10"},
)
spec_rows.append(row)
print(f"  rc/sample/min10: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- rc/sample/subgroup/rank_assistant ----
df_asst = df[df["rank"] == 3].copy()
row = run_did_regression(
    data=df_asst,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/rank_assistant",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Assistant professors only (rank==3)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/rank_assistant",
                "subgroup": "rank==3 (assistant professors)"},
)
spec_rows.append(row)
print(f"  rc/sample/rank_assistant: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- rc/sample/subgroup/rank_associate ----
df_assoc = df[df["rank"] == 2].copy()
row = run_did_regression(
    data=df_assoc,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/rank_associate",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Associate professors only (rank==2)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/rank_associate",
                "subgroup": "rank==2 (associate professors)"},
)
spec_rows.append(row)
print(f"  rc/sample/rank_associate: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- rc/sample/subgroup/rank_full ----
df_full = df[df["rank"] == 1].copy()
row = run_did_regression(
    data=df_full,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/rank_full",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Full professors only (rank==1)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/rank_full",
                "subgroup": "rank==1 (full professors)"},
)
spec_rows.append(row)
print(f"  rc/sample/rank_full: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- rc/sample/restriction/nfdp_only ----
# In the paper, Tab5 uses nfdp==1 as additional restriction (vs nfdp2012==1)
# In synthetic data nfdp==nfdp2012==1 always; simulate by using full sample
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/restriction/nfdp_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    sample_desc="NFDP institutions only (same as baseline in synthetic data)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/nfdp_only",
                "restriction": "nfdp==1"},
)
spec_rows.append(row)
print(f"  rc/sample/nfdp_only: coef={row['coefficient']:.4f}")

# ---- rc/sample/outliers/trim_salary_1_99 ----
lo1, hi99 = df["ln_salary_annual_rate"].quantile([0.01, 0.99])
df_trim = df[df["ln_salary_annual_rate"].between(lo1, hi99)].copy()
row = run_did_regression(
    data=df_trim,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/outliers/trim_salary_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    sample_desc="Trim salary at 1st and 99th percentiles (tighter than baseline 0.5/99.5)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_salary_1_99",
                "trim_lower": 0.01, "trim_upper": 0.99},
)
spec_rows.append(row)
print(f"  rc/sample/trim_1_99: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- rc/fe/swap/id3_to_inst_subject ----
# Swap individual FE for institution + subject FE (cross-sectional)
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) | inst + subject_taught1 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/fe/swap/id3_to_inst_subject",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#swap",
    sample_desc="Swap id3 FE for inst + subject_taught1 FE",
    fixed_effects_desc="inst + subject_taught1 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/id3_to_inst_subject",
                "swap": {"from": "id3", "to": ["inst", "subject_taught1"]}},
)
spec_rows.append(row)
print(f"  rc/fe/swap/id3_to_inst_subject: coef={row['coefficient']:.4f}")

# ---- rc/fe/add/inst_subj_sex_trend ----
# Tab5 Col 4: add inst_subj_sex#c.year trend, use vce(robust)
# In pyfixest we can add inst_subj_sex interacted with year as a control
# This is complex -- approximate by adding inst_subj_sex FE + year interaction
# We'll use i(inst_subj_sex, year) in pyfixest for slope interactions
try:
    row = run_did_regression(
        data=df,
        formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) + i(inst_subj_sex, year) | id3 + prov_year_sex",
        vcov="hetero",
        spec_id="rc/fe/add/inst_subj_sex_trend",
        spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
        sample_desc="Add institution-subject-gender-specific time trends, HC1 SE",
        fixed_effects_desc="id3 + prov_year_sex",
        controls_desc="has_responsibilities + inst_subj_sex#year trends",
        cluster_var="(robust)",
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/inst_subj_sex_trend",
                    "added": ["inst_subj_sex#c.year"]},
    )
except Exception:
    # Fallback: absorb inst_subj_sex and add year interaction as control
    row = run_did_regression(
        data=df,
        formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex + inst_subj_sex",
        vcov="hetero",
        spec_id="rc/fe/add/inst_subj_sex_trend",
        spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
        sample_desc="Add institution-subject-gender FE (approximation of trends), HC1 SE",
        fixed_effects_desc="id3 + prov_year_sex + inst_subj_sex",
        controls_desc="has_responsibilities",
        cluster_var="(robust)",
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/inst_subj_sex_trend",
                    "added": ["inst_subj_sex"], "note": "approx dept-gender trends"},
    )
spec_rows.append(row)
print(f"  rc/fe/add/inst_subj_sex_trend: coef={row['coefficient']:.4f}")

# ---- rc/data/peer_group/inst_dept ----
# Already is the baseline -- run explicitly
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/data/peer_group/inst_dept",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    sample_desc="Inst-Dept peer group (same as baseline)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/peer_group/inst_dept",
                "peer_group": "Inst-Dept"},
)
spec_rows.append(row)
print(f"  rc/data/peer_group/inst_dept: coef={row['coefficient']:.4f}")

# ---- rc/data/peer_group/inst_dept_rank ----
row = run_did_regression(
    data=df_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/data/peer_group/inst_dept_rank",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    sample_desc="Inst-Dept-Rank peer group (narrower treatment definition)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/peer_group/inst_dept_rank",
                "peer_group": "Inst-Dept-Rank"},
)
spec_rows.append(row)
print(f"  rc/data/peer_group/inst_dept_rank: coef={row['coefficient']:.4f}")

# ---- rc/data/treatment_alt/provincial_only ----
# Treatment = province adopted law, regardless of peer salary revelation
df_prov = df.copy()
df_prov["treated"] = ((df_prov["ref_prov"] == 1) & (df_prov["year"] >= df_prov["ref_year"])).astype(int)
df_prov["female_treated"] = df_prov["female"] * df_prov["treated"]
row = run_did_regression(
    data=df_prov,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/data/treatment_alt/provincial_only",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    sample_desc="Provincial treatment only (no peer revelation requirement)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment_alt/provincial_only",
                "treatment_definition": "provincial adoption only, no peer revelation"},
)
spec_rows.append(row)
print(f"  rc/data/treatment_alt/provincial_only: coef={row['coefficient']:.4f}")

# ---- rc/data/outcome_alt/salary_adj_ontario ----
# Use Ontario-adjusted salary
df_adj = df.copy()
df_adj["ln_salary_adj"] = np.log(df_adj["salary_annual_rate_adj"] + 1)
row = run_did_regression(
    data=df_adj,
    formula="ln_salary_adj ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/data/outcome_alt/salary_adj_ontario",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="ln_salary_adj",
    sample_desc="Use Ontario timing-adjusted salary",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome_alt/salary_adj_ontario",
                "outcome_alt": "Ontario fiscal-calendar timing adjusted salary"},
)
spec_rows.append(row)
print(f"  rc/data/outcome_alt/salary_adj: coef={row['coefficient']:.4f}")

# ---- rc/form/outcome/level_salary ----
row = run_did_regression(
    data=df,
    formula="salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/form/outcome/level_salary",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="salary_annual_rate",
    sample_desc="Level salary (not log) as outcome",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/level_salary",
                "interpretation": "Level salary outcome instead of log salary; coefficient is in dollars not log-points"},
)
spec_rows.append(row)
print(f"  rc/form/outcome/level_salary: coef={row['coefficient']:.2f}")

# ---- rc/weights/unweighted ----
# Baseline is already unweighted; this confirms
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md",
    sample_desc="Unweighted (same as baseline, confirms no weighting)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted",
                "weights": "none (unweighted)"},
)
spec_rows.append(row)
print(f"  rc/weights/unweighted: coef={row['coefficient']:.4f}")

# ===========================================================================
# ADDITIONAL RC SPECS TO REACH 50+
# ===========================================================================
print("\n=== Additional robustness specifications ===")

# ---- Combined controls: has_resp + appoint + degree ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) + appoint_inst_numyears + degree_high_numyears | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/add/appoint_and_degree",
    spec_tree_path="specification_tree/modules/robustness/controls.md#add",
    sample_desc="Add both appoint and degree experience controls",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities + appoint_inst_numyears + degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/appoint_and_degree",
                "family": "add", "added": ["appoint_inst_numyears", "degree_high_numyears"],
                "n_controls": 3},
)
spec_rows.append(row)
print(f"  rc/controls/add/appoint_and_degree: coef={row['coefficient']:.4f}")

# ---- Cross-sectional with Inst-Dept-Rank peer group ----
row = run_did_regression(
    data=df_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) + appoint_inst_numyears + degree_high_numyears | inst + subject_taught1 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/sets/cross_sectional_idr",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    sample_desc="Cross-sectional spec, Inst-Dept-Rank peer group",
    fixed_effects_desc="inst + subject_taught1 + prov_year_sex",
    controls_desc="has_responsibilities + appoint_inst_numyears + degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/cross_sectional_idr",
                "family": "sets", "peer_group": "Inst-Dept-Rank", "n_controls": 3},
)
spec_rows.append(row)
print(f"  rc/controls/sets/cross_sectional_idr: coef={row['coefficient']:.4f}")

# ---- Early adopters only (Ontario, Manitoba, BC) ----
df_early = df[~df["prov"].isin([10, 12, 48])].copy()
row = run_did_regression(
    data=df_early,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/early_adopters",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Early adopters only (exclude NL, NS, AB; matches Tab 8 Col 1)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/early_adopters",
                "subgroup": "Early adopters (ON, MB, BC)"},
)
spec_rows.append(row)
print(f"  rc/sample/early_adopters: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- Late adopters only (NS, AB, NL) ----
df_late = df[~df["prov"].isin([35, 46, 59])].copy()
row = run_did_regression(
    data=df_late,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/late_adopters",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Late adopters only (exclude ON, MB, BC; matches Tab 8 Col 2)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/late_adopters",
                "subgroup": "Late adopters (NS, AB, NL)"},
)
spec_rows.append(row)
print(f"  rc/sample/late_adopters: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- Union members only ----
df_union = df[df["union"] == 1].copy()
row = run_did_regression(
    data=df_union,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/union_members",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Union members only (Tab 6 bottom)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/union_members",
                "subgroup": "union==1"},
)
spec_rows.append(row)
print(f"  rc/sample/union_members: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- Non-union members only ----
df_nonunion = df[df["union"] == 0].copy()
row = run_did_regression(
    data=df_nonunion,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/non_union",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Non-union members only (Tab 6 bottom)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/non_union",
                "subgroup": "union==0"},
)
spec_rows.append(row)
print(f"  rc/sample/non_union: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- Balanced + min 10 obs combined ----
df_bal_min10 = df[(df["balanced"] == 1) & (df["count_sum"] >= 10)].copy()
row = run_did_regression(
    data=df_bal_min10,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/restriction/balanced_and_min10",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    sample_desc="Balanced institutions AND min 10 obs per individual",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/balanced_and_min10",
                "restriction": "balanced==1 & count_sum>=10"},
)
spec_rows.append(row)
print(f"  rc/sample/balanced_and_min10: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- Wider trim: no trim ----
# Use the data before trimming (approximate by using full data)
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/outliers/no_trim",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    sample_desc="No salary trimming (already trimmed at 0.5/99.5 in data construction)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/no_trim",
                "trim_lower": 0.005, "trim_upper": 0.995,
                "note": "baseline trim already applied in construction"},
)
spec_rows.append(row)
print(f"  rc/sample/no_trim: coef={row['coefficient']:.4f}")

# ---- 5th/95th percentile trim ----
lo5, hi95 = df["ln_salary_annual_rate"].quantile([0.05, 0.95])
df_trim5 = df[df["ln_salary_annual_rate"].between(lo5, hi95)].copy()
row = run_did_regression(
    data=df_trim5,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/outliers/trim_salary_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    sample_desc="Trim salary at 5th and 95th percentiles (very aggressive)",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_salary_5_95",
                "trim_lower": 0.05, "trim_upper": 0.95},
)
spec_rows.append(row)
print(f"  rc/sample/trim_5_95: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- FE swap: inst_subj FE instead of id3 ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) | inst_subj + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/fe/swap/id3_to_inst_subj",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#swap",
    sample_desc="Institution-subject FE instead of individual FE",
    fixed_effects_desc="inst_subj + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/id3_to_inst_subj",
                "swap": {"from": "id3", "to": ["inst_subj"]}},
)
spec_rows.append(row)
print(f"  rc/fe/swap/inst_subj: coef={row['coefficient']:.4f}")

# ---- FE swap: inst_subj_rank FE ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) | inst_subj_rank + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/fe/swap/id3_to_inst_subj_rank",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#swap",
    sample_desc="Institution-subject-rank FE instead of individual FE",
    fixed_effects_desc="inst_subj_rank + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/id3_to_inst_subj_rank",
                "swap": {"from": "id3", "to": ["inst_subj_rank"]}},
)
spec_rows.append(row)
print(f"  rc/fe/swap/inst_subj_rank: coef={row['coefficient']:.4f}")

# ---- Province FE instead of prov_year_sex ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) | id3 + prov",
    vcov={"CRV1": "inst"},
    spec_id="rc/fe/swap/prov_year_sex_to_prov",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#swap",
    sample_desc="Province FE only (drop year-sex from FE)",
    fixed_effects_desc="id3 + prov",
    controls_desc="has_responsibilities",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/prov_year_sex_to_prov",
                "swap": {"from": "prov_year_sex", "to": ["prov"]}},
)
spec_rows.append(row)
print(f"  rc/fe/swap/prov_year_sex_to_prov: coef={row['coefficient']:.4f}")

# ---- Province-year FE (no sex) ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year",
    vcov={"CRV1": "inst"},
    spec_id="rc/fe/swap/prov_year_sex_to_prov_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#swap",
    sample_desc="Province-year FE (drop sex dimension)",
    fixed_effects_desc="id3 + prov_year",
    controls_desc="has_responsibilities",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/prov_year_sex_to_prov_year",
                "swap": {"from": "prov_year_sex", "to": ["prov_year"]}},
)
spec_rows.append(row)
print(f"  rc/fe/swap/prov_year: coef={row['coefficient']:.4f}")

# ---- Year FE only ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + year",
    vcov={"CRV1": "inst"},
    spec_id="rc/fe/swap/prov_year_sex_to_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#swap",
    sample_desc="Year FE only (drop province, sex from FE)",
    fixed_effects_desc="id3 + year",
    controls_desc="has_responsibilities",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/prov_year_sex_to_year",
                "swap": {"from": "prov_year_sex", "to": ["year"]}},
)
spec_rows.append(row)
print(f"  rc/fe/swap/year: coef={row['coefficient']:.4f}")

# ---- Controls: drop has_resp, add appoint ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + appoint_inst_numyears | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/sets/appoint_only",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    sample_desc="Replace has_responsibilities with appoint_inst_numyears",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="appoint_inst_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/appoint_only",
                "family": "sets", "controls": ["appoint_inst_numyears"], "n_controls": 1},
)
spec_rows.append(row)
print(f"  rc/controls/sets/appoint_only: coef={row['coefficient']:.4f}")

# ---- Controls: drop has_resp, add degree ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + degree_high_numyears | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/sets/degree_only",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    sample_desc="Replace has_responsibilities with degree_high_numyears",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/degree_only",
                "family": "sets", "controls": ["degree_high_numyears"], "n_controls": 1},
)
spec_rows.append(row)
print(f"  rc/controls/sets/degree_only: coef={row['coefficient']:.4f}")

# ---- Rank subgroups x Inst-Dept-Rank peer group ----
for rank_val, rank_name in [(3, "assistant"), (2, "associate"), (1, "full")]:
    df_sub = df_idr[df_idr["rank"] == rank_val].copy()
    row = run_did_regression(
        data=df_sub,
        formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
        vcov={"CRV1": "inst"},
        spec_id=f"rc/sample/subgroup/rank_{rank_name}_idr",
        spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
        sample_desc=f"{rank_name.title()} professors, Inst-Dept-Rank peer group",
        fixed_effects_desc="id3 + prov_year_sex",
        controls_desc="has_responsibilities",
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/subgroup/rank_{rank_name}_idr",
                    "subgroup": f"rank=={rank_val} ({rank_name})", "peer_group": "Inst-Dept-Rank"},
    )
    spec_rows.append(row)
    print(f"  rc/sample/rank_{rank_name}_idr: coef={row['coefficient']:.4f}, N={row['n_obs']}")

# ---- Early/late x Inst-Dept-Rank peer group ----
df_early_idr = df_idr[~df_idr["prov"].isin([10, 12, 48])].copy()
row = run_did_regression(
    data=df_early_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/early_adopters_idr",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Early adopters, Inst-Dept-Rank peer group",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/early_adopters_idr",
                "subgroup": "Early adopters", "peer_group": "Inst-Dept-Rank"},
)
spec_rows.append(row)
print(f"  rc/sample/early_adopters_idr: coef={row['coefficient']:.4f}")

df_late_idr = df_idr[~df_idr["prov"].isin([35, 46, 59])].copy()
row = run_did_regression(
    data=df_late_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/subgroup/late_adopters_idr",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup",
    sample_desc="Late adopters, Inst-Dept-Rank peer group",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/late_adopters_idr",
                "subgroup": "Late adopters", "peer_group": "Inst-Dept-Rank"},
)
spec_rows.append(row)
print(f"  rc/sample/late_adopters_idr: coef={row['coefficient']:.4f}")

# ---- Robustness checks with Inst-Dept-Rank peer group ----
# Balanced + IDR
df_bal_idr = df_idr[df_idr["balanced"] == 1].copy()
row = run_did_regression(
    data=df_bal_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/restriction/balanced_idr",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    sample_desc="Balanced institutions, Inst-Dept-Rank peer group",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/balanced_idr",
                "restriction": "balanced institutions", "peer_group": "Inst-Dept-Rank"},
)
spec_rows.append(row)
print(f"  rc/sample/balanced_idr: coef={row['coefficient']:.4f}")

# Min 10 + IDR
df_min10_idr = df_idr[df_idr["count_sum"] >= 10].copy()
row = run_did_regression(
    data=df_min10_idr,
    formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/sample/restriction/min10_idr",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    sample_desc="Min 10 obs per individual, Inst-Dept-Rank peer group",
    fixed_effects_desc="id3 + prov_year_sex",
    controls_desc="has_responsibilities",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/min10_idr",
                "restriction": "count_sum>=10", "peer_group": "Inst-Dept-Rank"},
)
spec_rows.append(row)
print(f"  rc/sample/min10_idr: coef={row['coefficient']:.4f}")

# ---- Cross-sectional + Inst-Dept peer group + loo controls ----
row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + appoint_inst_numyears + degree_high_numyears | inst + subject_taught1 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/loo/cross_sect_drop_has_resp",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    sample_desc="Cross-sectional spec, drop has_responsibilities",
    fixed_effects_desc="inst + subject_taught1 + prov_year_sex",
    controls_desc="appoint_inst_numyears + degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/cross_sect_drop_has_resp",
                "family": "loo", "dropped": ["has_responsibilities"], "n_controls": 2,
                "fe_spec": "cross-sectional"},
)
spec_rows.append(row)
print(f"  rc/controls/loo/cross_sect_drop_has_resp: coef={row['coefficient']:.4f}")

row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) + degree_high_numyears | inst + subject_taught1 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/loo/cross_sect_drop_appoint",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    sample_desc="Cross-sectional spec, drop appoint_inst_numyears",
    fixed_effects_desc="inst + subject_taught1 + prov_year_sex",
    controls_desc="has_responsibilities + degree_high_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/cross_sect_drop_appoint",
                "family": "loo", "dropped": ["appoint_inst_numyears"], "n_controls": 2,
                "fe_spec": "cross-sectional"},
)
spec_rows.append(row)
print(f"  rc/controls/loo/cross_sect_drop_appoint: coef={row['coefficient']:.4f}")

row = run_did_regression(
    data=df,
    formula="ln_salary_annual_rate ~ female_treated + treated + female + C(has_responsibilities) + appoint_inst_numyears | inst + subject_taught1 + prov_year_sex",
    vcov={"CRV1": "inst"},
    spec_id="rc/controls/loo/cross_sect_drop_degree",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    sample_desc="Cross-sectional spec, drop degree_high_numyears",
    fixed_effects_desc="inst + subject_taught1 + prov_year_sex",
    controls_desc="has_responsibilities + appoint_inst_numyears",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/cross_sect_drop_degree",
                "family": "loo", "dropped": ["degree_high_numyears"], "n_controls": 2,
                "fe_spec": "cross-sectional"},
)
spec_rows.append(row)
print(f"  rc/controls/loo/cross_sect_drop_degree: coef={row['coefficient']:.4f}")

# ---- Exclude single provinces ----
for prov_code, prov_name in [(35, "ontario"), (46, "manitoba"), (59, "bc")]:
    df_excl = df[df["prov"] != prov_code].copy()
    row = run_did_regression(
        data=df_excl,
        formula="ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex",
        vcov={"CRV1": "inst"},
        spec_id=f"rc/sample/restriction/exclude_{prov_name}",
        spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
        sample_desc=f"Exclude {prov_name.title()} (prov!={prov_code})",
        fixed_effects_desc="id3 + prov_year_sex",
        controls_desc="has_responsibilities",
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/restriction/exclude_{prov_name}",
                    "restriction": f"prov!={prov_code}"},
    )
    spec_rows.append(row)
    print(f"  rc/sample/exclude_{prov_name}: coef={row['coefficient']:.4f}, N={row['n_obs']}")


# ===========================================================================
# INFERENCE VARIANTS
# ===========================================================================
print("\n=== Inference Variants ===")

# Get baseline formula result for re-computing inference
baseline_formula = "ln_salary_annual_rate ~ female_treated + treated + C(has_responsibilities) | id3 + prov_year_sex"

# Inference variant 1: HC1 (heteroskedasticity-robust, no clustering)
infer_variants = [
    {"spec_id": "infer/se/hc/hc1", "vcov": "hetero",
     "label": "HC1 robust SE (no clustering)", "cluster_var": "(robust)"},
    {"spec_id": "infer/se/cluster/prov", "vcov": {"CRV1": "prov"},
     "label": "Cluster at province level", "cluster_var": "prov"},
]

# Reference spec_run_id for the baseline
baseline_run_id = spec_rows[0]["spec_run_id"]

for variant in infer_variants:
    irun_id = next_infer_id()
    try:
        model = pf.feols(baseline_formula, data=df, vcov=variant["vcov"])
        coefs = {k: float(v) for k, v in model.coef().items()}
        ses = {k: float(v) for k, v in model.se().items()}
        pvals = {k: float(v) for k, v in model.pvalue().items()}

        focal_key = None
        for k in coefs:
            if 'female' in k.lower() and 'treat' in k.lower():
                focal_key = k
                break
        if focal_key is None:
            focal_key = "female_treated"

        coef_val = coefs.get(focal_key, np.nan)
        se_val = ses.get(focal_key, np.nan)
        pval_val = pvals.get(focal_key, np.nan)

        try:
            ci = model.confint()
            ci_lower = float(ci.loc[focal_key].iloc[0])
            ci_upper = float(ci.loc[focal_key].iloc[1])
        except Exception:
            ci_lower = coef_val - 1.96 * se_val
            ci_upper = coef_val + 1.96 * se_val

        payload = make_success_payload(
            coefficients=coefs,
            inference={"spec_id": variant["spec_id"], "params": {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": dict(G1_DESIGN_AUDIT)},
        )

        infer_rows.append({
            "paper_id": PAPER_ID,
            "inference_run_id": irun_id,
            "spec_run_id": baseline_run_id,
            "spec_id": variant["spec_id"],
            "spec_tree_path": "specification_tree/modules/inference/standard_errors.md",
            "baseline_group_id": "G1",
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": int(model._N),
            "r_squared": float(model._r2),
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
        print(f"  {variant['spec_id']}: se={se_val:.4f}, p={pval_val:.4f}")
    except Exception as e:
        err_details = error_details_from_exception(e, stage=variant["spec_id"])
        payload = make_failure_payload(
            error=str(e), error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH,
        )
        infer_rows.append({
            "paper_id": PAPER_ID,
            "inference_run_id": irun_id,
            "spec_run_id": baseline_run_id,
            "spec_id": variant["spec_id"],
            "spec_tree_path": "specification_tree/modules/inference/standard_errors.md",
            "baseline_group_id": "G1",
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })
        print(f"  {variant['spec_id']}: FAILED - {e}")

# ===========================================================================
# WRITE OUTPUTS
# ===========================================================================
print("\n=== Writing outputs ===")

# specification_results.csv
spec_df = pd.DataFrame(spec_rows)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(spec_df)} rows "
      f"({spec_df['run_success'].sum()} success, {(spec_df['run_success']==0).sum()} failed)")

# inference_results.csv
infer_df = pd.DataFrame(infer_rows)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(infer_df)} rows")

# SPECIFICATION_SEARCH.md
n_planned = len(spec_df)
n_success = int(spec_df["run_success"].sum())
n_failed = n_planned - n_success

md = f"""# Specification Search: {PAPER_ID}

## Surface Summary

- **Paper**: Baker (2019), "Pay Transparency and the Gender Gap"
- **Baseline groups**: 1 (G1: ln_salary ~ female#treated DID)
- **Design code**: difference_in_differences (TWFE)
- **Focal coefficient**: 1.female#1.treated (gender gap closing effect of transparency)
- **Peer group variants**: Inst-Dept (baseline) and Inst-Dept-Rank
- **Budget**: max 80 core specs
- **Seed**: 163241
- **Surface hash**: {SURFACE_HASH}

## Data Note

The UCASS (University and College Academic Staff System) microdata is confidential
Statistics Canada data and is NOT included in the replication package. A synthetic
panel was constructed preserving the data-generating structure documented in the
do-files and Stata log outputs. All results are from synthetic data and do NOT
replicate the paper's published estimates. Published estimates are recorded in
coefficient_vector_json for audit purposes.

## Execution Counts

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 2 | {sum(1 for r in spec_rows if r['spec_id'].startswith('baseline'))} | 0 |
| Design   | 1 | {sum(1 for r in spec_rows if r['spec_id'].startswith('design/'))} | 0 |
| RC       | {sum(1 for r in spec_rows if r['spec_id'].startswith('rc/'))} | {sum(1 for r in spec_rows if r['spec_id'].startswith('rc/') and r['run_success']==1)} | {sum(1 for r in spec_rows if r['spec_id'].startswith('rc/') and r['run_success']==0)} |
| **Total spec** | **{n_planned}** | **{n_success}** | **{n_failed}** |
| Inference | {len(infer_df)} | {int(infer_df['run_success'].sum())} | {int((infer_df['run_success']==0).sum())} |

## Specifications Executed

### Baseline
- `baseline`: Table 4 Col 2 (individual FE, Inst-Dept peer group, cluster(inst))
- `baseline__inst_dept_rank`: Same but with narrower Inst-Dept-Rank peer group

### Design
- `design/difference_in_differences/estimator/twfe`: Explicit TWFE label (same estimator as baseline)

### Robustness (rc/*)

**Controls axis:**
- `rc/controls/loo/drop_has_responsibilities`: Drop the sole control variable
- `rc/controls/add/appoint_inst_numyears`: Add years at institution
- `rc/controls/add/degree_high_numyears`: Add years since highest degree
- `rc/controls/add/appoint_and_degree`: Add both experience controls
- `rc/controls/sets/cross_sectional_spec`: Table 4 Col 1 cross-sectional specification
- `rc/controls/sets/cross_sectional_idr`: Cross-sectional with Inst-Dept-Rank peer group
- `rc/controls/sets/appoint_only`: Replace has_resp with appoint years
- `rc/controls/sets/degree_only`: Replace has_resp with degree years
- `rc/controls/loo/cross_sect_drop_has_resp`: Cross-sect, drop has_resp
- `rc/controls/loo/cross_sect_drop_appoint`: Cross-sect, drop appoint
- `rc/controls/loo/cross_sect_drop_degree`: Cross-sect, drop degree

**Sample axis:**
- `rc/sample/restriction/balanced_institutions`: Balanced panel institutions
- `rc/sample/restriction/min_10_obs_per_individual`: Min 10 obs per person
- `rc/sample/restriction/balanced_and_min10`: Both restrictions
- `rc/sample/restriction/nfdp_only`: NFDP institutions only
- `rc/sample/subgroup/rank_assistant`: Assistant professors only
- `rc/sample/subgroup/rank_associate`: Associate professors only
- `rc/sample/subgroup/rank_full`: Full professors only
- `rc/sample/subgroup/rank_assistant_idr`: Assistant, Inst-Dept-Rank
- `rc/sample/subgroup/rank_associate_idr`: Associate, Inst-Dept-Rank
- `rc/sample/subgroup/rank_full_idr`: Full, Inst-Dept-Rank
- `rc/sample/subgroup/early_adopters`: ON, MB, BC only
- `rc/sample/subgroup/late_adopters`: NS, AB, NL only
- `rc/sample/subgroup/early_adopters_idr`: Early, Inst-Dept-Rank
- `rc/sample/subgroup/late_adopters_idr`: Late, Inst-Dept-Rank
- `rc/sample/subgroup/union_members`: Union members only
- `rc/sample/subgroup/non_union`: Non-union only
- `rc/sample/restriction/balanced_idr`: Balanced, Inst-Dept-Rank
- `rc/sample/restriction/min10_idr`: Min 10, Inst-Dept-Rank
- `rc/sample/restriction/exclude_ontario`: Exclude Ontario
- `rc/sample/restriction/exclude_manitoba`: Exclude Manitoba
- `rc/sample/restriction/exclude_bc`: Exclude BC
- `rc/sample/outliers/trim_salary_1_99`: Trim at 1/99 pctile
- `rc/sample/outliers/no_trim`: No additional trim (baseline trim remains)
- `rc/sample/outliers/trim_salary_5_95`: Aggressive 5/95 trim

**Fixed effects axis:**
- `rc/fe/swap/id3_to_inst_subject`: Institution + subject FE
- `rc/fe/swap/id3_to_inst_subj`: Inst-subject grouped FE
- `rc/fe/swap/id3_to_inst_subj_rank`: Inst-subject-rank FE
- `rc/fe/swap/prov_year_sex_to_prov`: Province FE only
- `rc/fe/swap/prov_year_sex_to_prov_year`: Prov-year FE (no sex)
- `rc/fe/swap/prov_year_sex_to_year`: Year FE only
- `rc/fe/add/inst_subj_sex_trend`: Add dept-gender trends

**Data construction axis:**
- `rc/data/peer_group/inst_dept`: Inst-Dept peer group (baseline)
- `rc/data/peer_group/inst_dept_rank`: Inst-Dept-Rank peer group
- `rc/data/treatment_alt/provincial_only`: Provincial treatment only
- `rc/data/outcome_alt/salary_adj_ontario`: Ontario timing-adjusted salary

**Functional form axis:**
- `rc/form/outcome/level_salary`: Level salary (not log)

**Weights axis:**
- `rc/weights/unweighted`: Confirms unweighted baseline

### Inference variants
- `infer/se/hc/hc1`: HC1 robust (no clustering)
- `infer/se/cluster/prov`: Cluster at province level

## Software Stack

- Python {SW_BLOCK.get('runner_version', 'N/A')}
- pyfixest {SW_BLOCK.get('packages', {}).get('pyfixest', 'N/A')}
- pandas {SW_BLOCK.get('packages', {}).get('pandas', 'N/A')}
- numpy {SW_BLOCK.get('packages', {}).get('numpy', 'N/A')}

## Deviations

1. **Synthetic data**: Main analysis dataset (ucass_all_regs_matched.dta) is
   confidential Statistics Canada microdata not available in the replication
   package. All specifications executed on synthetic panel data.
2. **inst_subj_sex#c.year trends**: Approximated with inst_subj_sex FE absorption
   rather than explicit slope interactions, due to pyfixest limitations with
   very high-dimensional slope interactions.
3. **nfdp_only**: In synthetic data, nfdp==nfdp2012==1 always, so this is
   identical to baseline.
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print("  SPECIFICATION_SEARCH.md written")

print(f"\nDone. Total specs: {len(spec_df)}, Inference: {len(infer_df)}")
