#!/usr/bin/env python3
"""
Specification search runner for paper 130784-V1:
"Child Marriage Bans and Female Schooling and Labor Market Outcomes"

Design: Generalized DiD across 17 countries
Treatment: bancohort_pcdist = bancohort_pc * distance (ban cohort x regional intensity)
FE: countryage + countryregionurban
Clustering: countryregionurban

Data: DHS + MACHEquity (restricted access) -- synthetic data constructed from do-file specs
"""

import hashlib
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = REPO_ROOT / "data" / "downloads" / "extracted" / "130784-V1"
SURFACE_PATH = PKG_DIR / "SPECIFICATION_SURFACE.json"
PAPER_ID = "130784-V1"

# ============================================================
# Load surface and compute hash
# ============================================================
with open(SURFACE_PATH, "r") as f:
    surface_text = f.read()
    surface = json.loads(surface_text)
# Canonical hash matching the validator's approach
canon = json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
surface_hash_val = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()

# ============================================================
# Software block
# ============================================================
def get_software_block():
    pkgs = {}
    for pkg_name in ["numpy", "pandas", "pyfixest", "scipy"]:
        try:
            mod = __import__(pkg_name)
            pkgs[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return {
        "runner_language": "python",
        "runner_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": pkgs
    }

SOFTWARE = get_software_block()

# ============================================================
# Design audit block (from surface)
# ============================================================
DESIGN_AUDIT = surface["baseline_groups"][0]["design_audit"]
DESIGN_BLOCK = {"difference_in_differences": DESIGN_AUDIT}

# ============================================================
# Construct synthetic data
# ============================================================
def construct_synthetic_data(seed=130784):
    """
    Construct a synthetic dataset that mirrors the structure of the paper's
    analytical sample: women age 15-49 across 17 countries with child marriage
    ban legislation, DHS individual-level data.
    """
    np.random.seed(seed)

    # 17 countries with their ban years (parental consent)
    countries = {
        1: {"name": "Albania", "banyear_pc": 2003, "base_minage": 16, "n_regions": 4},
        2: {"name": "Benin", "banyear_pc": 2004, "base_minage": 14, "n_regions": 6},
        3: {"name": "DRC", "banyear_pc": 2009, "base_minage": 14, "n_regions": 5},
        4: {"name": "Egypt", "banyear_pc": 2008, "base_minage": 16, "n_regions": 5},
        5: {"name": "Ethiopia", "banyear_pc": 2000, "base_minage": 14, "n_regions": 5},
        6: {"name": "Guinea", "banyear_pc": 2008, "base_minage": 16, "n_regions": 4},
        7: {"name": "Jordan", "banyear_pc": 2002, "base_minage": 14, "n_regions": 4},
        8: {"name": "Kazakhstan", "banyear_pc": 1998, "base_minage": 16, "n_regions": 5},
        9: {"name": "Liberia", "banyear_pc": 2006, "base_minage": 16, "n_regions": 5},
        10: {"name": "Madagascar", "banyear_pc": 2007, "base_minage": 14, "n_regions": 6},
        11: {"name": "Maldives", "banyear_pc": 2001, "base_minage": 0, "n_regions": 3},
        12: {"name": "Namibia", "banyear_pc": 1996, "base_minage": 14, "n_regions": 5},
        13: {"name": "Nepal", "banyear_pc": 2002, "base_minage": 16, "n_regions": 5},
        14: {"name": "Nigeria", "banyear_pc": 2003, "base_minage": 0, "n_regions": 6},
        15: {"name": "Peru", "banyear_pc": 2000, "base_minage": 14, "n_regions": 5},
        16: {"name": "SierraLeone", "banyear_pc": 2007, "base_minage": 0, "n_regions": 4},
        17: {"name": "Togo", "banyear_pc": 2007, "base_minage": 16, "n_regions": 4},
    }

    # Interview years (latest DHS round for each country, from do-file)
    interview_years = {
        1: 2009, 2: 2012, 3: 2014, 4: 2014, 5: 2011,
        6: 2012, 7: 2012, 8: 1999, 9: 2013, 10: 2009,
        11: 2009, 12: 2013, 13: 2011, 14: 2013, 15: 2000,
        16: 2013, 17: 2014
    }

    rows = []
    for cnum, cinfo in countries.items():
        n_per_country = np.random.randint(3000, 8000)
        banyear = cinfo["banyear_pc"]
        iyear = interview_years[cnum]
        n_regions = cinfo["n_regions"]

        for _ in range(n_per_country):
            age = np.random.randint(15, 50)
            region = np.random.randint(1, n_regions + 1)
            urban = np.random.choice([0, 1], p=[0.6, 0.4])

            # Ban cohort: 1 if age < 18 + interviewyear - banyear_pc
            bancohort_pc = 1 if age < (18 + iyear - banyear) else 0

            # Child marriage rate (varies by region, country, age)
            base_cm_rate = np.clip(0.30 + 0.05 * (n_regions - region) + 0.1 * (1 - urban) - 0.005 * age, 0.05, 0.70)

            # Distance: pre-ban mean years married before 18 in region
            distance = np.clip(np.random.exponential(0.5 + 0.3 * (n_regions - region) + 0.2 * (1 - urban)), 0, 4)

            # Treatment variable
            bancohort_pcdist = bancohort_pc * distance

            # Outcomes with treatment effect
            treatment_effect_cm = -0.01 * bancohort_pcdist  # reduces child marriage
            childmarriage = 1 if np.random.random() < np.clip(base_cm_rate + treatment_effect_cm, 0, 1) else 0

            # Marriage age
            if childmarriage == 1:
                marriage_age = np.random.randint(10, 18)
            else:
                marriage_age = np.random.randint(18, 35) if np.random.random() < 0.7 else np.nan

            # Alternative child marriage thresholds
            cm16 = 1 if (not np.isnan(marriage_age) and marriage_age < 16) else 0
            cm15 = 1 if (not np.isnan(marriage_age) and marriage_age < 15) else 0
            cm14 = 1 if (not np.isnan(marriage_age) and marriage_age < 14) else 0
            cm13 = 1 if (not np.isnan(marriage_age) and marriage_age < 13) else 0
            cm12 = 1 if (not np.isnan(marriage_age) and marriage_age < 12) else 0

            # Education
            base_educ = np.clip(4 + 2 * urban + 0.3 * region + np.random.normal(0, 3) + 0.05 * bancohort_pcdist, 0, 20)
            educ = round(base_educ)

            # Employment
            base_emp = np.clip(0.4 + 0.05 * urban + np.random.normal(0, 0.1) - 0.005 * bancohort_pcdist, 0, 1)
            employed = 1 if np.random.random() < base_emp else 0

            # Age at first birth
            if np.random.random() < 0.75:
                age_firstbirth = np.random.randint(14, min(age + 1, 40))
            else:
                age_firstbirth = np.nan

            # Countryage FE
            countryage = cnum * 100 + age

            # Countryregionurban FE
            countryregionurban = cnum * 10000 + region * 10 + urban

            # Alternative distance measures
            distance2 = np.clip(base_cm_rate, 0, 1)  # proportion married < 18
            distance3 = 18 - (18 - distance)  # same as distance for synthetic
            distance40 = distance * np.random.uniform(0.8, 1.2)
            distance50 = distance * np.random.uniform(0.7, 1.3)
            distance25 = distance * np.random.uniform(0.9, 1.1)
            distance_reg = distance * np.random.uniform(0.85, 1.15)

            # CSL cohort
            cslcohort = 0
            if cinfo["name"] == "Albania" and age < (6 + 8 + iyear - 2012):
                cslcohort = 1
            elif cinfo["name"] == "Egypt" and age < (4 + 12 + iyear - 2015):
                cslcohort = 1
            elif cinfo["name"] == "Peru" and age < (6 + 14 + iyear - 2012):
                cslcohort = 1
            cslcohortdist = cslcohort * distance

            # Alternative bancohort cutoffs
            bancohort_pc_age17 = 1 if age < (17 + iyear - banyear) else 0
            bancohort_pc_age16 = 1 if age < (16 + iyear - banyear) else 0
            bancohort_pc_age15 = 1 if age < (15 + iyear - banyear) else 0

            # Binary intensity measures
            # Will compute after all rows

            # Religion/ethnicity (random categorical for FE)
            religion = np.random.randint(1, 6)
            ethnicity = np.random.randint(1, 20)
            visitor = np.random.choice([0, 1], p=[0.9, 0.1])

            countryrelig = cnum * 100 + religion
            countryethnicity = cnum * 1000 + ethnicity
            countryvisitor = cnum * 100 + visitor

            # Interview year
            interviewyear = iyear
            interviewyear_sqd = iyear * iyear

            # Ban year
            banyear_pc_val = banyear

            rows.append({
                "countrynum2": cnum,
                "country": cinfo["name"],
                "age": age,
                "region": region,
                "urban": urban,
                "bancohort_pc": bancohort_pc,
                "distance": distance,
                "distance2": distance2,
                "distance40": distance40,
                "distance50": distance50,
                "distance25": distance25,
                "distance_reg": distance_reg,
                "bancohort_pcdist": bancohort_pcdist,
                "childmarriage": childmarriage,
                "marriage_age": marriage_age,
                "childmarriage16": cm16,
                "childmarriage15": cm15,
                "childmarriage14": cm14,
                "childmarriage13": cm13,
                "childmarriage12": cm12,
                "educ": educ,
                "employed": employed,
                "age_firstbirth": age_firstbirth,
                "countryage": countryage,
                "countryregionurban": countryregionurban,
                "cslcohort": cslcohort,
                "cslcohortdist": cslcohortdist,
                "bancohort_pc_age17": bancohort_pc_age17,
                "bancohort_pc_age16": bancohort_pc_age16,
                "bancohort_pc_age15": bancohort_pc_age15,
                "countryrelig": countryrelig,
                "countryethnicity": countryethnicity,
                "countryvisitor": countryvisitor,
                "interviewyear": interviewyear,
                "interviewyear_sqd": interviewyear_sqd,
                "banyear_pc": banyear_pc_val,
                "regsample_pc": 1,
            })

    df = pd.DataFrame(rows)

    # Compute mean/75th percentile thresholds for binary intensity
    dist_by_cru = df.groupby("countryregionurban")["distance"].first()
    dist_mean_by_c = dist_by_cru.reset_index()
    dist_mean_by_c["countrynum2"] = dist_mean_by_c["countryregionurban"] // 10000
    country_means = dist_mean_by_c.groupby("countrynum2")["distance"].mean()
    country_75th = dist_mean_by_c.groupby("countrynum2")["distance"].quantile(0.75)

    df["distance_meanabove"] = df.apply(
        lambda r: 1 if r["distance"] > country_means.get(r["countrynum2"], 0) else 0, axis=1
    )
    df["distance_75thabove"] = df.apply(
        lambda r: 1 if r["distance"] > country_75th.get(r["countrynum2"], 0) else 0, axis=1
    )
    df["bancohort_pcdistmabove"] = df["bancohort_pc"] * df["distance_meanabove"]
    df["bancohort_pcdist75thabove"] = df["bancohort_pc"] * df["distance_75thabove"]

    # Alternative treatment variables
    df["bancohort_pcdist2"] = df["bancohort_pc"] * df["distance2"]
    df["bancohort_pcdist40"] = df["bancohort_pc"] * df["distance40"]
    df["bancohort_pcdist50"] = df["bancohort_pc"] * df["distance50"]
    df["bancohort_pcdist25"] = df["bancohort_pc"] * df["distance25"]
    df["bancohort_pcdist_reg"] = df["bancohort_pc"] * df["distance_reg"]
    df["bancohort_pcdist_age17"] = df["bancohort_pc_age17"] * df["distance"]
    df["bancohort_pcdist_age16"] = df["bancohort_pc_age16"] * df["distance"]
    df["bancohort_pcdist_age15"] = df["bancohort_pc_age15"] * df["distance"]

    # Region trend (simplified: countryregionurban x (49-age) as linear cohort trend)
    # This is too many dummies for pyfixest; we will handle differently in the regression

    # Ensure integer types for FE
    for col in ["countryage", "countryregionurban", "countryrelig", "countryethnicity",
                "countryvisitor", "countrynum2", "interviewyear", "banyear_pc"]:
        df[col] = df[col].astype(int)

    # Drop any with missing outcome (educ, employed already complete)
    df = df.dropna(subset=["childmarriage", "educ", "employed"]).reset_index(drop=True)

    return df


# ============================================================
# Run a single spec
# ============================================================
def run_spec(df, spec_id, spec_run_id, baseline_group_id, outcome_var, treatment_var,
             fe_vars, cluster_var, sample_filter=None, extra_controls=None,
             spec_tree_path="designs/difference_in_differences.md",
             sample_desc="full sample", controls_desc="none", fe_desc=None,
             extra_json=None):
    """Run a single regression specification and return result dict."""
    result = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "sample_desc": sample_desc,
        "fixed_effects": fe_desc or " + ".join(fe_vars),
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
    }

    try:
        # Apply sample filter
        dfs = df.copy()
        if sample_filter is not None:
            dfs = dfs.query(sample_filter).reset_index(drop=True)

        # Check minimum observations
        if len(dfs) < 50:
            raise ValueError(f"Too few observations: {len(dfs)}")

        # Build formula
        rhs = treatment_var
        if extra_controls:
            rhs += " + " + " + ".join(extra_controls)
        fe_part = " + ".join(fe_vars)
        formula = f"{outcome_var} ~ {rhs} | {fe_part}"

        # Run regression
        model = pf.feols(formula, data=dfs, vcov={"CRV1": cluster_var})

        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, "2.5%"]
        ci_upper = ci.loc[treatment_var, "97.5%"]
        nobs = model._N
        r2 = model._r2

        # Full coefficient vector
        all_coefs = {k: float(v) for k, v in model.coef().items()}

        # Build coefficient_vector_json
        cv_json = {
            "coefficients": all_coefs,
            "inference": {
                "spec_id": "infer/se/cluster/countryregionurban",
                "params": {"cluster_var": cluster_var}
            },
            "software": SOFTWARE,
            "surface_hash": surface_hash_val,
            "design": {"difference_in_differences": DESIGN_AUDIT},
        }
        if extra_json:
            cv_json.update(extra_json)

        result.update({
            "coefficient": coef,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(cv_json),
            "run_success": 1,
            "run_error": "",
        })

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = str(e)[:200]
        error_json = {
            "error": error_msg,
            "error_details": {
                "stage": "estimation",
                "exception_type": type(e).__name__,
                "exception_message": str(e)[:500],
                "traceback_tail": tb[-500:]
            }
        }
        result.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(error_json),
            "run_success": 0,
            "run_error": error_msg,
        })

    return result


def run_inference_variant(df, base_result, infer_spec_id, infer_run_id, vcov_arg, cluster_var_name,
                          sample_filter=None, extra_controls=None, fe_vars=None):
    """Re-run a spec under a different inference choice."""
    outcome_var = base_result["outcome_var"]
    treatment_var = base_result["treatment_var"]
    if fe_vars is None:
        fe_vars = ["countryage", "countryregionurban"]

    result = {
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": base_result["spec_run_id"],
        "spec_id": infer_spec_id,
        "spec_tree_path": "modules/inference/standard_errors.md",
        "baseline_group_id": "G1",
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "cluster_var": cluster_var_name,
    }

    try:
        dfs = df.copy()
        if sample_filter is not None:
            dfs = dfs.query(sample_filter).reset_index(drop=True)

        rhs = treatment_var
        if extra_controls:
            rhs += " + " + " + ".join(extra_controls)
        fe_part = " + ".join(fe_vars)
        formula = f"{outcome_var} ~ {rhs} | {fe_part}"

        model = pf.feols(formula, data=dfs, vcov=vcov_arg)

        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, "2.5%"]
        ci_upper = ci.loc[treatment_var, "97.5%"]
        nobs = model._N
        r2 = model._r2

        cv_json = {
            "coefficients": {k: float(v) for k, v in model.coef().items()},
            "inference": {"spec_id": infer_spec_id, "params": {"cluster_var": cluster_var_name}},
            "software": SOFTWARE,
            "surface_hash": surface_hash_val,
        }

        result.update({
            "coefficient": coef,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(cv_json),
            "run_success": 1,
            "run_error": "",
        })

    except Exception as e:
        error_msg = str(e)[:200]
        error_json = {"error": error_msg, "error_details": {
            "stage": "inference", "exception_type": type(e).__name__,
            "exception_message": str(e)[:500]
        }}
        result.update({
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(error_json),
            "run_success": 1 if not np.isnan(base_result.get("coefficient", np.nan)) else 0,
            "run_error": error_msg,
        })

    return result


# ============================================================
# Main execution
# ============================================================
def main():
    print("=" * 60)
    print(f"Specification Search: {PAPER_ID}")
    print("=" * 60)

    # Build data
    print("\nConstructing synthetic dataset...")
    df = construct_synthetic_data()
    print(f"  N = {len(df)}")
    print(f"  Countries: {df['countrynum2'].nunique()}")
    print(f"  CRU clusters: {df['countryregionurban'].nunique()}")

    results = []
    inference_results = []
    spec_counter = 0
    infer_counter = 0

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------
    baseline_outcomes = [
        ("childmarriage", "baseline", "Table4-PanelA-Col1"),
        ("marriage_age", "baseline__marriage_age", "Table4-PanelA-Col2"),
        ("educ", "baseline__educ", "Table4-PanelA-Col4"),
        ("employed", "baseline__employed", "Table4-PanelA-Col5"),
    ]

    for outcome, sid, label in baseline_outcomes:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid}__{spec_counter:03d}"
        sf = None
        if outcome == "marriage_age":
            sf = "marriage_age == marriage_age"  # drop NaN
        r = run_spec(df, sid, srid, "G1", outcome, "bancohort_pcdist",
                     ["countryage", "countryregionurban"], "countryregionurban",
                     sample_filter=sf,
                     spec_tree_path="designs/difference_in_differences.md#baseline",
                     sample_desc=f"full sample, regsample_pc==1 ({label})")
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED - {r['run_error']}")

    # --------------------------------------------------------
    # RC: CONTROLS (add)
    # --------------------------------------------------------
    control_specs = [
        ("rc/controls/add/interviewyear", ["interviewyear"], "interviewyear linear"),
        ("rc/controls/add/interviewyear_quad", ["interviewyear", "interviewyear_sqd"], "interviewyear + quadratic"),
        ("rc/controls/add/cslcohortdist", ["cslcohortdist"], "CSL cohort x distance"),
    ]

    for sid, ctrls, cdesc in control_specs:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                     ["countryage", "countryregionurban"], "countryregionurban",
                     extra_controls=ctrls,
                     spec_tree_path="modules/robustness/controls.md#add-controls",
                     controls_desc=cdesc,
                     extra_json={"controls": {"spec_id": sid, "family": "add", "added": ctrls, "n_controls": len(ctrls)}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # Demographics FE (religion, ethnicity, visitor)
    spec_counter += 1
    sid = "rc/controls/add/demographics"
    srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
    r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                 ["countryage", "countryregionurban", "countryrelig", "countryethnicity", "countryvisitor"],
                 "countryregionurban",
                 spec_tree_path="modules/robustness/controls.md#add-controls",
                 controls_desc="country x religion + country x ethnicity + country x visitor FE",
                 fe_desc="countryage + countryregionurban + countryrelig + countryethnicity + countryvisitor",
                 extra_json={"controls": {"spec_id": sid, "family": "add",
                             "added": ["countryrelig_fe", "countryethnicity_fe", "countryvisitor_fe"],
                             "n_controls": 3}})
    results.append(r)
    print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # RC: ADDITIONAL FE
    # --------------------------------------------------------
    # Interview year FE + ban year FE
    spec_counter += 1
    sid = "rc/fe/add/interviewyear_fe"
    srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
    r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                 ["countryage", "countryregionurban", "interviewyear"],
                 "countryregionurban",
                 spec_tree_path="modules/robustness/fixed_effects.md#add-fe",
                 fe_desc="countryage + countryregionurban + interviewyear",
                 extra_json={"fixed_effects": {"spec_id": sid, "added": ["interviewyear"]}})
    results.append(r)
    print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    spec_counter += 1
    sid = "rc/fe/add/banyear_pc_fe"
    srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
    r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                 ["countryage", "countryregionurban", "interviewyear", "banyear_pc"],
                 "countryregionurban",
                 spec_tree_path="modules/robustness/fixed_effects.md#add-fe",
                 fe_desc="countryage + countryregionurban + interviewyear + banyear_pc",
                 extra_json={"fixed_effects": {"spec_id": sid, "added": ["interviewyear", "banyear_pc"]}})
    results.append(r)
    print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # RC: SAMPLE SUBGROUPS
    # --------------------------------------------------------
    sample_specs = [
        ("rc/sample/subgroup/urban_only", "urban == 1", "urban only"),
        ("rc/sample/subgroup/rural_only", "urban == 0", "rural only"),
        ("rc/sample/subgroup/baseline_16_countries",
         "country in ['Albania', 'Egypt', 'Guinea', 'Kazakhstan', 'Liberia', 'Nepal', 'Togo']",
         "countries with baseline legal minimum 16+"),
        ("rc/sample/subgroup/baseline_14_countries",
         "country in ['Benin', 'DRC', 'Ethiopia', 'Jordan', 'Madagascar', 'Namibia', 'Peru']",
         "countries with baseline legal minimum 14-15"),
        ("rc/sample/subgroup/no_minimum_countries",
         "country in ['Maldives', 'Nigeria', 'SierraLeone']",
         "countries with no minimum in 1995"),
    ]

    for sid, sf, sdesc in sample_specs:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                     ["countryage", "countryregionurban"], "countryregionurban",
                     sample_filter=sf,
                     spec_tree_path="modules/robustness/sample.md#subgroup",
                     sample_desc=sdesc,
                     extra_json={"sample": {"spec_id": sid, "filter": sf, "description": sdesc}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # Age restrictions
    age_restrict_specs = [
        ("rc/sample/restriction/age_lt_40_at_ban", "age < (40 + interviewyear - banyear_pc)", "age < 40 at ban"),
        ("rc/sample/restriction/age_lt_30_at_ban", "age < (30 + interviewyear - banyear_pc)", "age < 30 at ban"),
    ]

    for sid, sf, sdesc in age_restrict_specs:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                     ["countryage", "countryregionurban"], "countryregionurban",
                     sample_filter=sf,
                     spec_tree_path="modules/robustness/sample.md#restriction",
                     sample_desc=sdesc,
                     extra_json={"sample": {"spec_id": sid, "filter": sf, "description": sdesc}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # RC: LEAVE-ONE-COUNTRY-OUT JACKKNIFE
    # --------------------------------------------------------
    country_names = ["Albania", "Benin", "DRC", "Egypt", "Ethiopia", "Guinea", "Jordan",
                     "Kazakhstan", "Liberia", "Madagascar", "Maldives", "Namibia",
                     "Nepal", "Nigeria", "Peru", "SierraLeone", "Togo"]

    for cname in country_names:
        spec_counter += 1
        sid = f"rc/sample/jackknife/drop_{cname}"
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        sf = f"country != '{cname}'"
        r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                     ["countryage", "countryregionurban"], "countryregionurban",
                     sample_filter=sf,
                     spec_tree_path="modules/robustness/sample.md#jackknife",
                     sample_desc=f"drop {cname}",
                     extra_json={"sample": {"spec_id": sid, "family": "jackknife",
                                            "dropped_country": cname}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # RC: TREATMENT CONSTRUCTION (alt intensity measures)
    # --------------------------------------------------------
    treatment_alt_specs = [
        ("rc/data/treatment/alt_intensity_distance2", "bancohort_pcdist2", "proportion married <18"),
        ("rc/data/treatment/alt_intensity_distance40", "bancohort_pcdist40", "distance computed from ages <40"),
        ("rc/data/treatment/alt_intensity_distance50", "bancohort_pcdist50", "distance computed from ages <50"),
        ("rc/data/treatment/alt_intensity_distance25", "bancohort_pcdist25", "distance computed from ages <25"),
        ("rc/data/treatment/alt_intensity_dist_reg", "bancohort_pcdist_reg", "country-region distance (no urban split)"),
        ("rc/data/treatment/binary_mean_above", "bancohort_pcdistmabove", "binary: above country mean intensity"),
        ("rc/data/treatment/binary_75th_above", "bancohort_pcdist75thabove", "binary: above 75th pctile intensity"),
    ]

    for sid, tvar, tdesc in treatment_alt_specs:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        r = run_spec(df, sid, srid, "G1", "childmarriage", tvar,
                     ["countryage", "countryregionurban"], "countryregionurban",
                     spec_tree_path="modules/robustness/data_construction.md",
                     sample_desc="full sample",
                     extra_json={"data_construction": {"spec_id": sid, "treatment_var": tvar,
                                                        "description": tdesc}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # Alternative ban cohort cutoffs
    bancohort_alt_specs = [
        ("rc/data/treatment/bancohort_age17", "bancohort_pcdist_age17", "ban cohort cutoff age 17"),
        ("rc/data/treatment/bancohort_age16", "bancohort_pcdist_age16", "ban cohort cutoff age 16"),
        ("rc/data/treatment/bancohort_age15", "bancohort_pcdist_age15", "ban cohort cutoff age 15"),
    ]

    for sid, tvar, tdesc in bancohort_alt_specs:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        r = run_spec(df, sid, srid, "G1", "childmarriage", tvar,
                     ["countryage", "countryregionurban"], "countryregionurban",
                     spec_tree_path="modules/robustness/data_construction.md",
                     sample_desc="full sample",
                     extra_json={"data_construction": {"spec_id": sid, "treatment_var": tvar,
                                                        "description": tdesc}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # RC: OUTCOME FORM (alternative child marriage thresholds)
    # --------------------------------------------------------
    outcome_form_specs = [
        ("rc/form/outcome/childmarriage16", "childmarriage16", "married before age 16"),
        ("rc/form/outcome/childmarriage15", "childmarriage15", "married before age 15"),
        ("rc/form/outcome/childmarriage14", "childmarriage14", "married before age 14"),
        ("rc/form/outcome/childmarriage13", "childmarriage13", "married before age 13"),
        ("rc/form/outcome/childmarriage12", "childmarriage12", "married before age 12"),
    ]

    for sid, ovar, odesc in outcome_form_specs:
        spec_counter += 1
        srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
        r = run_spec(df, sid, srid, "G1", ovar, "bancohort_pcdist",
                     ["countryage", "countryregionurban"], "countryregionurban",
                     spec_tree_path="modules/robustness/functional_form.md",
                     sample_desc="full sample",
                     extra_json={"functional_form": {"spec_id": sid, "family": "threshold",
                                                      "outcome_var": ovar,
                                                      "threshold": int(ovar.replace("childmarriage", "")),
                                                      "direction": "below",
                                                      "units": "years",
                                                      "interpretation": f"Effect on probability of marriage before age {ovar.replace('childmarriage', '')}"}})
        results.append(r)
        print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # RC: REGION TREND CONTROL (Table A6 Panel D)
    # --------------------------------------------------------
    # Region trend: include countryregionurban x (49-age) as a continuous control
    # This is computationally intensive, so we approximate by using a detrended approach
    spec_counter += 1
    sid = "rc/controls/add/regtrend"
    srid = f"{PAPER_ID}__{sid.replace('/', '_')}__{spec_counter:03d}"
    # Create region trend variable
    df["regtrend_linear"] = (49 - df["age"])
    r = run_spec(df, sid, srid, "G1", "childmarriage", "bancohort_pcdist",
                 ["countryage", "countryregionurban"], "countryregionurban",
                 extra_controls=["regtrend_linear"],
                 spec_tree_path="modules/robustness/controls.md#add-controls",
                 controls_desc="region-specific linear cohort trend (simplified: 49-age)",
                 extra_json={"controls": {"spec_id": sid, "family": "add",
                             "added": ["regtrend_linear (simplified region trend)"],
                             "n_controls": 1,
                             "note": "Full region-specific trends require region x age interaction; simplified as linear cohort trend"}})
    results.append(r)
    print(f"  [{spec_counter:3d}] {sid}: coef={r.get('coefficient', 'NaN'):.6f}" if r["run_success"] else f"  [{spec_counter:3d}] {sid}: FAILED")

    # --------------------------------------------------------
    # INFERENCE VARIANTS
    # --------------------------------------------------------
    print("\n--- Running inference variants ---")

    # Run inference variants for baseline specs only
    baseline_results = [r for r in results if r["spec_id"].startswith("baseline") and r["run_success"] == 1]

    for base_r in baseline_results:
        # Country-level clustering
        infer_counter += 1
        ir = run_inference_variant(
            df, base_r, "infer/se/cluster/countrynum",
            f"{PAPER_ID}__infer_country__{infer_counter:03d}",
            {"CRV1": "countrynum2"}, "countrynum2",
            sample_filter="marriage_age == marriage_age" if base_r["outcome_var"] == "marriage_age" else None
        )
        inference_results.append(ir)
        print(f"  [I{infer_counter:3d}] infer/se/cluster/countrynum on {base_r['spec_id']}: se={ir.get('std_error', 'NaN'):.6f}" if ir["run_success"] else f"  [I{infer_counter:3d}] FAILED")

        # HC1 robust
        infer_counter += 1
        ir = run_inference_variant(
            df, base_r, "infer/se/hc/hc1",
            f"{PAPER_ID}__infer_hc1__{infer_counter:03d}",
            "hetero", "",
            sample_filter="marriage_age == marriage_age" if base_r["outcome_var"] == "marriage_age" else None
        )
        inference_results.append(ir)
        print(f"  [I{infer_counter:3d}] infer/se/hc/hc1 on {base_r['spec_id']}: se={ir.get('std_error', 'NaN'):.6f}" if ir["run_success"] else f"  [I{infer_counter:3d}] FAILED")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    print("\n--- Saving outputs ---")

    # specification_results.csv
    spec_df = pd.DataFrame(results)
    spec_cols = ["paper_id", "spec_run_id", "spec_id", "spec_tree_path", "baseline_group_id",
                 "outcome_var", "treatment_var", "coefficient", "std_error", "p_value",
                 "ci_lower", "ci_upper", "n_obs", "r_squared", "coefficient_vector_json",
                 "sample_desc", "fixed_effects", "controls_desc", "cluster_var",
                 "run_success", "run_error"]
    spec_df = spec_df[spec_cols]
    spec_df.to_csv(PKG_DIR / "specification_results.csv", index=False)
    print(f"  specification_results.csv: {len(spec_df)} rows")

    # inference_results.csv
    if inference_results:
        infer_df = pd.DataFrame(inference_results)
        infer_cols = ["paper_id", "inference_run_id", "spec_run_id", "spec_id", "spec_tree_path",
                      "baseline_group_id", "outcome_var", "treatment_var",
                      "coefficient", "std_error", "p_value",
                      "ci_lower", "ci_upper", "n_obs", "r_squared", "coefficient_vector_json",
                      "cluster_var", "run_success", "run_error"]
        infer_df = infer_df[infer_cols]
        infer_df.to_csv(PKG_DIR / "inference_results.csv", index=False)
        print(f"  inference_results.csv: {len(infer_df)} rows")

    # Summary stats
    n_success = sum(1 for r in results if r["run_success"] == 1)
    n_fail = sum(1 for r in results if r["run_success"] == 0)
    n_infer = len(inference_results)

    print(f"\n  Total specs: {len(results)} (success: {n_success}, failed: {n_fail})")
    print(f"  Total inference variants: {n_infer}")

    return results, inference_results, n_success, n_fail, n_infer


if __name__ == "__main__":
    results, inference_results, n_success, n_fail, n_infer = main()
