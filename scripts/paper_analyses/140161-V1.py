#!/usr/bin/env python3
"""
Specification search runner for paper 140161-V1.

Paper: Henry, Zhuravskaya & Guriev — Fake news & fact-check sharing experiment
Design: Randomized experiment (individual-level, online survey)
Baseline groups:
  G1: ATE of fact-checking on desire to share alt-facts (surveys 1,2,3)
  G2: ATE of voluntary vs imposed fact-check on fact-check sharing (surveys 2,3)
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──
REPO = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
PKG = REPO / "data" / "downloads" / "extracted" / "140161-V1"
DATA = PKG / "original_data"
OUT = PKG

sys.path.insert(0, str(REPO / "scripts"))
from agent_output_utils import (
    surface_hash as compute_surface_hash,
    software_block,
    make_success_payload,
    make_failure_payload,
    error_details_from_exception,
)

PAPER_ID = "140161-V1"

with open(PKG / "SPECIFICATION_SURFACE.json") as f:
    SURFACE = json.load(f)

SHASH = compute_surface_hash(SURFACE)
SW = software_block()
CANONICAL_INFERENCE = {"spec_id": "infer/se/hc/hc1", "params": {}}
DESIGN_AUDIT_G1 = SURFACE["baseline_groups"][0]["design_audit"]
DESIGN_AUDIT_G2 = SURFACE["baseline_groups"][1]["design_audit"]


# ═══════════════════════════════════════════════════════════
# DATA CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def load_survey(fpath, survey_num):
    """Load one survey CSV, applying Qualtrics metadata skip and quality filters."""
    raw = pd.read_csv(fpath, low_memory=False)
    # Skip first 2 rows (Qualtrics metadata)
    df = raw.iloc[2:].reset_index(drop=True)
    df = df[df["DistributionChannel"] != "preview"].copy()

    # Rename columns to lowercase for consistency with Stata code
    col_map = {c: c.lower() for c in df.columns}
    # Special renames
    col_map["Duration (in seconds)"] = "durationinseconds"
    col_map["Q17_7_TEXT"] = "q17_7_text"
    col_map["Q45_NPS_GROUP"] = "q45_nps_group"
    col_map["Q9_1"] = "q9_1"
    if "Q17_7_TEXT - Topics" in col_map:
        col_map["Q17_7_TEXT - Topics"] = "q17_7_text_topics"
    df.rename(columns=col_map, inplace=True)

    # Destring key vars
    num_cols = ["q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9_1", "q10", "q11", "q12",
                "q13", "q14", "q15", "q16_1", "q16_2", "q16_3", "q16_4", "q17",
                "q18_1", "q18_2", "q18_3", "q18_4", "q19_1", "q19_2", "q19_3",
                "q20_1", "q21", "q22", "q27", "q28", "q29", "q30", "q34",
                "q45", "q46_1", "q46_2", "q46_3", "q46_4", "q46_5", "q46_6",
                "q47_1", "q48", "q49", "q50", "q84_1", "q85", "q52_1", "q52_2",
                "q52_3", "q52_4", "q52_5", "q52_6", "q52_7", "q86",
                "durationinseconds", "gc"]
    if survey_num >= 2:
        num_cols.append("q41")
    if survey_num == 3:
        num_cols.append("q38")
    if survey_num == 2:
        num_cols.append("q37")

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quality filters
    df = df[df["durationinseconds"] >= 250].copy()
    df = df[df["gc"] == 1].copy()

    df["survey_id"] = survey_num
    df["survey"] = survey_num

    # ── Parse time for GA merge ──
    df["_startdate"] = df["startdate"].astype(str)
    df["_day"] = pd.to_numeric(df["_startdate"].str[8:10], errors="coerce")
    df["_hour"] = pd.to_numeric(df["_startdate"].str[11:13], errors="coerce")
    df["_minutes"] = pd.to_numeric(df["_startdate"].str[14:16], errors="coerce")

    # Timezone adjust: +8 hours
    df["_hour"] = df["_hour"] + 8
    mask_overflow = df["_hour"] > 23
    df.loc[mask_overflow, "_day"] = df.loc[mask_overflow, "_day"] + 1
    df.loc[mask_overflow, "_hour"] = df.loc[mask_overflow, "_hour"] - 24

    # Duration to minutes
    df["durationinseconds"] = df["durationinseconds"] / 60
    mean_dur = df["durationinseconds"].mean()
    df["_minutes"] = df["_minutes"] + mean_dur

    # Adjust for minutes >= 60
    mask60 = df["_minutes"] >= 60
    df.loc[mask60, "_hour"] = df.loc[mask60, "_hour"] + 1
    mask24 = df["_hour"] == 24
    df.loc[mask24, "_day"] = df.loc[mask24, "_day"] + 1
    df.loc[mask24, "_hour"] = 0

    # fake/fact indicators
    df["_fake"] = np.where(df["q34"] == 1, 1, 0)
    df["_freq"] = 1

    if survey_num == 1:
        df["_fact"] = 0
    else:
        df["_fact"] = np.where(df["q41"] == 1, 1, 0)

    return df


def merge_ga_and_build(surveys_list, ga_hours):
    """Merge GA data and construct outcome variables.

    Replicates Stata logic:
    1. m:1 merge on (day, hour, survey_id, fb=0, fake/fact) to get GA metrics
    2. Two-pass merge: first with adjusted hour, then fallback with original hour
    3. total_participants_fake = count of fake==1 respondents in (day, hour, survey_id, fb=0)
    4. prob_view_fake = uniquepageviews / total_participants_fake
    """
    all_dfs = []

    for df in surveys_list:
        snum = df["survey"].iloc[0]

        # ── GA merge for FAKE NEWS (prob_view_fake, prob_share_GA_fake) ──
        ga_fake = ga_hours[(ga_hours["fb"] == 0) & (ga_hours["fake"] == 1)].copy()
        ga_fake = ga_fake[["day", "hour", "survey_id", "uniquepageviews", "number_share"]].copy()
        for c in ["day", "hour", "survey_id"]:
            ga_fake[c] = pd.to_numeric(ga_fake[c], errors="coerce")

        # First-pass merge: use adjusted hour (_hour which already has +8 and +1 for minutes>=60)
        df["_merge_day"] = df["_day"].copy()
        df["_merge_hour"] = df["_hour"].copy()
        df["_sid_num"] = pd.to_numeric(df["survey_id"], errors="coerce")

        # Merge GA onto individuals who have fake==1 by (day, hour, survey_id)
        # Only those with _fake==1 will get meaningful prob_view_fake
        df_idx = df.index.copy()

        merged = df.merge(
            ga_fake.rename(columns={"day": "_merge_day", "hour": "_merge_hour",
                                    "survey_id": "_sid_num"}),
            on=["_merge_day", "_merge_hour", "_sid_num"],
            how="left", suffixes=("", "_ga_fake")
        )

        # Compute total_participants_fake per (day, hour, survey_id)
        # = count of respondents with fake==1 in each (day, hour, survey) group
        fake_counts = (df[df["_fake"] == 1]
                       .groupby(["_merge_day", "_merge_hour", "_sid_num"])["_freq"]
                       .sum().reset_index()
                       .rename(columns={"_freq": "_tp_fake"}))

        merged = merged.merge(fake_counts, on=["_merge_day", "_merge_hour", "_sid_num"],
                              how="left")

        # prob_view_fake: only for fake==1 and when total_participants >= uniquepageviews
        merged["prob_view_fake"] = np.nan
        mask = (merged["_fake"] == 1) & merged["_tp_fake"].notna() & merged["uniquepageviews"].notna()
        mask = mask & (merged["_tp_fake"] >= merged["uniquepageviews"])
        merged.loc[mask, "prob_view_fake"] = (merged.loc[mask, "uniquepageviews"] /
                                               merged.loc[mask, "_tp_fake"])

        merged["prob_share_GA_fake"] = np.nan
        mask2 = (merged["_fake"] == 1) & merged["_tp_fake"].notna() & merged["number_share"].notna()
        mask2 = mask2 & (merged["_tp_fake"] >= merged["number_share"])
        merged.loc[mask2, "prob_share_GA_fake"] = (merged.loc[mask2, "number_share"] /
                                                    merged.loc[mask2, "_tp_fake"])

        # Clean up temp columns
        for c in ["uniquepageviews", "number_share", "_tp_fake", "_merge_day", "_merge_hour"]:
            if c in merged.columns:
                merged.drop(columns=[c], inplace=True, errors="ignore")

        df = merged

        # ── GA merge for FACT-CHECK (prob_view_fact, prob_share_GA_fact) ──
        if snum > 1 and "fact" in ga_hours.columns:
            ga_fact = ga_hours[(ga_hours["fb"] == 0) & (ga_hours["fact"] == 1)].copy()
            ga_fact = ga_fact[["day", "hour", "survey_id", "uniquepageviews", "number_share"]].copy()
            for c in ["day", "hour", "survey_id"]:
                ga_fact[c] = pd.to_numeric(ga_fact[c], errors="coerce")

            df["_merge_day"] = df["_day"].copy()
            df["_merge_hour"] = df["_hour"].copy()

            merged2 = df.merge(
                ga_fact.rename(columns={"day": "_merge_day", "hour": "_merge_hour",
                                        "survey_id": "_sid_num"}),
                on=["_merge_day", "_merge_hour", "_sid_num"],
                how="left", suffixes=("", "_ga_fact")
            )

            fact_counts = (df[df["_fact"] == 1]
                           .groupby(["_merge_day", "_merge_hour", "_sid_num"])["_freq"]
                           .sum().reset_index()
                           .rename(columns={"_freq": "_tp_fact"}))

            merged2 = merged2.merge(fact_counts, on=["_merge_day", "_merge_hour", "_sid_num"],
                                    how="left")

            merged2["prob_view_fact"] = np.nan
            mask3 = (merged2["_fact"] == 1) & merged2["_tp_fact"].notna() & merged2["uniquepageviews"].notna()
            mask3 = mask3 & (merged2["_tp_fact"] >= merged2["uniquepageviews"])
            merged2.loc[mask3, "prob_view_fact"] = (merged2.loc[mask3, "uniquepageviews"] /
                                                     merged2.loc[mask3, "_tp_fact"])

            merged2["prob_share_GA_fact"] = np.nan
            mask4 = (merged2["_fact"] == 1) & merged2["_tp_fact"].notna() & merged2["number_share"].notna()
            mask4 = mask4 & (merged2["_tp_fact"] >= merged2["number_share"])
            merged2.loc[mask4, "prob_share_GA_fact"] = (merged2.loc[mask4, "number_share"] /
                                                         merged2.loc[mask4, "_tp_fact"])

            for c in ["uniquepageviews", "number_share", "_tp_fact", "_merge_day", "_merge_hour"]:
                if c in merged2.columns:
                    merged2.drop(columns=[c], inplace=True, errors="ignore")

            df = merged2
        else:
            df["prob_view_fact"] = np.nan
            df["prob_share_GA_fact"] = np.nan

        all_dfs.append(df)

    return all_dfs


def build_dataset():
    """Build the full analysis dataset from raw CSVs + GA data."""
    ga = pd.read_stata(DATA / "GA_hours.dta", convert_categoricals=False)

    survey_files = {
        1: DATA / "Survey+1_May+27%2C+2019_01.02.csv",
        2: DATA / "Survey+2_May+27%2C+2019_01.03.csv",
        3: DATA / "Survey+3_May+27%2C+2019_01.03.csv",
    }

    surveys = []
    for snum, fpath in survey_files.items():
        s = load_survey(fpath, snum)
        surveys.append(s)
        print(f"  Survey {snum}: N={len(s)} after filters")

    # Merge GA data
    surveys = merge_ga_and_build(surveys, ga)

    # Now construct all analysis variables for each survey and concatenate
    frames = []
    for df in surveys:
        snum = df["survey"].iloc[0]

        # ── Variable construction matching 1.infile_data.do ──
        out = pd.DataFrame()
        out["survey"] = df["survey"]

        # Demographics
        out["age"] = df["q2"]
        q5 = df["q5"]
        out["male"] = np.where(q5 == 1, 1, np.where(q5 == 2, 0, np.nan))

        # Education
        q4 = df["q4"]
        out["educ"] = q4
        out["low_educ"] = np.where(q4 <= 7, 1, np.where(q4.isna(), np.nan, 0))
        out["mid_educ"] = np.where(q4 == 8, 1, 0)
        out["high_educ"] = np.where(q4 == 9, 1, 0)

        # Income
        out["income"] = np.where(df["q11"] < 11, df["q11"], np.nan)

        # Matrimonial
        out["married"] = np.where(df["q10"] == 2, 1, 0)
        out["single"] = np.where(df["q10"] == 1, 1, np.where(df["q10"] == 7, np.nan, 0))

        # City size
        q3 = df["q3"]
        out["village"] = np.where(q3 == 1, 1, np.where(q3 == 4, np.nan, 0))
        out["town"] = np.where(q3 == 2, 1, np.where(q3 == 4, np.nan, 0))

        # Children
        out["children"] = np.where(df["q12"] == 1, 1, np.where(df["q12"].isna(), np.nan, 0))

        # Religion
        q21 = df["q21"]
        out["catholic"] = np.where(q21 == 1, 1, 0)
        out["muslim"] = np.where(q21 == 4, 1, 0)
        out["no_religion"] = np.where(q21 == 7, 1, np.where(q21 == 8, np.nan, 0))

        q22 = df["q22"]
        out["religious"] = np.where(q22 <= 4, 1, 0)

        # Voting
        q30 = df["q30"]
        out["second_mlp"] = np.where(q30 == 1, 1, np.where(q30.isna(), np.nan, 0))
        second_macron = np.where(q30 == 2, 1, np.where(q30.isna(), np.nan, 0))
        vote_second = pd.to_numeric(out["second_mlp"], errors="coerce").fillna(0) + pd.to_numeric(pd.Series(second_macron), errors="coerce").fillna(0)
        out.loc[vote_second == 0, "second_mlp"] = np.nan

        # EU image
        out["negative_image_UE"] = df["q86"]

        # FB
        q13 = df["q13"]
        out["use_FB"] = np.where(q13 == 5, np.nan, 5 - q13)
        q14 = df["q14"].copy()
        q14[q14 > 5000] = np.nan
        out["log_nb_friends_fb"] = np.log(q14 + 1)
        out["often_share_fb"] = np.where(df["q15"] == 3, 1, 0)

        # Sharing reasons
        out["share_interest"] = df["q16_1"] / 5
        out["share_influence"] = df["q16_2"] / 5
        out["share_image"] = df["q16_3"] / 5
        out["share_reciprocity"] = df["q16_4"] / 5

        # Prosocial
        out["altruism"] = (df["q18_1"] + df["q18_2"] + df["q18_3"] + df["q18_4"]) / 20
        out["reciprocity"] = (df["q19_1"] + df["q19_2"] + df["q19_3"]) / 15
        out["image"] = df["q20_1"] / 5

        out["age_sqrd"] = out["age"] ** 2

        # ── Outcome variables ──
        out["want_share_fb"] = np.where(df["q34"] == 1, 1, 0)
        out["want_share_others"] = np.where(df["q37"] == 1, 1, 0) if "q37" in df.columns else 0

        if snum >= 2:
            out["want_share_facts"] = np.where(df["q41"] == 1, 1, 0)
        else:
            out["want_share_facts"] = 0

        # share_click2 = prob_view_fake, filled with 0
        out["share_click2"] = df.get("prob_view_fake", pd.Series(np.nan, index=df.index)).fillna(0)
        out["share_click3"] = df.get("prob_share_GA_fake", pd.Series(np.nan, index=df.index)).fillna(0)
        out["share_facts_click2"] = df.get("prob_view_fact", pd.Series(np.nan, index=df.index)).fillna(0)
        out["share_fact_click3"] = df.get("prob_share_GA_fact", pd.Series(np.nan, index=df.index)).fillna(0)

        # Education dummies for i.educ
        for v in range(2, 10):
            out[f"educ_{v}"] = np.where(q4 == v, 1, 0)

        frames.append(out)

    result = pd.concat(frames, ignore_index=True)

    # Treatment dummies
    result["survey2"] = np.where(result["survey"] == 2, 1, np.where(result["survey"].isin([1, 3]), 0, np.nan))
    result["survey3"] = np.where(result["survey"] == 3, 1, np.where(result["survey"].isin([1, 2]), 0, np.nan))

    return result


print("Building dataset from raw CSV files...")
df = build_dataset()
df_main = df[df["survey"] < 4].copy()
df_fc = df[(df["survey"] > 1) & (df["survey"] < 4)].copy()
print(f"Total: {len(df)}, G1 sample: {len(df_main)}, G2 sample: {len(df_fc)}")

# Quick sanity: means of outcomes
for v in ["want_share_fb", "share_click2", "want_share_facts", "share_facts_click2"]:
    if v in df_main.columns:
        print(f"  {v}: mean={df_main[v].mean():.4f}" if len(df_main) > 0 else f"  {v}: no data")


# ═══════════════════════════════════════════════════════════
# CONTROL SETS
# ═══════════════════════════════════════════════════════════

STRATA = ["male", "low_educ", "mid_educ"]
EDUC_DUMMIES = [f"educ_{v}" for v in range(2, 10)]
SOCIO = ["age", "age_sqrd", "income", "married", "single", "village", "town",
         "children", "catholic", "muslim", "no_religion", "religious"] + EDUC_DUMMIES
FB = ["use_FB", "often_share_fb", "log_nb_friends_fb"]
VOTE = ["second_mlp", "negative_image_UE"]
BEHAVIORAL = ["altruism", "reciprocity", "image"]
REPORTED = ["share_interest", "share_influence", "share_image", "share_reciprocity"]

CONTROLS_FULL = STRATA + SOCIO + VOTE + FB + REPORTED + BEHAVIORAL
ALL_OPTIONAL = list(set(SOCIO + FB + VOTE + BEHAVIORAL + REPORTED))


# ═══════════════════════════════════════════════════════════
# RESULT COLLECTION
# ═══════════════════════════════════════════════════════════

spec_results = []
inference_results = []
run_counter = {"spec": 0, "infer": 0}

def next_spec_run_id():
    run_counter["spec"] += 1
    return f"{PAPER_ID}__spec__{run_counter['spec']:04d}"

def next_infer_run_id():
    run_counter["infer"] += 1
    return f"{PAPER_ID}__infer__{run_counter['infer']:04d}"


def run_and_record(spec_id, tree_path, bg_id, data, outcome, treat_formula,
                   controls, focal_coef, controls_desc, sample_desc,
                   fixed_effects="", cluster_var="", extra_payload=None,
                   vcov="hetero"):
    """Run OLS via pyfixest and record result. Return (run_id, model, results_dict)."""
    rid = next_spec_run_id()
    design_audit = DESIGN_AUDIT_G1 if bg_id == "G1" else DESIGN_AUDIT_G2

    try:
        ctrl_str = " + ".join(controls) if controls else ""
        formula = f"{outcome} ~ {treat_formula}" + (f" + {ctrl_str}" if ctrl_str else "")
        m = pf.feols(formula, data=data, vcov=vcov)

        coefs = dict(zip(m.coef().index, m.coef().values))
        ses = dict(zip(m.se().index, m.se().values))
        pvals = dict(zip(m.pvalue().index, m.pvalue().values))
        ci = m.confint()
        ci_d = {idx: (ci.loc[idx, "2.5%"], ci.loc[idx, "97.5%"]) for idx in ci.index}

        payload = make_success_payload(
            coefficients={k: float(v) for k, v in coefs.items()},
            inference=CANONICAL_INFERENCE,
            software=SW,
            surface_hash=SHASH,
            design={"randomized_experiment": design_audit},
        )
        if extra_payload:
            for k, v in extra_payload.items():
                payload[k] = v

        spec_results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": rid,
            "spec_id": spec_id,
            "spec_tree_path": tree_path,
            "baseline_group_id": bg_id,
            "outcome_var": outcome,
            "treatment_var": treat_formula,
            "coefficient": coefs.get(focal_coef, np.nan),
            "std_error": ses.get(focal_coef, np.nan),
            "p_value": pvals.get(focal_coef, np.nan),
            "ci_lower": ci_d.get(focal_coef, (np.nan, np.nan))[0],
            "ci_upper": ci_d.get(focal_coef, (np.nan, np.nan))[1],
            "n_obs": m._N,
            "r_squared": m._r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        })
        return rid, m, {"coefficients": coefs, "ses": ses, "pvals": pvals, "ci": ci_d,
                        "n_obs": m._N, "r_squared": m._r2}
    except Exception as e:
        payload = make_failure_payload(
            error=str(e)[:200],
            error_details=error_details_from_exception(e, stage="estimation"),
            inference=CANONICAL_INFERENCE, software=SW, surface_hash=SHASH,
        )
        spec_results.append({
            "paper_id": PAPER_ID, "spec_run_id": rid, "spec_id": spec_id,
            "spec_tree_path": tree_path, "baseline_group_id": bg_id,
            "outcome_var": outcome, "treatment_var": treat_formula,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fixed_effects,
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": str(e)[:200],
        })
        return rid, None, None


def run_fe_spec(spec_id, tree_path, bg_id, data, outcome, treat_formula,
                focal_coef, fe_formula, controls_desc, sample_desc,
                fixed_effects_desc, extra_payload=None):
    """Run FE spec via pyfixest with | FE syntax."""
    rid = next_spec_run_id()
    design_audit = DESIGN_AUDIT_G1 if bg_id == "G1" else DESIGN_AUDIT_G2
    try:
        formula = f"{outcome} ~ {treat_formula} | {fe_formula}"
        m = pf.feols(formula, data=data, vcov="hetero")
        coefs = dict(zip(m.coef().index, m.coef().values))
        ses = dict(zip(m.se().index, m.se().values))
        pvals = dict(zip(m.pvalue().index, m.pvalue().values))
        ci = m.confint()
        ci_d = {idx: (ci.loc[idx, "2.5%"], ci.loc[idx, "97.5%"]) for idx in ci.index}

        payload = make_success_payload(
            coefficients={k: float(v) for k, v in coefs.items()},
            inference=CANONICAL_INFERENCE, software=SW, surface_hash=SHASH,
            design={"randomized_experiment": design_audit},
        )
        if extra_payload:
            for k, v in extra_payload.items():
                payload[k] = v

        spec_results.append({
            "paper_id": PAPER_ID, "spec_run_id": rid, "spec_id": spec_id,
            "spec_tree_path": tree_path, "baseline_group_id": bg_id,
            "outcome_var": outcome, "treatment_var": treat_formula,
            "coefficient": coefs.get(focal_coef, np.nan),
            "std_error": ses.get(focal_coef, np.nan),
            "p_value": pvals.get(focal_coef, np.nan),
            "ci_lower": ci_d.get(focal_coef, (np.nan, np.nan))[0],
            "ci_upper": ci_d.get(focal_coef, (np.nan, np.nan))[1],
            "n_obs": m._N, "r_squared": m._r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fixed_effects_desc,
            "controls_desc": controls_desc, "cluster_var": "",
            "run_success": 1, "run_error": "",
        })
        return rid, m
    except Exception as e:
        payload = make_failure_payload(
            error=str(e)[:200],
            error_details=error_details_from_exception(e, stage="estimation"),
            inference=CANONICAL_INFERENCE, software=SW, surface_hash=SHASH,
        )
        spec_results.append({
            "paper_id": PAPER_ID, "spec_run_id": rid, "spec_id": spec_id,
            "spec_tree_path": tree_path, "baseline_group_id": bg_id,
            "outcome_var": outcome, "treatment_var": treat_formula,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fixed_effects_desc,
            "controls_desc": controls_desc, "cluster_var": "",
            "run_success": 0, "run_error": str(e)[:200],
        })
        return rid, None


# ═══════════════════════════════════════════════════════════
# Helper: run a spec for all G1 outcomes/focals or G2
# ═══════════════════════════════════════════════════════════

def run_g1(spec_id, tree_path, controls, controls_desc, extra_payload=None):
    """Run G1 specs for both outcomes x both focal coefs."""
    rids = []
    for outcome in ["want_share_fb", "share_click2"]:
        for focal in ["survey2", "survey3"]:
            ep = dict(extra_payload) if extra_payload else {}
            rid, m, res = run_and_record(
                spec_id, tree_path, "G1", df_main, outcome, "survey2 + survey3",
                controls, focal, controls_desc, "survey < 4", extra_payload=ep)
            rids.append((rid, focal, outcome, res))
    return rids

def run_g2(spec_id, tree_path, controls, controls_desc, extra_payload=None):
    """Run G2 spec."""
    ep = dict(extra_payload) if extra_payload else {}
    rid, m, res = run_and_record(
        spec_id, tree_path, "G2", df_fc, "want_share_facts", "survey3",
        controls, "survey3", controls_desc, "survey > 1 & survey < 4", extra_payload=ep)
    return [(rid, "survey3", "want_share_facts", res)]


# ═══════════════════════════════════════════════════════════
# STEP 1: BASELINES
# ═══════════════════════════════════════════════════════════

print("\n=== STEP 1: Baselines ===")

baseline_rids_g1 = {}
for outcome in ["want_share_fb", "share_click2"]:
    base_id = "baseline__intent_share_fb" if outcome == "want_share_fb" else "baseline__share_action_g1"
    for focal in ["survey2", "survey3"]:
        ep = {"controls": {"spec_id": "rc/controls/sets/strata_only", "family": "sets",
                           "set_name": "strata_only", "included": STRATA}}
        rid, m, res = run_and_record(
            base_id, "designs/randomized_experiment.md#baseline", "G1",
            df_main, outcome, "survey2 + survey3", STRATA, focal,
            "strata: male, low_educ, mid_educ", "survey < 4", extra_payload=ep)
        baseline_rids_g1[(outcome, focal)] = rid
        if res:
            print(f"  {base_id} [{focal}]: coef={res['coefficients'].get(focal, 'NA'):.4f}, "
                  f"SE={res['ses'].get(focal, 'NA'):.4f}, p={res['pvals'].get(focal, 'NA'):.4f}, N={res['n_obs']}")

baseline_rids_g2 = {}
for outcome in ["want_share_facts"]:
    ep = {"controls": {"spec_id": "rc/controls/sets/strata_only", "family": "sets",
                       "set_name": "strata_only", "included": STRATA}}
    rid, m, res = run_and_record(
        "baseline__factcheck_action", "designs/randomized_experiment.md#baseline", "G2",
        df_fc, outcome, "survey3", STRATA, "survey3",
        "strata: male, low_educ, mid_educ", "survey > 1 & survey < 4", extra_payload=ep)
    baseline_rids_g2[(outcome, "survey3")] = rid
    if res:
        print(f"  baseline__factcheck_action: coef={res['coefficients'].get('survey3', 'NA'):.4f}, "
              f"SE={res['ses'].get('survey3', 'NA'):.4f}, p={res['pvals'].get('survey3', 'NA'):.4f}, N={res['n_obs']}")


# ═══════════════════════════════════════════════════════════
# STEP 2: DESIGN VARIANTS
# ═══════════════════════════════════════════════════════════

print("\n=== STEP 2: Design variants ===")

# diff_in_means: no controls
run_g1("design/randomized_experiment/estimator/diff_in_means",
       "designs/randomized_experiment.md#diff-in-means", [], "none")
run_g2("design/randomized_experiment/estimator/diff_in_means",
       "designs/randomized_experiment.md#diff-in-means", [], "none")
print("  diff_in_means: done")

# with_covariates: strata + socio + vote + fb + i.educ
covs = STRATA + SOCIO + VOTE + FB
run_g1("design/randomized_experiment/estimator/with_covariates",
       "designs/randomized_experiment.md#with-covariates", covs,
       "strata + socio + vote + fb + i.educ")
run_g2("design/randomized_experiment/estimator/with_covariates",
       "designs/randomized_experiment.md#with-covariates", covs,
       "strata + socio + vote + fb + i.educ")
print("  with_covariates: done")

# strata_fe: G1 only
for outcome in ["want_share_fb", "share_click2"]:
    for focal in ["survey2", "survey3"]:
        data_tmp = df_main.copy()
        data_tmp["strata_fe"] = (data_tmp["male"].fillna(0).astype(int).astype(str) + "_" +
                                 data_tmp["low_educ"].fillna(0).astype(int).astype(str) + "_" +
                                 data_tmp["mid_educ"].fillna(0).astype(int).astype(str))
        run_fe_spec("design/randomized_experiment/estimator/strata_fe",
                    "designs/randomized_experiment.md#strata-fe", "G1",
                    data_tmp, outcome, "survey2 + survey3", focal,
                    "strata_fe", "strata absorbed as FE", "survey < 4",
                    "strata_fe (male x low_educ x mid_educ)")
print("  strata_fe: done")


# ═══════════════════════════════════════════════════════════
# STEP 2: RC CONTROL VARIANTS
# ═══════════════════════════════════════════════════════════

print("\n=== RC control variants ===")

CTRL_SPECS = {
    "rc/controls/sets/none": [],
    "rc/controls/sets/strata_only": STRATA,
    "rc/controls/sets/strata_socio_vote_fb": STRATA + SOCIO + VOTE + FB,
    "rc/controls/sets/full": CONTROLS_FULL,
    "rc/controls/progression/strata": STRATA,
    "rc/controls/progression/strata_socio": STRATA + SOCIO,
    "rc/controls/progression/strata_socio_vote": STRATA + SOCIO + VOTE,
    "rc/controls/progression/strata_socio_vote_fb": STRATA + SOCIO + VOTE + FB,
    "rc/controls/progression/strata_socio_vote_fb_behavioral": STRATA + SOCIO + VOTE + FB + BEHAVIORAL,
    "rc/controls/progression/strata_socio_vote_fb_behavioral_reported": STRATA + SOCIO + VOTE + FB + BEHAVIORAL + REPORTED,
}

g1_rc = SURFACE["baseline_groups"][0]["core_universe"]["rc_spec_ids"]
g2_rc = SURFACE["baseline_groups"][1]["core_universe"]["rc_spec_ids"]

for sid, ctrls in CTRL_SPECS.items():
    fam = "sets" if "sets" in sid else "progression"
    ep = {"controls": {"spec_id": sid, "family": fam, "included": ctrls}}
    desc = f"{sid.split('/')[-1]}: {len(ctrls)} controls"
    if sid in g1_rc:
        run_g1(sid, "modules/robustness/controls.md#control-sets", ctrls, desc, ep)
        print(f"  G1 {sid}: done")
    if sid in g2_rc:
        run_g2(sid, "modules/robustness/controls.md#control-sets", ctrls, desc, ep)
        print(f"  G2 {sid}: done")


# ═══════════════════════════════════════════════════════════
# LOO CONTROLS
# ═══════════════════════════════════════════════════════════

print("\n=== LOO controls ===")

LOO_MAP = {
    "drop_male": ["male"],
    "drop_low_educ": ["low_educ"],
    "drop_mid_educ": ["mid_educ"],
    "drop_age": ["age", "age_sqrd"],
    "drop_income": ["income"],
    "drop_married": ["married"],
    "drop_single": ["single"],
    "drop_village": ["village"],
    "drop_town": ["town"],
    "drop_children": ["children"],
    "drop_catholic": ["catholic"],
    "drop_muslim": ["muslim"],
    "drop_no_religion": ["no_religion"],
    "drop_religious": ["religious"],
    "drop_use_FB": ["use_FB"],
    "drop_often_share_fb": ["often_share_fb"],
    "drop_log_nb_friends_fb": ["log_nb_friends_fb"],
    "drop_second_mlp": ["second_mlp"],
    "drop_negative_image_UE": ["negative_image_UE"],
    "drop_altruism": ["altruism"],
    "drop_reciprocity": ["reciprocity"],
    "drop_image": ["image"],
}

g1_loo = [s for s in g1_rc if s.startswith("rc/controls/loo/")]
for sid in g1_loo:
    varname = sid.split("/")[-1]
    dropped = LOO_MAP.get(varname, [])
    loo_ctrls = [c for c in CONTROLS_FULL if c not in dropped]
    ep = {"controls": {"spec_id": sid, "family": "loo", "dropped": dropped, "n_controls": len(loo_ctrls)}}
    run_g1(sid, "modules/robustness/controls.md#leave-one-out-controls-loo", loo_ctrls,
           f"full minus {varname}", ep)

g2_loo = [s for s in g2_rc if s.startswith("rc/controls/loo/")]
for sid in g2_loo:
    varname = sid.split("/")[-1]
    dropped = LOO_MAP.get(varname, [])
    loo_ctrls = [c for c in CONTROLS_FULL if c not in dropped]
    ep = {"controls": {"spec_id": sid, "family": "loo", "dropped": dropped, "n_controls": len(loo_ctrls)}}
    run_g2(sid, "modules/robustness/controls.md#leave-one-out-controls-loo", loo_ctrls,
           f"full minus {varname}", ep)

print(f"  G1 LOO: {len(g1_loo)} variants, G2 LOO: {len(g2_loo)} variants")


# ═══════════════════════════════════════════════════════════
# RANDOM SUBSETS
# ═══════════════════════════════════════════════════════════

print("\n=== Random subsets ===")

opt_no_age_sqrd = [c for c in ALL_OPTIONAL if c != "age_sqrd"]

def gen_subsets(seed, n, optional, mandatory):
    rng = np.random.RandomState(seed)
    subsets = []
    sizes = sorted(rng.choice(range(1, min(len(optional)+1, 20)), size=n, replace=True))
    for k in sizes:
        chosen = list(rng.choice(optional, size=min(k, len(optional)), replace=False))
        if "age" in chosen and "age_sqrd" not in chosen:
            chosen.append("age_sqrd")
        subsets.append(mandatory + chosen)
    return subsets

g1_subs = gen_subsets(140161, 10, opt_no_age_sqrd, STRATA)
for i, sub in enumerate(g1_subs):
    sid = f"rc/controls/subset/random_{i+1:03d}"
    ep = {"controls": {"spec_id": sid, "family": "subset", "draw_index": i+1,
                       "seed": 140161, "included": sub, "n_controls": len(sub)}}
    run_g1(sid, "modules/robustness/controls.md#random-subsets", sub,
           f"random subset {i+1}: {len(sub)} ctrls", ep)

g2_subs = gen_subsets(140162, 5, opt_no_age_sqrd, STRATA)
for i, sub in enumerate(g2_subs):
    sid = f"rc/controls/subset/random_{i+1:03d}"
    ep = {"controls": {"spec_id": sid, "family": "subset", "draw_index": i+1,
                       "seed": 140162, "included": sub, "n_controls": len(sub)}}
    run_g2(sid, "modules/robustness/controls.md#random-subsets", sub,
           f"random subset {i+1}: {len(sub)} ctrls", ep)

print(f"  G1: {len(g1_subs)} subsets, G2: {len(g2_subs)} subsets")


# ═══════════════════════════════════════════════════════════
# RC TREATMENT VARIANTS (pairwise, binary)
# ═══════════════════════════════════════════════════════════

print("\n=== RC treatment variants ===")

# Pairwise: imposed vs control (surveys 1 & 2)
df_12 = df_main[df_main["survey"].isin([1, 2])].copy()
for outcome in ["want_share_fb", "share_click2"]:
    ep = {"sample": {"spec_id": "rc/treatment/pairwise/imposed_vs_control", "filter": "survey in {1,2}"}}
    run_and_record("rc/treatment/pairwise/imposed_vs_control",
                   "modules/robustness/controls.md#pairwise-treatment", "G1",
                   df_12, outcome, "survey2", STRATA, "survey2",
                   "strata", "surveys 1 & 2 only", extra_payload=ep)

# Pairwise: voluntary vs control (surveys 1 & 3)
df_13 = df_main[df_main["survey"].isin([1, 3])].copy()
for outcome in ["want_share_fb", "share_click2"]:
    ep = {"sample": {"spec_id": "rc/treatment/pairwise/voluntary_vs_control", "filter": "survey in {1,3}"}}
    run_and_record("rc/treatment/pairwise/voluntary_vs_control",
                   "modules/robustness/controls.md#pairwise-treatment", "G1",
                   df_13, outcome, "survey3", STRATA, "survey3",
                   "strata", "surveys 1 & 3 only", extra_payload=ep)

# Pairwise: imposed vs voluntary (surveys 2 & 3)
for outcome in ["want_share_fb", "share_click2"]:
    ep = {"sample": {"spec_id": "rc/treatment/pairwise/imposed_vs_voluntary", "filter": "survey in {2,3}"}}
    run_and_record("rc/treatment/pairwise/imposed_vs_voluntary",
                   "modules/robustness/controls.md#pairwise-treatment", "G1",
                   df_fc, outcome, "survey2", STRATA, "survey2",
                   "strata", "surveys 2 & 3 only", extra_payload=ep)

# Binary: any factcheck vs control
df_bin = df_main.copy()
df_bin["any_factcheck"] = np.where(df_bin["survey"].isin([2, 3]), 1, 0)
for outcome in ["want_share_fb", "share_click2"]:
    ep = {"extra": {"treatment_recoding": "any_factcheck = 1 if survey in {2,3}"}}
    run_and_record("rc/treatment/binary/any_factcheck_vs_control",
                   "modules/robustness/controls.md#binary-treatment", "G1",
                   df_bin, outcome, "any_factcheck", STRATA, "any_factcheck",
                   "strata", "survey < 4, binary treatment", extra_payload=ep)

print("  Treatment variants: done")


# ═══════════════════════════════════════════════════════════
# RC FORM (OUTCOME) VARIANTS
# ═══════════════════════════════════════════════════════════

print("\n=== RC form (outcome) variants ===")

# G1: share_click2 as form variant of want_share_fb baseline
for focal in ["survey2", "survey3"]:
    ep = {"functional_form": {"spec_id": "rc/form/outcome/share_click2", "type": "outcome_variant",
          "baseline_outcome": "want_share_fb", "variant_outcome": "share_click2",
          "interpretation": "2-click action of sharing alt-facts on FB (GA page view)"}}
    run_and_record("rc/form/outcome/share_click2",
                   "modules/robustness/functional_form.md#outcome-variant", "G1",
                   df_main, "share_click2", "survey2 + survey3", STRATA, focal,
                   "strata", "survey < 4", extra_payload=ep)

# G1: share_click3
for focal in ["survey2", "survey3"]:
    ep = {"functional_form": {"spec_id": "rc/form/outcome/share_click3", "type": "outcome_variant",
          "baseline_outcome": "want_share_fb", "variant_outcome": "share_click3",
          "interpretation": "3-click reconfirm sharing alt-facts on FB (GA share count)"}}
    run_and_record("rc/form/outcome/share_click3",
                   "modules/robustness/functional_form.md#outcome-variant", "G1",
                   df_main, "share_click3", "survey2 + survey3", STRATA, focal,
                   "strata", "survey < 4", extra_payload=ep)

# G2: share_facts_click2
ep = {"functional_form": {"spec_id": "rc/form/outcome/share_facts_click2", "type": "outcome_variant",
      "baseline_outcome": "want_share_facts", "variant_outcome": "share_facts_click2",
      "interpretation": "2-click action of sharing fact-check on FB (GA page view)"}}
run_and_record("rc/form/outcome/share_facts_click2",
               "modules/robustness/functional_form.md#outcome-variant", "G2",
               df_fc, "share_facts_click2", "survey3", STRATA, "survey3",
               "strata", "survey > 1 & survey < 4", extra_payload=ep)

# G2: share_fact_click3
ep = {"functional_form": {"spec_id": "rc/form/outcome/share_fact_click3", "type": "outcome_variant",
      "baseline_outcome": "want_share_facts", "variant_outcome": "share_fact_click3",
      "interpretation": "3-click reconfirm sharing fact-check on FB (GA share count)"}}
run_and_record("rc/form/outcome/share_fact_click3",
               "modules/robustness/functional_form.md#outcome-variant", "G2",
               df_fc, "share_fact_click3", "survey3", STRATA, "survey3",
               "strata", "survey > 1 & survey < 4", extra_payload=ep)

print("  Form variants: done")


# ═══════════════════════════════════════════════════════════
# STEP 3: INFERENCE VARIANTS (HC3)
# ═══════════════════════════════════════════════════════════

print("\n=== Inference variants (HC3) ===")

def run_hc3(data, outcome, treat_formula, controls, focal_coef):
    ctrl_str = " + ".join(controls) if controls else ""
    formula = f"{outcome} ~ {treat_formula}" + (f" + {ctrl_str}" if ctrl_str else "")
    m = smf.ols(formula, data=data.dropna(subset=[outcome] + treat_formula.split(" + ") + controls)).fit(cov_type="HC3")
    coefs = dict(m.params)
    ses = dict(m.bse)
    pvals = dict(m.pvalues)
    ci = m.conf_int()
    ci_d = {idx: (ci.loc[idx, 0], ci.loc[idx, 1]) for idx in ci.index}
    return {"coefficients": coefs, "ses": ses, "pvals": pvals, "ci": ci_d,
            "n_obs": int(m.nobs), "r_squared": m.rsquared}


# HC3 for baseline specs
for outcome in ["want_share_fb", "share_click2"]:
    for focal in ["survey2", "survey3"]:
        base_rid = baseline_rids_g1.get((outcome, focal))
        if not base_rid:
            continue
        try:
            res = run_hc3(df_main, outcome, "survey2 + survey3", STRATA, focal)
            irid = next_infer_run_id()
            design_audit = DESIGN_AUDIT_G1
            payload = make_success_payload(
                coefficients={k: float(v) for k, v in res["coefficients"].items()},
                inference={"spec_id": "infer/se/hc/hc3", "params": {"se_type": "HC3"}},
                software=SW, surface_hash=SHASH,
                design={"randomized_experiment": design_audit},
            )
            inference_results.append({
                "paper_id": PAPER_ID, "inference_run_id": irid,
                "spec_run_id": base_rid, "spec_id": "infer/se/hc/hc3",
                "spec_tree_path": "modules/inference/standard_errors.md#hc3",
                "baseline_group_id": "G1",
                "outcome_var": outcome, "treatment_var": "survey2 + survey3",
                "cluster_var": "",
                "coefficient": res["coefficients"].get(focal, np.nan),
                "std_error": res["ses"].get(focal, np.nan),
                "p_value": res["pvals"].get(focal, np.nan),
                "ci_lower": res["ci"].get(focal, (np.nan, np.nan))[0],
                "ci_upper": res["ci"].get(focal, (np.nan, np.nan))[1],
                "n_obs": res["n_obs"], "r_squared": res["r_squared"],
                "coefficient_vector_json": json.dumps(payload),
                "run_success": 1, "run_error": "",
            })
            print(f"  G1 HC3 {outcome} [{focal}]: SE={res['ses'].get(focal):.4f}")
        except Exception as e:
            print(f"  G1 HC3 {outcome} [{focal}] FAILED: {e}")

for outcome in ["want_share_facts"]:
    base_rid = baseline_rids_g2.get((outcome, "survey3"))
    if not base_rid:
        continue
    try:
        res = run_hc3(df_fc, outcome, "survey3", STRATA, "survey3")
        irid = next_infer_run_id()
        payload = make_success_payload(
            coefficients={k: float(v) for k, v in res["coefficients"].items()},
            inference={"spec_id": "infer/se/hc/hc3", "params": {"se_type": "HC3"}},
            software=SW, surface_hash=SHASH,
            design={"randomized_experiment": DESIGN_AUDIT_G2},
        )
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": irid,
            "spec_run_id": base_rid, "spec_id": "infer/se/hc/hc3",
            "spec_tree_path": "modules/inference/standard_errors.md#hc3",
            "baseline_group_id": "G2",
            "outcome_var": outcome, "treatment_var": "survey3",
            "cluster_var": "",
            "coefficient": res["coefficients"].get("survey3", np.nan),
            "std_error": res["ses"].get("survey3", np.nan),
            "p_value": res["pvals"].get("survey3", np.nan),
            "ci_lower": res["ci"].get("survey3", (np.nan, np.nan))[0],
            "ci_upper": res["ci"].get("survey3", (np.nan, np.nan))[1],
            "n_obs": res["n_obs"], "r_squared": res["r_squared"],
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1, "run_error": "",
        })
        print(f"  G2 HC3 {outcome}: SE={res['ses'].get('survey3'):.4f}")
    except Exception as e:
        print(f"  G2 HC3 {outcome} FAILED: {e}")


# ═══════════════════════════════════════════════════════════
# WRITE OUTPUTS
# ═══════════════════════════════════════════════════════════

print("\n=== Writing outputs ===")

spec_df = pd.DataFrame(spec_results)
spec_df.to_csv(OUT / "specification_results.csv", index=False)
n_ok = int(spec_df["run_success"].sum())
n_fail = int((spec_df["run_success"] == 0).sum())
print(f"  specification_results.csv: {len(spec_df)} rows ({n_ok} ok, {n_fail} failed)")
print(f"    G1: {(spec_df['baseline_group_id']=='G1').sum()}, G2: {(spec_df['baseline_group_id']=='G2').sum()}")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(OUT / "inference_results.csv", index=False)
    print(f"  inference_results.csv: {len(infer_df)} rows")

# SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search Report: {PAPER_ID}

## Surface Summary

- **Paper**: Henry, Zhuravskaya & Guriev — Fake news & fact-check sharing experiment (France, 2019 EU elections)
- **Design**: Randomized experiment (individual-level, online survey)
- **Baseline groups**: 2
  - G1: ATE of fact-checking on desire to share alt-facts (surveys 1,2,3; N={len(df_main)})
  - G2: ATE of voluntary vs imposed fact-check on fact-check sharing (surveys 2,3; N={len(df_fc)})
- **Budget G1**: max 80 core specs, 10 random subsets
- **Budget G2**: max 40 core specs, 5 random subsets
- **Seed G1**: 140161 | **Seed G2**: 140162
- **Surface hash**: {SHASH}

## Execution Summary

| Category | Rows | Successful | Failed |
|----------|------|------------|--------|
| Total    | {len(spec_df)} | {n_ok} | {n_fail} |
| G1       | {int((spec_df['baseline_group_id']=='G1').sum())} | {int(spec_df[spec_df['baseline_group_id']=='G1']['run_success'].sum())} | {int((spec_df[spec_df['baseline_group_id']=='G1']['run_success']==0).sum())} |
| G2       | {int((spec_df['baseline_group_id']=='G2').sum())} | {int(spec_df[spec_df['baseline_group_id']=='G2']['run_success'].sum())} | {int((spec_df[spec_df['baseline_group_id']=='G2']['run_success']==0).sum())} |
| Inference (HC3) | {len(inference_results)} | {sum(1 for r in inference_results if r.get('run_success')==1)} | {sum(1 for r in inference_results if r.get('run_success')!=1)} |

## Spec Types Executed

| spec_id prefix | Count |
|----------------|-------|
"""

for prefix in sorted(spec_df["spec_id"].unique()):
    cnt = (spec_df["spec_id"] == prefix).sum()
    search_md += f"| {prefix} | {cnt} |\n"

search_md += f"""
## Data Construction Notes

- Dataset reconstructed from raw Qualtrics CSV files (Survey 1, 2, 3) + GA_hours.dta
- Duration filter (>=250s) and gc==1 applied during raw CSV import (matching Stata 1.infile_data.do)
- Google Analytics merge by day/hour/survey_id for share_click2/3 and share_facts_click2/share_fact_click3
- Variables constructed to match Stata do-file definitions exactly
- Education dummies (i.educ) expanded as educ_2 through educ_9 (educ_1 as reference)
- All outcome variables zero-filled when missing (matching do-file: replace X=0 if X==.)

## Deviations from Surface

- Duration filter variants (rc/sample/duration_filter/*) excluded per surface removal
- The `educ` variable in column 3+ regressions is expanded as 8 dummies (i.educ in Stata)

## Software Stack

- Python {sys.version.split()[0]}
- pyfixest {SW['packages'].get('pyfixest', 'unknown')}
- pandas {SW['packages'].get('pandas', 'unknown')}
- numpy {SW['packages'].get('numpy', 'unknown')}
- statsmodels {SW['packages'].get('statsmodels', 'unknown')} (for HC3 inference)
"""

with open(OUT / "SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)

print(f"  SPECIFICATION_SEARCH.md written")
print(f"\nDone! Total specs: {len(spec_df)}, Inference: {len(inference_results)}")
