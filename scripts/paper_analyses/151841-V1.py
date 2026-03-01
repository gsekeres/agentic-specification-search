"""
Specification Search Script for Beaman, Magruder, & Robinson (2023)
"Peer Ranking and Selective Capital Allocation"
American Economic Review

Paper ID: 151841-V1

Replicates the full Stata data pipeline from raw survey rounds, then runs 50+
specification variants around the main Table 2 claim: Winner*Rank in panel FE.

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json, sys, os, hashlib, traceback, warnings
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash, software_block
)

PAPER_ID = "151841-V1"
DATA_DIR = os.path.join(REPO_ROOT, "data/downloads/extracted/151841-V1")
RAW_DIR = os.path.join(DATA_DIR, "data/1_raw")
OUTPUT_DIR = DATA_DIR

with open(os.path.join(DATA_DIR, "SPECIFICATION_SURFACE.json")) as f:
    surface_obj = json.load(f)
SURFACE_HASH = compute_surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================================
# DATA PIPELINE: Replicate 3-step Stata data cleaning
# ============================================================================
def recode_missing(s, codes=(-555, -666, -999, -998, -777)):
    """Replace Stata missing codes with NaN, -777 -> 0."""
    s = s.copy()
    for c in codes:
        if c == -777:
            s = s.replace(c, 0)
        else:
            s = s.replace(c, np.nan)
    return s

def safe_col(df, col, default=np.nan):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

print("=== STEP 1: Append survey rounds ===")
rounds_raw = {}
for r in range(1, 6):
    fp = os.path.join(RAW_DIR, f"survey rounds/Survey Round {r}.dta")
    d = pd.read_stata(fp, convert_categoricals=False)
    for c in d.columns:
        if d[c].dtype == np.float32:
            d[c] = d[c].astype(np.float64)
    d["Survey_Version"] = r
    rounds_raw[r] = d

# Round 1: keep only business owners
r1 = rounds_raw[1][rounds_raw[1]["businesslist_5"].notna()].copy()
biz = pd.concat([r1] + [rounds_raw[r].copy() for r in range(2, 6)], ignore_index=True, sort=False)
print(f"Appended panel: {len(biz)} rows")

# Create balanced panel: expand (Id, Business_Owner_Id, Business_Name) x 5 rounds
key_combos = biz[["Id", "Business_Owner_Id", "Business_Name"]].drop_duplicates()
balanced = key_combos.loc[key_combos.index.repeat(5)].reset_index(drop=True)
balanced["Survey_Version"] = np.tile(np.arange(1, 6), len(key_combos))
biz = balanced.merge(biz, on=["Id", "Business_Owner_Id", "Business_Name", "Survey_Version"], how="left")

# Merge treatments (GroupNumber, LotteryWinner, etc.)
treats = pd.read_stata(os.path.join(RAW_DIR, "randomization and tracking/treatments and master trackers.dta"), convert_categoricals=False)
for c in treats.columns:
    if treats[c].dtype == np.float32:
        treats[c] = treats[c].astype(np.float64)
biz = biz.merge(treats[["Id","GroupNumber","Final_Randomization_Cluster","Public","Revealed",
                          "Incentives","LotteryWinner","Total_Num_Tickets",
                          "Status_Survey_1","Status_Survey_2","Status_Survey_3","Status_Survey_4","Status_Survey_5"]],
                 on="Id", how="left", suffixes=("","_t"))

# Attriter: Status_Survey_2 == 0 means attrited
biz["_att"] = np.where((biz["Status_Survey_2"] == 0) & (biz["Survey_Version"] == 1), 1, np.nan)
biz["Attriter"] = biz.groupby("Id")["_att"].transform("max").fillna(0)

# Business_Open
biz["Business_Open"] = safe_col(biz, "businesslist_5")
if "Intro_1" in biz.columns:
    biz.loc[biz["Intro_1"].notna(), "Business_Open"] = biz.loc[biz["Intro_1"].notna(), "Intro_1"]
    biz.loc[biz["Business_Open"] == 4, "Business_Open"] = 1  # transferred but open
if "q1" in biz.columns:
    mask3 = biz["Survey_Version"] >= 3
    biz.loc[mask3 & (biz["q1"] == 1) & biz["Business_Open"].isna(), "Business_Open"] = 1
    biz.loc[mask3 & (biz["q1"] != 1) & biz["q1"].notna() & biz["Business_Open"].isna(), "Business_Open"] = 0

print("=== STEP 2: Create business-level variables ===")

# Winner and Winner2
biz["Winner"] = biz["LotteryWinner"].copy()
biz.loc[biz["Survey_Version"] == 1, "Winner"] = 0
biz["Winner2"] = biz["LotteryWinner"].copy()

# Survey date and month
if "enddate" in biz.columns:
    biz["enddate"] = pd.to_datetime(biz["enddate"], errors="coerce")
    biz["survey_date"] = biz.groupby(["Id","Survey_Version"])["enddate"].transform("max")
    biz["survey_month"] = biz["survey_date"].dt.month
    biz["survey_month"] = biz["survey_month"].fillna(1).astype(int)
else:
    biz["survey_month"] = 1  # fallback

# Surveyor_Code: create group codes
if "Surveyor_Code" in biz.columns:
    biz["_sc2"] = biz["Surveyor_Code"]
    biz["Surveyor_Code"] = biz.groupby(["Id","Survey_Version"])["_sc2"].transform("max")
else:
    biz["Surveyor_Code"] = 1

# Client variable
if "client" in biz.columns:
    biz["client"] = biz["client"].fillna(0)
    biz["Client"] = biz.groupby(["Id","Business_Owner_Id"])["client"].transform("max")
else:
    biz["Client"] = 0

# --- Baseline characteristics from Round 1 ---
r1_chars = rounds_raw[1][rounds_raw[1]["businesslist_5"].notna()].copy()
# Demographics: hhroster_2=gender, hhroster_8=education, hhroster_4=age, hhroster_5=married, hhid_8=hh_size
if "hhroster_2" in r1_chars.columns:
    r1_chars["hhroster_2"] = r1_chars["hhroster_2"].replace(2, 0)  # female=0
for v, new in [("hhroster_2","_gender"), ("hhroster_8","_education"), ("hhroster_4","_age")]:
    if v in r1_chars.columns:
        r1_chars[new] = r1_chars[v]
if "hhroster_5" in r1_chars.columns:
    r1_chars["_married"] = (r1_chars["hhroster_5"] == 1).astype(float)
if "hhid_8" in r1_chars.columns:
    r1_chars["_hhsize"] = r1_chars["hhid_8"]
if "client" in r1_chars.columns:
    r1_chars["client"] = r1_chars["client"].fillna(0)
    for src, var in [("_gender","Gender_Followup"),("_education","Education_Followup"),
                     ("_age","Age_Followup"),("_married","Married_Followup"),("_hhsize","HH_Size_Followup")]:
        if src in r1_chars.columns:
            r1_chars[f"_{var}"] = np.where(r1_chars["client"]==1, r1_chars[src], np.nan)
            agg = r1_chars.groupby("Id")[f"_{var}"].max().reset_index().rename(columns={f"_{var}": var})
            biz = biz.merge(agg, on="Id", how="left", suffixes=("","_new"))
            if f"{var}_new" in biz.columns:
                biz[var] = biz[var].fillna(biz[f"{var}_new"])
                biz.drop(columns=[f"{var}_new"], inplace=True)

# Household Income
biz["Income"] = np.nan
if "hhroster_21" in biz.columns:
    biz.loc[biz["Survey_Version"]==1, "Income"] = biz.loc[biz["Survey_Version"]==1, "hhroster_21"]
if "credit1" in biz.columns:
    biz.loc[biz["Survey_Version"]==2, "Income"] = biz.loc[biz["Survey_Version"]==2, "credit1"]
if "q18" in biz.columns:
    for sv in [3,4,5]:
        biz.loc[biz["Survey_Version"]==sv, "Income"] = biz.loc[biz["Survey_Version"]==sv, "q18"]

# Profits_30Days (complex construction from business type-specific questions)
biz["Profits_30Days"] = np.nan
# Round 1-2: from RMS_24, Piecerate_19, LivestockFish profits, Construction_12, Commission_4, Farming_16
prof_vars_r12 = ["RMS_24", "Piecerate_19", "Construction_12", "Commission_4", "Farming_16"]
for v in prof_vars_r12:
    if v in biz.columns:
        biz[v] = recode_missing(biz[v])

# LivestockFish profits need aggregation
if "LivestockFish_25" in biz.columns:
    biz["LivestockFish_25"] = recode_missing(biz["LivestockFish_25"])
if "LivestockFish_33" in biz.columns:
    biz["LivestockFish_33"] = recode_missing(biz["LivestockFish_33"])
    biz["Total_LF33"] = biz.groupby(["Id","Business_Owner_Id","Business_Name","Survey_Version"])["LivestockFish_33"].transform("sum")
    biz["Total_LF33"] = biz["Total_LF33"].where(safe_col(biz, "LivestockFish_1").notna(), np.nan)
else:
    biz["Total_LF33"] = np.nan

# Combine livestock profits
biz["LF_Profits"] = np.nan
if "Total_LF33" in biz.columns and "LivestockFish_25" in biz.columns:
    biz["LF_Profits"] = biz[["Total_LF33","LivestockFish_25"]].sum(axis=1, min_count=1)

# Build Profits_30Days for rounds 1-2
existing_prof = [v for v in ["RMS_24","Piecerate_19","LF_Profits","Construction_12","Commission_4","Farming_16"] if v in biz.columns]
if existing_prof:
    biz.loc[biz["Survey_Version"].isin([1,2]), "Profits_30Days"] = biz.loc[biz["Survey_Version"].isin([1,2]), existing_prof].sum(axis=1, min_count=1)

# Rounds 3-5: q10
if "q10" in biz.columns:
    for sv in [3,4,5]:
        biz.loc[biz["Survey_Version"]==sv, "Profits_30Days"] = biz.loc[biz["Survey_Version"]==sv, "q10"]

# Set profits=0 for closed businesses
biz.loc[(biz["Business_Open"]==0) & biz["Profits_30Days"].isna(), "Profits_30Days"] = 0

# Inventory
biz["Inventory"] = np.nan
inv_r12 = ["RMS_3","RMS_4","RMS_7","Piecerate_3","LivestockFish_4","LivestockFish_5",
           "LivestockFish_6","LivestockFish_34","Construction_14"]
for v in inv_r12:
    if v in biz.columns:
        biz[v] = recode_missing(biz[v])
existing_inv = [v for v in inv_r12 if v in biz.columns]
if existing_inv:
    biz.loc[biz["Survey_Version"].isin([1,2]), "Inventory"] = biz.loc[biz["Survey_Version"].isin([1,2]), existing_inv].sum(axis=1, min_count=0)
if "q12" in biz.columns:
    for sv in [3,4,5]:
        biz.loc[biz["Survey_Version"]==sv, "Inventory"] = biz.loc[biz["Survey_Version"]==sv, "q12"]

# Owner Hours/Days
biz["Owner_Hours_Week"] = np.nan
biz["Owner_Days_Month"] = np.nan
if "hhroster_15" in biz.columns:
    biz.loc[(biz["Survey_Version"]==1) & (biz["Business_Open"]==1), "Owner_Hours_Week"] = biz.loc[(biz["Survey_Version"]==1) & (biz["Business_Open"]==1), "hhroster_15"]
if "Labor_Hours" in biz.columns:
    biz.loc[biz["Survey_Version"]==2, "Owner_Hours_Week"] = biz.loc[biz["Survey_Version"]==2, "Labor_Hours"]
if "q16" in biz.columns:
    for sv in [3,4,5]:
        biz.loc[biz["Survey_Version"]==sv, "Owner_Hours_Week"] = biz.loc[biz["Survey_Version"]==sv, "q16"]
biz["Owner_Hours_Week"] = recode_missing(biz["Owner_Hours_Week"])

if "hhroster_16" in biz.columns:
    biz.loc[(biz["Survey_Version"]==1) & (biz["Business_Open"]==1), "Owner_Days_Month"] = biz.loc[(biz["Survey_Version"]==1) & (biz["Business_Open"]==1), "hhroster_16"]
if "Labor_Day" in biz.columns:
    biz.loc[biz["Survey_Version"]==2, "Owner_Days_Month"] = biz.loc[biz["Survey_Version"]==2, "Labor_Day"]
if "q17" in biz.columns:
    for sv in [3,4,5]:
        biz.loc[biz["Survey_Version"]==sv, "Owner_Days_Month"] = biz.loc[biz["Survey_Version"]==sv, "q17"]
biz["Owner_Days_Month"] = recode_missing(biz["Owner_Days_Month"])

# Client profits/hours/days
biz["Client_Profits_30Days"] = np.where(biz["Client"]==1, biz["Profits_30Days"], np.nan)
biz["Client_Hours_Week"] = np.where(biz["Client"]==1, biz["Owner_Hours_Week"], np.nan)
biz["Client_Days_Month"] = np.where(biz["Client"]==1, biz["Owner_Days_Month"], np.nan)

# Labor variables
for v in ["Labor_1","Labor_2","Labor_3","Labor_4","Labor_5","Labor_6","Labor_7","Labor_8","Labor_9","Labor_10"]:
    if v in biz.columns:
        biz[v] = recode_missing(biz[v])
for v1, v2_list in [("Labor_1",["Labor_2"]), ("Labor_3",["Labor_4","Labor_5"]),
                     ("Labor_6",["Labor_7"]), ("Labor_8",["Labor_9","Labor_10"])]:
    if v1 in biz.columns:
        for v2 in v2_list:
            if v2 in biz.columns:
                biz.loc[biz[v1]==0, v2] = biz.loc[biz[v1]==0, v2].fillna(0)

nonhh_cols = [c for c in ["Labor_6","Labor_8"] if c in biz.columns]
biz["NonHH_Labor"] = biz[nonhh_cols].sum(axis=1, min_count=1) if nonhh_cols else np.nan
biz["NonHH_Labor_Dummy"] = (biz["NonHH_Labor"] != 0).astype(float).where(biz["NonHH_Labor"].notna() & biz["Survey_Version"].isin([1,2,5]))
nonhh_hrs_cols = [c for c in ["Labor_7","Labor_9"] if c in biz.columns]
biz["NonHH_Labor_Hrs"] = biz[nonhh_hrs_cols].sum(axis=1, min_count=1) if nonhh_hrs_cols else np.nan
hh_cols = [c for c in ["Labor_1","Labor_3"] if c in biz.columns]
biz["HH_Labor"] = biz[hh_cols].sum(axis=1, min_count=1) if hh_cols else np.nan
biz["HH_Labor_Dummy"] = (biz["HH_Labor"] != 0).astype(float).where(biz["HH_Labor"].notna() & biz["Survey_Version"].isin([1,2]))
hh_hrs_cols = [c for c in ["Labor_2","Labor_4"] if c in biz.columns]
biz["HH_Labor_Hrs"] = biz[hh_hrs_cols].sum(axis=1, min_count=1).where(biz["Survey_Version"].isin([1,2]) & biz["HH_Labor"].notna()) if hh_hrs_cols else np.nan

# Business sector classification
bus_class = pd.read_stata(os.path.join(RAW_DIR, "survey rounds/Business Classifications.dta"), convert_categoricals=False)
for c in bus_class.columns:
    if bus_class[c].dtype == np.float32:
        bus_class[c] = bus_class[c].astype(np.float64)
biz = biz.merge(bus_class[["Id","Business_Owner_Id","Business_Name","businessstatus_1_new","Business_Sector_new"]].drop_duplicates(),
                on=["Id","Business_Owner_Id","Business_Name"], how="left", suffixes=("","_bc"))

biz["B_manufacturing"] = ((biz["businessstatus_1_new"].isin([1,2])) & (biz["Survey_Version"]==1) & (biz["client"]==1) if "client" in biz.columns else (biz["Survey_Version"]==1)).astype(float)
biz["B_retail"] = ((biz["businessstatus_1_new"]==4) & (biz["Survey_Version"]==1) & (biz.get("client",0)==1)).astype(float)
biz["B_service"] = ((biz["businessstatus_1_new"].isin([3,5,6,7,10,11])) & (biz["Survey_Version"]==1) & (biz.get("client",0)==1)).astype(float)
biz["B_agriculture"] = ((biz["businessstatus_1_new"]==9) & (biz["Survey_Version"]==1) & (biz.get("client",0)==1)).astype(float)
# Propagate to household level
for v in ["B_manufacturing","B_retail","B_service","B_agriculture"]:
    biz[v] = biz.groupby("Id")[v].transform("max")

# Avg_Yrly_Profits from seasonality questions
seas_cols = [c for c in biz.columns if c.startswith("seasonality_1") and c != "seasonality_1"]
if seas_cols:
    for c in seas_cols:
        biz[c] = recode_missing(biz[c])
    biz["Avg_Yrly_Profits"] = biz[seas_cols].mean(axis=1)
else:
    biz["Avg_Yrly_Profits"] = np.nan

# B_ prefixed baseline characteristics from round 1 data
for v in ["digitspan_1","TV_hhincome_1","child_0to5","child_6to12"]:
    if v in biz.columns:
        biz[f"B_{v}"] = biz.groupby("Id")[v].transform("max")

# Total_Business_Assets (simplified - use round 1 business assets)
biz["Total_Business_Assets"] = np.nan
# For round 1, sum of all asset values
asset_val_cols = [c for c in biz.columns if "Asset_" in c and "Value" in c and "Name" not in c and "Number" not in c]
if asset_val_cols:
    for c in asset_val_cols:
        biz[c] = recode_missing(biz[c])
    biz.loc[biz["Survey_Version"]==1, "Total_Business_Assets"] = biz.loc[biz["Survey_Version"]==1, asset_val_cols].sum(axis=1, min_count=1)

# HH_Assets (simplified)
hh_asset_cols = [c for c in biz.columns if c.startswith("TV_hhasset") and c.endswith("_2")]
if hh_asset_cols:
    for c in hh_asset_cols:
        biz[c] = recode_missing(biz[c])
    biz["_hh_assets"] = biz[hh_asset_cols].sum(axis=1, min_count=1)
    biz["B_HH_Assets"] = biz.groupby("Id")["_hh_assets"].transform("max")
else:
    biz["B_HH_Assets"] = np.nan

# Baseline additional characteristics
if "Owner_Hist_1" in biz.columns:
    biz["Owner_Hist_1"] = recode_missing(biz["Owner_Hist_1"])
    tmp = biz.loc[biz.get("client",0)==1, ["Id","Owner_Hist_1"]].copy()
    agg = tmp.groupby("Id")["Owner_Hist_1"].max().reset_index().rename(columns={"Owner_Hist_1":"B_ChangeSales_2014"})
    biz = biz.merge(agg, on="Id", how="left", suffixes=("","_ch"))
else:
    biz["B_ChangeSales_2014"] = np.nan

if "Owner_Hist_2" in biz.columns:
    biz["Owner_Hist_2"] = recode_missing(biz["Owner_Hist_2"]).replace(-888, 0)
    tmp = biz.loc[biz.get("client",0)==1].copy()
    tmp["_pj5"] = (tmp["Owner_Hist_2"]==1).astype(float)
    agg = tmp.groupby("Id")["_pj5"].max().reset_index().rename(columns={"_pj5":"B_Primary_Job_5yrs"})
    biz = biz.merge(agg, on="Id", how="left", suffixes=("","_pj"))
else:
    biz["B_Primary_Job_5yrs"] = np.nan

if "Owner_Hist_3" in biz.columns:
    biz["Owner_Hist_3"] = recode_missing(biz["Owner_Hist_3"])
    tmp = biz.loc[biz.get("client",0)==1, ["Id","Owner_Hist_3"]].copy()
    agg = tmp.groupby("Id")["Owner_Hist_3"].max().reset_index().rename(columns={"Owner_Hist_3":"B_Wage_Exit_SE"})
    biz = biz.merge(agg, on="Id", how="left", suffixes=("","_we"))
else:
    biz["B_Wage_Exit_SE"] = np.nan

# Baseline aggregate variables
for v in ["Avg_Yrly_Profits"]:
    if v in biz.columns:
        biz[f"B_{v}"] = biz.groupby("Id")[v].transform("max")

# Number variables from round 1
if "hhroster_10" in biz.columns:
    biz["hhroster_10"] = biz["hhroster_10"].replace({-999:0, -777:0, 2:0})
if "hhroster_12" in biz.columns:
    biz["hhroster_12"] = biz["hhroster_12"].replace({-999:0, -777:0, 2:0})
if "hhroster_13" in biz.columns:
    biz["hhroster_13"] = biz["hhroster_13"].replace({-999:0, -777:0, 2:0})
for src, name in [("hhroster_10","B_Number_Fixed_Salary"),("hhroster_12","B_Number_Daily_Wage"),("hhroster_13","B_Number_Self_Employed")]:
    if src in biz.columns:
        biz[name] = biz.groupby("Id")[src].transform("max")
    else:
        biz[name] = np.nan
if "businesslist_5" in biz.columns:
    biz["_tnu"] = biz.groupby("Id")["businesslist_5"].transform("sum")
    biz["B_Total_Nu_Bus"] = biz.groupby("Id")["_tnu"].transform("max")
else:
    biz["B_Total_Nu_Bus"] = np.nan

# Psychometric variables
for i in range(1, 18):
    pvar = f"Psychometric_{i}"
    if pvar in biz.columns:
        biz[pvar] = recode_missing(biz[pvar])
        biz[f"B_{pvar}"] = biz.groupby("Id")[pvar].transform("max")

# Group_Size
biz["_id_sv_tag"] = ~biz.duplicated(subset=["GroupNumber","Id"])
biz["Group_Size"] = biz.groupby("GroupNumber")["_id_sv_tag"].transform("sum")
# Normalize to per-household count
id_per_group = biz.drop_duplicates(subset=["GroupNumber","Id"]).groupby("GroupNumber")["Id"].count().reset_index().rename(columns={"Id":"Group_Size"})
biz.drop(columns=["Group_Size","_id_sv_tag"], inplace=True)
biz = biz.merge(id_per_group, on="GroupNumber", how="left")

# Missing indicators
miss_vars = ["Profits_30Days","Client_Profits_30Days","Income","Inventory",
             "Total_Business_Assets","Owner_Hours_Week","Owner_Days_Month",
             "HH_Labor","HH_Labor_Hrs","Labor_5","NonHH_Labor",
             "NonHH_Labor_Hrs","Labor_10","NonHH_Labor_Dummy","HH_Labor_Dummy"]
for v in miss_vars:
    if v in biz.columns:
        biz[f"miss_{v}"] = biz[v].isna().astype(float)

# B_Capital from baseline
if "Inventory" in biz.columns and "Total_Business_Assets" in biz.columns:
    biz["_cap"] = biz[["Inventory","Total_Business_Assets"]].sum(axis=1, min_count=1)
    biz["B_Capital"] = biz.groupby("Id")["_cap"].transform("max")
else:
    biz["B_Capital"] = np.nan

# B_NonHH_Labor, B_HH_Labor etc from baseline
for v in ["NonHH_Labor","HH_Labor","Owner_Hours_Week","Owner_Days_Month"]:
    if v in biz.columns:
        tmp = biz.loc[biz["Survey_Version"]==1, ["Id",v]].groupby("Id")[v].sum().reset_index().rename(columns={v: f"B_{v}"})
        biz = biz.merge(tmp, on="Id", how="left", suffixes=("","_b"))

print("=== STEP 3: Collapse to household level ===")
# Aggregate: (Id, Survey_Version, GroupNumber) -> household panel

max_cols = [c for c in biz.columns if c.startswith("B_") or c.startswith("Winner") or
            c.startswith("miss_") or c in ["Attriter","Group_Size","digitspan_1","Public",
            "Revealed","Incentives","survey_month","Surveyor_Code","Income",
            "Gender_Followup","Education_Followup","Age_Followup","Married_Followup",
            "HH_Size_Followup","TV_hhincome_1","Client","Final_Randomization_Cluster",
            "LotteryWinner","Total_Num_Tickets","Status_Survey_2"]]
max_cols = list(set(c for c in max_cols if c in biz.columns))

sum_cols = [c for c in ["Profits_30Days","Income","Inventory","Total_Business_Assets",
            "NonHH_Labor","NonHH_Labor_Hrs","HH_Labor","HH_Labor_Hrs",
            "Owner_Hours_Week","Owner_Days_Month","Client_Hours_Week","Client_Days_Month",
            "Client_Profits_30Days","Avg_Yrly_Profits","NonHH_Labor_Dummy","HH_Labor_Dummy",
            "Labor_5","Labor_10"] if c in biz.columns]

# Remove Income from sum_cols if it's already in max_cols (it should be maxed for HH)
sum_cols_final = [c for c in sum_cols if c not in max_cols]

agg_dict = {}
for c in max_cols:
    agg_dict[c] = "max"
for c in sum_cols_final:
    if c not in agg_dict:
        agg_dict[c] = "sum"

# GroupBy
hh = biz.groupby(["Id","Survey_Version","GroupNumber"], as_index=False).agg(agg_dict)
print(f"HH panel: {len(hh)} rows, {hh['Id'].nunique()} households")

# Done = not attriter
hh["Done"] = (hh["Attriter"] == 0).astype(int)

# Recode 0 back to NaN for summed vars where all businesses had missing
for v in sum_cols_final:
    mv = f"miss_{v}"
    if mv in hh.columns and v in hh.columns:
        hh.loc[hh[mv]==1, v] = np.nan

# Trimming
hh = hh.sort_values(["Id","Survey_Version"])
hh["_lag_prof"] = hh.groupby("Id")["Profits_30Days"].shift(1)
hh["perchange_profits"] = 100*(hh["Profits_30Days"]-hh["_lag_prof"])/hh["_lag_prof"]
xtreme_high = hh["perchange_profits"].quantile(0.995)
hh["sample"] = ((hh["perchange_profits"].isna()) | (hh["perchange_profits"] <= xtreme_high)).astype(int)

# Trimmed outcomes
for v in ["Profits_30Days","Income","Owner_Hours_Week","Owner_Days_Month",
          "Client_Profits_30Days","Client_Hours_Week","Client_Days_Month",
          "NonHH_Labor","HH_Labor","Inventory","Total_Business_Assets","Avg_Yrly_Profits"]:
    if v in hh.columns:
        hh[f"Trim_{v}"] = hh[v].where(hh["sample"]==1)
        bl = hh.loc[hh["Survey_Version"]==1, ["Id",f"Trim_{v}"]].groupby("Id").max().reset_index().rename(columns={f"Trim_{v}": f"B_{v}_trim"})
        hh = hh.merge(bl, on="Id", how="left")

# Log/IHS
for v, raw in [("log_Income","Income"),("log_Profits","Profits_30Days")]:
    if raw in hh.columns:
        hh[v] = np.log(hh[raw].clip(lower=0).fillna(0) + 1)
        hh.loc[hh[raw].isna(), v] = np.nan

# Baseline outcomes
if "B_Income" not in hh.columns and "B_Income_trim" in hh.columns:
    hh["B_Income"] = hh["B_Income_trim"]
if "B_Profits_30Days" not in hh.columns and "B_Profits_30Days_trim" in hh.columns:
    hh["B_Profits_30Days"] = hh["B_Profits_30Days_trim"]
hh["B_Trim_Income"] = hh.get("B_Income", pd.Series(np.nan, index=hh.index))
hh["B_Trim_Profits_30Days"] = hh.get("B_Profits_30Days", pd.Series(np.nan, index=hh.index))

# Baseline log
for v in ["log_Profits","log_Income"]:
    if v in hh.columns:
        bl = hh.loc[hh["Survey_Version"]==1, ["Id",v]].groupby("Id").max().reset_index().rename(columns={v: f"B_{v}"})
        hh = hh.merge(bl, on="Id", how="left", suffixes=("","_bl"))

print("=== Merging rankings data ===")
rankings = pd.read_stata(os.path.join(RAW_DIR, "rankings data/Rankings Individual.dta"), convert_categoricals=False)
for c in rankings.columns:
    if rankings[c].dtype == np.float32:
        rankings[c] = rankings[c].astype(np.float64)

rankings = rankings[rankings["q"].notna() & rankings["q"].isin([1,2])].copy()
rankings["AllRank_NS"] = rankings["AllRank"].copy()
rankings.loc[rankings["Id"] == rankings["RespondentID"], "AllRank_NS"] = np.nan

rank_agg = rankings.groupby(["Id","q"]).agg(
    AllRank_sd=("AllRank_NS","std"),
    AllRank_med=("AllRank_NS","median"),
    AllRank=("AllRank","mean"),
    AllRank_NS=("AllRank_NS","mean"),
).reset_index()

rq1 = rank_agg[rank_agg["q"]==1][["Id","AllRank","AllRank_NS","AllRank_med","AllRank_sd"]].rename(
    columns={"AllRank":"Quintile_Rank","AllRank_NS":"Quint_Rank_NS","AllRank_med":"Q_Rank_med_NS","AllRank_sd":"Quint_Rank_sd_NS"})
rq2 = rank_agg[rank_agg["q"]==2][["Id","AllRank_NS"]].rename(columns={"AllRank_NS":"Rel_Rank_NS"})
rank_data = rq1.merge(rq2, on="Id", how="outer")
hh = hh.merge(rank_data, on="Id", how="left")

# Propensity score
hh["Total_Num_Tickets"] = hh["Total_Num_Tickets"] + 20
hh["Winner2"] = hh["LotteryWinner"] if "LotteryWinner" in hh.columns else hh.get("Winner",0)
if "Winner" not in hh.columns:
    hh["Winner"] = np.where(hh["Survey_Version"]==1, 0, hh["Winner2"])
hh["N_Grants_Group"] = hh.groupby(["GroupNumber","Survey_Version"])["Winner2"].transform("sum")
hh["Tot_Tix_Group"] = hh.groupby(["GroupNumber","Survey_Version"])["Total_Num_Tickets"].transform("sum")
mean_tix = hh["Total_Num_Tickets"].mean()
hh["P1"] = np.where(hh["N_Grants_Group"]==1, hh["Total_Num_Tickets"]/hh["Tot_Tix_Group"], np.nan)
hh["P2"] = np.where(hh["N_Grants_Group"]==2,
    hh["Total_Num_Tickets"]/hh["Tot_Tix_Group"] + (1-hh["Total_Num_Tickets"]/hh["Tot_Tix_Group"])*(hh["Total_Num_Tickets"]/(hh["Tot_Tix_Group"]-mean_tix)),
    np.nan)
hh["Prob_Winning"] = hh["P1"].fillna(hh["P2"])
hh["Propensity_Score"] = np.where(hh["Winner2"]==1, 1/hh["Prob_Winning"], 1/(1-hh["Prob_Winning"]))
hh["Propensity_Score"] = hh["Propensity_Score"].clip(upper=100)  # cap extreme weights

# Winner*Rank interactions
hh["Winner_Quint_Rank_NS"] = hh["Winner"] * hh["Quint_Rank_NS"]
hh["Winner_Quintile_Rank"] = hh["Winner"] * hh["Quintile_Rank"]
hh["Winner_Q_Rank_med_NS"] = hh["Winner"] * hh["Q_Rank_med_NS"]
hh["Winner_Rel_Rank_NS"] = hh["Winner"] * hh["Rel_Rank_NS"]
hh["Winner_Quint_Rank_sd_NS"] = hh["Winner"] * hh["Quint_Rank_sd_NS"]
hh["Winner_SD_Rank"] = hh["Winner"] * hh["Quint_Rank_sd_NS"] * hh["Quint_Rank_NS"]

# Tercile indicators
def make_terciles(df, var, prefix):
    valid = df[var].notna()
    df[f"{prefix}_Tercile_1"] = 0
    df[f"{prefix}_Tercile_2"] = 0
    df[f"{prefix}_Tercile_3"] = 0
    if valid.sum() > 10:
        # Use unique values at HH level for tercile cutoffs
        uvals = df.loc[valid & (df["Survey_Version"]==1), var].dropna()
        if len(uvals) < 10:
            uvals = df.loc[valid, var].dropna()
        cuts = uvals.quantile([1/3, 2/3]).values
        df.loc[valid & (df[var] <= cuts[0]), f"{prefix}_Tercile_1"] = 1
        df.loc[valid & (df[var] > cuts[0]) & (df[var] <= cuts[1]), f"{prefix}_Tercile_2"] = 1
        df.loc[valid & (df[var] > cuts[1]), f"{prefix}_Tercile_3"] = 1
    df[f"Winner_{prefix}_Tercile_3"] = df["Winner"] * df[f"{prefix}_Tercile_3"]
    df[f"Winner_{prefix}_Tercile_2"] = df["Winner"] * df[f"{prefix}_Tercile_2"]

make_terciles(hh, "Quint_Rank_NS", "Quint_Rank_NS")
make_terciles(hh, "Quintile_Rank", "Quintile_Rank")
make_terciles(hh, "Q_Rank_med_NS", "Q_Rank_med_NS")
make_terciles(hh, "Rel_Rank_NS", "Rel_Rank_NS")

# Ensure numeric FE columns
for c in ["Surveyor_Code","Survey_Version","survey_month","Id","GroupNumber","Final_Randomization_Cluster"]:
    if c in hh.columns:
        hh[c] = pd.to_numeric(hh[c], errors="coerce")

print(f"Final panel: {len(hh)} rows, {hh['Id'].nunique()} HHs")
print(f"  Trim_Income non-null: {hh['Trim_Income'].notna().sum()}" if "Trim_Income" in hh.columns else "  Trim_Income: MISSING")
print(f"  Trim_Profits_30Days non-null: {hh['Trim_Profits_30Days'].notna().sum()}" if "Trim_Profits_30Days" in hh.columns else "  Trim_Profits_30Days: MISSING")
print(f"  Winner_Quint_Rank_NS non-null: {hh['Winner_Quint_Rank_NS'].notna().sum()}")

# ============================================================================
# CONTROL PANEL DEFINITIONS
# ============================================================================
panela1 = [c for c in ["Gender_Followup","Education_Followup","Married_Followup","Age_Followup",
            "B_digitspan_1","B_ChangeSales_2014","B_Primary_Job_5yrs","B_Wage_Exit_SE"] if c in hh.columns]
panelb1 = [c for c in ["B_manufacturing","B_retail","B_service","B_agriculture"] if c in hh.columns]
panelc1 = [c for c in ["HH_Size_Followup","B_child_0to5","B_child_6to12","B_Number_Fixed_Salary",
            "B_Number_Daily_Wage","B_Total_Nu_Bus","B_TV_hhincome_1","B_HH_Assets"] if c in hh.columns]
paneld1 = [c for c in ["B_NonHH_Labor","B_HH_Labor","B_Owner_Hours_Week","B_Owner_Days_Month",
            "B_Avg_Yrly_Profits","B_Capital"] if c in hh.columns]
all_controls = panela1 + panelb1 + panelc1 + paneld1
psych_controls = [c for c in [f"B_Psychometric_{i}" for i in range(1,18)] if c in hh.columns]

def prepare_fe_controls(df, controls, prefix="AA"):
    """Mean-impute controls, create Winner*control and Winner*miss_control."""
    ctrl_vars = []
    for k, var in enumerate(controls, 1):
        aa = f"{prefix}{k}"; miss = f"miss_{prefix}{k}"; w = f"Winner_{prefix}{k}"; mw = f"mi_Winner_{prefix}{k}"
        df[aa] = df[var].copy()
        df[miss] = df[aa].isna().astype(float)
        mn = df[aa].mean()
        df[aa] = df[aa].fillna(mn if pd.notna(mn) else 0)
        df[w] = df["Winner"] * df[aa]
        df[mw] = df["Winner"] * df[miss]
        ctrl_vars.extend([w, mw])
    return ctrl_vars

def prepare_ancova_controls(df, controls, prefix="BB"):
    ctrl_vars = []
    for k, var in enumerate(controls, 1):
        bb = f"{prefix}{k}"; miss = f"miss_{prefix}{k}"
        df[bb] = df[var].copy()
        df[miss] = df[bb].isna().astype(float)
        mn = df[bb].mean()
        df[bb] = df[bb].fillna(mn if pd.notna(mn) else 0)
        ctrl_vars.extend([bb, miss])
    return ctrl_vars

# Samples
df_base = hh[(hh["Survey_Version"]!=5) & (hh["Done"]==1)].copy()
df_ancova = hh[(hh["Survey_Version"]!=1) & (hh["Survey_Version"]!=5) & (hh["Done"]==1)].copy()
df_all5 = hh[hh["Done"]==1].copy()
df_g5 = hh[(hh["Survey_Version"]!=5) & (hh["Done"]==1) & (hh["Group_Size"]==5)].copy()

fe_ctrl = prepare_fe_controls(df_base, all_controls, "AA")
fe_ctrl_psych = prepare_fe_controls(df_base, all_controls + psych_controls, "PP")
ancova_ctrl = prepare_ancova_controls(df_ancova, all_controls, "BB")

print(f"Base sample: {len(df_base)} obs")

# ============================================================================
# SPECIFICATION RUNNER
# ============================================================================
design_audit = surface_obj["baseline_groups"][0]["design_audit"]
infer_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
results, inference_results = [], []
spec_run_counter = 0

def run_spec(spec_id, tree_path, outcome, treatment, controls_list, fe, data, vcov,
             weights_col, sample_desc, controls_desc, cluster_var="GroupNumber",
             fe_str="Id + Surveyor_Code + Survey_Version + survey_month",
             axis_block_name=None, axis_block=None, design_override=None, bg="G1"):
    global spec_run_counter; spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    try:
        cstr = " + ".join(controls_list) if controls_list else ""
        fml = f"{outcome} ~ {treatment}" + (f" + {cstr}" if cstr else "") + (f" | {fe}" if fe else "")
        key_vars = [outcome, treatment]
        if weights_col: key_vars.append(weights_col)
        if cluster_var and cluster_var in data.columns: key_vars.append(cluster_var)
        if fe:
            for f in fe.split(" + "):
                f = f.strip()
                if f in data.columns: key_vars.append(f)
        rd = data.dropna(subset=[v for v in set(key_vars) if v in data.columns]).copy()
        kw = {"data": rd, "vcov": vcov}
        if weights_col and weights_col in rd.columns: kw["weights"] = weights_col
        m = pf.feols(fml, **kw)
        coef = float(m.coef().get(treatment, np.nan))
        se = float(m.se().get(treatment, np.nan))
        pv = float(m.pvalue().get(treatment, np.nan))
        try:
            ci = m.confint()
            cil = float(ci.loc[treatment, ci.columns[0]]) if treatment in ci.index else np.nan
            ciu = float(ci.loc[treatment, ci.columns[1]]) if treatment in ci.index else np.nan
        except: cil, ciu = np.nan, np.nan
        nobs = int(m._N); r2 = float(m._r2) if m._r2 is not None else np.nan
        allc = {k: float(v) for k, v in m.coef().items()}
        db = design_override or {"randomized_experiment": dict(design_audit)}
        pkw = dict(coefficients=allc, inference={"spec_id": infer_canonical["spec_id"], "params": infer_canonical["params"]},
                   software=SW_BLOCK, surface_hash=SURFACE_HASH, design=db)
        if axis_block_name and axis_block: pkw["axis_block_name"] = axis_block_name; pkw["axis_block"] = axis_block
        payload = make_success_payload(**pkw)
        row = {"paper_id": PAPER_ID, "spec_run_id": run_id, "spec_id": spec_id, "spec_tree_path": tree_path,
               "baseline_group_id": bg, "outcome_var": outcome, "treatment_var": treatment,
               "coefficient": coef, "std_error": se, "p_value": pv, "ci_lower": cil, "ci_upper": ciu,
               "n_obs": nobs, "r_squared": r2, "coefficient_vector_json": json.dumps(payload),
               "sample_desc": sample_desc, "fixed_effects": fe_str, "controls_desc": controls_desc,
               "cluster_var": cluster_var, "run_success": 1, "run_error": ""}
        results.append(row); print(f"  OK {run_id} ({spec_id}): b={coef:.4f} se={se:.4f} p={pv:.4f} n={nobs}"); return row
    except Exception as e:
        em = str(e)[:200]
        payload = make_failure_payload(error=em, error_details=error_details_from_exception(e, stage="estimation"),
                                       software=SW_BLOCK, surface_hash=SURFACE_HASH)
        row = {"paper_id": PAPER_ID, "spec_run_id": run_id, "spec_id": spec_id, "spec_tree_path": tree_path,
               "baseline_group_id": bg, "outcome_var": outcome, "treatment_var": treatment,
               "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
               "n_obs": np.nan, "r_squared": np.nan, "coefficient_vector_json": json.dumps(payload),
               "sample_desc": sample_desc, "fixed_effects": fe_str, "controls_desc": controls_desc,
               "cluster_var": cluster_var, "run_success": 0, "run_error": em}
        results.append(row); print(f"  FAIL {run_id}: {em[:80]}"); return row

BFE = "Id + Surveyor_Code + Survey_Version + survey_month"
BVCOV = {"CRV1": "GroupNumber"}
OC = [("Trim_Income","trimmed income"), ("Trim_Profits_30Days","trimmed profits")]

# ============================================================================
# BASELINES
# ============================================================================
print("\n=== BASELINES ===")
bl_rows = []
for ov, ol in OC:
    for ctrl, cd in [(["Winner"], "no controls"), (["Winner"]+fe_ctrl, "26 controls interacted with Winner")]:
        r = run_spec("baseline", "designs/randomized_experiment.md#panel-fe", ov, "Winner_Quint_Rank_NS",
                     ctrl, BFE, df_base, BVCOV, "Propensity_Score", f"SV!=5 & Done==1, {ol}", cd)
        bl_rows.append(r)

# ============================================================================
# DESIGN: ANCOVA
# ============================================================================
print("\n=== DESIGN: ANCOVA ===")
afe = "Final_Randomization_Cluster + Surveyor_Code + Survey_Version + survey_month"
ad = {"randomized_experiment": {**dict(design_audit), "estimator": "ancova"}}
for ov, ol in OC:
    bvar = f"B_Trim_{ov.replace('Trim_','')}" if f"B_Trim_{ov.replace('Trim_','')}" in df_ancova.columns else None
    extra = [bvar] if bvar else []
    run_spec("design/randomized_experiment/estimator/ancova", "designs/randomized_experiment.md#ancova",
             ov, "Winner_Quint_Rank_NS", ["Winner","Quint_Rank_NS"]+extra, afe, df_ancova, BVCOV,
             "Propensity_Score", f"ANCOVA {ol}", "strata FE + baseline outcome",
             fe_str=afe, design_override=ad)

# ============================================================================
# RC SPECS
# ============================================================================
print("\n=== RC: Controls ===")
# Psychometric
for ov, ol in OC:
    run_spec("rc/controls/add/psychometric_controls", "modules/robustness/controls.md#add-controls",
             ov, "Winner_Quint_Rank_NS", ["Winner"]+fe_ctrl_psych, BFE, df_base, BVCOV, "Propensity_Score",
             f"SV!=5 & Done==1, {ol}", "26+17 psych controls interacted with Winner",
             axis_block_name="controls", axis_block={"spec_id":"rc/controls/add/psychometric_controls","family":"add","n_controls":len(all_controls)+len(psych_controls)})

# LOO panels
for sid, panel, plab in [("rc/controls/loo/panela1_demographics",panela1,"demographics"),
                          ("rc/controls/loo/panelb1_business_type",panelb1,"business type"),
                          ("rc/controls/loo/panelc1_household",panelc1,"household"),
                          ("rc/controls/loo/paneld1_business_chars",paneld1,"business chars")]:
    rem = [c for c in all_controls if c not in panel]
    cv = prepare_fe_controls(df_base, rem, "LO")
    for ov, ol in OC:
        run_spec(sid, "modules/robustness/controls.md#leave-one-out-controls-loo", ov, "Winner_Quint_Rank_NS",
                 ["Winner"]+cv, BFE, df_base, BVCOV, "Propensity_Score", f"SV!=5, {ol}", f"drop {plab}",
                 axis_block_name="controls", axis_block={"spec_id":sid,"family":"loo","dropped":panel})

# Control subsets (profits only for budget)
for sid, sub, slab in [("rc/controls/subset/panela1_only",panela1,"demo only"),
                        ("rc/controls/subset/panelb1_only",panelb1,"biz type only"),
                        ("rc/controls/subset/panelc1_only",panelc1,"household only"),
                        ("rc/controls/subset/paneld1_only",paneld1,"biz chars only"),
                        ("rc/controls/subset/panela1_panelb1",panela1+panelb1,"demo+biz type"),
                        ("rc/controls/subset/panela1_panelc1",panela1+panelc1,"demo+household"),
                        ("rc/controls/subset/panelc1_paneld1",panelc1+paneld1,"hh+biz chars"),
                        ("rc/controls/subset/panelb1_panelc1",panelb1+panelc1,"biz type+hh"),
                        ("rc/controls/subset/panela1_panelb1_panelc1",panela1+panelb1+panelc1,"demo+biz+hh")]:
    cv = prepare_fe_controls(df_base, sub, "SU")
    run_spec(sid, "modules/robustness/controls.md#subset-controls", "Trim_Profits_30Days", "Winner_Quint_Rank_NS",
             ["Winner"]+cv, BFE, df_base, BVCOV, "Propensity_Score", "SV!=5, profits", slab,
             axis_block_name="controls", axis_block={"spec_id":sid,"family":"subset","n_controls":len(sub)})

print("\n=== RC: Rank construction ===")
for sid, tv, note in [("rc/data/rank_construction/include_self_rank","Winner_Quintile_Rank","with self"),
                       ("rc/data/rank_construction/relative_rank","Winner_Rel_Rank_NS","relative"),
                       ("rc/data/rank_construction/median_rank","Winner_Q_Rank_med_NS","median")]:
    for ov, ol in OC:
        run_spec(sid, "modules/robustness/data_construction.md", ov, tv, ["Winner"], BFE, df_base, BVCOV,
                 "Propensity_Score", f"SV!=5, {ol}", f"{note} rank",
                 axis_block_name="data_construction", axis_block={"spec_id":sid,"rank_type":note})

for ov, ol in OC:
    run_spec("rc/data/rank_construction/sd_rank_interaction", "modules/robustness/data_construction.md",
             ov, "Winner_Quint_Rank_NS", ["Winner","Winner_SD_Rank"], BFE, df_base, BVCOV,
             "Propensity_Score", f"SV!=5, {ol}", "SD rank interaction",
             axis_block_name="data_construction", axis_block={"spec_id":"rc/data/rank_construction/sd_rank_interaction"})

print("\n=== RC: Functional form ===")
for sid, ov, interp in [("rc/form/outcome/log_income","log_Income","log(Income+1)"),
                          ("rc/form/outcome/log_profits","log_Profits","log(Profits+1)")]:
    run_spec(sid, "modules/robustness/functional_form.md#log-outcome", ov, "Winner_Quint_Rank_NS",
             ["Winner"], BFE, df_base, BVCOV, "Propensity_Score", f"SV!=5, {ov}", "no controls, log outcome",
             axis_block_name="functional_form", axis_block={"spec_id":sid,"interpretation":interp})

for ov, ol in OC:
    run_spec("rc/form/treatment/tercile_rank", "modules/robustness/functional_form.md#treatment-parameterization",
             ov, "Winner_Quint_Rank_NS_Tercile_3", ["Winner_Quint_Rank_NS_Tercile_2","Winner"], BFE, df_base, BVCOV,
             "Propensity_Score", f"SV!=5, {ol}", "tercile rank focal=top",
             axis_block_name="functional_form", axis_block={"spec_id":"rc/form/treatment/tercile_rank","interpretation":"top tercile interaction"})

print("\n=== RC: Sample ===")
for ov, ol in OC:
    _ = prepare_fe_controls(df_all5, all_controls, "A5")
    run_spec("rc/sample/waves/all_5_waves", "modules/robustness/sample.md#sample-restriction",
             ov, "Winner_Quint_Rank_NS", ["Winner"], BFE, df_all5, BVCOV, "Propensity_Score",
             f"all 5 waves, {ol}", "no controls", axis_block_name="sample",
             axis_block={"spec_id":"rc/sample/waves/all_5_waves"})

for ov, ol in OC:
    _ = prepare_fe_controls(df_g5, all_controls, "G5")
    run_spec("rc/sample/restriction/groups_of_5_only", "modules/robustness/sample.md#sample-restriction",
             ov, "Winner_Quint_Rank_NS", ["Winner"], BFE, df_g5, BVCOV, "Propensity_Score",
             f"groups of 5, {ol}", "no controls", axis_block_name="sample",
             axis_block={"spec_id":"rc/sample/restriction/groups_of_5_only"})

# Trim 5-95
for ov, ol in OC:
    if ov in df_base.columns:
        p5, p95 = df_base[ov].quantile(0.05), df_base[ov].quantile(0.95)
        dt = df_base[(df_base[ov]>=p5)&(df_base[ov]<=p95)].copy()
        run_spec("rc/sample/outliers/trim_y_5_95", "modules/robustness/sample.md#outlier-handling",
                 ov, "Winner_Quint_Rank_NS", ["Winner"], BFE, dt, BVCOV, "Propensity_Score",
                 f"5-95% trim, {ol}", "no controls", axis_block_name="sample",
                 axis_block={"spec_id":"rc/sample/outliers/trim_y_5_95"})

# Winsorize 1-99
for ov, ol in OC:
    if ov in df_base.columns:
        p1, p99 = df_base[ov].quantile(0.01), df_base[ov].quantile(0.99)
        dw = df_base.copy(); dw[f"{ov}_w"] = dw[ov].clip(lower=p1, upper=p99)
        run_spec("rc/sample/outliers/winsorize_y_1_99", "modules/robustness/sample.md#outlier-handling",
                 f"{ov}_w", "Winner_Quint_Rank_NS", ["Winner"], BFE, dw, BVCOV, "Propensity_Score",
                 f"winsorize 1-99%, {ol}", "no controls", axis_block_name="sample",
                 axis_block={"spec_id":"rc/sample/outliers/winsorize_y_1_99"})

print("\n=== RC: Weights, FE ===")
for ov, ol in OC:
    run_spec("rc/weights/unweighted", "modules/robustness/weights.md", ov, "Winner_Quint_Rank_NS",
             ["Winner"], BFE, df_base, BVCOV, None, f"unweighted, {ol}", "no controls, no IPW",
             axis_block_name="weights", axis_block={"spec_id":"rc/weights/unweighted"})

for ov, ol in OC:
    run_spec("rc/fe/drop_surveyor_fe", "modules/robustness/fixed_effects.md", ov, "Winner_Quint_Rank_NS",
             ["Winner"], "Id + Survey_Version + survey_month", df_base, BVCOV, "Propensity_Score",
             f"drop surveyor FE, {ol}", "no controls", fe_str="Id + Survey_Version + survey_month",
             axis_block_name="fixed_effects", axis_block={"spec_id":"rc/fe/drop_surveyor_fe"})

for ov, ol in OC:
    run_spec("rc/fe/drop_survey_month_fe", "modules/robustness/fixed_effects.md", ov, "Winner_Quint_Rank_NS",
             ["Winner"], "Id + Surveyor_Code + Survey_Version", df_base, BVCOV, "Propensity_Score",
             f"drop month FE, {ol}", "no controls", fe_str="Id + Surveyor_Code + Survey_Version",
             axis_block_name="fixed_effects", axis_block={"spec_id":"rc/fe/drop_survey_month_fe"})

for ov, ol in OC:
    run_spec("rc/fe/strata_fe_instead_of_hh_fe", "modules/robustness/fixed_effects.md", ov, "Winner_Quint_Rank_NS",
             ["Winner","Quint_Rank_NS"], "Final_Randomization_Cluster + Surveyor_Code + Survey_Version + survey_month",
             df_base, BVCOV, "Propensity_Score", f"strata FE, {ol}", "strata FE instead of HH FE (HIGH LEVERAGE)",
             fe_str="Final_Randomization_Cluster + Surveyor_Code + Survey_Version + survey_month",
             axis_block_name="fixed_effects", axis_block={"spec_id":"rc/fe/strata_fe_instead_of_hh_fe","note":"HIGH LEVERAGE"})

print("\n=== RC: Joint ===")
# Joint specs — loop over a compact list
joint_specs = [
    # (spec_id, outcome_list, treatment, extra_ctrl_fn, fe, data, fe_str, desc)
]
# ANCOVA + controls / no controls
for ov, ol in OC:
    bv = f"B_Trim_{ov.replace('Trim_','')}" if f"B_Trim_{ov.replace('Trim_','')}" in df_ancova.columns else None
    ex = [bv] if bv else []
    run_spec("rc/joint/ancova_with_controls", "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS",
             ["Winner","Quint_Rank_NS"]+ex+ancova_ctrl, afe, df_ancova, BVCOV, "Propensity_Score",
             f"ANCOVA+ctrl, {ol}", "ANCOVA with controls", fe_str=afe,
             axis_block_name="joint", axis_block={"spec_id":"rc/joint/ancova_with_controls","axes_changed":["design","controls"]})
    run_spec("rc/joint/ancova_no_controls", "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS",
             ["Winner","Quint_Rank_NS"]+ex, afe, df_ancova, BVCOV, "Propensity_Score",
             f"ANCOVA no ctrl, {ol}", "ANCOVA no controls", fe_str=afe,
             axis_block_name="joint", axis_block={"spec_id":"rc/joint/ancova_no_controls","axes_changed":["design"]})

# Log outcomes +/- controls
for jid, use_ctrl in [("rc/joint/log_outcome_with_controls",True),("rc/joint/log_outcome_no_controls",False)]:
    for ov in ["log_Income","log_Profits"]:
        c = (["Winner"]+fe_ctrl) if use_ctrl else ["Winner"]
        run_spec(jid, "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS", c, BFE, df_base, BVCOV,
                 "Propensity_Score", f"{ov} {'ctrl' if use_ctrl else 'no ctrl'}", f"log {'ctrl' if use_ctrl else ''}",
                 axis_block_name="joint", axis_block={"spec_id":jid,"axes_changed":["form"]+["controls"] if use_ctrl else ["form"]})

# Rank variants +/- controls
for rank_name, tv in [("relative_rank","Winner_Rel_Rank_NS"),("median_rank","Winner_Q_Rank_med_NS"),
                       ("include_self_rank","Winner_Quintile_Rank")]:
    for suf, use_ctrl in [("with_controls",True),("no_controls",False)]:
        jid = f"rc/joint/{rank_name}_{suf}"
        for ov, ol in OC:
            c = (["Winner"]+fe_ctrl) if use_ctrl else ["Winner"]
            run_spec(jid, "modules/robustness/joint.md", ov, tv, c, BFE, df_base, BVCOV,
                     "Propensity_Score", f"{rank_name} {suf}, {ol}", f"{rank_name} {suf}",
                     axis_block_name="joint", axis_block={"spec_id":jid,"axes_changed":["data","controls"] if use_ctrl else ["data"]})

# Groups of 5 + controls, All 5 waves + controls
g5cv = prepare_fe_controls(df_g5, all_controls, "GC")
a5cv = prepare_fe_controls(df_all5, all_controls, "AC")
for ov, ol in OC:
    run_spec("rc/joint/groups_of_5_with_controls", "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS",
             ["Winner"]+g5cv, BFE, df_g5, BVCOV, "Propensity_Score", f"g5+ctrl, {ol}", "groups of 5 + controls",
             axis_block_name="joint", axis_block={"spec_id":"rc/joint/groups_of_5_with_controls","axes_changed":["sample","controls"]})
    run_spec("rc/joint/all_5_waves_with_controls", "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS",
             ["Winner"]+a5cv, BFE, df_all5, BVCOV, "Propensity_Score", f"5waves+ctrl, {ol}", "all 5 waves + controls",
             axis_block_name="joint", axis_block={"spec_id":"rc/joint/all_5_waves_with_controls","axes_changed":["sample","controls"]})

# Unweighted + controls
for ov, ol in OC:
    run_spec("rc/joint/unweighted_with_controls", "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS",
             ["Winner"]+fe_ctrl, BFE, df_base, BVCOV, None, f"unweighted+ctrl, {ol}", "unweighted + controls",
             axis_block_name="joint", axis_block={"spec_id":"rc/joint/unweighted_with_controls","axes_changed":["weights","controls"]})

# Tercile rank +/- controls
for suf, use_ctrl in [("no_controls",False),("with_controls",True)]:
    jid = f"rc/joint/tercile_rank_{suf}"
    for ov, ol in OC:
        c = (["Winner_Quint_Rank_NS_Tercile_2","Winner"]+fe_ctrl) if use_ctrl else ["Winner_Quint_Rank_NS_Tercile_2","Winner"]
        run_spec(jid, "modules/robustness/joint.md", ov, "Winner_Quint_Rank_NS_Tercile_3", c, BFE, df_base, BVCOV,
                 "Propensity_Score", f"tercile {suf}, {ol}", f"tercile rank {suf}",
                 axis_block_name="joint", axis_block={"spec_id":jid,"axes_changed":["form"]+["controls"] if use_ctrl else ["form"]})

# ============================================================================
# INFERENCE VARIANTS
# ============================================================================
print("\n=== INFERENCE VARIANTS ===")
for br in bl_rows:
    if br["run_success"] != 1: continue
    ov, tv, cd = br["outcome_var"], br["treatment_var"], br["controls_desc"]
    c = (["Winner"]+fe_ctrl) if "26" in cd else ["Winner"]
    cstr = " + ".join(c)
    fml = f"{ov} ~ {tv} + {cstr} | {BFE}"
    for iid, ipath, vcov_new, cvar in [
        ("infer/se/hc/hc1","modules/inference/standard_errors.md#heteroskedasticity-robust","hetero",""),
        ("infer/se/cluster/hh","modules/inference/standard_errors.md#cluster-robust",{"CRV1":"Id"},"Id")]:
        try:
            m = pf.feols(fml, data=df_base, vcov=vcov_new, weights="Propensity_Score")
            coef = float(m.coef().get(tv, np.nan)); se = float(m.se().get(tv, np.nan)); pv = float(m.pvalue().get(tv, np.nan))
            try:
                ci = m.confint()
                cil = float(ci.loc[tv, ci.columns[0]]) if tv in ci.index else np.nan
                ciu = float(ci.loc[tv, ci.columns[1]]) if tv in ci.index else np.nan
            except: cil, ciu = np.nan, np.nan
            allc = {k: float(v) for k, v in m.coef().items()}
            payload = make_success_payload(coefficients=allc, inference={"spec_id":iid,"params":{"cluster_var":cvar} if cvar else {}},
                                            software=SW_BLOCK, surface_hash=SURFACE_HASH, design={"randomized_experiment":dict(design_audit)})
            spec_run_counter += 1
            inference_results.append({"paper_id":PAPER_ID,"inference_run_id":f"{PAPER_ID}_infer_{spec_run_counter:03d}",
                "spec_run_id":br["spec_run_id"],"spec_id":iid,"spec_tree_path":ipath,"baseline_group_id":"G1",
                "outcome_var":ov,"treatment_var":tv,"cluster_var":cvar,
                "coefficient":coef,"std_error":se,"p_value":pv,"ci_lower":cil,"ci_upper":ciu,
                "n_obs":int(m._N),"r_squared":float(m._r2) if m._r2 else np.nan,
                "coefficient_vector_json":json.dumps(payload),"run_success":1,"run_error":""})
            print(f"  INFER OK {iid}: se={se:.4f}")
        except Exception as e:
            spec_run_counter += 1
            em = str(e)[:200]
            payload = make_failure_payload(error=em, error_details=error_details_from_exception(e, stage="inference"),
                                           software=SW_BLOCK, surface_hash=SURFACE_HASH)
            inference_results.append({"paper_id":PAPER_ID,"inference_run_id":f"{PAPER_ID}_infer_{spec_run_counter:03d}",
                "spec_run_id":br["spec_run_id"],"spec_id":iid,"spec_tree_path":ipath,"baseline_group_id":"G1",
                "outcome_var":ov,"treatment_var":tv,"cluster_var":cvar,
                "coefficient":np.nan,"std_error":np.nan,"p_value":np.nan,"ci_lower":np.nan,"ci_upper":np.nan,
                "n_obs":np.nan,"r_squared":np.nan,
                "coefficient_vector_json":json.dumps(payload),"run_success":0,"run_error":em})
            print(f"  INFER FAIL {iid}: {em[:60]}")

# ============================================================================
# WRITE OUTPUTS
# ============================================================================
print("\n=== WRITING OUTPUTS ===")
df_r = pd.DataFrame(results); df_r.to_csv(os.path.join(OUTPUT_DIR, "specification_results.csv"), index=False)
df_i = pd.DataFrame(inference_results); df_i.to_csv(os.path.join(OUTPUT_DIR, "inference_results.csv"), index=False)

ns = int(df_r["run_success"].sum()); nf = len(df_r)-ns; ni = len(df_i)
nis = int(df_i["run_success"].sum()) if ni else 0
print(f"specification_results.csv: {len(df_r)} rows ({ns} success, {nf} fail)")
print(f"inference_results.csv: {ni} rows ({nis} success)")

md = f"""# Specification Search Report: {PAPER_ID}

## Paper
Beaman, Magruder, & Robinson (2023) "Peer Ranking and Selective Capital Allocation"

## Surface Summary
- **Baseline groups**: 1 (G1: Heterogeneous ITT Winner*Rank panel FE)
- **Canonical inference**: Cluster SE at group level
- **Budget**: 80 core, 20 control subset
- **Seed**: 151841

## Counts
- **Spec rows**: {len(df_r)} ({ns} success, {nf} fail)
- **Inference rows**: {ni} ({nis} success)

## Failures
"""
fails = df_r[df_r["run_success"]==0]
for _, f in fails.iterrows():
    md += f"- `{f['spec_id']}` ({f['outcome_var']}): {f['run_error'][:100]}\n"
if len(fails)==0: md += "None.\n"

md += f"""
## Software
- Python {sys.version.split()[0]}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest','?')}
- pandas: {SW_BLOCK['packages'].get('pandas','?')}
"""
with open(os.path.join(OUTPUT_DIR, "SPECIFICATION_SEARCH.md"), "w") as f:
    f.write(md)
print("Done!")
