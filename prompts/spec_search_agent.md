# Specification Search Agent Instructions

Use this prompt when launching agents to run systematic specification searches on AEA replication packages.

---

## Your Task

Run a systematic specification search on paper **{PAPER_ID}** following the specification tree at `/agentic_specification_search/specification_tree/`.

**Package Directory**: `{EXTRACTED_PACKAGE_PATH}`

**Important**: We do NOT have Stata installed. Replicate all analyses using Python (pandas, statsmodels, linearmodels, pyfixest) and/or R (fixest, plm, lfe, did, synthdid).

---

## MINIMUM SPECIFICATION TARGET: 50+ SPECIFICATIONS

Based on the Institute for Replication (i4r) methodology, you MUST run **at least 50 specifications** per paper. The i4r average is 63 specs per paper, with many papers having 100+.

### Required Robustness Categories (i4r checklist)

| Category | Min Specs | Description |
|----------|-----------|-------------|
| **Control variations** | 10-15 | Drop each control, add incrementally, alternative sets |
| **Sample restrictions** | 10-15 | By time, geography, demographics, outlier handling |
| **Alternative outcomes** | 5-10 | Different DV codings, related outcomes |
| **Alternative treatments** | 3-5 | Binary vs continuous, intensity, thresholds |
| **Inference variations** | 5-8 | Different clustering, robust SEs, two-way |
| **Estimation method** | 3-5 | Different FE structures, OLS vs alternatives |
| **Functional form** | 3-5 | Log, IHS, levels, polynomials |
| **Weights** | 2-3 | Weighted vs unweighted, different weights |
| **Placebo tests** | 3-5 | Pre-treatment, fake timing, unaffected outcomes |
| **Heterogeneity** | 5-10 | Interactions with gender, age, income, etc. |

If you are finishing with fewer than 50 specifications, you have NOT been thorough enough. Go back and add more variations.

---

## Step 1: Understand the Paper

1. Read the README and any documentation
2. Identify:
   - Main hypothesis
   - Primary outcome variable(s)
   - Treatment/exposure variable(s)
   - Key control variables
   - Identification strategy

## Step 2: Classify Paper Method

Read the paper's do files/scripts and classify into one or more of:

| Method | Use When |
|--------|----------|
| `difference_in_differences` | Treatment varies across units AND time |
| `event_study` | Dynamic treatment effects with leads/lags |
| `regression_discontinuity` | Treatment assigned by cutoff/threshold |
| `instrumental_variables` | Endogenous treatment instrumented |
| `panel_fixed_effects` | Repeated observations, unit/time FE |
| `cross_sectional_ols` | Single time period, no panel |
| `discrete_choice` | Binary/categorical outcome |
| `dynamic_panel` | Lagged dependent variable on RHS |

If a major category does not exist here, please add it.

**Required output**: record the chosen method and the explicit method tree path you will follow.

```
method_code: difference_in_differences
method_tree_path: specification_tree/methods/difference_in_differences.md
```

## Step 3: Read Specification Tree

Open `specification_tree/methods/{paper_method}.md` and note ALL required specifications.

Also review the robustness checks in `specification_tree/robustness/`:
- `leave_one_out.md`
- `single_covariate.md`
- `sample_restrictions.md`
- `clustering_variations.md`
- `functional_form.md`

Create a **method map** before you run any models. This ensures every specification explicitly
references a node in the tree.

```json
{
  "method_code": "difference_in_differences",
  "method_tree_path": "specification_tree/methods/difference_in_differences.md",
  "specs_to_run": [
    {"spec_id": "baseline", "spec_tree_path": "methods/difference_in_differences.md#baseline"},
    {"spec_id": "did/fe/twoway", "spec_tree_path": "methods/difference_in_differences.md#fixed-effects"}
  ],
  "robustness_specs": [
    {"spec_id": "robust/loo/drop_age", "spec_tree_path": "robustness/leave_one_out.md"},
    {"spec_id": "robust/cluster/unit", "spec_tree_path": "robustness/clustering_variations.md"}
  ]
}
```

## Step 4: Load and Prepare Data

```python
import pandas as pd
import numpy as np
import pyfixest as pf
from linearmodels.panel import PanelOLS
import json

# Load data
df = pd.read_stata('data.dta')  # or pd.read_csv, pd.read_parquet

# Recreate variable transformations from original code
# Set up panel structure if applicable
```

## Step 5: Run Baseline Specification

Replicate the paper's main result EXACTLY. This is critical for validation.

```python
results = []

# Baseline - exact replication
baseline = pf.feols("outcome ~ treatment + controls | fe1 + fe2",
                    data=df, vcov={'CRV1': 'cluster_var'})

results.append({
    'spec_id': 'baseline',
    'spec_tree_path': 'methods/{method}.md',
    'outcome_var': 'outcome',
    'treatment_var': 'treatment',
    'coefficient': baseline.coef()['treatment'],
    'std_error': baseline.se()['treatment'],
    # ... all fields
})
```

## Step 6: Run Method-Specific Specifications

For each spec in the method file:
- Run the specification
- Record ALL coefficients (not just treatment)
- Save `coefficient_vector_json`
- Include `spec_tree_path` that points to the method tree header or table section

## Step 7: Run Robustness Checks (COMPREHENSIVE - aim for 40+ robustness specs)

Following the Institute for Replication (i4r) methodology, run ALL of the following robustness check categories. This is the core of the specification search.

### 7.1 Control Variable Variations (~10-15 specs)

**Leave-One-Out** - Drop each control one at a time:
```python
for control in all_controls:
    remaining = [c for c in all_controls if c != control]
    model = pf.feols(f"y ~ treat + {'+'.join(remaining)} | fe", data=df)
    results.append({'spec_id': f'robust/control/drop_{control}', ...})
```

**Add Controls Incrementally** - Build up from bivariate:
```python
# Bivariate (no controls)
model = pf.feols("y ~ treat | fe", data=df)
results.append({'spec_id': 'robust/control/none', ...})

# Add each control one at a time
for i, control in enumerate(all_controls):
    controls_so_far = all_controls[:i+1]
    model = pf.feols(f"y ~ treat + {'+'.join(controls_so_far)} | fe", data=df)
    results.append({'spec_id': f'robust/control/add_{control}', ...})
```

**Alternative Control Sets** - If paper uses different control sets in different tables:
```python
for control_set_name, controls in [('minimal', minimal_controls), ('full', full_controls), ('extended', extended_controls)]:
    model = pf.feols(f"y ~ treat + {'+'.join(controls)} | fe", data=df)
    results.append({'spec_id': f'robust/control/set_{control_set_name}', ...})
```

### 7.2 Sample Restrictions (~10-15 specs)

**Time Period Restrictions**:
```python
# Drop first/last year
for year in df['year'].unique():
    model = pf.feols("y ~ treat + controls | fe", data=df[df['year'] != year])
    results.append({'spec_id': f'robust/sample/drop_year_{year}', ...})

# Pre-2000 vs post-2000 (or relevant cutoff)
model = pf.feols("y ~ treat + controls | fe", data=df[df['year'] < 2000])
results.append({'spec_id': 'robust/sample/pre_2000', ...})
```

**Geographic Restrictions** (if applicable):
```python
# Drop each region/state
for region in df['region'].unique():
    model = pf.feols("y ~ treat + controls | fe", data=df[df['region'] != region])
    results.append({'spec_id': f'robust/sample/drop_{region}', ...})
```

**Demographic/Unit Restrictions**:
```python
# By gender, age groups, income levels, firm size, etc.
for group_name, condition in [('male', df['male']==1), ('female', df['male']==0),
                               ('young', df['age']<40), ('old', df['age']>=40)]:
    model = pf.feols("y ~ treat + controls | fe", data=df[condition])
    results.append({'spec_id': f'robust/sample/{group_name}', ...})
```

**Outlier Treatment**:
```python
# Winsorize at 1%, 5%, 10%
for pct in [1, 5, 10]:
    df_wins = df.copy()
    df_wins['y'] = df_wins['y'].clip(lower=df_wins['y'].quantile(pct/100),
                                      upper=df_wins['y'].quantile(1-pct/100))
    model = pf.feols("y ~ treat + controls | fe", data=df_wins)
    results.append({'spec_id': f'robust/sample/winsorize_{pct}pct', ...})

# Trim extreme values
model = pf.feols("y ~ treat + controls | fe",
                 data=df[(df['y'] > df['y'].quantile(0.01)) & (df['y'] < df['y'].quantile(0.99))])
results.append({'spec_id': 'robust/sample/trim_1pct', ...})
```

### 7.3 Alternative Outcomes (~5-10 specs)

If the paper has multiple outcome variables or the outcome can be measured differently:
```python
for outcome in ['outcome1', 'outcome2', 'log_outcome', 'outcome_binary']:
    model = pf.feols(f"{outcome} ~ treat + controls | fe", data=df)
    results.append({'spec_id': f'robust/outcome/{outcome}', ...})
```

### 7.4 Alternative Treatment Definitions (~3-5 specs)

If treatment can be coded differently:
```python
# Continuous vs binary treatment
# Different treatment thresholds
# Intensity vs any treatment
for treat_var in ['treat_binary', 'treat_continuous', 'treat_intensity']:
    model = pf.feols(f"y ~ {treat_var} + controls | fe", data=df)
    results.append({'spec_id': f'robust/treatment/{treat_var}', ...})
```

### 7.5 Inference/Clustering Variations (~5-8 specs)

```python
# Different clustering levels
for cluster in ['unit_id', 'time_id', 'region_id', 'state_id']:
    if cluster in df.columns:
        model = pf.feols("y ~ treat + controls | fe", data=df, vcov={'CRV1': cluster})
        results.append({'spec_id': f'robust/cluster/{cluster}', ...})

# Two-way clustering
model = pf.feols("y ~ treat + controls | fe", data=df, vcov={'CRV1': ['unit_id', 'time_id']})
results.append({'spec_id': 'robust/cluster/twoway', ...})

# Robust (heteroskedasticity-consistent) SEs
model = pf.feols("y ~ treat + controls | fe", data=df, vcov='hetero')
results.append({'spec_id': 'robust/cluster/robust_hc1', ...})

# Wild bootstrap (if small number of clusters)
# Note: may need to implement separately
```

### 7.6 Estimation Method Variations (~3-5 specs)

```python
# OLS vs Poisson (for count outcomes)
# OLS vs Logit/Probit (for binary outcomes)
# OLS vs Tobit (for censored outcomes)
# With and without fixed effects
# Different FE structures

# No fixed effects
model = pf.feols("y ~ treat + controls", data=df, vcov={'CRV1': 'cluster_var'})
results.append({'spec_id': 'robust/estimation/no_fe', ...})

# Only unit FE
model = pf.feols("y ~ treat + controls | unit_id", data=df, vcov={'CRV1': 'cluster_var'})
results.append({'spec_id': 'robust/estimation/unit_fe_only', ...})

# Only time FE
model = pf.feols("y ~ treat + controls | time_id", data=df, vcov={'CRV1': 'cluster_var'})
results.append({'spec_id': 'robust/estimation/time_fe_only', ...})
```

### 7.7 Functional Form (~3-5 specs)

```python
# Log outcome
model = pf.feols("np.log(y+1) ~ treat + controls | fe", data=df)
results.append({'spec_id': 'robust/funcform/log_outcome', ...})

# Inverse hyperbolic sine
model = pf.feols("np.arcsinh(y) ~ treat + controls | fe", data=df)
results.append({'spec_id': 'robust/funcform/ihs_outcome', ...})

# Levels vs logs for controls
# Polynomial terms
# Squared terms
```

### 7.8 Weights Variations (~2-3 specs)

If the paper uses weights or weighting is sensible:
```python
# Unweighted (if paper uses weights)
model = pf.feols("y ~ treat + controls | fe", data=df)  # no weights
results.append({'spec_id': 'robust/weights/unweighted', ...})

# Population weights
model = pf.feols("y ~ treat + controls | fe", data=df, weights='pop_weight')
results.append({'spec_id': 'robust/weights/population', ...})
```

### 7.9 Placebo Tests (~3-5 specs)

```python
# Placebo treatment (pre-treatment period for DiD)
# Placebo outcome (outcome that shouldn't be affected)
# Randomization inference / permutation test

# Pre-treatment placebo (for DiD/event study)
model = pf.feols("y ~ treat_placebo + controls | fe", data=df[df['period'] < 0])
results.append({'spec_id': 'robust/placebo/pre_treatment', ...})

# Fake treatment date
df['fake_treat'] = (df['year'] >= fake_cutoff).astype(int)
model = pf.feols("y ~ fake_treat + controls | fe", data=df)
results.append({'spec_id': 'robust/placebo/fake_timing', ...})
```

### 7.10 Heterogeneity Analysis (~5-10 specs)

```python
# Interactions with key variables
for het_var in ['male', 'high_income', 'urban', 'large_firm']:
    if het_var in df.columns:
        model = pf.feols(f"y ~ treat * {het_var} + controls | fe", data=df)
        results.append({'spec_id': f'robust/heterogeneity/{het_var}', ...})
```

---

**CHECKPOINT**: After running all robustness checks, count your specifications. If you have fewer than 50, go back and add more variations. Common areas to expand:
- More sample restrictions (different years, regions, subgroups)
- More control combinations
- More outcome variations
- More heterogeneity analyses

---

Every robustness run must include:
1. `spec_id`
2. `spec_tree_path`
3. explicit reference to the robustness file section (if applicable)

## Step 8: Save Estimation Script

Save your complete estimation script to:
`scripts/paper_analyses/{PAPER_ID}.py`

This script should be fully self-contained and reproducible.

## Step 9: Output Results

### Required CSV Columns

Save results to the package directory as `specification_results.csv`:

| Column | Description |
|--------|-------------|
| `paper_id` | e.g., "223561-V1" |
| `journal` | e.g., "AER", "AEJ-Applied" |
| `paper_title` | Full title |
| `spec_id` | e.g., "did/fe/twoway" or "baseline" |
| `spec_tree_path` | e.g., "methods/difference_in_differences.md#fixed-effects" |
| `outcome_var` | Dependent variable name |
| `treatment_var` | Main treatment/exposure variable |
| `coefficient` | Point estimate on treatment |
| `std_error` | Standard error |
| `t_stat` | t-statistic |
| `p_value` | p-value |
| `ci_lower` | 95% CI lower bound |
| `ci_upper` | 95% CI upper bound |
| `n_obs` | Sample size |
| `r_squared` | R² (if applicable) |
| `coefficient_vector_json` | Full coefficient vector as JSON |
| `sample_desc` | Sample description |
| `fixed_effects` | FE structure description |
| `controls_desc` | Control variables description |
| `cluster_var` | Clustering variable |
| `model_type` | OLS, FE, IV, PPML, etc. |
| `estimation_script` | Path to script |

### coefficient_vector_json Format

```json
{
  "treatment": {"var": "policy_dummy", "coef": 0.05, "se": 0.02, "pval": 0.01},
  "controls": [
    {"var": "age", "coef": 0.1, "se": 0.05, "pval": 0.04},
    {"var": "income", "coef": -0.02, "se": 0.01, "pval": 0.02}
  ],
  "fixed_effects": ["state", "year"],
  "diagnostics": {
    "first_stage_F": null,
    "overid_pval": null,
    "hausman_pval": null
  }
}
```

---

## Adding New Specifications

If you identify a reasonable specification NOT in the tree:

1. Run it anyway
2. Use `spec_id = "custom/{description}"`
3. Note in your summary that this should be added to the tree
4. **Add the spec directly to the appropriate method or robustness markdown file**
   in `specification_tree/` (include the spec_id, description, and any constraints)
5. If unsure where it belongs, add a short note in the Summary Report explaining why
   and propose a location

---

## Summary Report

Also create `SPECIFICATION_SEARCH.md` in the package directory:

```markdown
# Specification Search: {Paper Title}

## Paper Overview
- **Paper ID**: {PAPER_ID}
- **Topic**: [Brief description]
- **Hypothesis**: [Main hypothesis]
- **Method**: [DiD, IV, RD, etc.]
- **Data**: [Data description]

## Classification
- **Method Type**: [From specification tree]
- **Spec Tree Path**: methods/{method}.md

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total specifications | N |
| Positive coefficients | X (Y%) |
| Significant at 5% | X (Y%) |
| Significant at 1% | X (Y%) |
| Median coefficient | X.XX |
| Mean coefficient | X.XX |
| Range | [min, max] |

## Robustness Assessment

**STRONG / MODERATE / WEAK** support for the main hypothesis.

[Explanation]

## Specification Breakdown by Category (i4r format)

| Category | N | % Positive | % Sig 5% |
|----------|---|------------|----------|
| Baseline | 1 | - | - |
| Control variations | X | X% | X% |
| Sample restrictions | X | X% | X% |
| Alternative outcomes | X | X% | X% |
| Alternative treatments | X | X% | X% |
| Inference variations | X | X% | X% |
| Estimation method | X | X% | X% |
| Functional form | X | X% | X% |
| Weights | X | X% | X% |
| Placebo tests | X | X% | X% |
| Heterogeneity | X | X% | X% |
| **TOTAL** | **≥50** | **X%** | **X%** |

## Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Critical Caveats

1. [Caveat 1]
2. [Caveat 2]

## Files Generated

- `specification_results.csv`
- `scripts/paper_analyses/{PAPER_ID}.py`
```

---

## Step 10: Mark Paper as Complete

After successfully generating `specification_results.csv` and `SPECIFICATION_SEARCH.md`, update the tracking file:

```python
import json

status_file = 'data/tracking/spec_search_status.json'
with open(status_file, 'r') as f:
    status = json.load(f)

# Update status to completed
for pkg in status['packages_with_data']:
    if pkg['id'] == '{PAPER_ID}':
        pkg['status'] = 'completed'
        break
else:
    # Add new entry if not found
    status['packages_with_data'].append({
        'id': '{PAPER_ID}',
        'title': '{SHORT_TITLE}',
        'status': 'completed'
    })

with open(status_file, 'w') as f:
    json.dump(status, f, indent=2)
```

---

## Step 11: Disk Cleanup

**CRITICAL FINAL STEP**: After completing steps 9-10, run this cleanup to free disk space:

```bash
bash /Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/scripts/cleanup_after_spec_search.sh {PAPER_ID}
```

This script keeps only `specification_results.csv` and `SPECIFICATION_SEARCH.md` in the package directory, deleting all raw data files to save disk space.

**Do NOT skip this step** - the raw data files are large and no longer needed after analysis.

---

## Usage

Launch with:

```
Task tool with subagent_type="general-purpose"
prompt: [paste this template with {PAPER_ID} and {EXTRACTED_PACKAGE_PATH} filled in]
```
