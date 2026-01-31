# Specification Search Agent Instructions

Use this prompt when launching agents to run systematic specification searches on AEA replication packages.

---

## Your Task

Run a systematic specification search on paper **{PAPER_ID}** following the specification tree at `/agentic_specification_search/specification_tree/`.

**Package Directory**: `{EXTRACTED_PACKAGE_PATH}`

**Important**: We do NOT have Stata installed. Replicate all analyses using Python (pandas, statsmodels, linearmodels, pyfixest) and/or R (fixest, plm, lfe, did, synthdid).

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

## Step 3: Read Specification Tree

Open `specification_tree/methods/{paper_method}.md` and note ALL required specifications.

Also review the robustness checks in `specification_tree/robustness/`:
- `leave_one_out.md`
- `single_covariate.md`
- `sample_restrictions.md`
- `clustering_variations.md`
- `functional_form.md`

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

## Step 7: Run Robustness Checks

Apply specifications from `specification_tree/robustness/`:

### Leave-One-Out
```python
for control in all_controls:
    remaining = [c for c in all_controls if c != control]
    model = pf.feols(f"y ~ treat + {'+'.join(remaining)} | fe", data=df)
    results.append({'spec_id': f'robust/loo/drop_{control}', ...})
```

### Single Covariate
```python
# Bivariate
model = pf.feols("y ~ treat | fe", data=df)
results.append({'spec_id': 'robust/single/none', ...})

for control in all_controls:
    model = pf.feols(f"y ~ treat + {control} | fe", data=df)
    results.append({'spec_id': f'robust/single/{control}', ...})
```

### Clustering Variations
```python
for cluster_var in ['unit', 'time', 'region']:
    model = pf.feols("y ~ treat + controls | fe",
                     data=df, vcov={'CRV1': cluster_var})
    results.append({'spec_id': f'robust/cluster/{cluster_var}', ...})
```

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
4. Consider directly editing the appropriate markdown file in `specification_tree/`

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

## Specification Breakdown

| Category | N | % Significant |
|----------|---|---------------|
| Baseline | 1 | X% |
| Method variations | X | X% |
| Robustness checks | X | X% |
| Custom | X | X% |

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

## Usage

Launch with:

```
Task tool with subagent_type="general-purpose"
prompt: [paste this template with {PAPER_ID} and {EXTRACTED_PACKAGE_PATH} filled in]
```
