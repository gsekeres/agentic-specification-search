# Paper Classifier Prompt

Use this prompt to classify papers by their primary empirical method before running specification searches.

---

## Purpose

Given an AEA replication package, identify the primary empirical method used in the paper to determine which specification tree to follow.

---

## Prompt Template

```
Classify the empirical method used in this AEA replication package.

**Package Directory**: {EXTRACTED_PACKAGE_PATH}

## Instructions

1. Read the README and documentation
2. Examine the primary do files / scripts
3. Identify the main estimation approach

## Classification Categories

Classify into ONE of the following:

| Code | Method | Key Indicators |
|------|--------|----------------|
| `did` | Difference-in-Differences | `xtreg ... i.treat#i.post`, `reghdfe`, treatment×post interaction, two-way FE |
| `es` | Event Study | `eventdd`, `eventstudyinteract`, leads/lags relative to treatment |
| `rd` | Regression Discontinuity | `rdrobust`, `rddensity`, running variable, bandwidth |
| `iv` | Instrumental Variables | `ivreghdfe`, `ivreg2`, `2sls`, first stage, instrument |
| `panel` | Panel Fixed Effects | `xtreg, fe`, `reghdfe`, entity + time FE without DiD |
| `ols` | Cross-Sectional OLS | Single cross-section, `reg`, no panel structure |
| `discrete` | Discrete Choice | `logit`, `probit`, `mlogit`, `ologit`, binary/categorical Y |
| `dynpanel` | Dynamic Panel | `xtabond`, `xtdpdsys`, lagged dependent variable |

## Output Format

Return a JSON object:

```json
{
  "paper_id": "{project_id}",
  "method_code": "did",
  "method_name": "Difference-in-Differences",
  "confidence": "high|medium|low",
  "evidence": [
    "Uses reghdfe with unit and time fixed effects",
    "Has treat*post interaction term",
    "README mentions difference-in-differences"
  ],
  "secondary_methods": ["es"],
  "notes": "Also includes event study as robustness"
}
```

## Decision Rules

### DiD vs Panel FE
- **DiD**: Treatment status changes over time for some units (staggered or simultaneous)
- **Panel FE**: Treatment is time-invariant OR continuous policy variable

### Event Study vs DiD
- **Event Study**: Explicit dynamic specification with relative time indicators
- **DiD**: Single post-treatment indicator (or treat×post)

### IV vs OLS
- **IV**: Endogenous regressor instrumented
- **OLS**: No instrumental variable, all regressors exogenous

### Panel vs Cross-Sectional
- **Panel**: Multiple observations per unit over time
- **Cross-Sectional**: Single observation per unit

### When Uncertain
- If paper uses multiple methods, classify by the PRIMARY (main table) method
- Note secondary methods in the `secondary_methods` field
- If truly ambiguous, set `confidence: "low"` and explain in notes
```

---

## Usage

Launch with:

```
Task tool with subagent_type="general-purpose"
prompt: [paste template with {EXTRACTED_PACKAGE_PATH} filled in]
```

---

## Batch Classification

For classifying multiple papers:

```python
import json

def classify_papers(metadata_file, output_file):
    """
    Classify all papers in metadata file.
    """
    classifications = []

    with open(metadata_file) as f:
        for line in f:
            paper = json.loads(line)
            # Launch classifier agent for each paper
            # ... agent call ...
            classifications.append(result)

    with open(output_file, 'w') as f:
        for c in classifications:
            f.write(json.dumps(c) + '\n')
```

---

## Integration with Specification Search

After classification, the specification search agent should:

1. Read the classification result
2. Load the corresponding specification tree file
3. Run all specifications from that tree
4. Apply universal robustness checks from `robustness/` directory
