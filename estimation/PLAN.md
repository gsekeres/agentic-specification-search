# Estimation Code Plan

## Purpose

Structural estimation of publication bias and specification search behavior from agent-generated data. The goal is to:
1. Estimate a hierarchical model of z-statistic distributions across papers
2. Validate against external replication data (i4r)
3. Run counterfactual simulations of alternative editorial mechanisms

## Data Inputs

- `../unified_results.csv` - Agent specification search results (52 papers, ~2943 specs)
- `../i4r/Meta Database Public.xlsx` - Iowa Replication Database for validation

## Model Structure

**K=2 Type Hierarchical Model with Random Effects:**
- Paper types: Null (N) vs High-impact (H)
- Within-paper: AR(1) process for sequential z-statistics
- Parameters: φ (persistence), μ_N, μ_H (type means), τ_N, τ_H (random effect SDs), σ (innovation SD), π_N (null fraction)

## Analysis Pipeline

### Stage 0: Data Reality Check
- Verify observation completeness
- Check ordering/timestamps
- Assess stopping behavior (fixed horizon vs adaptive)

### Stage 1: Estimation
- MLE with bounded optimization
- Bootstrap standard errors
- Order sensitivity analysis (critical for identification)

### Stage 2: Calibration
- γ = cost/value ratio calibration
- Internal anchor (budget constraints) vs external anchor (dollar values)

### Stage 3: Validation
- i4r moment comparison
- Key moments: shrinkage slope, sign flip rate, significance drop rate

### Stage 4: Counterfactuals
- Status quo (single p < 0.05)
- Certificate rule (m tests in band [p_min, p_max])
- Purity vs throughput tradeoffs

## Output

- Tables: parameters, validation, counterfactuals
- Figures: z-distributions, order sensitivity, shrinkage, purity curves

## Key Issues to Address

1. **Order sensitivity**: φ estimates vary substantially with assumed specification ordering
2. **Model fit**: Current model has poor fit on shrinkage/sign flip moments
3. **Plotting**: PyPlot/LaTeX integration issues with Unicode characters

## Directory Structure

```
estimation/
├── PLAN.md           # This file
├── Project.toml      # Julia dependencies
├── main.jl           # Master script
├── src/
│   ├── data_preparation.jl
│   ├── hierarchical_model.jl
│   ├── estimation.jl
│   ├── validation.jl
│   ├── counterfactuals.jl
│   ├── plotting.jl
│   └── ...
└── output/
    ├── tables/
    └── figures/
```
