# Specification Surface: 112574-V1

## Paper Overview
- **Title**: How Much Is a Seat on the Security Council Worth? Foreign Aid and Bribery at the United Nations (Kuziemko & Werker, 2006, JDE)
- **Design**: Panel fixed effects (donor-recipient pair FE + time FE)
- **Key finding**: US foreign aid increases significantly to countries that vote with the US in the UN General Assembly during election years. The interaction of UN voting alignment with executive election timing (p_unvotes_elecex) is positive and significant, suggesting that politically aligned countries receive more aid around election time. A two-standard-deviation increase in UN alignment is associated with roughly a 59% increase in US aid during election years.

## Baseline Groups

### G1: Aid-for-Votes Effect (Main Table)

**Claim object**: The differential effect of UN voting alignment on ODA commitments during recipient-country executive election years. The paper argues this captures the "price" of political alignment -- countries that vote with the US receive more aid, especially around elections when political support is most valuable.

**Baseline specification** (Main Table, Column IV):
- Formula: `oda ~ unvotes + i_elecex + p_unvotes_elecex + year_dummies` (with pair FE via within-transformation)
- Outcome: `oda` (ODA commitments in 2000 USD thousands)
- Focal variable: `p_unvotes_elecex` (= `unvotes * i_elecex`)
- Additional regressors: `unvotes` (UN voting agreement), `i_elecex` (election year indicator)
- FE: Donor-recipient pair FE (absorbed via de-meaning) + year dummies
- Sample: All Big 5 donors x all recipients, 1975-2003, balanced panel (no gaps)
- Inference: 3-way clustering on donor, recipient, year (Cameron-Gelbach-Miller)

**Additional baseline-like specs** (all 9 columns of the main table):
- Cols I-III: Election effect only (without UN alignment interaction)
- Cols IV-VI: Main specification with UN voting interaction
- Cols VII-IX: UN votes decomposed into donor-average and residual components
- Within each triple: pair+year FE / pair+donor-year FE / pair FE + macro controls

## RC Axes Included

### Controls (macro block)
- **No controls**: Pair FE + year/donor-year dummies only (baseline Cols IV, V)
- **Macro block**: Add pop, gdp2000, pop_donor, gdp2000_donor (Col VI)
- **LOO within macro block**: Drop each of the 4 macro controls individually
- Note: The paper treats macro controls as a single block and does not use subsets

### FE structure
- **Pair + year dummies**: Standard (Cols I, IV, VII)
- **Pair + donor-year dummies**: More demanding; absorbs all donor-year variation (Cols II, V, VIII)
- **Pair only + macro controls**: No time dummies; time variation captured by macro controls (Cols III, VI, IX)
- **Pair + recipient-year**: A natural robustness check not in the paper; absorbs recipient-year variation

### Sample restrictions
- **Drop Big 3 recipients**: Exclude Egypt, Indonesia, India (largest aid recipients; sensitivity analysis)
- **Drop Big 5 recipients**: Exclude Egypt, Indonesia, India, Israel, China (sensitivity analysis)
- **US only**: Restrict to US as donor (the paper focuses on Big 5 but the US mechanism is most prominent)
- **Balanced panel**: Ensure no gaps in the panel for each pair (the paper's estsample function already enforces this)

### Outcome definition
- **Level ODA**: oda in USD thousands (baseline)
- **Log ODA**: log(oda), with zeros replaced by log(1/1000000); macro controls also logged (sensitivity analysis "loda" check)

### Treatment decomposition
- **UN voting agreement**: Single unvotes measure (baseline)
- **Decomposed UN votes**: Split into donor-average component (unvotes_rt) and residual (unvotes_resid), with separate interactions (Cols VII-IX)
- **Competitive vs. non-competitive elections (EIEC)**: Split election indicator by competitiveness using EIEC measure (Competitiveness Table, Cols IV-VI)
- **Competitive vs. non-competitive elections (vote share)**: Split by percent vote measure (Competitiveness Table, Cols I-III)
- **Early vs. late elections**: Split election year by whether election occurred in first 4 months (timing sensitivity)

### Joint variations
- Log ODA with each of the three FE structures
- Drop Big 3/Big 5 with pair+year FE
- Decomposed UN votes with pair+year and pair+donor-year
- Competitive elections (both EIEC and pct measures) with each FE structure

## What Is Excluded and Why

- **NED regressions (Table 6 in the paper)**: The National Endowment for Democracy analysis uses a different outcome variable (NEDtotal, NEDODA) at the recipient-year level (not donor-recipient-year). This is a distinct claim about a different aid channel and is treated as a separate analysis, not a robustness check of the main ODA result. The NED analysis also uses only 2-way clustering (recipient, year) since there is no donor dimension. It could be a second baseline group but the paper positions it as supplementary evidence.
- **Design variants**: No first-differencing or long-differencing variants tested; the within-estimator via de-meaning is the canonical approach. Correlated random effects is not meaningful here.
- **Exploration**: Heterogeneity by election competitiveness is already captured as an RC axis (treatment decomposition). No CATE or policy learning analysis is appropriate.
- **Sensitivity to unobserved confounding**: Not applicable in the same way as cross-sectional designs; the identifying variation is within-pair over time.

## Budgets and Sampling

- **G1 max core specs**: 100
- **Max control subsets**: 15 (only 4 macro controls in one block, so LOO + block is exhaustive at 5 combinations)
- **Seed**: 112574
- **Exhaustive enumeration**: Feasible for controls. Main combinatorial axis is the cross of FE structure (3) x treatment decomposition (5) x sample (4) x outcome form (2) = 120 potential cells. Budget requires sampling from joint combinations; priority is given to treatment decomposition x FE structure crosses (which the paper already explores) plus outcome form and sample restrictions.

## Inference Plan

- **Canonical**: 3-way clustering on donor, recipient, year (Cameron-Gelbach-Miller multi-way clustering)
- **Variants**: Cluster at pair level; cluster at recipient level; 2-way cluster (donor, recipient); HC1 robust
- All inference variants recorded in `inference_results.csv`

## Key Linkage Constraints

- When using the decomposed UN votes (unvotes_rt, unvotes_resid), both components and both interactions (p_unvotes_rt_elecex, p_unvotes_resid_elecex) must enter the regression together
- When splitting elections by competitiveness, both competitive and non-competitive election indicators and their interactions must enter together
- When using log ODA, all macro controls (pop, gdp2000, pop_donor, gdp2000_donor) must also be logged
- The interaction term p_unvotes_elecex must always be accompanied by its constituent terms (unvotes, i_elecex) in the regression
- The balanced panel restriction (no gaps) is always enforced by the estsample function
