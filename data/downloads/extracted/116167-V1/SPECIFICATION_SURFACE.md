# Specification Surface: 116167-V1

## Paper: Durante, Pinotti & Tesei -- "The Political Legacy of Entertainment TV" (AER)

This paper studies the causal effect of early exposure to Berlusconi's Mediaset TV on political outcomes in Italy, using quasi-random variation in TV signal availability driven by geographic/topographic factors.

---

## Baseline Groups

### G1: Effect of Mediaset exposure on Forza Italia vote share in 1994

- **Design**: Cross-sectional OLS with district FE
- **Outcome**: berl (Forza Italia vote share in 1994)
- **Treatment**: signal (continuous Mediaset TV signal exposure)
- **Estimand**: Causal effect of Mediaset exposure on voting for Berlusconi
- **Target population**: Italian municipalities (trimmed top/bottom 2.5% of signal distribution)
- **Selection story**: TV signal varies quasi-randomly due to geographic/topographic factors (terrain, transmitter locations)

The paper's Table 3 is the core result, showing the effect of signal on berl across progressively richer specifications:
- Column 1: Bivariate OLS (no controls)
- Column 2: Signal + signalfree
- Column 3: + land controls
- Column 4: + land controls + district FE + SLL FE
- Column 5: + socioeconomic controls (preferred baseline)
- Column 6: Unweighted
- Column 7: Capped signal

All regressions use population weights (pop81) and cluster at the district level.

---

## Baseline Specs

- **Table3-Col5**: signal + signalfree + land controls + socioeconomic controls, district FE + SLL FE, cluster(district), [w=pop81]
- **Table3-Col4**: signal + signalfree + land controls only, district FE + SLL FE
- Additional baselines: no controls, land only, unweighted

---

## Core Universe

### Design variants
None beyond OLS -- the paper's identification relies on the selection-on-observables argument that geographic signal variation is exogenous conditional on controls and FE.

### RC axes
- **Controls**: LOO drops of each covariate; control set progressions (none, signal only, land, full); adding civic81
- **FE**: Drop/add SLL FE, drop/add district FE, add province FE
- **Sample restrictions**: Exclude provincial capitals, population caps (100k, 50k, 10k) -- from Table 4
- **Signal trimming**: Trim top/bottom 5% or 10% of signal
- **Weights**: Weighted vs unweighted
- **Treatment definition**: Capped signal (capsignal) from Table 3 Col 7
- **Data/matching**: Matched-neighbor estimates from Table 4 Cols 6-8 (pairs of municipalities with similar signalfree but different signal)
- **Election years**: 1994, 1996, 2001, 2006, 2008, 2013 -- the paper's Table 5 time-series of effects

### Excluded from core
- Individual-level regressions (ITANES survey, PIAAC) -- different data/unit of analysis, better as exploration
- Party manifesto analysis, TV show content analysis -- mechanism exploration
- Civic engagement outcomes (Table A10) -- different outcome concept
- Table 5 multiple parties -- the focal claim is about Forza Italia/Berlusconi

---

## Constraints

- signalfree should generally be included when signal is included (geographic confound control)
- Control-count envelope: 0-9
- Land controls (area, area^2, altitude, altitude^2, ruggedness) are often treated as a block
- Population weights (pop81) are the default; unweighted is a robustness check

---

## Inference Plan

- **Canonical**: Cluster at district level (matching the code)
- **Variants**: SLL clustering, two-way clustering, and Conley spatial SEs at 10/30/50km (the paper's Table A3 computes spatial SEs using the acreg command)

The spatial SE variants are particularly important given the geographic nature of the treatment variation.

---

## Budget

- Total core specs: up to 80
- No controls-subset sampling needed (manageable control pool)
- Full enumeration is feasible
