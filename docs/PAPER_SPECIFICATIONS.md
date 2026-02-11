# Paper Specification Descriptions and Justifications

This file contains descriptions and justifications for the specification searches run on each paper.

> Note: This document is a **legacy narrative artifact** from earlier, non-surface-driven runs. The current workflow records per-paper choices in:
>
> - `{EXTRACTED_PACKAGE_PATH}/SPECIFICATION_SURFACE.md` (pre-run universe + constraints + budgets)
> - `{EXTRACTED_PACKAGE_PATH}/SPEC_SURFACE_REVIEW.md` (pre-run verifier critique/edits)
> - `{EXTRACTED_PACKAGE_PATH}/SPECIFICATION_SEARCH.md` (execution log)
> - `data/verification/{PAPER_ID}/VERIFICATION_REPORT.md` (post-run audit + core/non-core classification)
>
> We keep this file for context, but it may be removed once the repo is fully re-run under the surface-driven protocol.

---

## 114705-V1: The Role of Information in Disability Insurance Application

### Paper Summary
Studies the effect of the Social Security Statement -- which provides personalized SSDI benefit information -- on disability insurance application rates. Exploits the phased rollout of the Statement (1995-2000) as a natural experiment, using restricted-use HRS data merged with SSA administrative records.

### Baseline Specification
Linear probability model: `disability_applied ~ first_receipt + married + C(DEGREE) + C(GENDER) + C(age) + C(year) + ssa_redux [aw=fweight], cluster(BIRTHYR)`
- disability_applied = binary SSDI application indicator
- first_receipt = binary treatment (received SS Statement)
- Sample: alive, fully insured, cumulative applications <= 1, age 50-64

### Covariates
- **Treatment**: first_receipt (SS Statement receipt)
- **Controls**: married, DEGREE (education), GENDER, ssa_redux (benefit reduction factor)
- **Fixed effects**: age, year
- **Interactions**: any_hlthlm*Stat, working_lag*Stat, replacement_rate*Stat

### Specification Variations Justification
- **Core variations**: No FE, year-only FE, age-specific trends, 5yr-age x year FE, birth-year FE, health controls, working status, replacement rate, benefit expectations
- **Sample restrictions**: Work-limited, gender, age subgroups, time periods, working status, education, marital status
- **Estimation**: OLS, WLS, logit (MFX), probit, alternative clustering
- **Leave-one-out**: Drop married, DEGREE, GENDER, ssa_redux
- **Placebo**: 5 random permutations + pre-treatment placebo

### Results
- **Total specifications**: 69 (63 main + 6 placebo)
- **Significance rate**: 58.7% at 5% (main specs)
- **Sign consistency**: 98.4% positive (main specs)
- **Key finding**: Effect concentrated among work-limited individuals (coef ~0.036 vs ~0.011 full sample)

### Data Note
Restricted-use data not available; specification search uses synthetic data matching the paper's variable structure.

---

## 239292-V1: Jargon and Shared Language in Team Communication Networks

### Paper Summary
Studies jargon and shared language development in team communication networks using a laboratory experiment where teams of participants communicate to solve coordination tasks across 15 trials. Teams are assigned to different network structures (Fully Connected, Double Bridge, Consensus Builder, and Wheel networks).

### Baseline Specification
The original paper uses Stata's generalized structural equation model (gsem) with two simultaneous equations and team random effects:
- **Equation 1 (Shared Language)**: sl ~ network*lntrial + jargon*lntrial + trial10 + trial_term + shared1-5 + team[RE]
- **Equation 2 (Team Accuracy)**: correct ~ network*sl + jargon*sl + network*vel + lntrial + trial10 + trial_term + shared1-5 + team[RE], binomial(5) logit

### Covariates
- **Treatment**: jargon, jargon_lntrial, jargon_sl
- **Controls**: lntrial, net1/net2/net4 (network dummies), vel (message velocity), trial10, trial_term, shared1-5 (symbol FE)

### Specification Variations Justification
- **Core variations**: Controls-only model, OLS with clustered SE, team FE (more conservative), FC vs non-FC comparison, early/late trials split, count outcome, no shared FE, no network effects, linear trial
- **Leave-one-out**: Tests sensitivity of main effects to each control variable
- **Single-covariate**: Reveals potential confounding patterns

### Results
- **Total specifications**: 203
- **Baseline, Core, Leave-one-out, Single-covariate**: 6, 33, 82, 82

---

## 223561-V1: Life Cycle Wage Growth and Internal Migration

### Paper Summary
Studies the relationship between life-cycle wage growth (returns to work experience) and regional economic development. Key finding: workers in richer regions experience steeper wage growth over their careers. Uses microdata from IPUMS-International for the United States, Mexico, and Brazil.

### Baseline Specification
State-level regression: `avg_gain_s = alpha + beta * log(GDP_per_capita_s) + epsilon_s`
- avg_gain_s = Average wage growth (in %) relative to workers with <5 years experience
- Results: Brazil (coef=17.75, R²=0.617), Mexico (coef=9.10, R²=0.404), US (coef=17.58, R²=0.160)

### Covariates
- **Treatment**: log_y (log GDP per capita)
- **Counterfactuals**: counter_ed (education), counter_oc (occupation), counter_ind (industry)

### Specification Variations Justification
- **Core variations**: Counterfactual outcomes, gain_5 (returns at 25-30 years experience), pooled sample, country FE, quadratic form, GDP in levels, median regression, sample restrictions
- **Leave-one-out**: Jackknife dropping each state (107 specs: 25 Brazil + 32 Mexico + 50 US)
- **Single-covariate**: Counterfactual variables as sole predictors, log_y with counterfactual controls

### Results
- **Total specifications**: 163
- **Significance rate**: 95.7% (156/163 significant at 5%)
- **Jackknife stability**: Brazil (mean 17.77, std 0.50), Mexico (mean 9.12, std 0.53), US (mean 17.58, std 0.73)

---

## 237688-V4: Debt Types and Economic Growth

### Paper Summary
Studies the relationship between different types of debt (household, corporate, and public) and economic growth using panel data from 22 OECD countries (2000-2019). Uses Pooled Mean Group (PMG) estimation with ARDL error correction model.

### Baseline Specification
Effect of household debt on GDP growth: `dy ~ dhh + dp + dshort + country_FE`
- dy = first difference of log GDP (GDP growth)
- dhh = first difference of log household debt
- dp = inflation, dshort = short-term interest rate change

### Covariates
- **Treatment**: dhh (household debt), dcor (corporate debt), dpub (public debt)
- **Controls**: dp (inflation), dshort (short-term rate), dlong (long-term rate), L_dy (lagged growth)

### Specification Variations Justification
- **Core variations**: Different debt types as treatment, no interest rate controls, with lagged DV, long vs short-term rates, pooled OLS, pre/post-2008 splits, excluding PIIGS, reverse causality
- **Leave-one-out**: Drop dp (inflation), drop dshort (interest rate)
- **Single-covariate**: dhh + dp only, dhh + dshort only, dhh only

### Results
- **Total specifications**: 56
- **Key finding**: Sign reversal between pre-2008 (coef=0.008, ns) and post-2008 (coef=-0.122, sig negative)
- **Baseline**: Household debt coefficient 0.031 (marginally significant, p=0.05)

---

## 237888-V3: Regional Impacts of International Tourism Boycott (China-Japan)

### Paper Summary
Studies the regional economic impacts of the Chinese tourism boycott against Japan beginning August 2012, following the Senkaku/Diaoyu islands territorial dispute. Uses triple-difference (DDD) design exploiting variation in country of origin, time, and prefecture-level China dependency.

### Baseline Specification
PPML regression: `visitor_ipt ~ sCHNPost + sPost + sCHN + CHNPost | country + time + prefecture`
- sCHNPost: Triple interaction (main coefficient of interest)
- s: Prefecture's pre-boycott share of Chinese visitors
- Clustering: Prefecture level

### Covariates
- **Treatment**: sCHNPost (triple-difference term)
- **Controls**: sPost (China dependency × Post), sCHN (China dependency × China), CHNPost (China × Post)
- **FE**: Country, year-month, prefecture

### Specification Variations Justification
- **Core variations**: Time windows (6mo to 3yr), treatment definition (China only vs +HK vs +Taiwan), aggregate vs disaggregate, sample restrictions (exclude Tohoku, exclude top 4 prefectures), FE variations
- **Leave-one-out**: Drop each of sPost, sCHN, CHNPost individually
- **Single-covariate**: Treatment + each covariate alone

### Results
- **Total specifications**: 26
- **Baseline coefficient**: -1.286 (p=0.007), implying ~72% fewer visitors in high-dependency prefectures post-boycott
- **Variation**: Core specs mean -0.905 (std 1.05), some sensitivity in leave-one-out specs

---

## 233621-V3: High School Gender Composition and University Major Choice

### Paper Summary
Studies how the gender composition of a student's high school peers affects their choice of university major, particularly whether women exposed to more female peers are more likely to choose STEM fields. Uses administrative data from British Columbia, Canada linking K-12 records to university enrollment.

### Baseline Specification
`outcome = female_ratio_v2 + balance1 + balance2_vary + school_FE + year_FE + school_trend`
- Treatment: female_ratio_v2 (leave-one-out peer female ratio)
- Outcomes: STEM enrollment, University enrollment
- Clustering: School level

### Covariates
- **Individual controls (balance1)**: AGE_IN_YEARS, aboriginal, home_eng, special_need, gifted, FRENCH_IMM, HL_CHINESE, HL_PUNJABI, resident, earliest2_grd12
- **School-cohort controls (balance2_vary)**: Peer ratios of all individual controls + cohort_size

### Specification Variations Justification
- **Core variations**: No controls, individual controls only, school-cohort controls only, private schools, different grades (10, 11, 12)
- **Leave-one-out**: Drop each of 21 covariates one at a time (42 specs)
- **Single-covariate**: Treatment + each covariate alone (42 specs)

### Results
- **Total specifications**: 100
- **Baseline coefficient**: -0.048 for STEM, -0.096 for university enrollment
- **Robustness**: Highly consistent across specifications (std 0.024-0.043)
- **Note**: Uses synthetic data (actual data is confidential)

---

## 232122-V3: Climate and Migration in the United States

### Paper Summary
Examines how climate affects internal migration in the United States at the county level. Uses decadal census data (1950s-2000s) to estimate effects of temperature (HDD/CDD) and precipitation on net migration rates. Main finding: higher cooling degree days (hotter temperatures) lead to significant out-migration.

### Baseline Specification
`Net_Migration_Rate ~ HDD100s + CDD100s + prec100s | County_FE + Region_x_Decade_FE`
- Weights: 1950 county population
- Clustering: County level
- Baseline results: CDD100s = -0.916 (p<0.001), HDD100s = -0.087 (p=0.086), prec100s = -0.094 (p=0.049)

### Covariates
- **Treatment**: CDD100s (cooling degree days - heat exposure)
- **Controls**: HDD100s (heating degree days - cold exposure), prec100s (precipitation)
- **FE**: County, Region × Decade

### Specification Variations Justification
- **Core variations**: Different FE (county+decade, county+state-decade), outcome variations (winsorized, age-specific), unweighted, sample splits (by region, time period, urbanicity, income, baseline temperature), state-clustered SE, no FE
- **Leave-one-out**: Drop each covariate, drop each FE type, drop weights
- **Single-covariate**: Each climate variable alone, pairwise combinations

### Results
- **Total specifications**: 34 (90 coefficient estimates)
- **CDD100s robustness**: Baseline -0.92, consistently negative and significant
- **Heterogeneity**: Stronger in urban (-0.89) vs rural (-0.11 ns), high-income (-1.02) vs low-income (-0.31 ns)
- **Regional variation**: Effect insignificant in Northeast and Midwest

---

## 244567-V1: Interest Rate Misalignments and Monetary Policy (US States)

### Paper Summary
Studies how monetary policy affects state-level economic outcomes when the federal funds rate deviates from state-specific Taylor rules. Because the Fed sets a single national rate, some states experience policy that is too tight/loose relative to local conditions. Uses panel of 33 US states (1989-2017) with local projections.

### Baseline Specification
Local projection for headline inflation:
`reghdfe pi l(0/4).mps_raw_ffr l(1/4).pi l(0/4).pi_us l(0/4).gdp_us, absorb(time id) cluster(id)`
- Treatment: mps_raw_ffr = FFR - state-specific Taylor rule rate
- FE: State, time
- Clustering: State

### Covariates
- **Treatment**: mps_raw_ffr (policy rate gap), 4 lags
- **Controls**: Lagged outcome (pi_L1-L4), US inflation (pi_us + 4 lags), US GDP (gdp_us + 4 lags)

### Specification Variations Justification
- **Core variations**: Different outcomes (unemployment, tradable/non-tradable inflation), FE variations (state only, time only), sample splits (pre/post-2008), lag structure variations, alternative treatment (output gap Taylor rule), different SE clustering
- **Leave-one-out**: Drop each control group (8 specs)
- **Single-covariate**: Treatment + one control group at a time (6 specs)

### Results
- **Total specifications**: 27
- **Baseline coefficient**: -0.645 (p<0.001) - 1pp rate gap → 0.65pp inflation decrease
- **Significance rate**: 100% (all 27 specs significant at 5%)
- **Unemployment effect**: +0.974 (tighter policy increases unemployment)

---

## 237684-V2: Cultural Individualism and Working from Home

### Paper Summary
Studies the relationship between cultural individualism and working from home (WFH) using the epidemiological approach. Compares immigrants from different countries of origin living in the USA to identify causal effect of cultural values on work behavior.

### Baseline Specification
`wfh_any ~ idv_origin + female + age + age_sq + race | state_FE + year_month_FE, cluster(origin_country)`
- Treatment: idv_origin (Hofstede individualism score 0-100)
- Sample: CPS 1st/2nd generation immigrants, employed, ages 16-64
- N = 151,404

### Covariates
- **Treatment**: idv_origin (country of origin individualism score)
- **Controls**: female, age, age_sq, race dummies
- **FE**: State, Survey year × month

### Specification Variations Justification
- **Core variations**: WFH hours outcome, 1st vs 2nd generation only, gender subgroups, adding college control, FE variations (none, state only, full), clustering variations, additional controls (self-employment)
- **Leave-one-out**: Drop each of female, age, age_sq, race (5 specs)
- **Single-covariate**: Treatment + one control at a time (5 specs)

### Results
- **Total specifications**: 22
- **Baseline coefficient**: 0.0039 (10-point individualism increase → 3.9pp WFH increase)
- **Significance rate**: 100% (all 22 specs significant at 5%)
- **Key heterogeneity**: 1st gen (0.0050) vs 2nd gen (0.0016) - cultural persistence
- **Mechanism**: Adding college control reduces to 0.0023 (education as mechanism)

---

## 239462-V1: Nurse Practitioner Scope of Practice and WIC Enrollment

### Paper Summary
Studies the effect of Nurse Practitioner (NP) Scope of Practice (SOP) laws on WIC nutrition assistance program participation. Examines how state-level adoption of Full Practice Authority (FPA) for NPs affects WIC enrollment, using a difference-in-differences design with staggered treatment timing (2005-2019).

### Baseline Specification
`ln_total ~ did_NP_0_1 + did_NP_2_3 + did_NP_4_5 + did_NP_6_7 + did_NP_8_9 + ln_WIC_pop + covariates | state_FE + year_month_FE, cluster(state)`
- Treatment: Bucketed indicators for years 0-9 post-FPA adoption
- Outcome: Log total WIC enrollment at state-month level

### Covariates
- **Treatment**: FPA adoption indicators (bucketed by years since adoption)
- **Controls**: ln_WIC_pop, medicaid_expansion, percent_poverty, percent_under18, percent_65plus, percent_female

### Specification Variations Justification
- **Core variations**: No controls, population control only, state FE only, time FE only, exclude always-treated states, sample restrictions (pre-2015, post-2010), levels instead of logs
- **Leave-one-out**: Drop each covariate individually (6 specs)
- **Single-covariate**: Treatment + each covariate alone (6 specs)

### Results
- **Total specifications**: 26
- **Key finding**: Positive effect emerges in years 4-5 post-adoption and grows through years 8-9
- **Effect size**: Coefficients of 0.045, 0.045, and 0.063 for periods 4-5, 6-7, and 8-9 respectively

---

## 241417-V1: Household Debt and Safety-Net Participation

### Paper Summary
Studies the effect of household debt on safety-net participation, examining how state-level household debt-to-income ratios affect TANF and SNAP enrollment. Uses state-level quarterly panel (1999q4-2019q3) with dynamic panel models and two-way fixed effects.

### Baseline Specification
`reghdfe LN_TANF DEBT LN_SNAP UNEMP SNAPINDEX AFSillegal controls, absorb(state quarter) vce(cluster state)`
- Treatment: DEBT (purged household debt measure, residual from regressing log(debt/income) on log(personal income) with state FE)
- Outcome: Log TANF recipients

### Covariates
- **Treatment**: DEBT (purged debt measure)
- **Controls**: LN_SNAP, unemp, snapindex, AFSillegal, tanf1/tanf2/tanf3 (benefit levels), govdem, pov, BANK (bankruptcies), INCOME_1000

### Specification Variations Justification
- **Core variations**: SNAP as outcome, FE variations (state only, time only, pooled OLS), minimal controls, no TANF policy controls, sample splits (pre/post-2010), exclude always-AFS-illegal states
- **Leave-one-out**: Drop each of 11 covariates individually
- **Single-covariate**: Treatment + each covariate alone (12 specs)

### Results
- **Total specifications**: 33
- **Baseline coefficient**: 0.082 (SE: 0.067, p=0.226) - positive but not significant
- **Significance rate**: 0% at 5%, 27% at 10%
- **Sensitivity**: Pre-2010 shows negative effect, post-2010 near-zero; excluding AFS-illegal states increases coefficient to 0.127 (p=0.076)

---

## 114843-V1: On the Empirics of the EU Regional Policy

### Paper Summary
Studies the causal effect of EU Objective 1 structural funds on regional economic growth and investment. Regions with GDP per capita below 75% of the EU average are eligible for Objective 1 transfers. Uses a Fuzzy Regression Discontinuity Design with polynomial control functions (Wooldridge 2002, Procedure 18.1) on a panel of 257 EU regions across 3 programming periods (1989, 1994, 2000).

### Baseline Specification
Fuzzy RDD with probit first stage + 2SLS second stage:
- **First stage**: Period-by-period probit of `object1` on `elig_object1` + polynomial in `gdpcap` + `gdpcap_i` (interactions) + `hc_i`
- **Second stage**: IV regression of `growth` on `object1` + `hc_object1` (instrumented), polynomial controls, clustered by region
- Also: Panel-IV with Chamberlainian correlated random effects (Mundlak group means)

### Covariates
- **Treatment**: object1 (received Objective 1 funds)
- **Running variable**: gdpcap (GDP per capita relative to EU avg), elig_object1 (eligibility indicator)
- **Heterogeneity**: hc (human capital), hc_var, qog (quality of government), qog_var
- **Controls**: Polynomial in gdpcap (order 3-5), gdpcap*eligibility interactions, Mundlak terms

### Specification Variations Justification
- **Core variations**: Polynomial orders 3-5, investment vs growth outcome, different heterogeneity variables (hc, hc_var, qog, qog_var), pooled IV vs panel IV
- **Estimation method**: Naive OLS, LPM first stage, region FE, manual 2SLS
- **Reduced form**: Direct eligibility-to-outcome regressions
- **Sample restrictions**: Single periods, drop periods, bandwidth around threshold, donut hole, eligible only
- **Functional form**: Linear, quadratic, log outcome
- **Inference**: Robust vs clustered SE, year dummies
- **Excluded instruments**: literacy1870, protestant, capital_labor_85 (Table 8 approach)
- **Placebo**: Random treatment, first-period only

### Results
- **Total specifications**: 131
- **Baseline (Pooled IV, poly3)**: coef=0.0076, SE=0.004, p=0.061
- **Baseline (Panel IV, poly3)**: coef=0.0073, SE=0.005, p=0.111
- **Significance rate**: 64.9% at 5% overall; interaction term (hc*object1=0.053, p<0.001) is highly robust
- **Key finding**: Main treatment effect is positive but marginally significant; the interaction with human capital is the strongest result
- **Sensitivity**: Higher polynomial orders (4-5) produce numerically unstable estimates due to extreme variable magnitudes

---
