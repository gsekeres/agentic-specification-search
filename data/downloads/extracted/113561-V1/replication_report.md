# Replication Report: 113561-V1

## Summary
- **Paper**: "What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty" by Christina M. Fong and Erzo F.P. Luttmer, AEJ: Applied Economics, 2009, 1(2), 64-87.
- **Replication status**: full
- **Total regressions in original package**: 203
- **Regressions in scope (main + key robustness)**: 74
- **Successfully replicated**: 74
- **Match breakdown**: 0 exact, 72 close, 2 discrepant, 0 failed

## Data Description
- Files used: `katrina.dta` (single data file, 1530 observations)
- Key variables:
  - Outcomes: `giving` (experimental giving 0-100), `hypgiv_tc500` (hypothetical giving, topcoded at 500), `subjsupchar` (subjective support for charity spending, 1-7), `subjsupgov` (subjective support for government spending, 1-7), `per_hfhdif` (perceived racial difference in Habitat recipients)
  - Treatments: `picshowblack` (pictures show black victims), `picraceb` (pictures with black victims, whether race obscured or not), `picobscur` (race-obscured picture treatment)
- Sample sizes: 1343 after excluding respondents who could not hear audio (soundcheck!=3) and those with missing giving data (5 observations). Matches the original code's sample selection.

## Replication Details

### Table 3: Effects on Perceived Race and Giving (6 regressions)
- Column 1: Manipulation check (per_hfhdif ~ picshowblack), N=1321, weighted by tweight
- Column 2: Baseline giving regression, N=1343, weighted by tweight
- Columns 3-4: Interaction with respondent race and subjective identification, non-other sample
- Columns 5-6: White-only and black-only subsamples (unweighted)
- All estimates replicated with WLS + HC1 robust standard errors

### Table 4: Results by Race and Measure of Generosity (12 regressions)
- 4 panels (giving, hypothetical giving, charity support, government support) x 3 columns (all, white, black)
- Uses nraudworthy variable (combined worthiness manipulation count) instead of separate audio manipulation dummies
- All weighted by tweight with HC1 robust SEs

### Table 5: Robustness Checks (32 regressions)
- 4 panels x 8 specifications, white respondents only
- Specifications: baseline, main sample only (mweight), Slidell only, Biloxi only, no demographics, extra controls, censored reg/ordered probit, race-shown only
- OLS specifications (28): Replicated with WLS + HC1
- cnreg specifications (2): Replicated with manual MLE for censored normal regression. SEs from inverse Hessian (non-robust), may differ slightly from Stata cnreg which uses analytic weights.
- oprob specifications (2): Replicated with statsmodels OrderedModel. However, pweights are not supported in statsmodels; the model was estimated unweighted. Additionally, the full model with ~30 regressors and 7 outcome categories produced numerically unstable Hessians, resulting in NaN standard errors. Coefficients were estimated but are unreliable without proper weight handling.
- Race-shown subsample (s8): `picobscur` is constant 0 and `picraceb = picshowblack`, creating perfect collinearity. Our code automatically drops collinear variables, matching Stata's behavior.

### Table 6: Interactions with Subjective Racial Attitudes (24 regressions)
- 6 rows (3 attitude variables x 2 race panels) x 4 columns (4 outcome variables)
- Uses the `interactd` program which creates int1_* and int0_* interaction variables
- NOTE: The `interactd` program in the do-file runs unweighted OLS with robust SEs (no `[pw=tweight]`), unlike Tables 3-5
- The Stata `int?_*` glob expands alphabetically: int0_picobscur, int0_picraceb, int0_picshowb, int1_picobscur, int1_picraceb, int1_picshowb. Our code matches this ordering.
- Collinearity detection handles cases where interaction variables are linearly dependent

## Translation Notes
- Original language: Stata
- Translation approach: All OLS regressions translated using statsmodels WLS with HC1 robust standard errors, matching Stata's `reg ... [pw=w], robust`. The `cnreg` (censored normal regression) was translated using manual MLE with scipy.optimize. Ordered probit was translated using statsmodels OrderedModel.
- Known limitations:
  - Stata's `cnreg` supports analytic weights in the likelihood; our MLE implementation does include weights in the likelihood but computes SEs from the Hessian rather than using the sandwich estimator. This may produce slightly different SEs than Stata.
  - Stata's `oprob` with pweights uses pseudo-likelihood estimation. statsmodels OrderedModel does not support pweights. Our implementation is unweighted, which will produce different point estimates and SEs.
  - For the ordered probit with the full set of ~30 regressors, the Hessian was numerically unstable, resulting in NaN standard errors.

UNLISTED_METHOD: cnreg in 113561-V1 -- Stata censored normal regression (generalized Tobit with indicator for left/right/uncensored). Implemented via manual MLE.
UNLISTED_METHOD: oprob in 113561-V1 -- Stata ordered probit with pweights and robust SEs. Approximated with unweighted statsmodels OrderedModel.

## Software Stack
- Language: Python 3.12
- Key packages: statsmodels 0.14.6, pandas, numpy, scipy
