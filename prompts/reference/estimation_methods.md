# Estimation Methods Reference

This file maps Stata (and Matlab/R/Julia) estimation commands to Python equivalents.
Agents should consult this when translating replication code.

**If you encounter a command not listed here, flag it** in your output with:
```
UNLISTED_METHOD: <command> in <paper_id> — <brief description>
```
so it can be added to this reference.

---

## Python packages available

| Package | Version | Import |
|---------|---------|--------|
| pyfixest | 0.40+ | `import pyfixest as pf` |
| statsmodels | 0.14.6 | `import statsmodels.api as sm` / `import statsmodels.formula.api as smf` |
| linearmodels | 6.1 | `from linearmodels.iv import IV2SLS` / `from linearmodels.panel import PanelOLS` |
| rdrobust | 1.3 | `from rdrobust import rdrobust` |
| scipy | 1.10+ | `from scipy.optimize import minimize` |
| gEconpy | 2.0+ | `import gEconpy` (DSGE modeling — may have import issues) |
| dolo | 0.4+ | `from dolo import yaml_import` (DSGE toolkit — model files in YAML) |

---

## 1. OLS and Fixed Effects

### `reg y x1 x2, robust`
**Python (pyfixest):**
```python
m = pf.feols("y ~ x1 + x2", data=df, vcov="hetero")
```
**Notes:** Stata `robust` = HC1. pyfixest `vcov="hetero"` uses HC1 by default.

### `reg y x1 x2, cluster(cl)`
```python
m = pf.feols("y ~ x1 + x2", data=df, vcov={"CRV1": "cl"})
```
**Notes:** Stata `cluster()` uses small-sample-adjusted CR (equivalent to CRV1).

### `areg y x1 x2, absorb(fe) robust`
```python
m = pf.feols("y ~ x1 + x2 | fe", data=df, vcov="hetero")
```
**Notes:** `areg` absorbs one set of FE. pyfixest `| fe` does the same.

### `areg y x1 x2, absorb(fe) cluster(cl)`
```python
m = pf.feols("y ~ x1 + x2 | fe", data=df, vcov={"CRV1": "cl"})
```

### `reghdfe y x1 x2, absorb(fe1 fe2) cluster(cl)`
```python
m = pf.feols("y ~ x1 + x2 | fe1 + fe2", data=df, vcov={"CRV1": "cl"})
```
**Notes:** `reghdfe` drops singleton groups by default; pyfixest does too.

### `reg y x1 x2 [aw=w], robust`
```python
m = pf.feols("y ~ x1 + x2", data=df, vcov="hetero", weights="w")
```
**Notes:** Stata analytic weights `[aw=w]` → pyfixest `weights` parameter.

### `reg y x1 x2 [pw=w], cluster(cl)`
```python
m = pf.feols("y ~ x1 + x2", data=df, vcov={"CRV1": "cl"}, weights="w")
```

---

## 2. Panel Models

### `xtreg y x1 x2, fe cluster(id)`
```python
m = pf.feols("y ~ x1 + x2 | id", data=df, vcov={"CRV1": "id"})
```

### `xtreg y x1 x2, fe robust`
```python
m = pf.feols("y ~ x1 + x2 | id", data=df, vcov="hetero")
```

### `xtreg y x1 x2, re`
```python
from linearmodels.panel import RandomEffects
df_panel = df.set_index(["id", "time"])
m = RandomEffects(df_panel["y"], df_panel[["x1", "x2"]]).fit()
```

### `xtpcse y x1 x2, c(ar1)`
**Python:** No direct equivalent. Use `linearmodels.panel.PanelOLS` with appropriate covariance or `statsmodels` GLS.
```python
from linearmodels.panel import PanelOLS
m = PanelOLS(df_panel["y"], df_panel[["x1","x2"]], entity_effects=True).fit(cov_type="kernel")
```

### `xtregar y x1 x2, fe`
**Python:** Panel FE with AR(1) errors. No direct equivalent; approximate with:
```python
# Estimate FE model, then check residual autocorrelation
m = pf.feols("y ~ x1 + x2 | id", data=df, vcov="hetero")
# For Prais-Winsten-type correction, use iterative GLS manually
```

---

## 3. Instrumental Variables

### `ivreg2 y (x = z1 z2), robust`
```python
m = pf.feols("y ~ 1 | x ~ z1 + z2", data=df, vcov="hetero")
```

### `ivreg2 y x1 (x2 = z), robust cluster(cl)`
```python
m = pf.feols("y ~ x1 | x2 ~ z", data=df, vcov={"CRV1": "cl"})
```

### `ivreghdfe y (x = z), absorb(fe1 fe2) cluster(cl)`
```python
m = pf.feols("y ~ 1 | fe1 + fe2 | x ~ z", data=df, vcov={"CRV1": "cl"})
```

### `xtivreg y x1 (x2 = z), fe`
```python
m = pf.feols("y ~ x1 | id | x2 ~ z", data=df, vcov="hetero")
```

### Alternative: linearmodels IV
```python
from linearmodels.iv import IV2SLS
m = IV2SLS(df["y"], df[["x1"]], df[["x2"]], df[["z"]]).fit(cov_type="robust")
```

---

## 4. Limited Dependent Variable Models

### `logit y x1 x2, robust`
```python
m = smf.logit("y ~ x1 + x2", data=df).fit(cov_type="HC1")
```
**Notes:** `cov_type="HC1"` matches Stata `robust`.

### `probit y x1 x2, robust`
```python
m = smf.probit("y ~ x1 + x2", data=df).fit(cov_type="HC1")
```

### `logit y x1 x2, cluster(cl)`
```python
m = smf.logit("y ~ x1 + x2", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["cl"]})
```

### `ologit y x1 x2` (ordered logit)
```python
from statsmodels.miscmodels.orderedmodel import OrderedModel
m = OrderedModel(df["y"], df[["x1","x2"]], distr="logit").fit()
```

### `mlogit y x1 x2` (multinomial logit)
```python
m = smf.mnlogit("y ~ x1 + x2", data=df).fit()
```

### `clogit y x1 x2, group(g)` (conditional logit)
**Python:** No direct statsmodels equivalent. Use `FixedEffectLogit` if available, or manual conditional likelihood.

---

## 5. Tobit (Censored Regression)

### `tobit y x1 x2, ll(0)`
**Python:** statsmodels 0.14.6 does NOT have a built-in Tobit model. Use manual MLE:
```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def tobit_negll(params, y, X, lower=0):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    xb = X @ beta
    censored = (y <= lower)
    ll_cens = norm.logcdf((lower - xb[censored]) / sigma)
    ll_unc = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * ((y[~censored] - xb[~censored]) / sigma) ** 2
    return -(ll_cens.sum() + ll_unc.sum())

X = sm.add_constant(df[["x1", "x2"]].values)
y = df["y"].values
init = np.zeros(X.shape[1] + 1)  # betas + log_sigma
res = minimize(tobit_negll, init, args=(y, X, 0), method="BFGS")
# Standard errors via inverse Hessian
from scipy.optimize import approx_fprime
H = np.zeros((len(res.x), len(res.x)))
for i in range(len(res.x)):
    def f(p): return approx_fprime(p, lambda p: tobit_negll(p, y, X, 0), 1e-5)[i]
    H[i] = approx_fprime(res.x, f, 1e-5)
se = np.sqrt(np.diag(np.linalg.inv(H)))
```
**Notes:** This gives MLE estimates matching Stata's `tobit`. For `ll(0)`, set `lower=0`. For `ul(1)`, add upper censoring to the log-likelihood.

### `tobit y x1 x2, ll(0) ul(1)` (two-sided censoring)
Same as above but add upper-censoring term to the log-likelihood.

---

## 6. Count Models

### `poisson y x1 x2, robust`
```python
m = smf.poisson("y ~ x1 + x2", data=df).fit(cov_type="HC1")
```

### `nbreg y x1 x2, robust` (negative binomial)
```python
m = smf.negativebinomial("y ~ x1 + x2", data=df).fit(cov_type="HC1")
```
**Notes:** Convergence can be slow with many dummies. If it fails, try setting `start_params` or reducing the model.

### `xtpoisson y x1 x2, fe`
```python
m = pf.fepois("y ~ x1 + x2 | id", data=df)
```

### `zip y x1 x2, inflate(z1 z2)` (zero-inflated Poisson)
```python
from statsmodels.discrete.count_model import ZeroInflatedPoisson
m = ZeroInflatedPoisson(df["y"], df[["x1","x2"]], exog_infl=df[["z1","z2"]]).fit()
```

### `zinb y x1 x2, inflate(z1 z2)` (zero-inflated negative binomial)
```python
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
m = ZeroInflatedNegativeBinomialP(df["y"], df[["x1","x2"]], exog_infl=df[["z1","z2"]]).fit()
```

---

## 7. Duration / Survival Models

### `stcox x1 x2, robust`
```python
from statsmodels.duration.hazard_regression import PHReg
m = PHReg(df["duration"], df[["x1","x2"]], status=df["event"]).fit()
```
**Notes:** Stata's `stset` defines the survival time and failure indicator before `stcox`. Extract these from the stset commands.

### `streg x1 x2, dist(weibull)`
**Python:** Use `lifelines` (install separately) or manual MLE:
```python
# pip install lifelines
from lifelines import WeibullAFTFitter
wf = WeibullAFTFitter()
wf.fit(df, duration_col="T", event_col="E")
```

---

## 8. Heckman Selection / Treatment Models

### `heckman y x1 x2, select(s1 s2)`
**Python:** No built-in statsmodels Heckman. Two-step manual approach:
```python
# Step 1: Probit selection equation
probit = smf.probit("selected ~ s1 + s2", data=df).fit()
df["imr"] = norm.pdf(probit.fittedvalues) / norm.cdf(probit.fittedvalues)  # inverse Mills ratio
# Step 2: OLS with IMR
m = smf.ols("y ~ x1 + x2 + imr", data=df[df["selected"]==1]).fit(cov_type="HC1")
```
**Notes:** Two-step Heckman. For MLE, implement manually.

### `treatreg y x1, treat(d = z1 z2)`
**Python:** Similar two-step approach using probit first stage + OLS second stage with hazard correction.

---

## 9. Quantile Regression

### `qreg y x1 x2, quantile(0.5)`
```python
from statsmodels.regression.quantile_regression import QuantReg
m = QuantReg(df["y"], sm.add_constant(df[["x1","x2"]])).fit(q=0.5)
```

---

## 10. GLS / Feasible GLS

### `newey y x1 x2, lag(4)` (Newey-West HAC)
```python
m = smf.ols("y ~ x1 + x2", data=df).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
```

### `prais y x1 x2` (Prais-Winsten AR(1) GLS)
```python
from statsmodels.regression.linear_model import GLSAR
m = GLSAR(df["y"], sm.add_constant(df[["x1","x2"]]), rho=1).iterative_fit()
```

### `xtgls y x1 x2, panels(correlated)`
```python
from linearmodels.panel import PanelOLS
# SUR-type estimation; use linearmodels SUR or manual FGLS
```

---

## 11. Matching Estimators

### `nnmatch y d x1 x2, m(4) tc(att)`
**Python:** No direct single-function equivalent. Use:
```python
from sklearn.neighbors import NearestNeighbors
# Manual matching: find k nearest neighbors on covariates, compute ATT
nn = NearestNeighbors(n_neighbors=4).fit(df.loc[df["d"]==0, ["x1","x2"]])
distances, indices = nn.kneighbors(df.loc[df["d"]==1, ["x1","x2"]])
# Then compute matched difference in outcomes
```
**Notes:** This is approximate. For exact Abadie-Imbens matching with bias correction, no standard Python package exists. Flag and note the approximation.

### `psmatch2 d x1 x2, outcome(y) n(1) ate`
**Python:** Propensity score matching:
```python
# Step 1: Estimate propensity score
ps = smf.logit("d ~ x1 + x2", data=df).fit()
df["pscore"] = ps.predict()
# Step 2: Match on propensity score using nearest neighbors
```

### `teffects psmatch (y) (d x1 x2)`
Same as psmatch2 approach above.

---

## 12. Regression Discontinuity

### `rdrobust y x, c(0)`
```python
from rdrobust import rdrobust
result = rdrobust(df["y"], df["x"], c=0)
```

### `rdplot y x, c(0)`
```python
from rdrobust import rdplot
rdplot(df["y"], df["x"], c=0)
```

---

## 13. Panel VAR

### `pvar y1 y2 y3, lag(2) gmm` (Stata `pvar` by Inessa Love)

Stata's `pvar` implements system-GMM panel VAR with Helmert (forward orthogonal deviations) transformation and Arellano-Bover instruments.

**Python — Method A: Within-transformation + VAR (simpler, approximate)**
```python
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd

# Within-transform (demean by entity)
endogs = ["y1", "y2", "y3"]
df_dm = df.copy()
for v in endogs:
    df_dm[v] = df.groupby("entity")[v].transform(lambda x: x - x.mean())

# Fit VAR on stacked demeaned data (treats panel as single time series — approximate)
m = VAR(df_dm.groupby("entity")[endogs].apply(lambda g: g)).fit(maxlags=2)
irf = m.irf(periods=20)
```

**Python — Method B: Helmert + equation-by-equation GMM (closer to Stata `pvar`)**
```python
import numpy as np
import pandas as pd
from scipy.linalg import inv

def helmert_transform(df, entity_col, time_col, vars):
    """Forward orthogonal deviations (Arellano-Bover 1995)."""
    out = df.copy()
    for v in vars:
        vals = []
        for _, g in df.groupby(entity_col):
            g = g.sort_values(time_col)
            x = g[v].values
            T = len(x)
            h = np.full(T, np.nan)
            for t in range(T - 1):
                s = T - t - 1
                fwd_mean = x[t+1:].mean()
                h[t] = np.sqrt(s / (s + 1)) * (x[t] - fwd_mean)
            vals.extend(h)
        out[v + "_h"] = vals
    return out.dropna(subset=[v + "_h" for v in vars])

# Apply Helmert, then run VAR on transformed variables
df_h = helmert_transform(df, "entity", "time", endogs)
m = VAR(df_h[[v + "_h" for v in endogs]]).fit(maxlags=2)
irf = m.irf(periods=20)

# For Monte Carlo confidence bands on IRFs:
irf_err = m.irf_errband_mc(orth=True, repl=500, steps=20)
```

**Notes:**
- Method A is a rough approximation; Method B with Helmert is closer to Stata's `pvar`.
- Neither method exactly replicates `pvar`'s GMM instrumentation (which uses lagged levels as instruments for the Helmert-transformed equations). For exact replication, manual GMM with `scipy.optimize` is needed.
- Stata's `pvar` also computes Granger causality tests and forecast error variance decomposition (FEVD). Use `m.test_causality()` and `m.fevd()` from statsmodels VAR.

### `xtvar y1 y2 y3, lags(2)` (LSDV panel VAR)
```python
# LSDV = entity fixed effects + VAR lags (Nickell bias for short T)
import pyfixest as pf

# Equation-by-equation with entity FE and lags
for dep in endogs:
    lag_terms = " + ".join([f"L{l}_{v}" for l in range(1,3) for v in endogs])
    m = pf.feols(f"{dep} ~ {lag_terms} | entity", data=df_with_lags, vcov="hetero")
```

---

## 14. Mixed / Multilevel Models

### `mixed y x1 x2 || group:`
```python
import statsmodels.formula.api as smf
m = smf.mixedlm("y ~ x1 + x2", data=df, groups=df["group"]).fit()
```

### `meglm y x1 x2 || group:, family(binomial)`
```python
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
m = GEE.from_formula("y ~ x1 + x2", groups="group", data=df, family=Binomial()).fit()
```

---

## 15. Structural / Calibration (Matlab → Python)

### `fminunc(fun, x0)` (unconstrained optimization)
```python
from scipy.optimize import minimize
res = minimize(fun, x0, method="BFGS")
```

### `fminsearch(fun, x0)` (Nelder-Mead)
```python
res = minimize(fun, x0, method="Nelder-Mead")
```

### `fmincon(fun, x0, A, b)` (constrained optimization)
```python
from scipy.optimize import minimize
res = minimize(fun, x0, method="SLSQP", constraints={"type": "ineq", "fun": lambda x: b - A @ x})
```

### Dynare `.mod` files

**Python — For simple linear DSGE models (first-order perturbation):**
```python
import numpy as np
from scipy.linalg import ordqz, solve

def solve_linear_dsge(A, B):
    """
    Solve E_t[A * x_{t+1}] = B * x_t via generalized Schur (QZ) decomposition.
    Returns policy function x_t = P * x_{t-1} + Q * e_t.
    """
    AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='ouc')
    # Partition by stable/unstable eigenvalues
    n_stable = np.sum(np.abs(beta) > np.abs(alpha))
    Z11 = Z[:n_stable, :n_stable]
    Z21 = Z[n_stable:, :n_stable]
    P = Z21 @ np.linalg.inv(Z11)  # Policy function for jump vars
    return P
```

**Python — For medium-scale DSGE (dolo toolkit):**
```python
# pip install dolo (may have compatibility issues)
# dolo uses YAML model files, not .mod files — requires manual translation
from dolo import yaml_import
model = yaml_import("model.yaml")
from dolo.algos.perturbation import approximate_controls
dr = approximate_controls(model, order=1)
```

**Python — For `stoch_simul` (model simulation / moments):**
```python
# After solving for policy matrices P, Q:
def simulate_dsge(P, Q, T=1000, shock_cov=None):
    n = P.shape[0]
    x = np.zeros((T, n))
    shocks = np.random.multivariate_normal(np.zeros(Q.shape[1]), shock_cov, T)
    for t in range(1, T):
        x[t] = P @ x[t-1] + Q @ shocks[t]
    return x

# Model moments
sim = simulate_dsge(P, Q, T=10000, shock_cov=Sigma)
model_moments = np.cov(sim.T)  # Compare to paper's Table
```

**Notes:**
- Most Dynare papers use medium-to-large-scale nonlinear DSGE models that require exact Dynare replication. Mark as "not possible" if the paper's main results depend on complex Dynare simulation (multi-sector, nonlinear, higher-order perturbation, Bayesian estimation).
- For papers where Dynare is used ONLY for simple first-order perturbation of a small model (< 10 equations), manual Python replication is feasible using the QZ approach above.
- Bayesian DSGE estimation (`estimation` command in Dynare) can potentially use PyMC or `pydsge` (`pip install pydsge`), but exact replication is unlikely.

---

## 16. Post-Estimation Commands

### `predict yhat, xb` → `m.predict()` or `m.fittedvalues`
### `predict resid, residuals` → `m.resid`
### `test x1 = x2` → `m.wald_test("x1 = x2")` or `from scipy.stats import f as f_dist`
### `lincom x1 + x2` → manual: `coef = m.params["x1"] + m.params["x2"]`; SE via delta method
### `nlcom` → manual delta method or `scipy.optimize` for nonlinear transformations
### `margins` → manual computation of marginal effects; for logit/probit: `m.get_margeff()`
### `outreg2` / `estout` / `esttab` → ignore (output formatting only)

---

## 17. Stata Data/Sample Commands to Watch For

| Stata | Python Equivalent |
|-------|-------------------|
| `if condition` | `df[df["condition"]]` or `df.query("condition")` |
| `[aw=w]` | `weights="w"` in pyfixest |
| `[fw=w]` | Expand rows: `df.loc[df.index.repeat(df["w"])]` |
| `[pw=w]` | Same as `aw` for regression |
| `i.var` (factor) | `C(var)` in formula |
| `c.var` (continuous) | Just use `var` |
| `c.x1#c.x2` (interaction) | `x1:x2` in formula or create manually |
| `i.x1#i.x2` | `C(x1):C(x2)` |
| `xi: i.year` | Create dummies: `pd.get_dummies(df["year"])` |
| `egen` | `df.groupby().transform()` |
| `collapse` | `df.groupby().agg()` |
| `xtset id time` | `df = df.set_index(["id", "time"])` |

---

## 18. Common Gotchas

1. **Singleton FE**: Stata's `reghdfe` drops singletons by default. pyfixest does too, but `areg` does NOT. When replicating `areg`, check if singletons matter.

2. **Missing values**: Stata drops rows with ANY missing value in the regression variables. Ensure your Python code does the same via `dropna(subset=[...])` on all variables in the formula.

3. **Factor variables**: Stata's `i.var` auto-detects the base category (usually the most frequent or lowest). Python `C(var)` defaults to the first category alphabetically. Explicitly set base: `C(var, Treatment(reference=base_val))`.

4. **Degrees of freedom**: Stata adjusts SE for absorbed FE in `areg` (dof correction). pyfixest handles this automatically. `reg` with dummies does NOT adjust — compare carefully.

5. **Convergence**: Stata's `logit`/`probit`/`nbreg` use different default optimization methods than Python. If convergence differs, try `method="bfgs"` or `method="newton"` in statsmodels `.fit()`.

6. **Clustering with FE**: Stata's `areg y x, absorb(fe) cluster(cl)` adjusts the cluster VCE for the absorbed FE. pyfixest handles this correctly with `vcov={"CRV1": "cl"}`.

7. **Standard error types**:
   - Stata `robust` = HC1 (small-sample adjusted)
   - Stata `cluster(cl)` = CRV1 (small-sample adjusted)
   - pyfixest `vcov="hetero"` = HC1 (matches)
   - pyfixest `vcov={"CRV1": "cl"}` = CRV1 (matches)
   - statsmodels default `cov_type="nonrobust"` = classical OLS SE (NOT robust)

8. **Reading `.dta` files**: Use `pd.read_stata()`. For very old Stata formats, may need `convert_categoricals=False`.
