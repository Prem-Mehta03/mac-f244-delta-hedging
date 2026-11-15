# Theory notes

One short section per step, in the notation of the class notes. These sections are the markdown cells of the final notebook.

**Notation.** Brownian motion $W(t)$ on $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}, \mathbb{P})$; stock $S(t)$ with $S(0) = S_0$, drift $\alpha$ and volatility $\sigma$; risk-free rate $r$; strike $K$, maturity $T$, time to maturity $\tau = T - t$; hedge ratio $\Delta(t)$ (shares held); hedging wealth $X(t)$ with money-market part $X(t) - \Delta(t)S(t)$; call price $C(t, x)$ with $x = S(t)$ and Greeks $C_x$ (delta), $C_t$ (theta), $C_{xx}$ (gamma); normal CDF $N(\cdot)$ with density $N'(\cdot)$; time step $\Delta t_i = t_{i+1} - t_i$; realised volatility $\hat\sigma$. In the Itô-formula section the notes also write $dX = \Delta(t)\,dt + \Theta(t)\,dW$ for a general Itô process; in every hedging section $\Delta(t_i) = C_x(t_i, S(t_i))$ only.


## Step 2: BSM equation, price, Delta, Theta and Gamma (src/bs.py)

**Concept (derivation in the order of the notes).** The stock is $dS(t) = \alpha S(t)\,dt + \sigma S(t)\,dW(t)$. A hedger holds $\Delta(t)$ shares and the rest of the wealth $X(t)$ in the money market at rate $r$:

$$dX(t) = r\big(X(t) - \Delta(t)S(t)\big)dt + \Delta(t)\,dS(t) = rX(t)\,dt + \Delta(t)(\alpha - r)S(t)\,dt + \sigma\Delta(t)S(t)\,dW(t).$$

Discounted wealth satisfies $d\big(e^{-rt}X(t)\big) = \Delta(t)\,d\big(e^{-rt}S(t)\big)$. Now apply the Itô formula to $e^{-rt}C(t, S(t))$:

$$d\big(e^{-rt}C\big) = e^{-rt}\Big[-rC + C_t + \alpha S C_x + \tfrac12\sigma^2S^2C_{xx}\Big]dt + e^{-rt}\sigma S C_x\,dW.$$

Require $X(t) = C(t, S(t))$ for all $t$, starting from $X(0) = C(0, S_0)$. Matching the $dW$ terms gives $\Delta(t) = C_x(t, S(t))$; matching the $dt$ terms then gives the BSM equation

$$rC(t,x) = C_t(t,x) + rxC_x(t,x) + \tfrac12\sigma^2x^2C_{xx}(t,x), \qquad C(T,x) = (x-K)^+,$$

with $C(t,0) = 0$ and $\lim_{x\to\infty}\big[C(t,x) - (x - Ke^{-r(T-t)})\big] = 0$. The drift $\alpha$ cancels, so the price does not depend on it.

**Solution and Greeks.** With $\tau = T - t$,

$$d_\pm(\tau, x) = \frac{1}{\sigma\sqrt\tau}\Big[\log\frac{x}{K} + \Big(r \pm \frac{\sigma^2}{2}\Big)\tau\Big], \quad d_- = d_+ - \sigma\sqrt\tau, \quad C(t,x) = xN(d_+) - Ke^{-r\tau}N(d_-),$$

$$C_x = N(d_+), \qquad C_t = -rKe^{-r\tau}N(d_-) - \frac{x\sigma}{2\sqrt\tau}N'(d_+), \qquad C_{xx} = \frac{N'(d_+)}{x\sigma\sqrt\tau}.$$

The notes name gamma as $C_{xx}$ but I could not read its closed form off the pages, so I derived it from $C_x = N(d_+)$: $d_+$ depends on $x$ only through $\log(x/K)$, so $\partial d_+/\partial x = 1/(x\sigma\sqrt\tau)$ and $C_{xx} = N'(d_+)/(x\sigma\sqrt\tau)$. The test suite confirms it against a finite difference.

$C_{xx}$ is the key quantity for hedging error: the hedge ratio changes by about $C_{xx}\,dS$, so a large gamma means the hedge goes stale quickly when rebalancing is discrete.

**Result.** For $x = K = 100$, $r = 5\%$, $\sigma = 20\%$, $\tau = 1$: $C = 10.4506$ (the standard closed-form value), $C_x = 0.6368$, $C_t = -6.4140$, $C_{xx} = 0.01876$. The BSM equation holds numerically: $rC = 0.52253$ and $C_t + rxC_x + \tfrac12\sigma^2x^2C_{xx} = 0.52253$. The tests check put-call parity ($C - P = x - Ke^{-r\tau}$), $C_x \in (0,1)$, $C_x$, $C_t$ and $C_{xx}$ against finite differences, the BSM equation, and the boundary conditions. The BSM-equation test checks that the code satisfies the equation the theory is built on, not just that it reproduces one number.

## Step 3: GBM simulator and hedging engine (src/simulators.py, src/hedging.py)

**Concept.** Geometric Brownian motion has the exact solution

$$S(t) = S_0\exp\Big\{\sigma W(t) + \big(\alpha - \tfrac12\sigma^2\big)t\Big\}, \quad E[S(t)] = S_0e^{\alpha t}, \quad \mathrm{Var}(S(t)) = S_0^2e^{2\alpha t}\big(e^{\sigma^2t} - 1\big),$$

so paths are sampled from it on the grid $0 = t_0 < t_1 < \dots < t_n = T$ with $\Delta W(t_i) \sim N(0, \Delta t_i)$. This is exact: there is no discretisation error, and every hedging error we see comes from the hedging, not the simulation. The simulated mean and variance are tested against the two formulas above.

The hedger is the seller of the call. The seller receives $X(0) = C(0, S_0)$, holds $\Delta(t_i)$ shares, and keeps the rest in the money market at rate $r$. Self-financing means the only cash flows after time 0 are share trades:

$$\text{cash}_i = \text{cash}_{i-1}e^{r\Delta t_{i-1}} - \big(\Delta(t_i) - \Delta(t_{i-1})\big)S(t_i), \qquad \text{P\&L} = X(T) - (S(T) - K)^+ = \text{cash}_n + \Delta(t_{n-1})S(T) - (S(T)-K)^+.$$

If the hedge could be rebalanced continuously under GBM, this would be exactly 0 (replication, Step 2). With $n$ finite dates the portfolio is only hedged to first order between dates, and the second-order term $\tfrac12 C_{xx}(\Delta S)^2$ is what is left over.

**Result.** $S_0 = K = 100$, $\alpha = r = 5\%$, $\sigma = 20\%$, $T = 1$, 20,000 paths, seed 42:

| n | mean P&L | std P&L |
| --- | --- | --- |
| 4 | -0.040 | 3.224 |
| 52 | -0.002 | 0.952 |
| 252 | 0.002 | 0.432 |
| 1000 | 0.002 | 0.219 |

The mean is about 0 at every $n$ (the hedge is unbiased under the true model) and the standard deviation falls as $n$ rises. From $n = 252$ to $1000$ it falls by a factor of 1.97, close to $\sqrt4 = 2$ as the $\sqrt{\Delta t}$ law predicts. The formal log-log fit is E1.

## Step 4: E1, rebalancing frequency (experiments/e1_rebalancing.py)

**Concept: the error is a consequence of quadratic variation.** Over one step $[t_i, t_{i+1}]$ the hedge ratio is held at $\Delta(t_i) = C_x$. A second-order Taylor expansion gives $\Delta C \approx C_t\Delta t_i + C_x\Delta S + \tfrac12C_{xx}(\Delta S)^2$; the first-order term is removed by the hedge, and replacing $C_t$ with the BSM equation leaves (up to discounting)

$$\text{error}_i \approx \tfrac12C_{xx}\big[(\Delta S)^2 - \sigma^2S^2\Delta t_i\big] \approx \tfrac12C_{xx}\,\sigma^2S^2\Big[\big(\Delta W(t_i)\big)^2 - \Delta t_i\Big].$$

This is nonzero only because $(\Delta W)^2 \neq \Delta t$ at a finite step: the rule $(dW)^2 = dt$ comes from the quadratic variation $[W,W](T) = T$, which holds in the limit of a fine partition. Since $E[(\Delta W)^2] = \Delta t$ the bracket has mean 0, so the hedge is unbiased, and $\mathrm{Var}[(\Delta W)^2] = 2(\Delta t)^2$, so each step has variance $\tfrac12(C_{xx}\sigma^2S^2)^2(\Delta t)^2$. Steps are uncorrelated and there are $T/\Delta t$ of them, so

$$\mathrm{Var}[\text{P\&L}] \sim \frac{T}{\Delta t}\,(\Delta t)^2\,c = Tc\,\Delta t, \qquad \text{std} \sim \sqrt{\Delta t} \sim n^{-1/2}.$$

The simulation also compounds each step's error to time $T$ with $e^{r(T - t_i)}$ and averages the dollar gamma $S^2C_{xx}$ over paths.

**Result.** 20,000 paths, $S_0 = K = 100$, $\alpha = r = 5\%$, $\sigma = 20\%$, $T = 1$, seed 42:

| n | mean P&L | std P&L | predicted std |
| --- | --- | --- | --- |
| 4 | -0.0402 | 3.2238 | 3.0295 |
| 12 | 0.0145 | 1.8987 | 1.8480 |
| 52 | 0.0026 | 0.9547 | 0.9350 |
| 252 | 0.0000 | 0.4347 | 0.4317 |
| 1000 | 0.0023 | 0.2178 | 0.2187 |

Fitted log-log slope: $-0.487$ over all $n$, $-0.500$ over $n \ge 52$. The slope of the predicted std is $-0.477$.

**Interpretation.** The slope matches $-1/2$, which ties the simulation to the quadratic-variation argument above. The slope is slightly flatter at small $n$ because the formula is a small-$\Delta t$ expansion, so at $n = 4$ the higher-order terms still matter (the simulated std is about 6% above the prediction there). The agreement with the per-step formula is within 1% for $n \ge 252$. Practical meaning: to halve the hedging risk you need four times as many rebalances.

**Further reading.** Boyle and Emanuel (1980) analyse the discrete hedging error of a call; Bertsimas, Kogan and Lo (2000) study how that error behaves as the step size goes to zero. I cite them as further reading and have not reproduced their results.

## Step 5: E2, wrong volatility (experiments/e2_wrong_vol.py)

**Concept (the same derivation as Step 2, with $\sigma_h \neq \sigma$).** The stock has true volatility $\sigma$, but the hedger, the **seller** of the call, prices at $\sigma_h$: $X(0) = C(0, S_0)$ with $C$ the BSM price at $\sigma_h$, and holds $\Delta(t) = C_x(t, S(t))$ computed at $\sigma_h$, so $C$ satisfies the BSM equation at $\sigma_h$ while $S$ moves with $\sigma$. The discounted wealth is $d(e^{-rt}X) = C_x\,d(e^{-rt}S)$, and Itô on $e^{-rt}C(t,S(t))$ under the true dynamics gives

$$d\big(e^{-rt}C\big) = e^{-rt}\Big[-rC + C_t + \alpha SC_x + \tfrac12\sigma^2S^2C_{xx}\Big]dt + e^{-rt}\sigma SC_x\,dW.$$

The $dW$ terms cancel in $d\big(e^{-rt}(X - C)\big)$, and using the BSM equation at $\sigma_h$, $rC - C_t - rSC_x = \tfrac12\sigma_h^2S^2C_{xx}$, the $dt$ terms leave

$$d\Big(e^{-rt}\big(X(t) - C(t,S(t))\big)\Big) = \tfrac12\big(\sigma_h^2 - \sigma^2\big)e^{-rt}S(t)^2C_{xx}(t, S(t))\,dt.$$

So the seller's discounted terminal P&L is (El Karoui, Jeanblanc and Shreve, 1998)

$$e^{-rT}\,\text{P\&L} = \tfrac12\big(\sigma_h^2 - \sigma^2\big)\int_0^Te^{-rt}S(t)^2C_{xx}(t, S(t))\,dt,$$

positive when $\sigma_h > \sigma$. **Sign convention:** the hedger is the seller; the option holder has the opposite sign. The code returns the P&L at time $T$, which is $e^{rT}$ times the right-hand side. The integrand $S^2C_{xx}$ is the dollar gamma, always positive for a call, so the sign of the P&L is fixed by the sign of $\sigma_h - \sigma$. Selling too dear ($\sigma_h > \sigma$) locks in a profit; selling too cheap loses. The P&L is random only through the path of the dollar gamma, so it is not riskless even though it has a definite sign.

**Result.** $n = 1000$, 20,000 paths, seed 42, $S_0 = K = 100$, $\alpha = r = 5\%$, $\sigma = 20\%$, $T = 1$:

| sigma_h / sigma | mean sim | mean formula | std error | std sim |
| --- | --- | --- | --- | --- |
| 0.50 | -3.818 | -3.823 | 0.014 | 2.043 |
| 0.75 | -1.947 | -1.949 | 0.006 | 0.893 |
| 1.00 | 0.002 | 0.000 | 0.002 | 0.219 |
| 1.25 | 1.979 | 1.978 | 0.005 | 0.699 |
| 1.50 | 3.967 | 3.966 | 0.009 | 1.208 |

**Interpretation.** Simulated means agree with the formula to within about one standard error everywhere, which checks the sign and size of the result by simulation. The distributions are shifted left or right and are much wider than the correctly specified case, and their spread comes mainly from the dollar-gamma integral varying across paths: paths that end near the strike accumulate more gamma. The pathwise RMSE between simulation and formula (0.17 to 0.33) is the discrete-hedging noise from E1 and is about the same size as the $n = 1000$ std at $\sigma_h = \sigma$ (0.219). Takeaway: a volatility error of 25% costs about 2 on a premium of 10.45, far more than the discretisation error at $n = 1000$.

## Step 6: E3 and E4, wrong dynamics (src/simulators.py, experiments/e3_heston.py, e4_jumps.py)

**Concept: incomplete markets.** BSM works because one traded asset spans the single source of risk $dW$. Under Heston there are two sources of risk ($dW_1$ for the stock, $dW_2$ for the variance) but only the stock is traded, so the variance risk cannot be hedged away. Under jumps the Poisson risk cannot be offset by a continuous Delta hedge, since the stock moves discretely and $\Delta$ cannot adjust in time. In both cases the market is incomplete: the option cannot be replicated, no unique price exists, and a BSM Delta hedge leaves residual risk that does not vanish as rebalancing becomes continuous.

**Heston (E3).** It extends GBM by a stochastic variance $V(t)$ that follows the CIR process of the notes, $dR(t) = (\alpha_R - \beta R(t))\,dt + \sigma_R\sqrt{R(t)}\,dW(t)$, with $E[R(t)] = R(0)e^{-\beta t} + \tfrac{\alpha_R}{\beta}(1 - e^{-\beta t})$ and $\mathrm{Var}(R(t)) \to \alpha_R\sigma_R^2/(2\beta^2)$. To avoid clashing with the stock's $\alpha$ and $\sigma$ the mapping is $R \to V$, $\alpha_R \to a_V$, $\sigma_R \to \xi$ (in the usual Heston notation $a_V = \kappa\theta$, $\beta = \kappa$):

$$dS(t) = \alpha S(t)\,dt + \sqrt{V(t)}\,S(t)\,dW_1(t), \qquad dV(t) = \big(a_V - \beta V(t)\big)dt + \xi\sqrt{V(t)}\,dW_2(t), \qquad dW_1\,dW_2 = \rho\,dt.$$

The variance uses the full-truncation Euler scheme, with $V^+ = \max(V, 0)$ in every coefficient: $V_{i+1} = V_i + (a_V - \beta V_i^+)\Delta t + \xi\sqrt{V_i^+\Delta t}\,Z_2$. The stock then takes the exact lognormal step with the variance frozen at $V_i^+$ (Euler is used only for the variance). The tests check the simulated $E[V(t)]$ and the long-run variance against the two CIR formulas above, and that Heston with $\xi = 0$ reduces to GBM.

**Merton jumps (E4).** The notes only mention that a Poisson process is not a simple process; the model adds compound Poisson jumps to GBM:

$$\frac{dS}{S} = (\alpha - \lambda\bar\kappa)\,dt + \sigma\,dW + (J - 1)\,dN, \qquad \ln J \sim N(m, \delta^2), \quad \bar\kappa = e^{m + \delta^2/2} - 1,$$

with $N$ a Poisson process of rate $\lambda$. It is sampled exactly (a sum of $n$ log-jumps is $N(nm, n\delta^2)$), and the compensator $\lambda\bar\kappa$ keeps $E[S(T)] = S_0e^{\alpha T}$. Tests: Heston and Merton keep $E[S(T)]$; Merton log-variance equals $(\sigma^2 + \lambda(m^2 + \delta^2))T$.

**Setup.** The hedge and premium always use BSM at $\sigma = 20\%$ (the same trader as E1). To make the comparison fair, Heston starts at $V(0) = a_V/\beta = 0.04$ (average variance matches), and Merton's diffusion variance is reduced so that $\sigma_d^2 + \lambda(m^2 + \delta^2) = 0.04$. So differences from GBM are not caused by a higher or lower variance level. Heston paths are simulated with 1,008 Euler steps and the hedge uses every $(1008/n)$-th date, so the simulation error does not change with the hedging frequency. Parameters: $\beta = 2$, $a_V = 0.08$, 20,000 paths, seed 42, $S_0 = K = 100$, $\alpha = r = 5\%$, $T = 1$.

**E3 result (Heston, $n = 252$).** VaR and ES (5% tail losses) are reported in one table of tail statistics. An earlier version simulated Heston on the hedging grid itself. Simulating on the fine grid changed the standard deviations by under 1% (for $\xi = 0.4$, $\rho = -0.7$: 1.960 became 1.952), but small means moved more (0.195 became 0.157; for $\xi = 0.2$ the mean went from $+0.005$ to $-0.018$), which shows part of those small means was Euler path error.

| scenario | mean | std | skew | ex. kurt | VaR | ES |
| --- | --- | --- | --- | --- | --- | --- |
| GBM | 0.002 | 0.432 | -0.15 | 1.92 | 0.70 | 0.99 |
| xi 0.2, rho -0.7 | -0.018 | 1.102 | -0.62 | 1.61 | 1.95 | 2.72 |
| xi 0.4, rho -0.7 | 0.157 | 1.952 | -0.93 | 2.10 | 3.35 | 4.86 |
| xi 0.6, rho -0.7 | 0.453 | 2.647 | -1.28 | 3.24 | 4.46 | 6.70 |
| xi 0.4, rho -0.9 | 0.146 | 1.481 | -0.75 | 0.66 | 2.50 | 3.40 |
| xi 0.4, rho -0.5 | 0.181 | 2.242 | -1.02 | 2.78 | 3.84 | 5.72 |
| xi 0.4, rho 0.0 | 0.300 | 2.549 | -1.09 | 3.71 | 4.24 | 6.44 |
| xi 0.4, rho 0.5 | 0.516 | 2.359 | -0.82 | 2.51 | 3.52 | 5.44 |

Std against $n$ for $\xi = 0.4$, $\rho = -0.7$: $n = 12$: 2.83, 48: 2.17, 252: 1.95, 1008: 1.91 (GBM falls from 1.94 to 0.22). The Heston error flattens at about 1.9 instead of going to 0.

**E3 interpretation.** Std, VaR and ES grow with the vol-of-vol: even $\xi = 0.2$ multiplies the std by about 2.6 relative to GBM. The residual does not vanish with $n$, as the incomplete-market argument predicts. The error comes from the fact that the BSM Delta and gamma use a constant 20% while the true instantaneous volatility is $\sqrt{V(t)}$, so the seller is effectively in the wrong-volatility situation of E2 with a randomly varying volatility (E2 gives P&L $= \tfrac12(\sigma_h^2 - V(t))S^2C_{xx}\,dt$ integrated over time). The left skew deepens with $\xi$. The mean is about zero at $\xi = 0.2$ and positive at $\xi = 0.4$ and $0.6$ (0.16 and 0.45). A pricing bias is one possible reason: the premium was computed at constant 20%, and the Heston at-the-money call may be worth less because the at-the-money price is concave in variance (Jensen), so the seller has been paid too much. I have not isolated this, and the means are sensitive to the Euler grid, as noted above. Correlation changes the shape: $\rho = -0.9$ gives the smallest std of the $\xi = 0.4$ runs (1.48), and the largest std is at $\rho = 0$ (2.55). Strong correlation makes the variance shocks mostly a function of the stock shocks, so the Delta hedge captures part of them; at $\rho = 0$ the variance risk is orthogonal to the stock and entirely unhedgeable.

**E4 result (Merton, $n = 252$, jump std 0.1, variance-matched).**

| scenario | mean | std | skew | ex. kurt | VaR | ES |
| --- | --- | --- | --- | --- | --- | --- |
| GBM | 0.002 | 0.432 | -0.15 | 1.92 | 0.70 | 0.99 |
| lam 0.25, m -0.1 | 0.062 | 1.523 | -4.31 | 25.81 | 2.67 | 5.39 |
| lam 0.5, m -0.1 | 0.106 | 2.183 | -3.29 | 15.31 | 4.39 | 7.38 |
| lam 1.0, m -0.1 | 0.184 | 3.053 | -2.04 | 5.37 | 6.28 | 9.16 |
| lam 0.5, m -0.05 | 0.058 | 1.559 | -3.93 | 24.41 | 2.89 | 5.30 |
| lam 0.5, m -0.2 | 0.272 | 4.279 | -1.93 | 4.00 | 9.17 | 12.54 |

Std against $n$ for $\lambda = 0.5$, $m = -0.1$: $n = 12$: 2.78, 52: 2.30, 252: 2.18, 1000: 2.18 (GBM 1.94 to 0.22).

**E4 interpretation.** The seller is short gamma, so a large move in either direction loses money: a jump of size $J$ costs roughly $\tfrac12C_{xx}S^2(J-1)^2$ plus the first-order mismatch, and the Delta cannot be adjusted in time. The distribution has a peak at small positive P&L (paths with no jump, where diffusion volatility is below 20% and the seller collects the excess premium) and a long left tail (paths with jumps). Hence the very large negative skew and kurtosis, and the ES much larger than VaR. Rarer jumps (small $\lambda$) give the heaviest tails in skew and kurtosis terms (kurtosis 25.8 at $\lambda = 0.25$), while more frequent jumps give larger variance and larger VaR and ES. The std flattens at about 2.18 as $n$ grows, so more frequent hedging cannot remove jump risk. Larger jumps make everything worse: at $m = -0.2$ the VaR is 9.17.

**Caveat.** Variance matching is my design choice, not a calibration. I chose parameters for illustration (not fitted to market data), so the numbers show direction and rough size, not market values.

**Monte Carlo error.** With 20,000 paths, bootstrap standard errors over the paths (`bootstrap_standard_errors`, 500 resamples) are, at $n = 252$:

| scenario | mean | std | skew | ex. kurt | ES |
| --- | --- | --- | --- | --- | --- |
| GBM | 0.0028 | 0.0031 | 0.046 | 0.24 | 0.013 |
| Heston $\xi = 0.4$, $\rho = -0.7$ | 0.014 | 0.014 | 0.039 | 0.24 | 0.066 |
| Merton $\lambda = 0.5$, $m = -0.1$ | 0.015 | 0.031 | 0.093 | 1.26 | 0.127 |

The standard deviation is accurate to about 0.7% (Heston $1.95 \pm 0.014$) and the ES to 1 to 2% ($4.86 \pm 0.066$), but the skewness is only good to $\pm 0.04$ to $\pm 0.09$ and the excess kurtosis to $\pm 0.24$ (Heston) or $\pm 1.3$ (Merton). Small differences in skewness or kurtosis between scenarios are therefore not interpreted.

## Step 7: E5, transaction costs (src/costs.py, experiments/e5_costs.py)

**Leland (1985).** It extends the BSM hedging argument of Step 2 (hold $\Delta(t) = C_x$ and rebalance) by a cost on every trade; the Step 4 error formula supplies the trade sizes.

**Concept: cost versus risk.** With proportional costs the seller pays a fraction of every trade, so hedging more often reduces discretisation risk (E1: std $\sim n^{-1/2}$) but raises costs. Per step the hedge ratio changes by about $C_{xx}\,dS$, and $dS \sim S\sigma\sqrt{\Delta t}\,Z$, so the shares traded have mean absolute size $C_{xx}S\sigma\sqrt{\Delta t}\,E|Z| = C_{xx}S\sigma\sqrt{\Delta t}\sqrt{2/\pi}$. With one-way cost $\tfrac{k}{2}S$ per share ($k$ is Leland's round-trip rate):

$$\text{cost per step} \sim \tfrac{k}{2}\sqrt{\tfrac2\pi}\,\sigma\sqrt{\Delta t}\,C_{xx}S^2, \qquad \text{total expected cost} \sim \tfrac{k}{2}\sqrt{\tfrac2\pi}\,\frac{\sigma}{\sqrt{\Delta t}}\int_0^TS^2C_{xx}\,dt.$$

So the cost grows like $1/\sqrt{\Delta t} = \sqrt n$, while the hedging risk falls like $1/\sqrt n$. There must be an interior optimum.

**Leland's adjusted volatility.** The extra cost acts like an extra gamma term in the E2 formula, $\tfrac12(\hat\sigma^2 - \sigma^2)\int S^2C_{xx}\,dt$, with

$$\hat\sigma^2 = \sigma^2\Big(1 + \sqrt{\tfrac2\pi}\,\frac{k}{\sigma\sqrt{\Delta t}}\Big),$$

so pricing and hedging at $\hat\sigma$ covers the expected cost. The correction is of order $k/\sqrt{\Delta t}$, not of order $k$, which is why frequent trading is so expensive. The code charges $\tfrac{k}{2}|\text{shares traded}|\,S$ per trade, so $k$ is the round-trip rate, as in Leland. The initial purchase pays the cost too; the final delivery of shares does not.

**Optimal $n$.** I measure risk with the RMSE of the net P&L about zero, $\sqrt{\text{bias}^2 + \text{variance}}$. The bias is the expected cost. With std $\sim s/\sqrt n$ and cost $\sim a\sqrt n$ ($a$ proportional to $k$):

$$\text{RMSE}^2 = a^2n + \frac{s^2}{n}, \quad \text{minimised at } n^* = \frac{s}{a} \;\;(\text{so } n^* \text{ falls roughly like } 1/k).$$

The choice of RMSE is mine. Other criteria (a mean-variance objective with another risk aversion, or ES) give a different $n^*$, so "optimal" here means optimal for this criterion.

**Result.** GBM, $S_0 = K = 100$, $\alpha = r = 5\%$, $\sigma = 20\%$, 20,000 paths, seed 42.

Cost against the Leland formula (selected rows):

| k | n | mean cost | Leland cost | std P&L |
| --- | --- | --- | --- | --- |
| 0.005 | 52 | 0.710 | 0.705 | 1.001 |
| 0.005 | 252 | 1.373 | 1.370 | 0.656 |
| 0.005 | 1000 | 2.580 | 2.578 | 0.979 |
| 0.01 | 252 | 2.746 | 2.739 | 1.062 |

The simulated cost matches the formula to within 2% in every row (best 0.1% at $n = 1000$), and grows by about 1.9 for each 4x in $n$ ($\sqrt4 = 2$ asymptotically, a bit less because the initial purchase is constant). The std of net P&L stops falling at large $n$, because the random cost adds variance: at $k = 0.005$ it is 1.00 at $n = 52$, 0.66 at $n = 252$ and 0.98 at $n = 1000$, against 0.22 at $n = 1000$ with no costs.

Optimal $n$ (searched over the grid $n = 12, 26, 52, 78, 104, 156, 208, 252, 378, 504, 756, 1000$):

| k | n* simulated | n* closed form s/a | min RMSE | Leland correction at n* |
| --- | --- | --- | --- | --- |
| 0.001 | 378 | 398 | 0.509 | 0.078 |
| 0.0025 | 156 | 159 | 0.834 | 0.125 |
| 0.005 | 78 | 80 | 1.205 | 0.176 |
| 0.01 | 26 | 40 | 1.801 | 0.203 |

The closed form agrees with the simulation to a few percent for $k$ up to 0.005. At $k = 0.01$ the simulated optimum (26) is on a flat RMSE curve between $n = 26$ and 52 (1.80 and 1.83), and the grid is coarse, so the gap to 40 is within the resolution. The closed-form $n^*$ has log-log slope exactly $-1$ in $k$, which is built in by the cost being proportional to $k$. The simulated optima (378, 156, 78, 26) fall only roughly like $1/k$, and only for small $k$: from $k = 0.005$ to $0.01$ the simulated $n^*$ falls by a factor of 3, not 2, because the grid is coarse and the RMSE is flat near the minimum there.

Leland check at $k = 0.005$ (premium and Deltas both at $\sigma$ or at $\hat\sigma$):

| n | sigma hat | mean P&L at sigma | mean P&L at sigma hat | std at sigma | std at sigma hat |
| --- | --- | --- | --- | --- | --- |
| 26 | 0.2099 | -0.563 | -0.162 | 1.361 | 1.356 |
| 52 | 0.2139 | -0.734 | -0.164 | 1.001 | 0.981 |
| 252 | 0.2295 | -1.410 | -0.163 | 0.656 | 0.460 |

**Interpretation.** Pricing at $\hat\sigma$ removes the $n$-dependent cost bias: the mean P&L goes from $-0.56$, $-0.73$, $-1.41$ to a constant $-0.16$ for all three $n$. The remaining $-0.16$ is the initial purchase cost, $\tfrac{k}{2}\Delta(0)S_0 = 0.0025 \times 0.64 \times 100 = 0.16$, which Leland's formula does not include (it covers the rebalancing). So the Leland scaling is confirmed for the cost bias. Note that Leland's $\hat\sigma$ is not the optimal-$n$ criterion: Leland's $\sqrt{2/\pi}\,k/(\sigma\sqrt{\Delta t})$ is about 0.08 to 0.20 at the optimum, so the optimal frequency is where the correction is a moderate share of the volatility, not where it equals it. Practical meaning: with $k = 0.5\%$ (round trip), weekly to daily hedging ($n = 52$ to 252) is already too frequent; the best is about $n = 78$ under this RMSE criterion.

## Step 8: E6, moneyness and maturity (experiments/e6_moneyness_maturity.py)

**Concept.** The per-step error $\tfrac12C_{xx}\sigma^2S^2\big[(\Delta W)^2 - \Delta t\big]$ is proportional to the dollar gamma $S^2C_{xx}$ along the path, so the hedging error is large where and when the option has high gamma: close to the strike, and close to expiry. From Step 4 the variance of the total error is a sum over steps of $\tfrac12(\Delta t)^2(\sigma^2S^2C_{xx})^2$ (compounded to $T$), now split by step in `error_variance_profile`. For an at-the-money option, $C_{xx}S^2$ grows like $1/\sqrt\tau$ as $\tau = T - t \to 0$, so late steps contribute most. A deep in- or out-of-the-money option has small gamma, and so a small error in absolute terms.

**Setup.** Weekly hedging ($\Delta t = 1/52$, so $n = 52T$), GBM with $S_0 = 100$, $\alpha = r = 5\%$, $\sigma = 20\%$, 20,000 paths, seed 42.

**Result: strike sweep ($T = 1$, $n = 52$).**

| K | premium | S^2 C_xx at t = 0 | std sim | std Ito formula | std / premium |
| --- | --- | --- | --- | --- | --- |
| 80 | 24.59 | 68.1 | 0.465 | 0.450 | 0.019 |
| 90 | 16.70 | 135.8 | 0.745 | 0.729 | 0.045 |
| 100 | 10.45 | 187.6 | 0.952 | 0.936 | 0.091 |
| 110 | 6.04 | 197.9 | 1.017 | 0.995 | 0.168 |
| 120 | 3.25 | 170.4 | 0.955 | 0.934 | 0.294 |

**Result: maturity sweep ($K = 100$, weekly hedging).**

| T | n | premium | S^2 C_xx at t = 0 | std sim | std Ito formula | std / premium |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 5 | 2.77 | 626.9 | 0.937 | 0.875 | 0.338 |
| 0.25 | 13 | 4.62 | 392.9 | 0.944 | 0.896 | 0.205 |
| 0.5 | 26 | 6.89 | 273.6 | 0.943 | 0.917 | 0.137 |
| 1.0 | 52 | 10.45 | 187.6 | 0.952 | 0.936 | 0.091 |
| 2.0 | 104 | 16.13 | 124.8 | 0.952 | 0.938 | 0.059 |

For $K = 100$, $T = 1$, the last weekly step alone carries about 7% of the total error variance, and the first step about 1.3% (third panel of the figure; $K = 80$ and 120 look similar, with the final step at 8.5% and 7.5%).

**Interpretation.**
- Against strike, the std rises from 0.47 ($K = 80$) to a peak of 1.02 near $K = 110$ and falls again. This follows the gamma profile: the dollar gamma at $t = 0$ peaks near $K = 110$ for $T = 1$, and the std peaks at the same strike. The $t = 0$ gamma is only a rough guide ($K = 80$ has a higher std than its $t = 0$ gamma suggests) because the error accumulates over the whole path, and the stock can drift toward the strike. The Itô formula follows the simulation at all strikes, about 2 to 3% below it. The remaining gap is the higher-order terms that the second-order formula leaves out.
- The absolute error does not tell the whole story: relative to the premium, the error grows steadily with the strike, from 2% at $K = 80$ to 29% at $K = 120$. Out-of-the-money options are cheap and their small errors are large relative to what was paid.
- Against maturity at fixed weekly $\Delta t$, the std is nearly flat (0.94 to 0.95 from $T = 0.1$ to 2). I checked this only numerically, and the following explanation is my own reasoning, not tested separately: with dollar gamma proportional to $1/\sqrt\tau$, each step has variance of order $(\Delta t)^2/\tau_i$, and the sum over $i$ is of order $\Delta t\log n$, so the std depends on $T$ only slowly through $\log n$. The Itô formula understates the simulated std at short maturities (7% below at $T = 0.1$, where $n = 5$ steps are too coarse for the second-order expansion), which is consistent with the small-$\Delta t$ nature of the formula noted in E1.
- Relative to the premium, short-dated options are much harder to hedge weekly: 34% at $T = 0.1$ against 6% at $T = 2$.

## Step 9: E7, backtest on real listed option prices (src/backtest.py, experiments/e7_backtest.py)

**Concept.** The simulations assume a model for the stock. The backtest asks how a BSM Delta hedge, $\Delta(t_i) = C_x$, performs on the real NIFTY path when the option sold is a real listed contract, so that both the stock path and the option price come from the market. The theory it tests is the one already derived: the hedging error is the discretisation error of E1 plus the wrong-volatility term of E2, where the "true" volatility is whatever NIFTY realised over the option's life.

**Realised and implied volatility (which is used where).** The realised volatility is the notes' formula, computed from the stock path over the option's life $[T_1, T_2]$,

$$\hat\sigma = \sqrt{\frac{1}{T_2 - T_1}\sum_{i=0}^{n-1}\Big[\ln\frac{S(t_{i+1})}{S(t_i)}\Big]^2},$$

which estimates $\sigma$ because $\sum(\Delta W(t_i))^2 \to T_2 - T_1$ (quadratic variation). The implied volatility is the $\sigma$ for which $\text{BSM}(\tau, x, K, r, \sigma)$ equals the listed market price. The listed call is priced and hedged at its **implied** volatility ($\sigma_h$ in E2); the **realised** volatility plays the role of the true $\sigma$ in the E2 formula.

**Data (see data/SOURCE.md).** Listed prices: NSE F&O bhavcopy, one file per trading day from 2 January 2015 to 11 November 2026 (2,920 days; the older bhavcopy format up to 5 July 2024, the UDiFF format from 8 July 2024), downloaded on 15 November 2026. Index path and India VIX: Yahoo Finance via yfinance (^NSEI, ^INDIAVIX), same date.

**Method.** An entry date is used when a NIFTY expiry falls exactly 20 trading days later ($T = 20/252$ years, about 28 calendar days), which gives the same 20-step uniform grid as the simulations. Before 2019 NIFTY had monthly expiries only (about one window a month, not overlapping); weekly expiries start in 2019 (about one window a week, overlapping). The traded call with the strike closest to the NIFTY close is sold at its listed close `ClsPric`. Its own implied volatility ($r = 6\%$, no dividends) is used for the Deltas, so the engine's premium equals the market price. The hedge is rebalanced daily (20 steps) or weekly (4 steps) along the real NIFTY path, and the call is settled at the NIFTY close on the expiry date. P&L is divided by the premium. The simulations use $S_0 = K = 100$, $\alpha = r = 6\%$, $T = 20/252$ and $\sigma = 14.77\%$, the sample's mean implied volatility, with the premium and Deltas at that $\sigma$ and 20 daily steps (weekly hedging takes every fifth date). Heston ($\beta = 2$, $a_V = \beta\sigma^2$, $V(0) = \sigma^2$, $\xi = 0.4$, $\rho = -0.7$) uses the E3 settings, not the $\sigma = 20\%$, $T = 1$ of that experiment. Merton is not simulated for the 20-day window: the E4 jump parameters were chosen for one-year options, and in a 20-day window almost every path has no jump while a few lose far more than the premium (an earlier run gave skewness $-8$ and excess kurtosis 84), which says nothing about NIFTY. A single $\sigma$ is a simplification, since the real windows run from calm periods to the 2020 crash. Dividing by the premium (roughly proportional to $\sigma$ for an at-the-money option) puts the windows on one scale, and the residual comparison below removes the main regime effect.

**Data-quality filter (important).** The older files have no underlying price and I keep only calls that traded that day, so on days when no near-the-money call traded, the nearest traded strike can be far from the money. A first run without a filter had 23 of 395 windows with a strike more than 1% from spot (one 44% away, 20 March 2020). These are not at-the-money options, and they distorted the result (skewness $+0.71$ and excess kurtosis 9.4, against $-1.10$ and 2.5 without them). I now require $|K/S - 1| \le 1\%$, which leaves 372 windows. The 1% tolerance is my choice, so the sensitivity is reported below. I found the problem from the 44% gap, not by searching for nicer numbers.

**The VIX baseline.** To check how well a model-priced option reproduces the listed results (it extends the same BSM pricing of Step 2), the same hedge is run on a synthetic at-the-money call priced with Black-Scholes at India VIX$/100$ on the entry date. It needs no option data, so it covers every trading day (2,884 windows, January 2015 to October 2026, windows without a VIX value dropped), and it is also reported on exactly the dates of the listed windows.

**Result.** 372 listed windows, 28 January 2015 to 9 October 2026; strikes within 0.98% of spot (median 0.09%); median premium 353 index points. P&L / premium (VaR and ES are 5% losses):

| sample | mean | std | skew | ex. kurt | VaR 5% | ES 5% |
| --- | --- | --- | --- | --- | --- | --- |
| listed daily | 0.058 | 0.247 | -1.10 | 2.46 | 0.449 | 0.625 |
| listed weekly | 0.008 | 0.398 | -1.57 | 5.83 | 0.640 | 1.099 |
| VIX baseline daily, same dates | 0.111 | 0.231 | -1.81 | 6.78 | 0.232 | 0.570 |
| VIX baseline weekly, same dates | 0.069 | 0.372 | -1.61 | 6.26 | 0.506 | 0.930 |
| VIX baseline daily, all days | 0.127 | 0.228 | -1.63 | 5.85 | 0.234 | 0.542 |
| VIX baseline weekly, all days | 0.089 | 0.372 | -1.58 | 6.75 | 0.524 | 0.923 |
| GBM daily | 0.001 | 0.167 | -0.37 | 1.71 | 0.277 | 0.392 |
| GBM weekly | -0.001 | 0.352 | -0.59 | 1.28 | 0.603 | 0.850 |
| Heston daily | 0.007 | 0.226 | -0.84 | 2.77 | 0.379 | 0.572 |
| Heston weekly | 0.004 | 0.387 | -1.03 | 2.84 | 0.684 | 1.008 |

Mean implied volatility of the listed calls 14.77%, mean realised volatility $\hat\sigma$ 13.50%; implied above realised in 68.3% of windows. The E2 formula applied window by window ($\hat\sigma$ as the true $\sigma$, the listed implied volatility as $\sigma_h$): mean 23.0 index points against 23.6 in the backtest, correlation 0.868 across windows. Sensitivity to the at-the-money tolerance (daily hedging):

| max strike gap | windows | mean | std | skew | ex. kurt |
| --- | --- | --- | --- | --- | --- |
| 0.5% | 356 | 0.060 | 0.242 | -1.07 | 2.46 |
| 1% | 372 | 0.058 | 0.247 | -1.10 | 2.46 |
| 2% | 376 | 0.058 | 0.247 | -1.09 | 2.40 |

Mean P&L / premium by entry year (windows in brackets): 2015 0.041 (12), 2016 -0.004 (12), 2017 0.024 (12), 2018 -0.057 (11), 2019 0.069 (17), 2020 0.022 (20), 2021 -0.014 (39), 2022 0.040 (52), 2023 0.125 (51), 2024 0.054 (52), 2025 0.152 (53), 2026 0.026 (41).

**Statistical uncertainty (moving-block bootstrap).** Windows that start on nearby dates overlap in time, so I resample blocks of 5 consecutive windows (2,000 resamples; blocks of 3 and 10 give almost the same intervals) and recompute each statistic. 95% intervals for the listed P&L / premium, with the simulated values:

| statistic | daily: listed (95% interval) | GBM | Heston | weekly: listed (95% interval) | GBM | Heston |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 0.058 (0.017, 0.096) | 0.001 | 0.007 | 0.008 (-0.047, 0.069) | -0.001 | 0.004 |
| std | 0.247 (0.213, 0.280) | 0.167 | 0.226 | 0.398 (0.336, 0.462) | 0.352 | 0.387 |
| skewness | -1.10 (-1.47, -0.66) | -0.37 | -0.84 | -1.57 (-2.15, -0.67) | -0.59 | -1.03 |
| excess kurtosis | 2.46 (0.90, 4.06) | 1.71 | 2.77 | 5.83 (1.39, 9.14) | 1.28 | 2.84 |
| ES 5% | 0.625 (0.464, 0.728) | 0.392 | 0.572 | 1.099 (0.774, 1.425) | 0.850 | 1.008 |

For daily hedging GBM lies outside the interval for the mean, standard deviation, skewness and ES, and inside only for the kurtosis; Heston lies inside for all of the shape statistics. For weekly hedging GBM is outside for skewness and kurtosis (just) and Heston is inside for all. The means of the simulations are about 0 by construction, so a mean outside the interval says only that the real seller gained from the implied-realised gap. This raw comparison also mixes in the E2 spread that the simulations lack; see the gamma-weighted comparison below.

**Trading costs on the real backtest** (the Leland cost of Step 7 applied to the index hedge). The engine charges a round-trip cost $k$ ($k/2$ per trade). The rates below are illustrative (5, 10 and 20 basis points round trip); I did not check actual NIFTY futures costs. $n$ is the number of hedges in the 20-day life ($n = 20$ is daily, 4 is weekly).

| k | n | mean cost / premium | mean net P&L / premium | RMSE | share of windows losing |
| --- | --- | --- | --- | --- | --- |
| 0 | 20 | 0 | 0.058 | 0.254 | 33% |
| 0.0005 | 20 | 0.025 | 0.033 | 0.252 | 37% |
| 0.001 | 20 | 0.050 | 0.008 | 0.253 | 42% |
| 0.002 | 20 | 0.100 | -0.042 | 0.263 | 52% |
| 0.002 | 10 | 0.080 | -0.046 | 0.309 | 52% |
| 0.002 | 4 | 0.059 | -0.051 | 0.408 | 50% |
| 0.002 | 2 | 0.045 | -0.033 | 0.530 | 47% |

Total cost grows with $n$ (at $k = 0.001$ it is 5.0% of the premium at $n = 20$ and 2.9% at $n = 4$, a factor of 1.7 for a factor of 5 in $n$, less than the square-root growth of E5 because of the constant initial purchase). A cost of 10 basis points removes the seller's average gain, and 20 basis points turns it into a loss. At these small cost rates daily hedging still has the lowest RMSE for 20-day options, so costs do not move the optimum away from $n = 20$ (E5 predicts the optimum moves to coarser hedging as $k$ rises, which would need larger $k$). The main effect of costs here is on the mean.

**Sensitivity to the interest rate and the dividend yield.** The base case uses a constant $r = 6\%$ and ignores dividends, as in the notes. The engine and the implied-volatility solver accept a continuous dividend yield $q$ (1.3% is my rough figure for NIFTY, not taken from data). Daily hedging:

| r | q | mean implied vol | mean P&L / premium | std | skew | ex. kurt | ES 5% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4% | 0 | 15.5% | 0.105 | 0.246 | -1.03 | 2.36 | 0.568 |
| 6% | 0 | 14.8% | 0.058 | 0.247 | -1.10 | 2.46 | 0.625 |
| 8% | 0 | 14.0% | 0.009 | 0.250 | -1.17 | 2.50 | 0.688 |
| 4% | 1.3% | 16.0% | 0.135 | 0.246 | -0.97 | 2.26 | 0.532 |
| 6% | 1.3% | 15.3% | 0.090 | 0.246 | -1.05 | 2.39 | 0.586 |
| 8% | 1.3% | 14.5% | 0.042 | 0.248 | -1.12 | 2.47 | 0.646 |

The spread and shape of the error hardly change, but the mean does: each 2 percentage points of the assumed rate moves the mean P&L / premium by about 0.05, because a higher $r$ lowers the implied volatility backed out of the same market price (15.5% at $r = 4\%$ against 14.0% at $r = 8\%$) and so shrinks the implied-realised gap that drives the gain. Including a 1.3% dividend yield raises the mean by about 0.03. The sign of the daily mean is positive in all six cases; the weekly mean is negative at $r = 8\%$ ($-0.042$ with $q = 0$, $-0.008$ with $q = 1.3\%$). So the existence of a positive gain is fairly robust, its size is not. The check of the E2 formula and the 68% share of windows with implied above realised volatility were computed at $r = 6\%$, $q = 0$. A time-varying rate from data would remove part of this sensitivity.

**Gamma-weighted formula (a like-for-like comparison with the simulations).** The raw comparison above is not like for like. Each real window is hedged at its own implied volatility $\sigma_h$, so windows differ in the gap between implied and realised volatility, and by E2 this adds spread and skew; the simulations hedge at the true $\sigma$ and have no such term. The E2 formula with one window-average $\hat\sigma$ removes only part of it. The exact pathwise version (`pathwise_pnl_formula` in src/hedging.py) weights each day's squared return by that day's dollar gamma:

$$\text{P\&L} \approx \tfrac12\sum_i e^{r(T-t_i)}\,S(t_i)^2C_{xx}(t_i,S(t_i))\Big[\sigma_h^2\Delta t_i - \Big(\ln\frac{S(t_{i+1})}{S(t_i)}\Big)^2\Big].$$

It is the per-step error of Step 4 with the realised squared return in place of $\sigma^2\Delta t_i$ (second order, on the hedging grid, with $C_{xx}$ at $\sigma_h$). The window-average version replaces every squared return by $\hat\sigma^2\Delta t_i$, which is accurate only when the variance is spread evenly over the window.

The entry of 26 February 2020 shows why that fails. The index crashed during the option's life while the option was far out of the money, so its gamma was close to zero on the crash days and those days cost little. The window-average formula spreads the realised volatility (82%) evenly over the window, including the high-gamma early days, and predicts a loss of 3.6 times the premium; the actual loss was 0.44 times, and the gamma-weighted formula predicts 0.56.

Results for the listed windows (daily hedging, 372 windows): correlation of the P&L with the gamma-weighted formula 0.946 (0.868 for the constant-$\hat\sigma$ formula), average formula 23.7 index points against 23.6 observed (22.98 for the constant-$\hat\sigma$ formula). Standard deviation of the residual (P&L $-$ formula) / premium, with the same formula applied to the simulated paths:

| sample | daily: correlation with formula | daily: residual std | weekly: correlation | weekly: residual std |
| --- | --- | --- | --- | --- |
| listed, all windows (372) | 0.946 | 0.090 | 0.908 | 0.222 |
| listed, no daily move above 4% (355) | 0.985 | 0.041 | 0.946 | 0.117 |
| listed, a daily move above 4% (17) | 0.681 | 0.374 | 0.767 | 0.889 |
| GBM (simulated) | 0.969 | 0.041 | 0.942 | 0.118 |
| Heston (simulated) | 0.974 | 0.051 | 0.932 | 0.141 |

The 4% threshold is my choice (daily log moves above 4% are 0.6% of all days). For the daily residual of the windows without a larger move the standard deviation is 0.039 at a 3% threshold (327 windows), 0.041 at 4% and 0.061 at 5% (363 windows).

The constant-$\hat\sigma$ residual used before is the sum of this higher-order residual and a timing term (the gamma-weighted minus the equal-weighted realised variance, times the dollar gamma). Its standard deviation is 0.189, dominated by the 26 February 2020 window, where the timing term alone is 3.2 times the premium; the gamma-weighted residual is 0.090.

**Interpretation.**
- The positive mean is the E2 effect on real data. The seller sells at the market's implied volatility, and NIFTY realised less volatility than implied in 68% of windows. The E2 formula, which uses only each window's implied and realised volatility, explains most of the window-to-window variation in the P&L (correlation 0.87). The correlation between P&L / premium and $(\text{implied} - \hat\sigma)/\text{implied}$ is 0.72 (figure e7_timeline). The worst windows are those where realised volatility exceeded implied: for example the entry on 27 August 2019 (realised 25.4% against implied 14.2%) lost 108% of the premium, and the entry on 26 February 2020 (realised 82% against implied 15.6%) lost 44%.
- The simulations have no gap between implied and realised volatility, so their means are about 0 and are not comparable with the backtest mean. Shape and spread can be compared. The raw listed spread is above GBM (std 0.247 against 0.167 daily, 0.398 against 0.352 weekly) and close to Heston (0.226, 0.387), but this includes the E2 spread that the simulations lack; see the gamma-weighted comparison above.
- Tails: the raw P&L is clearly larger and more skewed than GBM predicts (GBM lies outside the 95% intervals for the daily standard deviation, skewness and ES). That comparison mixes two things. The gamma-weighted comparison above shows that, once the gap between implied and realised volatility is accounted for window by window, what is left is the higher-order error of large single-day moves: in the 355 windows without a daily move above 4% the residual is as small as GBM's (0.041 against 0.041 daily, 0.117 against 0.118 weekly; Heston 0.051, 0.141), and the 17 windows with such a move have residuals about ten times larger. Because the formula uses each path's own squared returns it absorbs any volatility clustering, so it cannot separate Heston from GBM. A stochastic-volatility effect is consistent with the raw difference, but volatility-regime variation across windows also contributes and explains most of it; the data cannot show a volatility effect beyond that.
- Daily hedging has lower relative error than weekly (0.247 against 0.398, a factor of 1.6). The $\sqrt{\Delta t}$ law would give 2.2, so part of the real error is not discretisation error.
- Baseline comparison. On the 371 shared dates the listed implied volatility and India VIX have correlation 0.889 (listed 14.75% on average against VIX 15.80%), and the P&L / premium of the two versions has correlation 0.852. The baseline gets the sign and rough size of the result right, and it is available for all days, but it is biased: its mean P&L is about twice the listed one (0.111 against 0.058), consistent with VIX being above the at-the-money implied volatility so the baseline seller collects more premium, and its tails are heavier (kurtosis 6.8 against 2.5, skew $-1.81$ against $-1.10$). Its all-days version (2,884 windows) has the same shape as the same-dates version (skew $-1.63$, kurtosis 5.85), so the difference is not a matter of which dates are used. I have not isolated why the baseline has heavier tails. The baseline is therefore a reasonable stand-in for direction and rough size, not for tail estimates.

**Limits.** Closing prices only, not intraday quotes or bid-ask (`ClsPric` may be stale for thinly traded strikes, mitigated by requiring positive volume and a strike within 1% of spot). The older files have no underlying price, so the strike filter uses the Yahoo NIFTY close as the reference level. The 2020 crash is thinly represented: only 7 windows were entered between mid-January and April 2020, and 4 of them pass the at-the-money filter, because exactly-20-day expiries with a traded near-the-money call were sparse then. The rate is a constant 6% over a 12-year period when rates varied, and dividends are ignored; the sensitivity analysis shows this changes the level of the mean P&L materially (about 0.05 per 2 percentage points of rate) though not the spread or shape. No file exists on the NSE archive for 30 March 2021, so no window can be entered on that day. The windows overlap after 2019.
