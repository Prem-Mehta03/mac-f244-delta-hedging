"""Discrete delta hedging of a short European call, in the notation of the class notes.

The hedger is the seller of the call. Wealth X(t) is invested in Delta(t) shares
and a money-market account earning r, with money-market part X(t) - Delta(t) S(t):

    dX(t) = r (X(t) - Delta(t) S(t)) dt + Delta(t) dS(t).

The seller receives the premium X(0) = C(0, S_0) and rebalances to
Delta(t_i) = C_x(t_i, S(t_i)) on every grid date t_i. The hedging error is the
terminal P&L X(T) - (S(T) - K)^+. With continuous trading under GBM and no
costs it is zero (replication); with finitely many dates it is not.
"""

import numpy as np
from scipy.stats import kurtosis, skew

from src import bs
from src.costs import trading_cost


def hedge_pnl_and_cost(
    S_paths: np.ndarray,
    K: float,
    r: float,
    sigma_price: float,
    T: float,
    sigma_h: float | None = None,
    cost_rate: float = 0.0,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Terminal P&L X(T) - (S(T) - K)^+ and total trading cost of a short call hedged with BSM Deltas.

    At time 0 the seller sells the call for X(0) = C(0, S_0) computed at
    `sigma_price`, buys Delta(0) shares and puts the rest in the money market.
    At each later grid date t_i the money market grows by exp(r Delta t_i), then
    the holding is moved to Delta(t_i), paying (Delta(t_i) - Delta(t_{i-1})) S(t_i)
    plus the trading cost (k / 2) |Delta(t_i) - Delta(t_{i-1})| S(t_i) (zero when cost_rate = 0). At T the shares are sold (or
    delivered) without cost and the payoff is paid, so
        P&L = X(T) - (S(T) - K)^+ = cash_T + Delta(t_{n-1}) S(T) - (S(T) - K)^+.
    The Deltas use the hedge volatility `sigma_h`, which is `sigma_price` unless a
    different value is passed (the wrong-volatility case of E2).

    Args:
        S_paths: array (n_paths, n_steps + 1) of S(t_i) on a uniform grid over [0, T].
        K: strike (a float, or an array with one strike per path).
        r: risk-free rate.
        sigma_price: volatility used for the premium C(0, S_0) (float or one per path).
        T: option maturity in years.
        sigma_h: volatility used for the Deltas; defaults to `sigma_price`.
        cost_rate: round-trip proportional cost k (0 for no costs).
        q: continuous dividend yield of the stock (default 0, as in the notes). The Deltas
            and the premium use the dividend-adjusted BSM formulas, and the shares held
            earn dividends of q S dt per share, added to the money market account.

    Returns:
        (pnl, total_cost): arrays of length n_paths. `total_cost` is the plain
        sum of all trading costs (not compounded), already included in `pnl`.
    """
    if sigma_h is None:
        sigma_h = sigma_price
    n_steps = S_paths.shape[1] - 1
    dt = T / n_steps
    S = S_paths[:, 0]

    delta = bs.call_delta(S, K, r, sigma_h, T, q)
    total_cost = trading_cost(delta, S, cost_rate)
    cash = bs.call_price(S, K, r, sigma_price, T, q) - delta * S - total_cost
    previous_S = S
    for i in range(1, n_steps):
        S = S_paths[:, i]
        new_delta = bs.call_delta(S, K, r, sigma_h, T - i * dt, q)
        cost = trading_cost(new_delta - delta, S, cost_rate)
        dividends = delta * previous_S * q * dt
        cash = cash * np.exp(r * dt) + dividends - (new_delta - delta) * S - cost
        total_cost = total_cost + cost
        delta = new_delta
        previous_S = S

    S_T = S_paths[:, -1]
    cash = cash * np.exp(r * dt) + delta * previous_S * q * dt
    return cash + delta * S_T - np.maximum(S_T - K, 0.0), total_cost


def hedge_pnl(
    S_paths: np.ndarray,
    K: float,
    r: float,
    sigma_price: float,
    T: float,
    sigma_h: float | None = None,
    cost_rate: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Terminal P&L X(T) - (S(T) - K)^+ of the hedged short call; see `hedge_pnl_and_cost`."""
    return hedge_pnl_and_cost(S_paths, K, r, sigma_price, T, sigma_h, cost_rate, q)[0]


def error_variance_profile(S_paths: np.ndarray, K: float, r: float, sigma: float, T: float) -> np.ndarray:
    """Variance contributed to the hedging error by each rebalancing interval.

    Over the step [t_i, t_{i+1}] the error is about
        0.5 C_xx sigma^2 S^2 [(Delta W(t_i))^2 - Delta t_i],
    which is nonzero only because (Delta W)^2 differs from Delta t at a finite step
    size (quadratic variation [W, W](T) = T only in the limit). Its conditional
    variance is 0.5 (C_xx sigma^2 S^2)^2 (Delta t_i)^2 because Var[(Delta W)^2] = 2 (Delta t)^2.
    Errors of different steps are uncorrelated, and each is compounded to time T
    by exp(r (T - t_i)), so step i contributes
        0.5 (Delta t_i)^2 E[(exp(r (T - t_i)) C_xx sigma^2 S(t_i)^2)^2].
    The expectation is estimated by averaging over the supplied paths.

    Returns:
        Array of length n_steps with the variance contributed by each step.
    """
    n_steps = S_paths.shape[1] - 1
    dt = T / n_steps
    times = dt * np.arange(n_steps)
    gammas = bs.gamma(S_paths[:, :-1], K, r, sigma, T - times)
    weights = np.exp(r * (T - times)) * sigma**2 * S_paths[:, :-1] ** 2 * gammas
    return 0.5 * dt**2 * np.mean(weights**2, axis=0)


def predicted_pnl_std(S_paths: np.ndarray, K: float, r: float, sigma: float, T: float) -> float:
    """Std of the hedging error predicted by the per-step Ito formula.

    The square root of the sum of `error_variance_profile` over all steps, so
    Var[P&L] ~ sum_i 0.5 (Delta t_i)^2 E[(exp(r (T - t_i)) C_xx sigma^2 S(t_i)^2)^2],
    which is of order Delta t, i.e. the standard deviation is of order sqrt(Delta t).
    """
    return float(np.sqrt(error_variance_profile(S_paths, K, r, sigma, T).sum()))


def wrong_vol_pnl_formula(
    S_paths: np.ndarray, K: float, r: float, sigma: float, sigma_h: float, T: float
) -> np.ndarray:
    """Pathwise terminal P&L of a short call sold and hedged at sigma_h while the stock has volatility sigma.

    With the BSM equation holding at sigma_h and the stock moving at sigma, Ito's
    formula on exp(-r t) (X(t) - C(t, S(t))) cancels the dW terms and leaves
        d(exp(-r t) (X(t) - C(t, S(t)))) = 0.5 (sigma_h^2 - sigma^2) exp(-r t) S(t)^2 C_xx(t, S(t)) dt,
    so the seller's discounted terminal P&L is
        exp(-r T) P&L = 0.5 (sigma_h^2 - sigma^2) int_0^T exp(-r t) S(t)^2 C_xx(t, S(t)) dt,
    positive when sigma_h > sigma (El Karoui, Jeanblanc-Picque and Shreve, 1998).
    This function returns P&L at time T, i.e. exp(rT) times the right-hand side,
    with C_xx the BSM Gamma at sigma_h. The integral is a left-endpoint sum over
    the grid of `S_paths`.
    """
    n_steps = S_paths.shape[1] - 1
    dt = T / n_steps
    times = dt * np.arange(n_steps)
    gammas = bs.gamma(S_paths[:, :-1], K, r, sigma_h, T - times)
    integrand = np.exp(r * (T - times)) * S_paths[:, :-1] ** 2 * gammas
    return 0.5 * (sigma_h**2 - sigma**2) * integrand.sum(axis=1) * dt


def pnl_summary(pnl: np.ndarray, tail: float = 0.05) -> dict[str, float]:
    """Summary statistics of a P&L sample.

    Returns the mean (bias), standard deviation, skewness, excess kurtosis, and
    the tail risk measures VaR and ES at level `tail`.
    Both are reported as positive losses: VaR = -(tail-quantile of P&L), ES = -(mean
    of the P&L values at or below that quantile).
    """
    cutoff = np.quantile(pnl, tail)
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "skew": float(skew(pnl)),
        "excess_kurtosis": float(kurtosis(pnl)),
        "VaR_5": float(-cutoff),
        "ES_5": float(-pnl[pnl <= cutoff].mean()),
    }


def bootstrap_standard_errors(pnl: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Monte Carlo standard errors of the statistics in `pnl_summary`, by bootstrapping the simulated paths.

    The paths are independent, so each of the `n_boot` resamples draws len(pnl)
    paths with replacement and recomputes the mean, standard deviation, skewness,
    excess kurtosis and 5% expected shortfall; the standard error is the standard
    deviation of each statistic over the resamples. Skewness, kurtosis and ES depend
    on the tails and are much noisier than the standard deviation.
    """
    rng = np.random.default_rng(seed)
    samples = pnl[rng.integers(0, len(pnl), size=(n_boot, len(pnl)))]
    cutoff = np.quantile(samples, 0.05, axis=1, keepdims=True)
    in_tail = samples <= cutoff
    replicates = {
        "mean": samples.mean(axis=1),
        "std": samples.std(axis=1),
        "skew": skew(samples, axis=1),
        "excess_kurtosis": kurtosis(samples, axis=1),
        "ES_5": -(np.where(in_tail, samples, 0.0).sum(axis=1) / in_tail.sum(axis=1)),
    }
    return {name: float(values.std()) for name, values in replicates.items()}


def pathwise_pnl_formula(S_paths: np.ndarray, K, r: float, sigma_h, T: float) -> np.ndarray:
    """Pathwise P&L at T of the seller hedged at sigma_h, from the realised squared returns of the path.

    Over the step [t_i, t_{i+1}] the hedging error is, to second order,
    0.5 C_xx [(Delta S)^2 - sigma_h^2 S^2 Delta t_i] with the sign of the seller's P&L reversed, so
        P&L = 0.5 sum_i exp(r (T - t_i)) S(t_i)^2 C_xx(t_i, S(t_i)) [sigma_h^2 Delta t_i - (ln S(t_{i+1}) / S(t_i))^2],
    with C_xx the BSM gamma at sigma_h. Each step's squared return is weighted by the
    dollar gamma of that step, unlike the constant-sigma formula of E2, which replaces
    every squared return by one window-average sigma^2 Delta t. It needs no assumption
    about the true volatility, so it holds for any path, but it is a second-order
    expansion and fails when a single step has a large move.

    Args:
        S_paths: array (n_paths, n_steps + 1) of S(t_i) on the hedging grid.
        K: strike (float, or one per path).
        r: risk-free rate.
        sigma_h: hedge volatility (float, or one per path).
        T: maturity in years.

    Returns:
        Array of length n_paths with the approximate P&L at time T.
    """
    n_steps = S_paths.shape[1] - 1
    dt = T / n_steps
    times = dt * np.arange(n_steps)
    K = np.asarray(K, dtype=float)
    sigma_h = np.asarray(sigma_h, dtype=float)
    if K.ndim == 1:
        K = K[:, None]
    if sigma_h.ndim == 1:
        sigma_h = sigma_h[:, None]
    gammas = bs.gamma(S_paths[:, :-1], K, r, sigma_h, T - times)
    weights = np.exp(r * (T - times)) * S_paths[:, :-1] ** 2 * gammas
    squared_returns = np.log(S_paths[:, 1:] / S_paths[:, :-1]) ** 2
    return 0.5 * (weights * (sigma_h**2 * dt - squared_returns)).sum(axis=1)
