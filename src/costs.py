"""Proportional transaction costs and Leland's (1985) results.

It extends the BSM hedging argument (hold Delta(t) =
C_x shares and rebalance) by a cost on every trade.

The cost rate k is the round-trip proportional cost used by Leland: buying and
then selling a share costs a fraction k of its price, so each one-way trade
pays (k / 2) * |shares traded| * price.
"""

import numpy as np

from src import bs


def trading_cost(shares_traded, price, k: float):
    """Cost of one trade: (k / 2) |shares traded| price."""
    return 0.5 * k * np.abs(shares_traded) * price


def leland_vol(sigma: float, k: float, dt: float) -> float:
    """Leland's adjusted volatility for a call hedged every dt.

    sigma_hat^2 = sigma^2 (1 + sqrt(2 / pi) k / (sigma sqrt(dt))).
    Pricing and hedging at sigma_hat covers the expected cost of rebalancing,
    since each step trades about C_xx S sigma sqrt(dt) sqrt(2 / pi) shares on average.
    """
    return float(sigma * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * k / (sigma * np.sqrt(dt))))


def leland_expected_cost(S_paths: np.ndarray, K: float, r: float, sigma: float, k: float, T: float) -> float:
    """Expected total (undiscounted) cost of hedging along the given paths.

    The initial purchase costs (k / 2) Delta(0) S_0. Each later rebalance moves
    Delta by about C_xx S sigma sqrt(dt) |Z| with E|Z| = sqrt(2 / pi), so
        E[cost] ~ (k / 2) Delta(0) S_0
                  + (k / 2) sqrt(2 / pi) sigma sqrt(dt) sum_{i >= 1} E[C_xx(t_i, S(t_i)) S(t_i)^2].
    The expectation is estimated by averaging over the supplied paths.
    """
    n_steps = S_paths.shape[1] - 1
    dt = T / n_steps
    times = dt * np.arange(1, n_steps)
    gammas = bs.gamma(S_paths[:, 1:-1], K, r, sigma, T - times)
    dollar_gamma_sum = np.mean(gammas * S_paths[:, 1:-1] ** 2, axis=0).sum()
    initial = trading_cost(bs.call_delta(S_paths[0, 0], K, r, sigma, T), S_paths[0, 0], k)
    return float(initial + 0.5 * k * np.sqrt(2.0 / np.pi) * sigma * np.sqrt(dt) * dollar_gamma_sum)
