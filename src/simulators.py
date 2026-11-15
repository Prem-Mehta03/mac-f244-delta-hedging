"""Stock-price path generators: GBM, Heston and Merton jump-diffusion.

Paths are arrays of shape (n_paths, n_steps + 1), with column 0 equal to the
initial price S_0. All simulators take a NumPy random generator for
reproducibility. GBM is sampled from its exact solution; Euler is used only for
the Heston variance process.
"""

import numpy as np


def gbm_paths(
    S0: float,
    alpha: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample GBM dS(t) = alpha S(t) dt + sigma S(t) dW(t) from its exact solution.

    S(t) = S_0 exp{sigma W(t) + (alpha - sigma^2 / 2) t}, evaluated on the grid
    t_i = i T / n_steps with W(t_{i+1}) - W(t_i) = Delta W(t_i) ~ N(0, Delta t),
    so that there is no discretisation error whatever the step size. Hence
    E[S(t)] = S_0 exp(alpha t) and Var(S(t)) = S_0^2 exp(2 alpha t) (exp(sigma^2 t) - 1).

    Args:
        S0: initial price S_0.
        alpha: stock drift (growth rate).
        sigma: volatility.
        T: horizon in years.
        n_steps: number of time steps n (Delta t = T / n_steps).
        n_paths: number of Monte Carlo paths.
        rng: NumPy random generator, used for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) of stock prices.
    """
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    log_increments = sigma * dW + (alpha - 0.5 * sigma**2) * dt
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def heston_paths(
    S0: float,
    alpha: float,
    v0: float,
    a_v: float,
    beta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    return_variance: bool = False,
):
    """Simulate Heston paths: GBM with a stochastic variance V(t) that follows a CIR process.

    The variance V is the CIR process of the notes, dR = (a - beta R) dt + sigma sqrt(R) dW,
    with the mapping R -> V, a -> a_V, sigma -> xi:

        dS(t) = alpha S(t) dt + sqrt(V(t)) S(t) dW_1(t),
        dV(t) = (a_V - beta V(t)) dt + xi sqrt(V(t)) dW_2(t),   dW_1 dW_2 = rho dt.

    (In the usual Heston notation a_V = kappa theta and beta = kappa.) The
    variance uses the full-truncation Euler scheme, with V+ = max(V, 0) in every
    coefficient,
        V_{i+1} = V_i + (a_V - beta V_i+) Delta t + xi sqrt(V_i+ Delta t) Z_2,
    and the stock then takes the exact lognormal step with the variance frozen at V_i+,
        log S_{i+1} = log S_i + (alpha - V_i+ / 2) Delta t + sqrt(V_i+ Delta t) Z_1,
    where Z_2 = rho Z_1 + sqrt(1 - rho^2) Z_perp.

    Args:
        S0: initial price S_0.
        alpha: stock drift.
        v0: initial variance V(0).
        a_v: CIR constant a_V (long-run mean of V is a_V / beta).
        beta: CIR mean-reversion speed.
        xi: volatility of the variance process.
        rho: correlation between the stock and variance shocks.
        T: horizon in years.
        n_steps: number of Euler steps.
        n_paths: number of Monte Carlo paths.
        rng: NumPy random generator.
        return_variance: if True, also return the variance paths.

    Returns:
        Array of shape (n_paths, n_steps + 1) of stock prices, or the pair
        (stock prices, variance paths) if `return_variance` is True.
    """
    dt = T / n_steps
    z_stock = rng.standard_normal((n_paths, n_steps))
    z_perp = rng.standard_normal((n_paths, n_steps))
    z_var = rho * z_stock + np.sqrt(1.0 - rho**2) * z_perp

    log_paths = np.zeros((n_paths, n_steps + 1))
    var_paths = np.zeros((n_paths, n_steps + 1)) if return_variance else None
    V = np.full(n_paths, v0, dtype=float)
    if return_variance:
        var_paths[:, 0] = V
    for i in range(n_steps):
        V_plus = np.maximum(V, 0.0)
        log_paths[:, i + 1] = log_paths[:, i] + (alpha - 0.5 * V_plus) * dt + np.sqrt(V_plus * dt) * z_stock[:, i]
        V = V + (a_v - beta * V_plus) * dt + xi * np.sqrt(V_plus * dt) * z_var[:, i]
        if return_variance:
            var_paths[:, i + 1] = V
    stock = S0 * np.exp(log_paths)
    return (stock, var_paths) if return_variance else stock


def merton_paths(
    S0: float,
    alpha: float,
    sigma: float,
    jump_rate: float,
    jump_mean: float,
    jump_std: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Merton (1976) jump-diffusion paths exactly on a uniform grid.

    Extends GBM by compound Poisson jumps:

        dS / S = (alpha - lam kbar) dt + sigma dW + (J - 1) dN,

    where N is a Poisson process with rate lam, ln J ~ N(m, delta^2) and
    kbar = E[J - 1] = exp(m + delta^2 / 2) - 1. The compensator lam kbar keeps
    E[S(T)] = S_0 exp(alpha T). Over one step with n jumps, the log-increment is
        (alpha - lam kbar - sigma^2 / 2) Delta t + sigma Delta W + n m + delta sqrt(n) Z',
    since a sum of n independent N(m, delta^2) variables is N(n m, n delta^2).

    Args:
        S0: initial price S_0.
        alpha: total expected return of the stock.
        sigma: diffusion volatility (jumps come on top of it).
        jump_rate: jump intensity lam per year.
        jump_mean: mean m of the log jump size.
        jump_std: standard deviation delta of the log jump size.
        T: horizon in years.
        n_steps: number of time steps.
        n_paths: number of Monte Carlo paths.
        rng: NumPy random generator.

    Returns:
        Array of shape (n_paths, n_steps + 1) of stock prices.
    """
    dt = T / n_steps
    kbar = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    diffusion = (alpha - jump_rate * kbar - 0.5 * sigma**2) * dt + sigma * dW
    counts = rng.poisson(jump_rate * dt, size=(n_paths, n_steps))
    jumps = counts * jump_mean + jump_std * np.sqrt(counts) * rng.standard_normal((n_paths, n_steps))
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(diffusion + jumps, axis=1)], axis=1)
    return S0 * np.exp(log_paths)
