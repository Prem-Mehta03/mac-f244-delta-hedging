"""Black-Scholes-Merton formulas in the notation of the class notes.

The stock follows dS(t) = alpha S(t) dt + sigma S(t) dW(t) and the risk-free
rate r is constant. The call price C(t, x), with x = S(t), solves the BSM equation

    r C = C_t + r x C_x + 0.5 sigma^2 x^2 C_xx,    C(T, x) = (x - K)^+,

whose solution is C(t, x) = BSM(tau, x, K, r, sigma) with tau = T - t. The
hedge ratio is Delta(t) = C_x(t, S(t)). All functions accept scalars or NumPy
arrays for x and tau.

An optional continuous dividend yield q (default 0)
replaces x by x exp(-q tau) in the price and r by r - q inside d_+ and d_-.
With q = 0 every formula below is the one in the notes.
"""

import numpy as np
from scipy.stats import norm


def d_plus_minus(x, K, r, sigma, tau, q=0.0):
    """Return (d_+, d_-) of the BSM formula.

    d_+(tau, x) = [log(x / K) + (r - q + sigma^2 / 2) tau] / (sigma sqrt(tau)),
    d_-(tau, x) = d_+(tau, x) - sigma sqrt(tau).

    Args:
        x: stock price S(t).
        K: strike.
        r: risk-free rate.
        sigma: volatility.
        tau: time to maturity T - t, must be positive.
        q: continuous dividend yield (default 0, as in the notes).
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sqrt_tau = np.sqrt(tau)
    d_plus = (np.log(x / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    return d_plus, d_plus - sigma * sqrt_tau


def call_price(x, K, r, sigma, tau, q=0.0):
    """BSM call price C(t, x) = x N(d_+) - K exp(-r tau) N(d_-).

    At tau = 0 the payoff (x - K)^+ is returned.
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    d_plus, d_minus = d_plus_minus(x, K, r, sigma, safe_tau, q)
    price = x * np.exp(-q * safe_tau) * norm.cdf(d_plus) - K * np.exp(-r * safe_tau) * norm.cdf(d_minus)
    return np.where(tau > 0, price, np.maximum(x - K, 0.0))


def put_price(x, K, r, sigma, tau, q=0.0):
    """BSM put price K exp(-r tau) N(-d_-) - x N(-d_+), so that C - P = x - K exp(-r tau) (put-call parity).

    At tau = 0 the payoff (K - x)^+ is returned.
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    d_plus, d_minus = d_plus_minus(x, K, r, sigma, safe_tau, q)
    price = K * np.exp(-r * safe_tau) * norm.cdf(-d_minus) - x * np.exp(-q * safe_tau) * norm.cdf(-d_plus)
    return np.where(tau > 0, price, np.maximum(K - x, 0.0))


def call_delta(x, K, r, sigma, tau, q=0.0):
    """Delta C_x(t, x) = N(d_+(tau, x)), the number of shares that hedges the call.

    At tau = 0 the step function 1{x > K} is returned.
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    d_plus, _ = d_plus_minus(x, K, r, sigma, safe_tau, q)
    return np.where(tau > 0, np.exp(-q * safe_tau) * norm.cdf(d_plus), (x > K).astype(float))


def call_theta(x, K, r, sigma, tau, q=0.0):
    """Theta C_t(t, x) = -r K exp(-r tau) N(d_-) - x sigma N'(d_+) / (2 sqrt(tau)).

    (With q > 0 the extra term q x exp(-q tau) N(d_+) is added and N'(d_+) is
    multiplied by exp(-q tau).) Requires tau > 0.
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d_plus, d_minus = d_plus_minus(x, K, r, sigma, tau, q)
    return (
        q * x * np.exp(-q * tau) * norm.cdf(d_plus)
        - r * K * np.exp(-r * tau) * norm.cdf(d_minus)
        - x * sigma * np.exp(-q * tau) * norm.pdf(d_plus) / (2.0 * np.sqrt(tau))
    )


def gamma(x, K, r, sigma, tau, q=0.0):
    """Gamma C_xx(t, x) = N'(d_+(tau, x)) / (x sigma sqrt(tau)).

    It follows from C_x = N(d_+) and d_+ depending on x through log(x / K), so
    d(d_+)/dx = 1 / (x sigma sqrt(tau)). The same for calls and puts. At tau = 0
    the value is set to 0.
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    d_plus, _ = d_plus_minus(x, K, r, sigma, safe_tau, q)
    value = np.exp(-q * safe_tau) * norm.pdf(d_plus) / (x * sigma * np.sqrt(safe_tau))
    return np.where(tau > 0, value, 0.0)
