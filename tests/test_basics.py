"""Sanity checks, in the notation of the class notes.

BSM (price, Delta C_x, Theta C_t, Gamma C_xx, put-call parity, BSM equation),
GBM moments, quadratic variation, the CIR/Heston variance, the hedging engine,
the wrong-volatility formula, and the backtest helpers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bs

S0, K, R, SIGMA, TAU = 100.0, 100.0, 0.05, 0.2, 1.0


# ---------------------------------------------------------------- BSM
def test_call_price_known_value():
    """Standard closed-form value: x = K = 100, r = 5%, sigma = 20%, tau = 1 gives 10.4506."""
    assert bs.call_price(S0, K, R, SIGMA, TAU) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    """C - P = x - K exp(-r tau) across a grid of prices and maturities."""
    xs = np.linspace(60, 140, 9)
    for tau in (0.1, 0.5, 2.0):
        lhs = bs.call_price(xs, K, R, SIGMA, tau) - bs.put_price(xs, K, R, SIGMA, tau)
        assert lhs == pytest.approx(xs - K * np.exp(-R * tau), abs=1e-10)


def test_delta_between_zero_and_one():
    xs = np.linspace(20, 300, 50)
    delta = bs.call_delta(xs, K, R, SIGMA, 0.5)
    assert np.all((delta > 0) & (delta < 1))


def test_delta_matches_finite_difference():
    """C_x(t, x) = N(d_+) against a central difference of C in x."""
    h = 1e-4
    up = bs.call_price(S0 + h, K, R, SIGMA, TAU)
    down = bs.call_price(S0 - h, K, R, SIGMA, TAU)
    assert bs.call_delta(S0, K, R, SIGMA, TAU) == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_gamma_matches_finite_difference():
    """C_xx(t, x) = N'(d_+) / (x sigma sqrt(tau)) against a central difference of C_x."""
    h = 1e-2
    up = bs.call_delta(S0 + h, K, R, SIGMA, TAU)
    down = bs.call_delta(S0 - h, K, R, SIGMA, TAU)
    assert bs.gamma(S0, K, R, SIGMA, TAU) == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_theta_matches_finite_difference():
    """C_t = -dC/dtau, against a central difference of C in tau."""
    h = 1e-4
    c_tau = (bs.call_price(S0, K, R, SIGMA, TAU + h) - bs.call_price(S0, K, R, SIGMA, TAU - h)) / (2 * h)
    assert bs.call_theta(S0, K, R, SIGMA, TAU) == pytest.approx(-c_tau, abs=1e-6)


def test_bsm_equation_holds():
    """r C = C_t + r x C_x + 0.5 sigma^2 x^2 C_xx, using the closed-form Greeks of the notes."""
    for x in (70.0, 100.0, 130.0):
        lhs = R * bs.call_price(x, K, R, SIGMA, TAU)
        rhs = (
            bs.call_theta(x, K, R, SIGMA, TAU)
            + R * x * bs.call_delta(x, K, R, SIGMA, TAU)
            + 0.5 * SIGMA**2 * x**2 * bs.gamma(x, K, R, SIGMA, TAU)
        )
        assert lhs == pytest.approx(rhs, abs=1e-9)


def test_boundary_conditions():
    """C(T, x) = (x - K)^+, C(t, 0) = 0 and C(t, x) - (x - K exp(-r tau)) -> 0 as x -> infinity."""
    assert bs.call_price(110.0, K, R, SIGMA, 0.0) == pytest.approx(10.0)
    assert bs.call_price(90.0, K, R, SIGMA, 0.0) == 0.0
    assert bs.call_price(1e-6, K, R, SIGMA, TAU) == pytest.approx(0.0, abs=1e-9)
    x = 1e4
    assert bs.call_price(x, K, R, SIGMA, TAU) - (x - K * np.exp(-R * TAU)) == pytest.approx(0.0, abs=1e-6)


def test_expiry_delta():
    assert bs.call_delta(90.0, K, R, SIGMA, 0.0) == 0.0
    assert bs.call_delta(110.0, K, R, SIGMA, 0.0) == 1.0


def test_dividend_yield_parity_and_zero_default():
    """With dividend yield q, C - P = x exp(-q tau) - K exp(-r tau); q = 0 gives the notes' formulas."""
    q = 0.03
    lhs = bs.call_price(S0, K, R, SIGMA, TAU, q) - bs.put_price(S0, K, R, SIGMA, TAU, q)
    assert lhs == pytest.approx(S0 * np.exp(-q * TAU) - K * np.exp(-R * TAU), abs=1e-10)
    assert bs.call_price(S0, K, R, SIGMA, TAU, 0.0) == bs.call_price(S0, K, R, SIGMA, TAU)
    h = 1e-4
    delta_fd = (bs.call_price(S0 + h, K, R, SIGMA, TAU, q) - bs.call_price(S0 - h, K, R, SIGMA, TAU, q)) / (2 * h)
    assert bs.call_delta(S0, K, R, SIGMA, TAU, q) == pytest.approx(delta_fd, abs=1e-6)
    c_tau = (bs.call_price(S0, K, R, SIGMA, TAU + h, q) - bs.call_price(S0, K, R, SIGMA, TAU - h, q)) / (2 * h)
    assert bs.call_theta(S0, K, R, SIGMA, TAU, q) == pytest.approx(-c_tau, abs=1e-6)


# ---------------------------------------------------------------- GBM and quadratic variation
def test_gbm_mean_and_variance():
    """E[S(t)] = S_0 exp(alpha t) and Var(S(t)) = S_0^2 exp(2 alpha t)(exp(sigma^2 t) - 1)."""
    from src.simulators import gbm_paths

    alpha, n_paths = 0.08, 400_000
    S_T = gbm_paths(S0, alpha, SIGMA, TAU, 4, n_paths, np.random.default_rng(0))[:, -1]
    mean = S0 * np.exp(alpha * TAU)
    var = S0**2 * np.exp(2 * alpha * TAU) * (np.exp(SIGMA**2 * TAU) - 1)
    assert S_T.mean() == pytest.approx(mean, rel=3 * np.sqrt(var / n_paths) / mean)
    assert S_T.var() == pytest.approx(var, rel=0.02)


def test_brownian_increment_squared_has_variance_two_dt_squared():
    """(Delta W)^2 has mean Delta t and variance 2 (Delta t)^2, the source of the sqrt(Delta t) law."""
    dt = 0.01
    dW = np.sqrt(dt) * np.random.default_rng(9).standard_normal(2_000_000)
    assert (dW**2).mean() == pytest.approx(dt, rel=0.01)
    assert (dW**2).var() == pytest.approx(2 * dt**2, rel=0.02)


def test_realised_volatility_recovers_sigma():
    """Quadratic variation: sum of [ln S(t_{i+1}) / S(t_i)]^2 / (T_2 - T_1) tends to sigma^2."""
    from src.backtest import realised_vol
    from src.simulators import gbm_paths

    T = 10.0
    paths = gbm_paths(S0, 0.07, SIGMA, T, 2520, 200, np.random.default_rng(10))
    assert realised_vol(paths, T).mean() == pytest.approx(SIGMA, abs=0.005)
    assert realised_vol(np.array([100.0, 101.0, 100.0]), 2 / 252) == pytest.approx(
        np.sqrt((np.log(1.01) ** 2 + np.log(100 / 101) ** 2) * 252 / 2)
    )


# ---------------------------------------------------------------- hedging
def test_hedge_error_small_with_many_steps():
    """Under GBM the hedge converges to replication: error near 0 for n = 2000 steps."""
    from src.hedging import hedge_pnl
    from src.simulators import gbm_paths

    paths = gbm_paths(S0, 0.05, SIGMA, TAU, 2000, 2000, np.random.default_rng(1))
    pnl = hedge_pnl(paths, K, R, SIGMA, TAU)
    assert abs(pnl.mean()) < 0.05
    assert pnl.std() < 0.5


def test_wrong_volatility_pnl_sign_and_formula():
    """The seller hedging at sigma_h > sigma gains 0.5 (sigma_h^2 - sigma^2) int exp(-rt) S^2 C_xx dt (discounted)."""
    from src.hedging import hedge_pnl, wrong_vol_pnl_formula
    from src.simulators import gbm_paths

    paths = gbm_paths(S0, R, SIGMA, TAU, 500, 5000, np.random.default_rng(11))
    for sigma_h, sign in ((0.25, 1), (0.15, -1)):
        pnl = hedge_pnl(paths, K, R, sigma_h, TAU)
        formula = wrong_vol_pnl_formula(paths, K, R, SIGMA, sigma_h, TAU)
        assert np.all(sign * formula > 0)
        assert pnl.mean() == pytest.approx(formula.mean(), abs=0.15)


def test_zero_cost_rate_changes_nothing_and_costs_reduce_pnl():
    from src.costs import leland_vol
    from src.hedging import hedge_pnl, hedge_pnl_and_cost
    from src.simulators import gbm_paths

    paths = gbm_paths(S0, 0.05, SIGMA, TAU, 52, 1000, np.random.default_rng(5))
    free, free_cost = hedge_pnl_and_cost(paths, K, R, SIGMA, TAU)
    costly, cost = hedge_pnl_and_cost(paths, K, R, SIGMA, TAU, cost_rate=0.01)
    assert np.all(free_cost == 0.0)
    assert np.all(costly < free)
    assert np.all(cost > 0.0)
    assert np.all(costly >= free - cost * np.exp(R * TAU))
    assert hedge_pnl(paths, K, R, SIGMA, TAU) == pytest.approx(free)
    assert leland_vol(SIGMA, 0.0, 1 / 52) == pytest.approx(SIGMA)


def test_hedge_with_dividends_converges():
    """Price index with drift alpha - q, dividends earned on the shares: small hedging error."""
    from src.hedging import hedge_pnl
    from src.simulators import gbm_paths

    q = 0.03
    paths = gbm_paths(S0, R - q, SIGMA, TAU, 2000, 2000, np.random.default_rng(8))
    pnl = hedge_pnl(paths, K, R, SIGMA, TAU, q=q)
    assert abs(pnl.mean()) < 0.05
    assert pnl.std() < 0.5


# ---------------------------------------------------------------- Heston (CIR variance) and Merton
def test_cir_variance_mean_and_long_run_variance():
    """CIR: E[V(t)] = V(0) e^{-beta t} + (a_V / beta)(1 - e^{-beta t}) and Var(V(t)) -> a_V xi^2 / (2 beta^2)."""
    from src.simulators import heston_paths

    a_v, beta, xi, v0, T = 0.08, 2.0, 0.3, 0.10, 6.0
    _, V = heston_paths(S0, 0.05, v0, a_v, beta, xi, -0.5, T, 1200, 100_000, np.random.default_rng(12),
                        return_variance=True)
    t = np.linspace(0.0, T, V.shape[1])
    expected_mean = v0 * np.exp(-beta * t) + (a_v / beta) * (1 - np.exp(-beta * t))
    for i in (100, 300, 600, 1200):
        assert V[:, i].mean() == pytest.approx(expected_mean[i], rel=0.01)
    assert V[:, -1].var() == pytest.approx(a_v * xi**2 / (2 * beta**2), rel=0.05)


def test_heston_reduces_to_gbm_when_vol_of_vol_is_zero():
    """With xi = 0 and V(0) = a_V / beta the variance is constant, so ln S(T) is Gaussian."""
    from src.simulators import heston_paths

    S_T = heston_paths(S0, 0.05, 0.04, 0.08, 2.0, 0.0, -0.7, TAU, 50, 200_000, np.random.default_rng(2))[:, -1]
    log_ret = np.log(S_T / S0)
    assert log_ret.mean() == pytest.approx(0.05 - 0.02, abs=0.002)
    assert log_ret.std() == pytest.approx(0.2, rel=0.01)


def test_heston_stock_mean():
    """E[S(T)] = S_0 exp(alpha T) for Heston; the Euler scheme for V has only a small bias."""
    from src.simulators import heston_paths

    S_T = heston_paths(S0, 0.05, 0.04, 0.08, 2.0, 0.5, -0.7, TAU, 252, 100_000, np.random.default_rng(3))[:, -1]
    assert np.all(np.isfinite(S_T))
    assert S_T.mean() == pytest.approx(S0 * np.exp(0.05), rel=0.005)


def test_merton_mean_and_log_variance():
    """E[S(T)] = S_0 exp(alpha T); Var[ln S(T)] = (sigma^2 + lam (m^2 + delta^2)) T."""
    from src.simulators import merton_paths

    lam, m, delta, sigma = 1.0, -0.1, 0.1, 0.15
    S_T = merton_paths(S0, 0.05, sigma, lam, m, delta, TAU, 52, 300_000, np.random.default_rng(4))[:, -1]
    assert S_T.mean() == pytest.approx(S0 * np.exp(0.05), rel=0.003)
    assert np.log(S_T).var() == pytest.approx((sigma**2 + lam * (m**2 + delta**2)) * TAU, rel=0.01)


# ---------------------------------------------------------------- backtest helpers
def test_listed_backtest_recovers_volatility_on_model_prices():
    """Calls priced by BSM at 20% on exact GBM data: implied vol is recovered and the hedge error is small."""
    import pandas as pd

    from src.backtest import make_windows, run_listed_backtest
    from src.simulators import gbm_paths

    prices = np.arange(1.0, 11.0)
    windows = make_windows(prices, 4)
    assert windows.shape == (6, 5)
    assert windows[1].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]

    path = gbm_paths(S0, 0.06, SIGMA, 20.0, 5040, 1, np.random.default_rng(6))[0]
    dates = pd.bdate_range("2000-01-03", periods=len(path))
    strikes = np.round(path)
    prices = bs.call_price(path, strikes, 0.06, SIGMA, 20 / 252)
    options = pd.DataFrame(
        {"TradDt": dates[:600], "XpryDt": dates[20:620], "StrkPric": strikes[:600], "ClsPric": prices[:600],
         "SttlmPric": 0.0, "UndrlygPric": path[:600], "OpnIntrst": 0, "TtlTradgVol": 1}
    )
    result = run_listed_backtest(pd.DataFrame({"nifty": path}, index=dates), options, n_days=20, r=0.06)
    assert len(result) == 600
    assert result["implied_vol"].to_numpy() == pytest.approx(SIGMA, abs=1e-6)
    assert abs(result["rel_pnl"].mean()) < 0.1


def test_implied_vol_round_trip():
    """The implied volatility of the BSM price at sigma = 27% is 27%; a worthless price has none."""
    from src.backtest import implied_vol

    price = float(bs.call_price(S0, 105.0, R, 0.27, 0.25))
    assert implied_vol(price, S0, 105.0, R, 0.25) == pytest.approx(0.27, abs=1e-8)
    assert np.isnan(implied_vol(0.0, S0, 105.0, R, 0.25))


def test_implied_vol_round_trip_with_dividends():
    from src.backtest import implied_vol

    price = float(bs.call_price(S0, 105.0, R, 0.27, 0.25, 0.02))
    assert implied_vol(price, S0, 105.0, R, 0.25, 0.02) == pytest.approx(0.27, abs=1e-8)


def test_vix_proxy_backtest_on_gbm_data():
    """On exact GBM data with VIX equal to the true 20% volatility the proxy hedge error is near 0."""
    import pandas as pd

    from src.backtest import run_vix_proxy_backtest
    from src.simulators import gbm_paths

    path = gbm_paths(S0, 0.06, SIGMA, 20.0, 5040, 1, np.random.default_rng(6))[0]
    data = pd.DataFrame({"nifty": path, "india_vix": 20.0}, index=pd.bdate_range("2000-01-03", periods=len(path)))
    result = run_vix_proxy_backtest(data, n_days=20)
    assert len(result) == len(path) - 20
    assert abs(result["rel_pnl"].mean()) < 0.1
    data.iloc[0, 1] = np.nan
    assert len(run_vix_proxy_backtest(data, n_days=20)) == len(path) - 21


def test_block_bootstrap_intervals():
    """On iid N(0.1, 1) data the 95% interval for the mean is about 0.1 +- 0.2 for n = 400, and contains the estimate."""
    from src.backtest import block_bootstrap_summary

    values = np.random.default_rng(7).normal(0.1, 1.0, 400)
    summary = block_bootstrap_summary(values, block_length=5, n_boot=1000, seed=1)
    row = summary.loc["mean"]
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["ci_high"] - row["ci_low"] == pytest.approx(2 * 1.96 / np.sqrt(400), rel=0.25)
    assert summary.loc["skew", "ci_low"] < 0.0 < summary.loc["skew", "ci_high"]


def test_bootstrap_standard_errors_match_normal_theory():
    """For N(0,1) data and n paths: se(mean) = 1/sqrt(n), se(std) = 1/sqrt(2n), se(skew) = sqrt(6/n), se(kurt) = sqrt(24/n)."""
    from src.hedging import bootstrap_standard_errors

    n = 20_000
    se = bootstrap_standard_errors(np.random.default_rng(13).standard_normal(n), n_boot=400, seed=2)
    assert se["mean"] == pytest.approx(1 / np.sqrt(n), rel=0.2)
    assert se["std"] == pytest.approx(1 / np.sqrt(2 * n), rel=0.2)
    assert se["skew"] == pytest.approx(np.sqrt(6 / n), rel=0.2)
    assert se["excess_kurtosis"] == pytest.approx(np.sqrt(24 / n), rel=0.2)


def test_pathwise_formula_explains_the_hedging_pnl():
    """The gamma-weighted formula tracks the simulated P&L of the seller (here sigma_h = 25% against sigma = 20%)."""
    from src.hedging import hedge_pnl, pathwise_pnl_formula
    from src.simulators import gbm_paths

    paths = gbm_paths(S0, R, SIGMA, TAU, 252, 4000, np.random.default_rng(14))
    pnl = hedge_pnl(paths, K, R, 0.25, TAU)
    formula = pathwise_pnl_formula(paths, K, R, 0.25, TAU)
    assert np.corrcoef(pnl, formula)[0, 1] > 0.99
    assert pnl.mean() == pytest.approx(formula.mean(), rel=0.05)
    assert np.std(pnl - formula) < 0.1 * np.std(pnl)
