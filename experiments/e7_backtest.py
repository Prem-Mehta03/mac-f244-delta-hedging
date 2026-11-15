"""E7: backtest of daily and weekly delta hedging on real listed NIFTY call options.

A short NIFTY call is sold at its NSE closing price on every day when an
expiry falls exactly 20 trading days later, and hedged with BSM Deltas at its
own implied volatility along the real NIFTY 50 path until expiry (see
src/backtest.py). The P&L is divided by the option premium. Its distribution
is compared with GBM and Heston simulations at the sample's average
implied volatility, hedged in the same way, and its mean is checked against
the wrong-volatility formula of E2. A baseline hedges a synthetic at-the-money
call priced at India VIX, on the same entry dates and on every trading day, to
check how well that model-priced version reproduces the listed-option results.
Finally the pathwise formula, which weights each day's squared return by that
day's dollar gamma, is compared with the P&L of the listed windows and of the
simulated paths; what is left over comes from large single-day moves.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bs
from src.backtest import (OPTIONS_PATH, TRADING_DAYS, block_bootstrap_summary, download_option_chain,
                          load_market_data, make_windows, realised_vol, run_listed_backtest,
                          run_vix_proxy_backtest)
from src.hedging import hedge_pnl, pathwise_pnl_formula, pnl_summary, wrong_vol_pnl_formula
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths, heston_paths

N_DAYS = 20
R = 0.06
HEDGE_EVERY = {"daily": 1, "weekly": 5}
FIRST_DATE = "2015-01-01"
MAX_STRIKE_GAP = 0.01
LARGE_MOVE = 0.04
SENSITIVITY_GAPS = [0.005, 0.01, 0.02]
BLOCK_LENGTHS = [3, 5, 10]
COST_RATES = [0.0005, 0.001, 0.002]
COST_STEPS = [1, 2, 5, 10]
RATE_GRID = [(0.04, 0.0), (0.06, 0.0), (0.08, 0.0), (0.04, 0.013), (0.06, 0.013), (0.08, 0.013)]
N_PATHS = 20_000
SEED = 42
BETA, XI, RHO = 2.0, 0.4, -0.7


def simulated_paths(sigma: float) -> dict[str, np.ndarray]:
    """Daily paths of the 20-day simulations, by model (GBM and Heston).

    Both start at S_0 = K = 100, with alpha = r, T = 20/252 and total variance
    sigma^2 (the sample's mean implied volatility). Heston uses the settings of
    E3: beta = 2, a_V = beta sigma^2, V(0) = sigma^2, xi = 0.4, rho = -0.7. The
    Merton jump parameters of E4 were chosen for one-year options and do not
    carry over to a 20-day window, so Merton is not simulated here.
    """
    T = N_DAYS / TRADING_DAYS
    generators = {
        "GBM": lambda rng: gbm_paths(100.0, R, sigma, T, N_DAYS, N_PATHS, rng),
        "Heston": lambda rng: heston_paths(100.0, R, sigma**2, BETA * sigma**2, BETA, XI, RHO, T, N_DAYS, N_PATHS, rng),
    }
    return {name: generate(np.random.default_rng(SEED)) for name, generate in generators.items()}


def simulated_rel_pnl(sigma: float, paths: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """P&L / premium of the hedge (premium and Deltas at `sigma`) on the simulated paths, by model and frequency.

    Paths are daily; weekly hedging takes every 5th date.
    """
    T = N_DAYS / TRADING_DAYS
    premium = float(bs.call_price(100.0, 100.0, R, sigma, T))
    return {
        name: {freq: hedge_pnl(p[:, ::step], 100.0, R, sigma, T) / premium for freq, step in HEDGE_EVERY.items()}
        for name, p in paths.items()
    }


def e2_formula_pnl(data: pd.DataFrame, listed: pd.DataFrame, step: int) -> np.ndarray:
    """The E2 formula (P&L at T) for each listed window, with its realised vol as sigma and its implied vol as sigma_h."""
    T = N_DAYS / TRADING_DAYS
    windows = make_windows(data["nifty"].to_numpy(), N_DAYS)[data.index.get_indexer(listed.index)]
    return np.array([
        wrong_vol_pnl_formula(w[None, ::step], K, R, rv, iv, T)[0]
        for w, K, rv, iv in zip(windows, listed["strike"], listed["realised_vol"], listed["implied_vol"])
    ])


def pathwise_table(data: pd.DataFrame, results: dict[str, pd.DataFrame], sigma: float,
                   paths: dict[str, np.ndarray]) -> pd.DataFrame:
    """How well the pathwise (gamma-weighted) formula explains the P&L, for the listed windows and the simulations.

    For every sample the table gives the correlation between the P&L and the
    formula and the standard deviation (and skewness, kurtosis) of the residual
    (P&L - formula) / premium. The listed windows are also split by whether the
    window contains a daily move larger than LARGE_MOVE (4% in log terms), because
    the second-order formula fails on such days.
    """
    T = N_DAYS / TRADING_DAYS
    premium_sim = float(bs.call_price(100.0, 100.0, R, sigma, T))
    all_windows = make_windows(data["nifty"].to_numpy(), N_DAYS)
    rows = []

    def add_row(freq, label, pnl, formula, residual, mean_in_points):
        rows.append({"frequency": freq, "sample": label, "windows": len(pnl),
                     "corr_with_pnl": float(np.corrcoef(pnl, formula)[0, 1]),
                     "mean_pnl": float(np.mean(pnl)) if mean_in_points else np.nan,
                     "mean_formula": float(np.mean(formula)) if mean_in_points else np.nan,
                     "resid_std": float(np.std(residual)), "resid_skew": float(skew(residual)),
                     "resid_excess_kurtosis": float(kurtosis(residual))})

    for freq, step in HEDGE_EVERY.items():
        listed = results[freq]
        windows = all_windows[data.index.get_indexer(listed.index)]
        formula = pathwise_pnl_formula(windows[:, ::step], listed["strike"].to_numpy(), R,
                                       listed["implied_vol"].to_numpy(), T)
        pnl = listed["pnl"].to_numpy()
        residual = (pnl - formula) / listed["premium"].to_numpy()
        large = np.abs(np.diff(np.log(windows), axis=1)).max(axis=1) > LARGE_MOVE
        for label, mask in ((f"listed, all windows", np.ones(len(pnl), bool)),
                            (f"listed, no daily move above {LARGE_MOVE:.0%}", ~large),
                            (f"listed, a daily move above {LARGE_MOVE:.0%}", large)):
            add_row(freq, label, pnl[mask], formula[mask], residual[mask], True)
        for name, p in paths.items():
            hedged = p[:, ::step]
            pnl_sim = hedge_pnl(hedged, 100.0, R, sigma, T)
            formula_sim = pathwise_pnl_formula(hedged, 100.0, R, sigma, T)
            add_row(freq, f"{name} (simulated)", pnl_sim, formula_sim, (pnl_sim - formula_sim) / premium_sim, False)
    return pd.DataFrame(rows)


def bootstrap_table(results: dict[str, pd.DataFrame], table: pd.DataFrame) -> pd.DataFrame:
    """Block-bootstrap intervals of the listed statistics, next to the simulated GBM and Heston values.

    `table` is the summary table of E7. The columns `gbm` and `heston` hold the
    simulated value of the same statistic and `heston_inside` / `gbm_inside`
    say whether it lies inside the 95% interval of the listed sample.
    """
    names = {"mean": "mean", "std": "std", "skew": "skew", "excess_kurtosis": "excess_kurtosis", "ES_5": "ES_5"}
    rows = []
    for freq in HEDGE_EVERY:
        values = results[freq]["rel_pnl"].to_numpy()
        for block in BLOCK_LENGTHS:
            summary = block_bootstrap_summary(values, block_length=block, seed=SEED)
            for stat in names:
                gbm, heston = table.loc[f"GBM {freq}", stat], table.loc[f"Heston {freq}", stat]
                low, high = summary.loc[stat, "ci_low"], summary.loc[stat, "ci_high"]
                rows.append({"frequency": freq, "block": block, "statistic": stat,
                             "estimate": summary.loc[stat, "estimate"], "ci_low": low, "ci_high": high,
                             "gbm": gbm, "gbm_inside": low <= gbm <= high,
                             "heston": heston, "heston_inside": low <= heston <= high})
    return pd.DataFrame(rows)


def cost_table(data: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """Listed-option backtest with proportional trading costs, for several rebalancing frequencies.

    Rebalancing every 1, 2, 5 or 10 trading days gives N = 20, 10, 4 or 2 hedges
    in the 20-day life. For each round-trip cost rate k the table gives the mean
    total cost and the RMSE of the net P&L, both relative to the premium, which is
    the real-data counterpart of the cost-risk trade-off in E5.
    """
    rows = []
    for k in [0.0] + COST_RATES:
        for step in COST_STEPS:
            result = run_listed_backtest(data, options, N_DAYS, step, R, MAX_STRIKE_GAP, cost_rate=k)
            rel = result["rel_pnl"]
            rows.append({"k": k, "N": N_DAYS // step, "windows": len(result),
                         "mean_cost": (result["cost"] / result["premium"]).mean(), "mean_pnl": rel.mean(),
                         "std_pnl": rel.std(), "rmse": float(np.sqrt(np.mean(rel**2))),
                         "share_losing": float((rel < 0).mean())})
    return pd.DataFrame(rows)


def rate_dividend_table(data: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """Listed backtest (daily and weekly hedging) for several risk-free rates r and dividend yields q.

    The rate and the dividend yield enter the implied volatility, the Deltas and
    the cash account. q = 0 is the base case (dividends ignored); 1.3% is my
    rough figure for the NIFTY 50 dividend yield, an assumption not taken from data.
    """
    rows = []
    for r, q in RATE_GRID:
        for freq, step in HEDGE_EVERY.items():
            result = run_listed_backtest(data, options, N_DAYS, step, r, MAX_STRIKE_GAP, q=q)
            rows.append({"r": r, "q": q, "frequency": freq, "windows": len(result),
                         "mean_implied_vol": result["implied_vol"].mean(), **pnl_summary(result["rel_pnl"].to_numpy())})
    return pd.DataFrame(rows)


def main() -> None:
    """Run E7, print and save the comparison tables, save the figures."""
    data = load_market_data()
    options = download_option_chain(data.index[data.index >= FIRST_DATE], OPTIONS_PATH)
    results = {freq: run_listed_backtest(data, options, N_DAYS, step, R, MAX_STRIKE_GAP)
               for freq, step in HEDGE_EVERY.items()}
    proxy_all = {freq: run_vix_proxy_backtest(data, N_DAYS, step, R) for freq, step in HEDGE_EVERY.items()}
    common = results["daily"].index.intersection(proxy_all["daily"].index)
    proxy_same = {freq: proxy_all[freq].loc[common] for freq in HEDGE_EVERY}
    daily = results["daily"]
    gap = (daily["strike"] / daily["spot"] - 1).abs()
    print(f"NIFTY days: {len(data)}; option days cached: {options['TradDt'].nunique()}; windows: {len(daily)} "
          f"from {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"strike within {gap.max():.2%} of spot (median {gap.median():.2%}); "
          f"median premium {daily['premium'].median():.1f} index points")
    print(f"mean implied vol {daily['implied_vol'].mean():.4f}, mean realised vol {daily['realised_vol'].mean():.4f}; "
          f"implied above realised in {(daily['implied_vol'] > daily['realised_vol']).mean():.1%} of windows")

    formula = e2_formula_pnl(data, daily, 1)
    print(f"E2 formula (realised vol per window): mean {formula.mean():.2f} vs backtest {daily['pnl'].mean():.2f} points; "
          f"correlation {np.corrcoef(formula, daily['pnl'])[0, 1]:.3f}")

    print(f"VIX baseline on the {len(common)} listed entry dates: corr(listed implied vol, VIX) "
          f"{np.corrcoef(daily.loc[common, 'implied_vol'], proxy_same['daily']['implied_vol'])[0, 1]:.3f}, mean implied "
          f"{daily.loc[common, 'implied_vol'].mean():.4f} vs mean VIX {proxy_same['daily']['implied_vol'].mean():.4f}; "
          f"corr(rel. P&L, listed vs baseline) "
          f"{np.corrcoef(results['daily'].loc[common, 'rel_pnl'], proxy_same['daily']['rel_pnl'])[0, 1]:.3f}")
    print(f"VIX baseline on all days: {len(proxy_all['daily'])} windows from {proxy_all['daily'].index.min().date()} "
          f"to {proxy_all['daily'].index.max().date()}")

    sigma_sim = daily["implied_vol"].mean()
    sim_paths = simulated_paths(sigma_sim)
    sims = simulated_rel_pnl(sigma_sim, sim_paths)
    print(f"simulations: sigma = {sigma_sim:.4f} (mean implied vol), alpha = r = {R}, T = {N_DAYS}/{TRADING_DAYS}, "
          f"S0 = K = 100, {N_DAYS} daily steps, {N_PATHS} paths")
    samples = {}
    for freq in HEDGE_EVERY:
        samples[f"listed {freq}"] = results[freq]["rel_pnl"].to_numpy()
        samples[f"VIX baseline {freq} (same dates)"] = proxy_same[freq]["rel_pnl"].to_numpy()
        samples[f"VIX baseline {freq} (all days)"] = proxy_all[freq]["rel_pnl"].to_numpy()
    for name, by_freq in sims.items():
        for freq, values in by_freq.items():
            samples[f"{name} {freq}"] = values
    table = pd.DataFrame({name: pnl_summary(values) for name, values in samples.items()}).T
    table.index.name = "sample"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e7_backtest.csv")
    pd.concat(results, names=["frequency"]).to_csv(TABLES_DIR / "e7_backtest_windows.csv")
    print("P&L / premium")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    pathwise = pathwise_table(data, results, sigma_sim, sim_paths)
    pathwise.to_csv(TABLES_DIR / "e7_pathwise.csv", index=False)
    print("Pathwise (gamma-weighted) formula: correlation with the P&L and residual (P&L - formula) / premium")
    print(pathwise.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    one = results["daily"].index.get_loc(pd.Timestamp("2020-02-26"))
    window = make_windows(data["nifty"].to_numpy(), N_DAYS)[data.index.get_indexer(results["daily"].index)][one:one + 1]
    row = results["daily"].iloc[one]
    print(f"26 Feb 2020 (daily hedging), in units of the premium: P&L {row['rel_pnl']:.2f}, "
          f"constant-sigma formula {e2_formula_pnl(data, results['daily'].iloc[one:one + 1], 1)[0] / row['premium']:.2f}, "
          f"gamma-weighted formula {pathwise_pnl_formula(window, row['strike'], R, row['implied_vol'], N_DAYS / TRADING_DAYS)[0] / row['premium']:.2f}")
    sensitivity = []
    for max_gap in SENSITIVITY_GAPS:
        sample = run_listed_backtest(data, options, N_DAYS, 1, R, max_gap)["rel_pnl"]
        sensitivity.append({"max_strike_gap": max_gap, "windows": len(sample), **pnl_summary(sample.to_numpy())})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(TABLES_DIR / "e7_strike_sensitivity.csv", index=False)
    print("Sensitivity to the at-the-money tolerance (daily hedging)")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    boot = bootstrap_table(results, table)
    boot.to_csv(TABLES_DIR / "e7_bootstrap.csv", index=False)
    print("Block bootstrap (block of 5 windows), 95% intervals of the listed statistics")
    print(boot[boot["block"] == 5].drop(columns="block").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Skewness and kurtosis intervals for other block lengths")
    other = boot[boot["statistic"].isin(["skew", "excess_kurtosis"])]
    print(other[["frequency", "block", "statistic", "ci_low", "ci_high"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    costs = cost_table(data, options)
    costs.to_csv(TABLES_DIR / "e7_costs.csv", index=False)
    print("Trading costs on the listed backtest (k = round-trip cost rate, N = hedges in 20 days)")
    print(costs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    rates = rate_dividend_table(data, options)
    rates.to_csv(TABLES_DIR / "e7_rate_dividend.csv", index=False)
    print("Sensitivity to the risk-free rate r and the dividend yield q")
    print(rates.drop(columns=["VaR_5"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    by_year = daily.groupby(daily.index.year)["rel_pnl"].agg(["mean", "std", "count"])
    print(by_year.to_string(float_format=lambda x: f"{x:.3f}"))

    styles = {"listed": ("black", 2.2, "-"), "GBM": ("tab:blue", 1.6, "--"), "Heston": ("tab:orange", 1.6, "-"),
              "VIX baseline": ("tab:green", 1.4, ":")}
    for with_baseline, figure_name in ((False, "e7_distribution"), (True, "e7_distribution_with_baseline")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, freq in zip(axes, HEDGE_EVERY):
            curves = {"listed": ("listed NIFTY options", samples[f"listed {freq}"])}
            if with_baseline:
                curves["VIX baseline"] = ("VIX baseline, same dates", samples[f"VIX baseline {freq} (same dates)"])
            for model in sims:
                curves[model] = (f"{model} (simulated)", sims[model][freq])
            for key, (label, values) in curves.items():
                color, width, style = styles[key]
                ax.hist(values, bins=45, range=(-2.5, 1.5), density=True, histtype="step", color=color,
                        linewidth=width, linestyle=style, label=label)
            ax.set_xlim(-2.5, 1.5)
            ax.set_xlabel("hedging P&L / premium", fontsize=12)
            ax.set_title(f"{freq} hedging, listed 20-day call", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(loc="upper left", fontsize=11)
        axes[0].set_ylabel("density", fontsize=12)
        fig.tight_layout()
        save_figure(fig, figure_name)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily["rel_pnl"], lw=0.8, label="P&L / premium")
    ax.plot(daily.index, (daily["implied_vol"] - daily["realised_vol"]) / daily["implied_vol"], lw=0.8,
            label="(implied - realised) / implied sigma")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="entry date of the 20-day window", title="E7: hedging P&L and the implied-realised volatility gap")
    ax.legend()
    save_figure(fig, "e7_timeline")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, group in costs.groupby("k"):
        ax.plot(group["N"], group["rmse"], "o-", label=f"k = {k:g}")
    ax.set(xlabel="hedges in the 20-day life, n", ylabel="RMSE of net P&L / premium",
           title="E7: cost-risk trade-off on listed options")
    ax.legend()
    save_figure(fig, "e7_costs")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(proxy_same["daily"]["implied_vol"], daily.loc[common, "implied_vol"], s=10)
    limits = [proxy_same["daily"]["implied_vol"].min(), proxy_same["daily"]["implied_vol"].max()]
    ax.plot(limits, limits, "--", color="gray", label="implied = VIX")
    ax.set(xlabel="India VIX at entry", ylabel="implied sigma of the listed call", title="E7: listed implied sigma against VIX")
    ax.legend()
    save_figure(fig, "e7_implied_vs_vix")


if __name__ == "__main__":
    main()
