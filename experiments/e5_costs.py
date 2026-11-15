"""E5: proportional transaction costs, the optimal rebalancing frequency and Leland's scaling.

GBM paths, short at-the-money call hedged with BSM Deltas (20%) on N dates, for
several round-trip cost rates k. Three things are measured:
  1. total trading cost vs N, against the Leland expected-cost formula;
  2. the std of the net P&L vs N;
  3. the RMSE of the net P&L (bias^2 + variance) vs N. Its minimum over N is the
     optimal rebalancing frequency, compared with the closed-form N* = s / a that
     follows from std ~ s / sqrt(n) and cost ~ a sqrt(n).
A last table tests Leland's claim that pricing and hedging at sigma_hat removes
the cost bias.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.costs import leland_expected_cost, leland_vol
from src.hedging import hedge_pnl_and_cost, predicted_pnl_std
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
COST_RATES = [0.001, 0.0025, 0.005, 0.01]
STEP_COUNTS = [12, 26, 52, 78, 104, 156, 208, 252, 378, 504, 756, 1000]
LELAND_COST_RATE = 0.005
LELAND_STEP_COUNTS = [26, 52, 252]
N_PATHS = 20_000
SEED = 42


def main() -> None:
    """Run E5, print and save the tables, save the figures."""
    rows = []
    closed_form = {}
    for n_steps in STEP_COUNTS:
        paths = gbm_paths(S0, R, SIGMA, T, n_steps, N_PATHS, np.random.default_rng(SEED))
        pred_std = predicted_pnl_std(paths, K, R, SIGMA, T)
        for k in COST_RATES:
            pnl, cost = hedge_pnl_and_cost(paths, K, R, SIGMA, T, cost_rate=k)
            pred_cost = leland_expected_cost(paths, K, R, SIGMA, k, T)
            rows.append(
                {
                    "k": k,
                    "N": n_steps,
                    "mean_cost": cost.mean(),
                    "leland_cost": pred_cost,
                    "mean_pnl": pnl.mean(),
                    "std_pnl": pnl.std(),
                    "rmse": np.sqrt(np.mean(pnl**2)),
                    "pred_rmse": np.hypot(pred_cost, pred_std),
                }
            )
            if n_steps == 252:
                closed_form[k] = (pred_std * np.sqrt(n_steps)) / (pred_cost / np.sqrt(n_steps))
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e5_costs.csv", index=False)
    print(table[table["N"].isin([12, 52, 252, 1000])].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    optimum = pd.DataFrame(
        [
            {
                "k": k,
                "N_star_sim": int(g.loc[g["rmse"].idxmin(), "N"]),
                "N_star_leland_model": int(g.loc[g["pred_rmse"].idxmin(), "N"]),
                "N_star_closed_form": closed_form[k],
                "min_rmse": g["rmse"].min(),
                "leland_Le_at_N_star": np.sqrt(2 / np.pi) * k / (SIGMA * np.sqrt(T / g.loc[g["rmse"].idxmin(), "N"])),
            }
            for k, g in table.groupby("k")
        ]
    )
    optimum.to_csv(TABLES_DIR / "e5_optimal_n.csv", index=False)
    slope = np.polyfit(np.log(optimum["k"]), np.log(optimum["N_star_closed_form"]), 1)[0]
    print(optimum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"log-log slope of closed-form N* against k: {slope:.3f}")

    leland_rows = []
    for n_steps in LELAND_STEP_COUNTS:
        paths = gbm_paths(S0, R, SIGMA, T, n_steps, N_PATHS, np.random.default_rng(SEED))
        hat = leland_vol(SIGMA, LELAND_COST_RATE, T / n_steps)
        for label, sigma_used in [("sigma", SIGMA), ("sigma_hat", hat)]:
            pnl, _ = hedge_pnl_and_cost(paths, K, R, sigma_used, T, cost_rate=LELAND_COST_RATE)
            leland_rows.append({"N": n_steps, "premium_and_hedge_vol": label, "vol": sigma_used,
                                "mean_pnl": pnl.mean(), "std_pnl": pnl.std()})
    leland_table = pd.DataFrame(leland_rows)
    leland_table.to_csv(TABLES_DIR / "e5_leland.csv", index=False)
    print(f"Leland check, k = {LELAND_COST_RATE}")
    print(leland_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, g in table.groupby("k"):
        line, = axes[0].loglog(g["N"], g["mean_cost"], "o-", label=f"k = {k}")
        axes[0].loglog(g["N"], g["leland_cost"], "--", color=line.get_color())
        axes[1].loglog(g["N"], g["std_pnl"], "o-", label=f"k = {k}")
    axes[0].set(xlabel="rebalancing dates n", ylabel="mean total trading cost",
                title="Cost grows like sqrt(n) (dashed: Leland formula)")
    axes[1].set(xlabel="rebalancing dates n", ylabel="std of net P&L", title="Risk falls like 1/sqrt(n)")
    axes[0].legend()
    save_figure(fig, "e5_cost_and_risk")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, g in table.groupby("k"):
        line, = ax.semilogx(g["N"], g["rmse"], "o-", label=f"k = {k}")
        best = g.loc[g["rmse"].idxmin()]
        ax.plot(best["N"], best["rmse"], "*", ms=14, color=line.get_color())
    ax.set(xlabel="rebalancing dates n", ylabel="RMSE of net P&L", title="E5: optimal rebalancing frequency (stars)")
    ax.legend()
    save_figure(fig, "e5_optimal_n")


if __name__ == "__main__":
    main()
