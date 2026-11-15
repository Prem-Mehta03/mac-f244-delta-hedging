"""E1: how the std of the hedging P&L shrinks with the number of rebalancing dates n.

Simulates exact GBM, hedges a short at-the-money call with BSM Deltas, and
fits log(std) = a + b log(N). Theory says b is close to -1/2. The std predicted
by the per-step Ito formula is shown alongside.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hedging import hedge_pnl, predicted_pnl_std
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
STEP_COUNTS = [4, 12, 52, 252, 1000]
N_PATHS = 20_000
SEED = 42


def main() -> None:
    """Run E1, print the table and slope, save the table and the log-log figure."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n_steps in STEP_COUNTS:
        paths = gbm_paths(S0, R, SIGMA, T, n_steps, N_PATHS, rng)
        pnl = hedge_pnl(paths, K, R, SIGMA, T)
        rows.append(
            {
                "N": n_steps,
                "mean_pnl": pnl.mean(),
                "std_pnl": pnl.std(),
                "predicted_std": predicted_pnl_std(paths, K, R, SIGMA, T),
            }
        )
    table = pd.DataFrame(rows)

    log_n = np.log(table["N"])
    slope, intercept = np.polyfit(log_n, np.log(table["std_pnl"]), 1)
    slope_tail = np.polyfit(log_n[2:], np.log(table["std_pnl"][2:]), 1)[0]
    pred_slope = np.polyfit(log_n, np.log(table["predicted_std"]), 1)[0]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e1_rebalancing.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"fitted slope (all N):         {slope:.3f}")
    print(f"fitted slope (N >= 52):       {slope_tail:.3f}")
    print(f"slope of predicted std:       {pred_slope:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(table["N"], table["std_pnl"], "o-", label=f"simulated (slope {slope:.2f})")
    ax.loglog(table["N"], table["predicted_std"], "s--", label="per-step Ito formula")
    ax.loglog(table["N"], np.exp(intercept) * table["N"] ** -0.5, ":", color="gray", label="slope -0.5")
    ax.set_xlabel("rebalancing dates n")
    ax.set_ylabel("std of hedging P&L")
    ax.set_title("E1: hedging error vs rebalancing frequency (GBM)")
    ax.legend()
    save_figure(fig, "e1_rebalancing")


if __name__ == "__main__":
    main()
