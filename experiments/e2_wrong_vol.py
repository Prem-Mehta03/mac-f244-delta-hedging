"""E2: hedging with the wrong volatility.

The stock follows GBM with true volatility sigma. The seller sells the call
and computes Deltas with sigma_h = ratio * sigma, with ratio from 0.5 to 1.5.
The simulated P&L is compared path by path with the El Karoui et al. formula.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hedging import hedge_pnl, wrong_vol_pnl_formula
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
RATIOS = [0.5, 0.75, 1.0, 1.25, 1.5]
N_STEPS = 1000
N_PATHS = 20_000
SEED = 42


def main() -> None:
    """Run E2, print the comparison table, save the table and two figures."""
    rng = np.random.default_rng(SEED)
    paths = gbm_paths(S0, R, SIGMA, T, N_STEPS, N_PATHS, rng)

    rows, pnls = [], {}
    for ratio in RATIOS:
        hedge_vol = ratio * SIGMA
        pnl = hedge_pnl(paths, K, R, hedge_vol, T)
        formula = wrong_vol_pnl_formula(paths, K, R, SIGMA, hedge_vol, T)
        pnls[ratio] = pnl
        rows.append(
            {
                "sigma_h/sigma": ratio,
                "mean_sim": pnl.mean(),
                "mean_formula": formula.mean(),
                "std_error": pnl.std() / np.sqrt(N_PATHS),
                "std_sim": pnl.std(),
                "std_formula": formula.std(),
                "pathwise_rmse": np.sqrt(np.mean((pnl - formula) ** 2)),
            }
        )
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e2_wrong_vol.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(table["sigma_h/sigma"], table["mean_sim"], yerr=1.96 * table["std_error"],
                fmt="o", capsize=3, label="simulated mean (95% CI)")
    ax.plot(table["sigma_h/sigma"], table["mean_formula"], "s--", label="formula")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("hedge volatility / true volatility")
    ax.set_ylabel("mean P&L of the seller")
    ax.set_title("E2: mean hedging P&L vs hedge volatility")
    ax.legend()
    save_figure(fig, "e2_mean_pnl")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ratio in RATIOS:
        ax.hist(pnls[ratio], bins=80, histtype="step", density=True, label=f"{ratio:g} sigma")
    ax.set_xlabel("P&L of the seller")
    ax.set_ylabel("density")
    ax.set_title("E2: P&L distribution by hedge volatility")
    ax.legend(title="sigma_h")
    save_figure(fig, "e2_distribution")


if __name__ == "__main__":
    main()
