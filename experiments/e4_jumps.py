"""Delta hedging with BSM when the stock follows a Merton jump-diffusion.

The diffusion volatility is lowered as jumps are added so that the total
variance sigma_d^2 + lam (m^2 + delta^2) always equals 0.04, the BSM variance
used for the premium and the Deltas (20% volatility). The comparison with GBM
then isolates the effect of jumps on the shape of the P&L distribution, not on
its variance level. Two sweeps: jump intensity (size fixed) and jump size
(intensity fixed). A final sweep over N shows the error does not vanish.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hedging import bootstrap_standard_errors, hedge_pnl, pnl_summary
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths, merton_paths

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_STEPS = 252
N_PATHS = 20_000
SEED = 42
JUMP_STD = 0.1
RATES = [0.25, 0.5, 1.0]
BASE_RATE, SIZES = 0.5, [-0.05, -0.1, -0.2]
BASE_SIZE = -0.1
STEP_COUNTS = [12, 52, 252, 1000]


def merton_pnl(jump_rate: float, jump_mean: float, n_steps: int) -> np.ndarray:
    """Hedging P&L of the short call on variance-matched Merton paths."""
    diffusion_var = SIGMA**2 - jump_rate * (jump_mean**2 + JUMP_STD**2)
    rng = np.random.default_rng(SEED)
    paths = merton_paths(S0, R, np.sqrt(diffusion_var), jump_rate, jump_mean, JUMP_STD,
                         T, n_steps, N_PATHS, rng)
    return hedge_pnl(paths, K, R, SIGMA, T)


def gbm_pnl(n_steps: int) -> np.ndarray:
    """Hedging P&L of the short call under GBM (baseline)."""
    paths = gbm_paths(S0, R, SIGMA, T, n_steps, N_PATHS, np.random.default_rng(SEED))
    return hedge_pnl(paths, K, R, SIGMA, T)


def main() -> None:
    """Run the experiment: print and save the summary table, save the figures."""
    scenarios = {"GBM": gbm_pnl(N_STEPS)}
    for rate in RATES:
        scenarios[f"lam={rate}, m={BASE_SIZE}"] = merton_pnl(rate, BASE_SIZE, N_STEPS)
    for size in SIZES:
        if size != BASE_SIZE:
            scenarios[f"lam={BASE_RATE}, m={size}"] = merton_pnl(BASE_RATE, size, N_STEPS)

    table = pd.DataFrame({name: pnl_summary(pnl) for name, pnl in scenarios.items()}).T
    table.index.name = "scenario"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e4_jumps.csv")
    print(f"n = {N_STEPS}, {N_PATHS} paths, jump std = {JUMP_STD}")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    errors = pd.DataFrame({name: bootstrap_standard_errors(scenarios[name])
                           for name in ("GBM", f"lam={BASE_RATE}, m={BASE_SIZE}")}).T
    errors.to_csv(TABLES_DIR / "e4_mc_error.csv")
    print(f"Monte Carlo standard errors with {N_PATHS} paths (bootstrap over the paths)")
    print(errors.to_string(float_format=lambda x: f"{x:.4f}"))

    n_table = pd.DataFrame(
        [{"N": n, "gbm_std": gbm_pnl(n).std(), "merton_std": merton_pnl(BASE_RATE, BASE_SIZE, n).std()}
         for n in STEP_COUNTS]
    )
    n_table.to_csv(TABLES_DIR / "e4_jumps_vs_n.csv", index=False)
    print(n_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ["GBM"] + [f"lam={rate}, m={BASE_SIZE}" for rate in RATES]:
        ax.hist(scenarios[name], bins=120, histtype="step", density=True, label=name)
    ax.set_xlim(-8, 6)
    ax.set_xlabel("P&L of the seller")
    ax.set_ylabel("density")
    ax.set_title(f"Hedging P&L under Merton jumps, n = {N_STEPS}")
    ax.legend()
    save_figure(fig, "e4_distribution")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(n_table["N"], n_table["gbm_std"], "o-", label="GBM")
    ax.loglog(n_table["N"], n_table["merton_std"], "s-", label=f"Merton (lam={BASE_RATE}, m={BASE_SIZE})")
    ax.set_xlabel("rebalancing dates n")
    ax.set_ylabel("std of hedging P&L")
    ax.set_title("Residual error does not vanish with jumps")
    ax.legend()
    save_figure(fig, "e4_std_vs_n")


if __name__ == "__main__":
    main()
