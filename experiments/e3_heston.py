"""Delta hedging with BSM when the stock follows Heston stochastic volatility.

The Heston variance V(t) starts at V(0) = a_V / beta = sigma^2 = 0.04, so the average
variance matches the BSM volatility used for the premium and the Deltas (20%).
Two sweeps: vol-of-vol xi (rho fixed) and correlation rho (xi fixed). A GBM
run is the baseline, and a final sweep over N shows the error does not vanish
as rebalancing becomes more frequent. Heston paths are simulated on a fine grid
of 1008 Euler steps and the hedge uses every (1008 / N)-th date, so the
discretisation error of the simulation does not change with the hedging frequency.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hedging import bootstrap_standard_errors, hedge_pnl, pnl_summary
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths, heston_paths

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
BETA, V_BAR = 2.0, SIGMA**2
A_V = BETA * V_BAR
N_STEPS = 252
FINE_STEPS = 1008
N_PATHS = 20_000
SEED = 42
XI_VALUES = [0.2, 0.4, 0.6]
RHO_VALUES = [-0.9, -0.5, 0.0, 0.5]
BASE_XI, BASE_RHO = 0.4, -0.7
STEP_COUNTS = [12, 48, 252, 1008]


def heston_pnl(xi: float, rho: float, n_steps: int) -> np.ndarray:
    """Hedging P&L of the short call on Heston paths, hedged with BSM at 20% on n_steps dates.

    The paths are simulated with FINE_STEPS Euler steps and subsampled to the
    n_steps hedging dates (n_steps must divide FINE_STEPS).
    """
    rng = np.random.default_rng(SEED)
    paths = heston_paths(S0, R, V_BAR, A_V, BETA, xi, rho, T, FINE_STEPS, N_PATHS, rng)
    return hedge_pnl(paths[:, :: FINE_STEPS // n_steps], K, R, SIGMA, T)


def main() -> None:
    """Run the experiment: print and save the summary table, save the figures."""
    rng = np.random.default_rng(SEED)
    gbm = hedge_pnl(gbm_paths(S0, R, SIGMA, T, N_STEPS, N_PATHS, rng), K, R, SIGMA, T)

    scenarios = {"GBM": gbm}
    for xi in XI_VALUES:
        scenarios[f"xi={xi}, rho={BASE_RHO}"] = heston_pnl(xi, BASE_RHO, N_STEPS)
    for rho in RHO_VALUES:
        scenarios[f"xi={BASE_XI}, rho={rho}"] = heston_pnl(BASE_XI, rho, N_STEPS)

    table = pd.DataFrame({name: pnl_summary(pnl) for name, pnl in scenarios.items()}).T
    table.index.name = "scenario"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "e3_heston.csv")
    print(f"n = {N_STEPS}, {N_PATHS} paths")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    errors = pd.DataFrame({name: bootstrap_standard_errors(scenarios[name])
                           for name in ("GBM", f"xi={BASE_XI}, rho={BASE_RHO}")}).T
    errors.to_csv(TABLES_DIR / "e3_mc_error.csv")
    print(f"Monte Carlo standard errors with {N_PATHS} paths (bootstrap over the paths)")
    print(errors.to_string(float_format=lambda x: f"{x:.4f}"))

    n_rows = [{"N": n, "gbm_std": hedge_pnl(gbm_paths(S0, R, SIGMA, T, n, N_PATHS, np.random.default_rng(SEED)),
                                            K, R, SIGMA, T).std(),
               "heston_std": heston_pnl(BASE_XI, BASE_RHO, n).std()} for n in STEP_COUNTS]
    n_table = pd.DataFrame(n_rows)
    n_table.to_csv(TABLES_DIR / "e3_heston_vs_n.csv", index=False)
    print(n_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ["GBM"] + [f"xi={xi}, rho={BASE_RHO}" for xi in XI_VALUES]:
        ax.hist(scenarios[name], bins=100, histtype="step", density=True, label=name)
    ax.set_xlim(-8, 6)
    ax.set_xlabel("P&L of the seller")
    ax.set_ylabel("density")
    ax.set_title(f"Hedging P&L under Heston, n = {N_STEPS}")
    ax.legend()
    save_figure(fig, "e3_distribution")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(n_table["N"], n_table["gbm_std"], "o-", label="GBM")
    ax.loglog(n_table["N"], n_table["heston_std"], "s-", label=f"Heston (xi={BASE_XI}, rho={BASE_RHO})")
    ax.set_xlabel("rebalancing dates n")
    ax.set_ylabel("std of hedging P&L")
    ax.set_title("Residual error does not vanish under Heston")
    ax.legend()
    save_figure(fig, "e3_std_vs_n")


if __name__ == "__main__":
    main()
