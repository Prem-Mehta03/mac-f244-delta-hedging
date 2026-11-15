"""E6: hedging error against K and T, and its link to the gamma profile.

GBM paths, short call hedged weekly (dt = 1/52, so N = 52 T) with BSM Deltas at
the true volatility. The simulated std of the P&L is compared with the std from
the per-step Ito formula, which is driven by the dollar gamma S^2 Gamma. A third
panel shows when in the option's life the error variance is created.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bs
from src.hedging import error_variance_profile, hedge_pnl, predicted_pnl_std
from src.plots import TABLES_DIR, save_figure
from src.simulators import gbm_paths

S0, R, SIGMA = 100.0, 0.05, 0.2
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
MATURITIES = [0.1, 0.25, 0.5, 1.0, 2.0]
BASE_STRIKE, BASE_MATURITY = 100.0, 1.0
STEPS_PER_YEAR = 52
N_PATHS = 20_000
SEED = 42


def run_case(K: float, T: float) -> tuple[dict, np.ndarray]:
    """Simulate one (K, T) case; return its table row and variance profile."""
    n_steps = round(STEPS_PER_YEAR * T)
    paths = gbm_paths(S0, R, SIGMA, T, n_steps, N_PATHS, np.random.default_rng(SEED))
    pnl = hedge_pnl(paths, K, R, SIGMA, T)
    premium = float(bs.call_price(S0, K, R, SIGMA, T))
    row = {
        "K": K,
        "T": T,
        "N": n_steps,
        "premium": premium,
        "dollar_gamma_0": S0**2 * float(bs.gamma(S0, K, R, SIGMA, T)),
        "std_sim": pnl.std(),
        "std_pred": predicted_pnl_std(paths, K, R, SIGMA, T),
        "std_over_premium": pnl.std() / premium,
    }
    return row, error_variance_profile(paths, K, R, SIGMA, T)


def main() -> None:
    """Run E6, print and save the tables, save the three-panel figure."""
    strike_rows = [run_case(k, BASE_MATURITY)[0] for k in STRIKES]
    maturity_rows = [run_case(BASE_STRIKE, t)[0] for t in MATURITIES]
    strike_table, maturity_table = pd.DataFrame(strike_rows), pd.DataFrame(maturity_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    strike_table.to_csv(TABLES_DIR / "e6_strike.csv", index=False)
    maturity_table.to_csv(TABLES_DIR / "e6_maturity.csv", index=False)
    fmt = lambda x: f"{x:.4f}"
    print("Strike sweep, T = 1, weekly hedging")
    print(strike_table.to_string(index=False, float_format=fmt))
    print("Maturity sweep, K = 100, weekly hedging")
    print(maturity_table.to_string(index=False, float_format=fmt))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.plot(strike_table["K"], strike_table["std_sim"], "o-", label="simulated std")
    ax.plot(strike_table["K"], strike_table["std_pred"], "s--", label="Ito formula")
    scale = strike_table["std_sim"].max() / strike_table["dollar_gamma_0"].max()
    ax.plot(strike_table["K"], scale * strike_table["dollar_gamma_0"], "^:", color="gray",
            label="dollar gamma at t = 0 (rescaled)")
    ax.set(xlabel="K K (S0 = 100, T = 1)", ylabel="std of hedging P&L", title="Error against K")
    ax.legend()

    ax = axes[1]
    ax.plot(maturity_table["T"], maturity_table["std_sim"], "o-", label="simulated std")
    ax.plot(maturity_table["T"], maturity_table["std_pred"], "s--", label="Ito formula")
    ax.set(xlabel="T T (K = 100, weekly hedging)", ylabel="std of hedging P&L", title="Error against T")
    ax.set_ylim(0, 1.2)
    ax.legend()

    ax = axes[2]
    for K in (80.0, 100.0, 120.0):
        profile = run_case(K, BASE_MATURITY)[1]
        ax.plot(np.arange(len(profile)) / STEPS_PER_YEAR, profile / profile.sum(), label=f"K = {K:g}")
    ax.set(xlabel="time t (T = 1)", ylabel="share of total error variance per step",
           title="When the error is created")
    ax.legend()
    save_figure(fig, "e6_moneyness_maturity")


if __name__ == "__main__":
    main()
