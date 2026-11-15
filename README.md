# Delta Hedging of European Options under Model Misspecification

Individual project for MAC F244 (Stochastic Calculus and Application to Finance), BITS Pilani K K Birla Goa Campus, AY 2026-27.

Author: Mehta Prem (2024ADPS0851G). Instructor: Mayank Goel.

## What this is

Black-Scholes-Merton (BSM) says a European option can be replicated perfectly by continuously holding Delta shares. This project measures how large and how risky the hedging error becomes when each assumption is relaxed:

- discrete rebalancing (E1)
- wrong hedge volatility (E2)
- Heston stochastic volatility (E3) and Merton jumps (E4), hedged with BSM deltas
- proportional transaction costs, compared with Leland's scaling (E5)
- strike and maturity, compared with the gamma profile (E6)

The simulations are then compared with a rolling delta-hedge backtest on real market data (E7).

## Layout

| Folder | Contents |
| --- | --- |
| `src/` | BSM formulas, path simulators, hedging engine, costs, backtest, plotting helpers |
| `experiments/` | One thin script per experiment (`e1_...` to `e7_...`) |
| `tests/` | Sanity checks (BSM values, put-call parity, GBM moments) |
| `data/` | Downloaded market data (NIFTY and India VIX closes, NSE option prices) and `SOURCE.md` with sources and download dates |
| `results/` | Generated figures and tables |
| `notes/` | `theory_notes.md`, the theory behind each step |
| `notebooks/` | Final notebook, built last |
| `report/` | LaTeX report (`report.tex`) and its compiled PDF |

## How to run

```
pip install -r requirements.txt
python -m pytest tests
python experiments/e1_rebalancing.py
```

Each experiment script saves its figure and table under `results/`. Random seeds are fixed, so results are reproducible.

`notebooks/final_notebook.ipynb` re-runs every experiment from top to bottom (about 3 minutes) with the theory notes, figures, tables and interpretations inline. Open it in Jupyter or VS Code and run all cells.

## Status

Code, experiments E1 to E7, tests, theory notes, report (`report/report.tex`, compiled to `report/report.pdf`) and the final notebook are done. The market data source and download date are in `data/SOURCE.md`. The E7 backtest uses real listed NSE option prices on NIFTY 50 (2015 to 2026, 372 windows), with a baseline that prices the same options at India VIX.
