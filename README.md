# Delta Hedging of European Options under Model Misspecification

Individual project for MAC F244 (Stochastic Calculus and Application to Finance), BITS Pilani K K Birla Goa Campus, AY 2026-27.

Author: Mehta Prem (2024ADPS0851G). Instructor: Mayank Goel.

## What this is

Black-Scholes-Merton (BSM) says a European option can be replicated perfectly by continuously holding Delta shares. This project measures how large and how risky the hedging error becomes when each assumption is relaxed:

- discrete rebalancing
- wrong hedge volatility
- Heston stochastic volatility and Merton jumps, hedged with BSM deltas
- proportional transaction costs, compared with Leland's scaling
- strike and maturity, compared with the gamma profile

The simulations are then compared with a rolling delta-hedge backtest on real listed NIFTY 50 options (NSE data, 2015 to 2026).

## Layout

| Folder | Contents |
| --- | --- |
| `src/` | BSM formulas, path simulators, hedging engine, costs, backtest, plotting helpers |
| `experiments/` | One thin script per experiment (`e1_...` to `e7_...`) |
| `tests/` | Sanity checks (BSM values, put-call parity, GBM moments, hedging formulas) |
| `data/` | Downloaded market data (NIFTY and India VIX closes, NSE option prices) and `SOURCE.md` with sources and download dates |
| `results/` | Generated figures and tables |
| `notebooks/` | Final notebook that re-runs every experiment |
| `report/` | LaTeX report (`report.tex`) and its compiled PDF |

## How to run

```
pip install -r requirements.txt
python -m pytest tests
python experiments/e1_rebalancing.py
```

Each experiment script saves its figure and table under `results/`. Random seeds are fixed, so results are reproducible.

`notebooks/final_notebook.ipynb` re-runs every experiment from top to bottom (about 3 minutes) with the figures, tables and interpretations inline. Open it in Jupyter or VS Code and run all cells.

The report is `report/report.tex`; `pdflatex report.tex`, run twice from inside `report/`, rebuilds `report/report.pdf`.
