"""Real-data delta-hedging backtests on NIFTY 50.

Main backtest: each window sells a listed NIFTY call at its market closing
price and hedges it with Delta(t_i) = C_x(t_i, S(t_i)), at the call's own
implied volatility, along the real NIFTY 50 path until expiry. Option prices come from the NSE
F&O bhavcopy and the index path from Yahoo Finance.

Baseline: the same hedge on a synthetic at-the-money call priced with
Black-Scholes at India VIX, which needs no option data and so covers every
trading day. It is used to check the listed-option results.
"""

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import kurtosis, skew

from src import bs
from src.hedging import hedge_pnl, hedge_pnl_and_cost, pnl_summary

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "nifty.csv"
OPTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "nifty_calls_bhavcopy.csv"
BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
OLD_BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/fo{day}{month}{year}bhav.csv.zip"
UDIFF_START = pd.Timestamp("2024-07-08")
OPTION_COLUMNS = ["TradDt", "XpryDt", "StrkPric", "ClsPric", "SttlmPric", "UndrlygPric", "OpnIntrst", "TtlTradgVol"]
TRADING_DAYS = 252


def download_market_data(path: Path = DATA_PATH, start: str = "2015-01-01") -> pd.DataFrame:
    """Download daily NIFTY 50 (^NSEI) and India VIX (^INDIAVIX) closes from Yahoo Finance.

    Saved as CSV with columns `nifty` and `india_vix`, indexed by `date`. The
    trading calendar is that of NIFTY: days where only India VIX is missing are
    kept, with a NaN VIX.
    """
    import yfinance as yf

    closes = yf.download(["^NSEI", "^INDIAVIX"], start=start, auto_adjust=False, progress=False)["Close"]
    data = closes.rename(columns={"^NSEI": "nifty", "^INDIAVIX": "india_vix"})[["nifty", "india_vix"]]
    data = data[data["nifty"].notna()]
    data.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)
    return data


def load_market_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the saved NIFTY and India VIX closes, downloading them first if the file is missing."""
    if not path.exists():
        return download_market_data(path)
    return pd.read_csv(path, index_col="date", parse_dates=True)


def make_windows(prices: np.ndarray, n_days: int) -> np.ndarray:
    """All overlapping windows of `n_days + 1` consecutive prices, one per start day.

    Returns an array of shape (len(prices) - n_days, n_days + 1).
    """
    return np.lib.stride_tricks.sliding_window_view(prices, n_days + 1)


def fetch_nifty_calls(date: pd.Timestamp, session: requests.Session) -> pd.DataFrame | None:
    """Download one day of NSE F&O bhavcopy and keep the NIFTY index calls, in a common layout.

    From 8 July 2024 the UDiFF file is used and calls with strikes within 10%
    of the underlying are kept. Before that the older bhavcopy file is used and
    only calls that traded that day are kept (it has no underlying price, so
    `UndrlygPric` is NaN). Returns None if the file does not exist (holiday).
    """
    if date >= UDIFF_START:
        url = BHAVCOPY_URL.format(date=date.strftime("%Y%m%d"))
    else:
        url = OLD_BHAVCOPY_URL.format(year=date.year, month=date.strftime("%b").upper(), day=date.strftime("%d"))
    for _ in range(3):
        try:
            response = session.get(url, timeout=30)
            break
        except requests.RequestException:
            time.sleep(2)
    else:
        return None
    if response.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        table = pd.read_csv(archive.open(archive.namelist()[0]), index_col=False)
    if date >= UDIFF_START:
        calls = table[(table["TckrSymb"] == "NIFTY") & (table["FinInstrmTp"] == "IDO") & (table["OptnTp"] == "CE")]
        near = (calls["StrkPric"] / calls["UndrlygPric"] - 1).abs() < 0.10
        return calls.loc[near, OPTION_COLUMNS]
    calls = table[(table["SYMBOL"] == "NIFTY") & (table["INSTRUMENT"] == "OPTIDX") & (table["OPTION_TYP"] == "CE")]
    calls = calls[calls["CONTRACTS"] > 0]
    return pd.DataFrame(
        {
            "TradDt": pd.to_datetime(calls["TIMESTAMP"], format="%d-%b-%Y").dt.strftime("%Y-%m-%d"),
            "XpryDt": pd.to_datetime(calls["EXPIRY_DT"], format="%d-%b-%Y").dt.strftime("%Y-%m-%d"),
            "StrkPric": calls["STRIKE_PR"],
            "ClsPric": calls["CLOSE"],
            "SttlmPric": calls["SETTLE_PR"],
            "UndrlygPric": np.nan,
            "OpnIntrst": calls["OPEN_INT"],
            "TtlTradgVol": calls["CONTRACTS"],
        }
    )


def download_option_chain(dates, path: Path = OPTIONS_PATH, pause: float = 0.3) -> pd.DataFrame:
    """Download NIFTY call prices from the NSE archive for each date and cache them as CSV.

    Dates already present in the cache are skipped, so an interrupted download
    can be resumed by calling the function again. The cache is saved every 40 files.
    """
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=OPTION_COLUMNS)
    done = set(cached["TradDt"])
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"})
    frames, count = [cached], 0
    for date in dates:
        if date.strftime("%Y-%m-%d") in done:
            continue
        day = fetch_nifty_calls(date, session)
        if day is not None:
            frames.append(day)
        count += 1
        if count % 40 == 0:
            pd.concat(frames).to_csv(path, index=False)
        time.sleep(pause)
    result = pd.concat(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def implied_vol(price: float, x: float, K: float, r: float, tau: float, q: float = 0.0) -> float:
    """Implied volatility of a call: the sigma for which BSM(tau, x, K, r, sigma) equals the market price.

    Found by root finding on [0.01%, 500%]. `q` is an optional continuous
    dividend yield (default 0, as in the notes). Returns NaN if the price is
    below the no-arbitrage lower bound x exp(-q tau) - K exp(-r tau) or no
    volatility reproduces it.
    """
    if price <= max(x * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0):
        return float("nan")
    try:
        return brentq(lambda sigma: float(bs.call_price(x, K, r, sigma, tau, q)) - price, 1e-4, 5.0)
    except ValueError:
        return float("nan")


def realised_vol(prices: np.ndarray, years: float) -> np.ndarray:
    """Realised volatility of the notes, computed along the last axis of `prices`.

    sigma_hat = sqrt( (1 / (T_2 - T_1)) sum_i [ln S(t_{i+1}) / S(t_i)]^2 ),
    which estimates sigma because the sum of squared increments of ln S tends to
    sigma^2 (T_2 - T_1) (quadratic variation). `years` is T_2 - T_1 in years.
    Realised volatility is measured from the stock path; implied volatility (see
    `implied_vol`) is the one backed out of an option price.
    """
    log_returns = np.log(prices[..., 1:] / prices[..., :-1])
    return np.sqrt((log_returns**2).sum(axis=-1) / years)


def run_listed_backtest(
    data: pd.DataFrame,
    options: pd.DataFrame,
    n_days: int = 20,
    rebalance_every: int = 1,
    r: float = 0.06,
    max_strike_gap: float = 0.01,
    cost_rate: float = 0.0,
    q: float = 0.0,
) -> pd.DataFrame:
    """Hedge a short listed NIFTY call, sold at its market close, until its expiry.

    A window is used when a NIFTY expiry falls exactly `n_days` trading days
    after the entry date and a call near the money traded that day. The call
    with the strike closest to the NIFTY close is sold at its listed closing
    price `ClsPric`; if even that strike is more than `max_strike_gap` away from
    the NIFTY close (as a fraction of it), the window is skipped, since the
    option is then not at the money. Its own implied volatility (found with `implied_vol`) is
    used for the Deltas, so the premium in the hedging engine equals the market
    price. The call is settled at the NIFTY close on the expiry date.

    Args:
        data: DataFrame with column `nifty` (daily closes), indexed by date.
        options: cached bhavcopy calls (columns `OPTION_COLUMNS`).
        n_days: trading days from entry to expiry.
        rebalance_every: 1 for daily hedging, 5 for weekly (n_days must be a multiple).
        r: constant risk-free rate (assumed).
        max_strike_gap: largest allowed |K / S - 1| for the chosen call.
        cost_rate: round-trip proportional trading cost k on the index hedge (0 for none).
        q: continuous dividend yield of the index (0 means dividends are ignored, as in the notes).

    Returns:
        DataFrame indexed by entry date with the spot, strike, implied vol,
        realised vol over the window (the notes' formula, annualised), the
        listed premium, the P&L in index points (net of trading costs), the total
        trading cost in index points, and `rel_pnl` = P&L / premium.
    """
    calls = options.assign(TradDt=pd.to_datetime(options["TradDt"]), XpryDt=pd.to_datetime(options["XpryDt"]))
    calls = calls[calls["TtlTradgVol"] > 0]
    by_day = {key: group for key, group in calls.groupby(["TradDt", "XpryDt"])}
    prices = data["nifty"].to_numpy()
    T = n_days / TRADING_DAYS

    rows, paths = [], []
    for i in range(len(data) - n_days):
        chain = by_day.get((data.index[i], data.index[i + n_days]))
        if chain is None:
            continue
        S0 = prices[i]
        best = chain.iloc[(chain["StrkPric"] - S0).abs().argmin()]
        if abs(best["StrkPric"] / S0 - 1) > max_strike_gap:
            continue
        sigma_implied = implied_vol(best["ClsPric"], S0, best["StrkPric"], r, T, q)
        if np.isnan(sigma_implied):
            continue
        window = prices[i : i + n_days + 1]
        paths.append(window)
        rows.append(
            {
                "date": data.index[i],
                "spot": S0,
                "strike": best["StrkPric"],
                "implied_vol": sigma_implied,
                "realised_vol": float(realised_vol(window, T)),
                "premium": best["ClsPric"],
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    paths = np.array(paths)
    result["pnl"], result["cost"] = hedge_pnl_and_cost(
        paths[:, ::rebalance_every], result["strike"].to_numpy(), r, result["implied_vol"].to_numpy(),
        T, cost_rate=cost_rate, q=q,
    )
    result["rel_pnl"] = result["pnl"] / result["premium"]
    return result


def run_vix_proxy_backtest(
    data: pd.DataFrame, n_days: int = 20, rebalance_every: int = 1, r: float = 0.06
) -> pd.DataFrame:
    """Baseline: hedge a synthetic at-the-money call started on every trading day.

    For the window starting on day t: strike = NIFTY close on t, volatility =
    India VIX(t) / 100 for both the Black-Scholes premium C(0, S_0) and the Deltas,
    maturity T = n_days / 252 years. Windows whose entry VIX is missing are dropped.

    Args:
        data: DataFrame with columns `nifty` and `india_vix`, indexed by date.
        n_days: option life in trading days.
        rebalance_every: 1 for daily hedging, 5 for weekly (n_days must be a multiple).
        r: constant risk-free rate (assumed).

    Returns:
        DataFrame indexed by start date with the entry spot, implied vol (the
        VIX), realised vol over the window (the notes' formula), the model premium, the P&L in index
        points, and `rel_pnl` = P&L / premium.
    """
    windows = make_windows(data["nifty"].to_numpy(), n_days)
    start_spot = windows[:, 0]
    implied = data["india_vix"].to_numpy()[: len(windows)] / 100.0
    T = n_days / TRADING_DAYS

    pnl = hedge_pnl(windows[:, ::rebalance_every], start_spot, r, implied, T)
    premium = bs.call_price(start_spot, start_spot, r, implied, T)
    realised = realised_vol(windows, T)
    result = pd.DataFrame(
        {
            "spot": start_spot,
            "implied_vol": implied,
            "realised_vol": realised,
            "premium": premium,
            "pnl": pnl,
            "rel_pnl": pnl / premium,
        },
        index=data.index[: len(windows)],
    )
    return result.dropna(subset=["implied_vol"])


def block_bootstrap_summary(
    values: np.ndarray, block_length: int = 5, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """95% moving-block bootstrap intervals for the summary statistics of a time-ordered P&L sample.

    Windows that start on nearby dates overlap in time, so they are not
    independent. The bootstrap therefore resamples blocks of `block_length`
    consecutive windows (which keeps their dependence) and recomputes the mean,
    standard deviation, skewness, excess kurtosis and 5% expected shortfall on
    each resample.

    Args:
        values: P&L / premium per window, in time order.
        block_length: number of consecutive windows per block.
        n_boot: number of bootstrap resamples.
        seed: seed of the random generator.

    Returns:
        DataFrame with one row per statistic and columns `estimate`, `ci_low`, `ci_high`.
    """
    n = len(values)
    starts = np.random.default_rng(seed).integers(0, n - block_length + 1, size=(n_boot, -(-n // block_length)))
    index = (starts[:, :, None] + np.arange(block_length)).reshape(n_boot, -1)[:, :n]
    samples = values[index]
    cutoff = np.quantile(samples, 0.05, axis=1, keepdims=True)
    in_tail = samples <= cutoff
    replicates = {
        "mean": samples.mean(axis=1),
        "std": samples.std(axis=1),
        "skew": skew(samples, axis=1),
        "excess_kurtosis": kurtosis(samples, axis=1),
        "ES_5": -(np.where(in_tail, samples, 0.0).sum(axis=1) / in_tail.sum(axis=1)),
    }
    point = pnl_summary(values)
    rows = {
        name: {"estimate": point[name], "ci_low": np.percentile(rep, 2.5), "ci_high": np.percentile(rep, 97.5)}
        for name, rep in replicates.items()
    }
    return pd.DataFrame(rows).T
