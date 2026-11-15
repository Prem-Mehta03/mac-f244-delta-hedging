# Market data sources

## NIFTY 50 index and India VIX

- File: `nifty.csv` (columns `date`, `nifty`, `india_vix`)
- Series: NIFTY 50 index daily close (Yahoo Finance ticker `^NSEI`) and India VIX daily close (`^INDIAVIX`)
- Provider: Yahoo Finance, downloaded with the `yfinance` Python library (version 1.8.0)
- Downloaded: 15 November 2026, 2 January 2015 to 13 November 2026 (2,921 NIFTY trading days; India VIX is missing on 17 of them)
- India VIX is used only for the baseline comparison in E7, not for the main (listed-option) results.

## NIFTY index call option prices

- File: `nifty_calls_bhavcopy.csv` (columns as in the NSE UDiFF F&O bhavcopy: `TradDt`, `XpryDt`, `StrkPric`, `ClsPric`, `SttlmPric`, `UndrlygPric`, `OpnIntrst`, `TtlTradgVol`)
- Source: NSE India, daily F&O bhavcopy files, one per trading day, downloaded on 15 November 2026 for 2 January 2015 to 11 November 2026 (2,920 days, 910,361 call rows):
  - 8 July 2024 onwards: UDiFF format, `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip`. NIFTY index calls with strikes within 10% of the underlying are kept.
  - Before 8 July 2024: older format, `https://nsearchives.nseindia.com/content/historical/DERIVATIVES/<YYYY>/<MON>/fo<DD><MON><YYYY>bhav.csv.zip`. This format has no underlying price (`UndrlygPric` is empty), and only NIFTY index calls that traded that day are kept.
- Coverage notes: NIFTY had monthly expiries only until about 2019 and weekly expiries since. No file was available for 30 March 2021 (a Yahoo trading day).
- Prices are end-of-day closes, not intraday quotes. Contracts with zero traded volume are ignored in the backtest.
