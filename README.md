# ATR Trend Bands + Normalized MACD backtest

A Streamlit website that backtests a long-only rule on Netflix (or any other stock):

- **Buy** when the ATR Trend Bands trend is down, the close is near the lower band, and the
  Normalized MACD line crosses above its trigger line after being near -1.
- **Sell** when the Normalized MACD line hits +1, when the ATR Trend Bands turn green (the trend flips
  up), or whichever of the two comes first. A stop loss and a time limit are optional.
- **Buy signals while already holding** are either ignored (one position at a time, all the capital in
  each trade) or each one buys more (a chosen % of the account, up to a chosen number of buys). Any sell
  then sells all the buys at once.

It shows P&L, win rate, profit factor, drawdowns, an equity curve against buy & hold, the price
chart with the bands and every buy and sell, the NMACD panel, a downloadable trade log, a count of
how many bars pass each condition, and how the result changes with the vague thresholds.

Prices: the Netflix file in `data/` (Nasdaq export, Oct 2016 to Oct 2026), any uploaded CSV of daily
prices (Nasdaq, Yahoo or TradingView export), or any Yahoo Finance ticker on a timeframe from 1 minute
to 1 month. Yahoo keeps 1-minute bars for the last 7 days, 5 to 30-minute bars for 60 days and 1-hour
bars for 2 years; daily, weekly and monthly bars go back to the listing. Yahoo has no 4-hour bars, so
those are built from 1-hour bars.

## Put it online

Upload everything in this folder, including `data/` and `.streamlit/`, to a GitHub repository, then
create an app on https://share.streamlit.io with the main file `app.py`.

## Run it on a computer with Python

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `backtest.py`: the rules, the trade simulation and the statistics
- `nmacd.py`: the Normalized MACD, line by line from the TradingView script
- `atr_bands.py`: the ATR Trend Bands replica (same as the ATR Trend Bands app)
- `data_io.py`: reads the CSV formats and Yahoo Finance at the chosen timeframe
- `app.py`: the web page
