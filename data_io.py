"""Load prices from the Nasdaq-style CSV, a Yahoo-style CSV, or Yahoo Finance at a chosen timeframe."""
from __future__ import annotations

import io

import pandas as pd

# Timeframes offered for Yahoo Finance: label shown in the app -> Yahoo interval.
# Yahoo has no 4-hour bars, so those are built from 1-hour bars (see _merge_hours).
TIMEFRAMES = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m", "1 hour": "1h",
              "4 hours": "4h", "1 day": "1d", "1 week": "1wk", "1 month": "1mo"}

# Yahoo only serves recent intraday bars: 1m for the last 7 days (8 per request), 5m to 30m for 60 days,
# 1h for 730 days. Days to ask for, a little under each limit.
_INTRADAY_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729}

MIN_BARS = 120


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                         errors="coerce")


def read_csv(source) -> pd.DataFrame:
    """Accepts Nasdaq exports (Date, Close/Last, Volume, Open, High, Low with $ signs) and
    Yahoo/TradingView exports (Date, Open, High, Low, Close[, Adj Close, Volume])."""
    raw = pd.read_csv(source if not isinstance(source, bytes) else io.BytesIO(source))
    cols = {c.strip().lower(): c for c in raw.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return raw[cols[n]]
        raise ValueError(f"The file needs a column named one of {names}; it has {list(raw.columns)}.")
    df = pd.DataFrame({
        "Date": pd.to_datetime(pick("date", "time", "datetime"), errors="coerce"),
        "Open": _num(pick("open")), "High": _num(pick("high")), "Low": _num(pick("low")),
        "Close": _num(pick("close/last", "close", "last")),
    })
    df = df.dropna().drop_duplicates("Date").sort_values("Date").set_index("Date")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else df.index
    if len(df) < MIN_BARS:
        raise ValueError(f"Only {len(df)} usable rows; the indicators need at least {MIN_BARS} bars.")
    return df


def from_yahoo(ticker: str, timeframe: str = "1 day") -> pd.DataFrame:
    """All the bars Yahoo Finance serves for `timeframe` (a key of TIMEFRAMES). Intraday bars cover regular
    trading hours only, with times in the exchange's local time."""
    import yfinance as yf
    interval = TIMEFRAMES[timeframe]
    fetch = "1h" if interval == "4h" else interval
    if fetch in _INTRADAY_DAYS:
        start = pd.Timestamp.now().normalize() - pd.Timedelta(days=_INTRADAY_DAYS[fetch])
        attempts = [{"start": start.to_pydatetime()}, {"period": "max"}]
    else:
        attempts = [{"period": "max"}]
    df, error = None, None
    for kwargs in attempts:                        # the second form is a fallback for other yfinance versions
        try:
            df = yf.Ticker(ticker).history(interval=fetch, auto_adjust=False, actions=False, **kwargs)
        except Exception as e:  # noqa: BLE001
            df, error = None, e
        if df is not None and not df.empty:
            break
    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance returned no prices for {ticker} at the {timeframe} timeframe."
                         + (f" ({error})" if error else ""))
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df = df[["Open", "High", "Low", "Close"]].dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if interval == "4h":
        df = _merge_hours(df, 4)
    if len(df) < MIN_BARS:
        raise ValueError(f"Yahoo Finance has only {len(df)} bars for {ticker} at the {timeframe} timeframe; the "
                         f"indicators need at least {MIN_BARS}. Pick a shorter timeframe.")
    return df


def _merge_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Merge 1-hour bars into `hours`-hour bars counted from the usual first bar of the day: 09:30 for US
    stocks (bars at 09:30 and 13:30, like TradingView), 00:00 for markets that trade around the clock."""
    day = df.index.normalize()
    tod = df.index - day
    anchor = pd.Series(tod, index=df.index).groupby(day).min().mode().iloc[0]
    step = pd.Timedelta(hours=hours)
    start = day + anchor + ((tod - anchor) // step) * step
    g = df.groupby(start)
    return pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                         "Close": g["Close"].last()})


def is_intraday(index: pd.DatetimeIndex) -> bool:
    """True when there's usually more than one bar per day."""
    return len(index) > 1 and pd.Series(1, index=index.normalize()).groupby(level=0).size().median() > 1


def bar_size(index: pd.DatetimeIndex) -> str:
    """The bar size read from the timestamps: '5-minute', '1-hour', 'daily', 'weekly', 'monthly'..."""
    if len(index) < 2:
        return "daily"
    day = pd.Series(index.normalize())
    if is_intraday(index):
        step = pd.Series(index).diff()[day.duplicated()].median()          # spacing inside a day
        minutes = int(round(step.total_seconds() / 60))
        return f"{minutes // 60}-hour" if minutes % 60 == 0 else f"{minutes}-minute"
    gap = day.drop_duplicates().diff().dt.total_seconds().median() / 86_400
    if gap < 4:
        return "daily"
    if gap < 10:
        return "weekly"
    if gap < 45:
        return "monthly"
    return "quarterly" if gap < 120 else f"{gap:.0f}-day"
