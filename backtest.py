"""Long-only backtest: buy a bullish Normalized MACD turn near the lower ATR Trend Band, sell on NMACD +1
or when the ATR Trend Bands turn green.

Entry, checked at each bar's close (all on the same bar):
  1. ATR Trend Bands trend is down.
  2. The close is near the lower band: within `near_band` of the corridor, measured from the lower
     band (0 = on the lower band, 1 = on the upper band).
  3. The NMACD line crosses above its trigger line on this bar.
  4. The NMACD line came from near -1: its lowest value over the last `lookback` bars is at or
     below `near_low`.
Exit (`exit_rule`): "nmacd" = the NMACD line reaches `exit_level` (it can't print exactly +1; see nmacd.py),
"atr" = the ATR Trend Bands turn green (the trend flips from down to up), "either" = whichever comes first.
Plus an optional stop loss / time limit.

Repeat buy signals (`pyramid`): off = one position at a time with all the cash, and buy signals while
holding are ignored. On = every buy signal buys another lot worth `entry_size` of the account value (never
more than the cash left), up to `max_entries` lots, and a sell signal sells all the lots together.
Orders fill at the next bar's open (TradingView's default) or the signal bar's close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import atr_bands
import nmacd

EXIT_RULES = ("nmacd", "atr", "either")
EXIT_REASONS = {"nmacd": "NMACD hit +1", "atr": "ATR turned green"}


@dataclass(frozen=True)
class Settings:
    atr_length: int = 5
    atr_mult: float = 7.6
    atr_source: str = "close"
    fast: int = 13
    slow: int = 21
    trigger: int = 9
    normalize: int = 50
    ma_type: int = 1
    near_band: float = 0.25
    near_low: float = -0.8
    lookback: int = 5
    exit_rule: str = "nmacd"        # "nmacd" = NMACD hits +1, "atr" = ATR bands turn green, "either"
    exit_level: float = 0.99
    stop_loss: float = 0.0          # fraction below the (average) entry price, 0 = off
    max_bars: int = 0               # bars since the (first) buy, 0 = off
    pyramid: bool = False           # True = buy again on every signal while holding, sell everything together
    entry_size: float = 0.25        # with pyramid: share of the account value each buy uses
    max_entries: int = 4            # with pyramid: most buys held at once
    fill: str = "open"              # "open" = next bar's open, "close" = the signal bar's close
    cost_bps: float = 5.0           # per side
    capital: float = 100_000.0


def indicators(prices: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if s.exit_rule not in EXIT_RULES:
        raise ValueError(f"exit_rule must be one of {EXIT_RULES}, not {s.exit_rule!r}.")
    bands = atr_bands.compute(prices, s.atr_source, s.atr_length, s.atr_mult)
    nm = nmacd.compute(prices["Close"], s.fast, s.slow, s.trigger, s.normalize, s.ma_type)
    d = pd.concat([prices[["Open", "High", "Low", "Close"]],
                   bands[["Upper", "Lower", "Width", "Trend", "Signal", "ATR"]].rename(columns={"Signal": "ATR signal"}),
                   nm[["NMACD", "Trigger", "Hist"]]], axis=1)
    d["Band position"] = (d["Close"] - d["Lower"]) / d["Width"]
    prev_n, prev_t = d["NMACD"].shift(1), d["Trigger"].shift(1)
    d["Cross up"] = (d["NMACD"] > d["Trigger"]) & (prev_n <= prev_t)
    d["Recent NMACD low"] = d["NMACD"].rolling(s.lookback, min_periods=1).min()
    d["Downtrend"] = d["Trend"] == "Down"
    d["Near lower band"] = d["Band position"] <= s.near_band
    d["From near -1"] = d["Recent NMACD low"] <= s.near_low
    d["Entry signal"] = d["Downtrend"] & d["Near lower band"] & d["Cross up"] & d["From near -1"]
    d["NMACD exit"] = d["NMACD"] >= s.exit_level
    # Green bands = uptrend. Buys only happen in a downtrend, so the first green bar after a buy is the bar
    # the trend flips up (where the indicator prints its L).
    d["ATR exit"] = d["Trend"] == "Up"
    d["Exit signal"] = {"nmacd": d["NMACD exit"], "atr": d["ATR exit"],
                        "either": d["NMACD exit"] | d["ATR exit"]}[s.exit_rule]
    return d


def run(prices: pd.DataFrame, s: Settings):
    """Bar-by-bar simulation, long only, compounding.

    Without pyramiding: one position at a time with 100% of the cash; buy signals while holding are
    ignored. With pyramiding: each buy signal adds a lot (see the module notes) and every exit sells all
    lots at once; the stop loss is measured from the lots' average price and the time limit from the
    first buy. Each lot is one row of the trade log.
    """
    d = indicators(prices, s)
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    entry_sig = d["Entry signal"].to_numpy()
    nm_exit, atr_exit = d["NMACD exit"].to_numpy(), d["ATR exit"].to_numpy()
    use_nm, use_atr = s.exit_rule in ("nmacd", "either"), s.exit_rule in ("atr", "either")
    dates, n, cost = d.index, len(d), s.cost_bps / 1e4
    most = max(int(s.max_entries), 1) if s.pyramid else 1
    equity = np.empty(n)
    cash, lots, trades, positions = s.capital, [], [], 0
    pending_entry = pending_exit = None

    def held():
        return sum(t["shares"] for t in lots)

    def stop_price():
        if s.stop_loss <= 0 or not lots:
            return -np.inf
        avg = lots[0]["entry_px"] if len(lots) == 1 else sum(t["shares"] * t["entry_px"] for t in lots) / held()
        return avg * (1 - s.stop_loss)

    def exit_reason(i):
        why = [EXIT_REASONS["nmacd"]] if use_nm and nm_exit[i] else []
        why += [EXIT_REASONS["atr"]] if use_atr and atr_exit[i] else []
        if why:
            return " and ".join(why)
        return "Time limit" if s.max_bars > 0 and i - lots[0]["entry_i"] >= s.max_bars else None

    def buy(i_signal, i_fill, px):
        nonlocal cash, positions
        amount = min(cash, s.entry_size * (cash + held() * px)) if s.pyramid else cash
        if amount <= 1e-9 * s.capital:             # no cash left: the signal is skipped
            return
        if not lots:
            positions += 1
        lots.append({"position": positions, "buy_no": len(lots) + 1, "signal_i": i_signal, "entry_i": i_fill,
                     "entry_px": px, "invested": amount, "shares": amount * (1 - cost) / px, "low": px})
        cash -= amount

    def sell_all(i_fill, px, reason, status="Closed"):
        nonlocal cash
        for t in lots:
            value = t["shares"] * px * (1 - cost)
            sig = t["signal_i"]
            row = {
                "Signal date": dates[sig], "Entry date": dates[t["entry_i"]], "Entry price": t["entry_px"],
                "Invested ($)": t["invested"], "Exit date": dates[i_fill], "Exit price": px, "Exit reason": reason,
                "Bars held": i_fill - t["entry_i"], "Return": value / t["invested"] - 1,
                "P&L ($)": value - t["invested"], "Worst drawdown in trade": t["low"] / t["entry_px"] - 1,
                "Band position at signal": d["Band position"].iloc[sig], "NMACD at signal": d["NMACD"].iloc[sig],
                "Recent NMACD low": d["Recent NMACD low"].iloc[sig], "Status": status}
            trades.append({"Position": t["position"], "Buy #": t["buy_no"], **row} if s.pyramid else row)
            if status == "Closed":
                cash += value
        if status == "Closed":
            lots.clear()

    def track_low(i):
        for t in lots:
            t["low"] = min(t["low"], l[i])

    for i in range(n):
        # Orders decided at the previous close fill at this bar's open.
        if pending_exit and lots:
            sell_all(i, o[i], pending_exit)
            pending_exit = None
        if lots and i > lots[0]["signal_i"] and o[i] <= stop_price():   # gapped through the stop overnight
            track_low(i)
            sell_all(i, o[i], "Stop loss")
        if pending_entry is not None:
            if len(lots) < most:
                buy(pending_entry, i, o[i])
            pending_entry = None
        if lots:
            stop_px = stop_price()
            track_low(i)
            if i > lots[0]["signal_i"] and l[i] <= stop_px:
                sell_all(i, min(o[i], stop_px), "Stop loss")
        # Decisions at this bar's close: sell everything...
        if lots and i > lots[0]["signal_i"]:
            reason = exit_reason(i)
            if reason:
                if s.fill == "close":
                    sell_all(i, c[i], reason)
                elif i + 1 < n:
                    pending_exit = reason
        # ... and buy (again).
        if entry_sig[i] and pending_entry is None and not pending_exit and len(lots) < most:
            if s.fill == "close":
                buy(i, i, c[i])
            elif i + 1 < n:
                pending_entry = i
        equity[i] = cash + held() * c[i]
    if lots:                                       # still open: mark at the last close
        sell_all(n - 1, c[-1], "Still open", status="Open")
    return d, pd.Series(equity, index=dates), pd.DataFrame(trades)


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Bars in a year, to annualise the Sharpe ratio: 252 for daily stock prices, about 52 weekly and 12
    monthly, and bars per session x 252 intraday (365 days for markets that also trade at weekends)."""
    if len(index) < 3:
        return 252.0
    day = index.normalize()
    days = day.unique()
    per_year = 365.0 if (days.dayofweek >= 5).mean() > 0.1 else 252.0
    per_day = pd.Series(1, index=day).groupby(level=0).size().median()
    if per_day > 1:                                # intraday
        return float(per_day * per_year)
    gap = pd.Series(days).diff().dt.total_seconds().median() / 86_400
    years = (days[-1] - days[0]).days / 365.25
    return per_year if gap < 4 or years <= 0 else (len(days) - 1) / years


def stats(eq: pd.Series, trades: pd.DataFrame, prices: pd.DataFrame, capital: float) -> dict:
    r = eq.pct_change().fillna(0.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    bh = prices["Close"] / prices["Close"].iloc[0] * capital
    closed = trades[trades["Status"] == "Closed"] if len(trades) else trades
    wins = closed[closed["Return"] > 0] if len(closed) else closed
    losses = closed[closed["Return"] <= 0] if len(closed) else closed
    if len(trades):
        # Buys made while already holding belong to the same position and leave with it.
        pos_id = trades["Position"] if "Position" in trades else pd.Series(np.arange(len(trades)), index=trades.index)
        in_mkt = trades.groupby(pos_id)["Bars held"].max().sum()
        n_pos, n_closed_pos = pos_id.nunique(), pos_id[trades["Status"] == "Closed"].nunique()
    else:
        in_mkt = n_pos = n_closed_pos = 0
    out = {
        "Trades": len(trades), "Closed trades": len(closed), "Positions": n_pos, "Closed positions": n_closed_pos,
        "Win rate": len(wins) / len(closed) if len(closed) else np.nan,
        "Average win": wins["Return"].mean() if len(wins) else np.nan,
        "Average loss": losses["Return"].mean() if len(losses) else np.nan,
        "Average trade": closed["Return"].mean() if len(closed) else np.nan,
        "Best trade": closed["Return"].max() if len(closed) else np.nan,
        "Worst trade": closed["Return"].min() if len(closed) else np.nan,
        "Profit factor": (wins["P&L ($)"].sum() / -losses["P&L ($)"].sum()) if len(losses) and losses["P&L ($)"].sum() < 0 else np.nan,
        "Total P&L ($)": eq.iloc[-1] - capital,
        "Total return": eq.iloc[-1] / capital - 1,
        "CAGR": (eq.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Sharpe": r.mean() / r.std() * np.sqrt(bars_per_year(eq.index)) if r.std() > 0 else np.nan,
        "Average bars held": trades["Bars held"].mean() if len(trades) else np.nan,
        "Time in market": in_mkt / max(len(eq) - 1, 1),
        "Worst drawdown inside a trade": trades["Worst drawdown in trade"].min() if len(trades) else np.nan,
        "Buy & hold return": bh.iloc[-1] / capital - 1,
        "Buy & hold max drawdown": (bh / bh.cummax() - 1).min(),
        "Years": years,
    }
    return out


def funnel(d: pd.DataFrame, unit: str = "bars") -> pd.DataFrame:
    """How many bars pass each entry condition, cumulatively."""
    steps = [(f"All {unit} with both indicators", d["NMACD"].notna() & d["Trend"].ne("")),
             ("ATR trend is down", d["Downtrend"]),
             ("... and close near the lower band", d["Downtrend"] & d["Near lower band"]),
             ("... and NMACD crosses above trigger", d["Downtrend"] & d["Near lower band"] & d["Cross up"]),
             ("... and NMACD came from near -1", d["Entry signal"])]
    return pd.DataFrame({"Condition": [s[0] for s in steps], unit.capitalize(): [int(s[1].sum()) for s in steps]})


def sensitivity(prices: pd.DataFrame, s: Settings, bands=(0.10, 0.25, 0.50, 1.00), lows=(-0.9, -0.8, -0.6, -0.4)):
    from dataclasses import replace
    rows = []
    for b in bands:
        for lo in lows:
            q = replace(s, near_band=b, near_low=lo)
            _, eq, tr = run(prices, q)
            st = stats(eq, tr, prices, s.capital)
            rows.append({"Near lower band (within)": b, "Near -1 (at or below)": lo, "Trades": st["Trades"],
                         "Win rate": st["Win rate"], "Average trade": st["Average trade"],
                         "Total return": st["Total return"], "Max drawdown": st["Max drawdown"]})
    return pd.DataFrame(rows)
