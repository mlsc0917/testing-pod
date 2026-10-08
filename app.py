"""
ATR Trend Bands + Normalized MACD: long-only backtest with a full trade log.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest as B
import data_io

st.set_page_config(page_title="ATR Bands + NMACD backtest", page_icon="📈", layout="wide")

FONT = "IBM Plex Sans"
INK, MUTED, GRID = "#1A2530", "#6B7780", "#E3E8EB"
UP, DOWN, NEUTRAL, ACCENT, TRIG = "#2E8B57", "#B23A48", "#8C969E", "#2C5F8A", "#C08A2E"
BUNDLED = Path(__file__).resolve().parent / "data" / "Netflix_Historical_Data.csv"
EXITS = {"NMACD hits +1": "nmacd", "ATR Trend Bands turn green": "atr", "Whichever of the two comes first": "either"}
TIME_UNITS = {"1 day": "trading days", "1 week": "weeks", "1 month": "months"}   # other timeframes: bars

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, p, li, label, input, h1, h2, h3, h4, h5, .stMarkdown,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ font-family: '{FONT}', sans-serif; }}
    h1 {{ font-weight: 600; letter-spacing: -0.02em; }}
    [data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 500; }}
    .rule-label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 0.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def spct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def money(x):
    return "–" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=dict(family=FONT, color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None))
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def segments(x, y, state, want):
    xs, ys = [], []
    for i in range(1, len(y)):
        if state[i] == want and np.isfinite(y[i - 1]) and np.isfinite(y[i]):
            xs += [x[i - 1], x[i], None]
            ys += [y[i - 1], y[i], None]
    return xs, ys


@st.cache_data(show_spinner="Reading prices…")
def load_file(content: bytes | None) -> pd.DataFrame:
    return data_io.read_csv(content if content is not None else BUNDLED)


@st.cache_data(ttl=900, show_spinner="Downloading prices from Yahoo Finance…")
def load_yahoo(ticker: str, timeframe: str) -> pd.DataFrame:
    return data_io.from_yahoo(ticker, timeframe)


@st.cache_data(show_spinner="Running the backtest…")
def run_cached(prices: pd.DataFrame, settings: dict):
    s = B.Settings(**settings)
    d, eq, tr = B.run(prices, s)
    return d, eq, tr, B.stats(eq, tr, prices, s.capital)


@st.cache_data(show_spinner="Testing other thresholds…")
def sensitivity_cached(prices: pd.DataFrame, settings: dict):
    return B.sensitivity(prices, B.Settings(**settings))


# ----------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Prices", ["Netflix file you sent", "Upload a CSV", "Yahoo Finance ticker"])
    upload, ticker, timeframe = None, "", "1 day"
    if source == "Upload a CSV":
        upload = st.file_uploader("Daily prices (Nasdaq, Yahoo or TradingView export)", type=["csv"])
    elif source == "Yahoo Finance ticker":
        ticker = st.text_input("Ticker", "NFLX").strip().upper()
        timeframe = st.selectbox(
            "Timeframe", list(data_io.TIMEFRAMES), index=list(data_io.TIMEFRAMES).index("1 day"),
            help="Yahoo keeps 1-minute bars for the last 7 days, 5 to 30-minute bars for 60 days and 1-hour bars "
                 "for 2 years. Daily, weekly and monthly bars go back to the listing. Yahoo has no 4-hour bars, so "
                 "they're built from 1-hour bars (09:30 and 13:30 for US stocks, like TradingView). Intraday bars "
                 "cover regular trading hours only.")
    unit = "bars" if source == "Upload a CSV" else TIME_UNITS.get(timeframe, "bars")

    st.header("Entry rules")
    near_band = st.slider("Close to the lower band: within this share of the corridor", 0.05, 1.0, 0.25, 0.05,
                          help="0 = on the lower band, 1 = on the upper band. 0.25 means the bottom quarter.")
    near_low = st.slider("NMACD came from near -1: its recent low is at or below", -1.0, 0.0, -0.8, 0.05)
    lookback = st.slider("Recent = the last this many bars", 1, 20, 5)

    st.header("Position")
    repeat = st.radio("A buy signal while already holding", ["Ignore it: one position at a time",
                                                             "Buy more, then sell everything together"],
                      help="'Buy more' buys again on every new signal. The next sell signal, stop loss or time "
                           "limit sells all the buys at once.")
    pyramid = repeat.startswith("Buy more")
    entry_size, max_entries = 0.25, 4
    if pyramid:
        entry_size = st.slider("Each buy uses this % of the account", 5, 100, 25, 5,
                               help="Of the account value at the fill (cash plus the shares already held), and "
                                    "never more than the cash left, so the test never borrows.") / 100
        max_entries = st.slider("Most buys held at once", 2, 20, 4)

    st.header("Exit rules")
    exit_rule = EXITS[st.radio("Sell when", list(EXITS),
                               help="The ATR Trend Bands turn green when their trend flips from down to up: the "
                                    "bar where the indicator prints an L.")]
    exit_level = 0.99
    if exit_rule != "atr":
        exit_level = st.slider("NMACD 'hits +1': at or above", 0.80, 1.0, 0.99, 0.01,
                               help="The line can't print exactly +1 because of a tiny constant in its formula.")
    stop_loss = st.slider("Optional stop loss (% below entry, 0 = off)", 0, 50, 0, 1,
                          help="With 'Buy more' it's measured from the average price of the buys and sells all "
                               "of them.")
    max_bars = st.slider(f"Optional time limit ({unit}, 0 = off)", 0, 250, 0, 5,
                         help="With 'Buy more' it counts from the first buy.")

    st.header("Execution")
    fill_label = st.radio("Orders fill at", ["Next bar's open (TradingView default)", "The signal bar's close"],
                          help="A bar is one candle of the timeframe: one day on daily prices.")
    cost_bps = st.number_input("Commission and slippage per side (basis points)", 0.0, 100.0, 5.0, 0.5)
    capital = st.number_input("Starting capital ($)", 1_000, 100_000_000, 100_000, 1_000)
    log_scale = st.toggle("Log scale on the equity chart", True)

    with st.expander("Indicator settings"):
        st.markdown("**ATR Trend Bands**")
        atr_source = st.selectbox("Source", ("close", "open", "high", "low", "hl2", "hlc3", "ohlc4"))
        atr_length = st.number_input("Length ATR", 1, 100, 5)
        atr_mult = st.number_input("Multiplier ATR", 0.1, 50.0, 7.6, 0.1, format="%.1f")
        st.markdown("**Normalized MACD**")
        fast = st.number_input("Fast MA", 1, 200, 13)
        slow = st.number_input("Slow MA", 2, 400, 21)
        trigger = st.number_input("Trigger", 1, 100, 9)
        normalize = st.number_input("Normalize", 1, 500, 50)
        ma_type = st.number_input("1=Ema, 2=Wma, 3=Sma", 1, 3, 1)

# ----------------------------------------------------------------------------------------
# Data and backtest
# ----------------------------------------------------------------------------------------
if source == "Upload a CSV" and upload is None:
    st.info("Upload a CSV of daily prices in the sidebar.")
    st.stop()
if source == "Yahoo Finance ticker" and not ticker:
    st.info("Type a ticker in the sidebar.")
    st.stop()
prices = None
try:
    if source == "Netflix file you sent":
        prices, label = load_file(None), "Netflix (NFLX), from the file you sent"
    elif source == "Upload a CSV":
        prices, label = load_file(upload.getvalue()), upload.name
    else:
        prices, label = load_yahoo(ticker, timeframe), f"{ticker} from Yahoo Finance"
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't load prices: {e}")
if prices is None:
    st.stop()

intraday = data_io.is_intraday(prices.index)
bars_desc = data_io.bar_size(prices.index)                                   # "daily", "1-hour", ...
bar_word = {"daily": "day", "weekly": "week", "monthly": "month"}.get(bars_desc, "bar")


def when(ts) -> str:
    return f"{pd.Timestamp(ts):%b %d, %Y %H:%M}" if intraday else f"{pd.Timestamp(ts):%b %d, %Y}"


def X(ts):
    """x values for the charts. Intraday bars go on a category axis, so nights and weekends take no space."""
    idx = pd.DatetimeIndex(ts)
    return list(idx.strftime("%Y-%m-%d %H:%M")) if intraday else idx


settings = B.Settings(atr_length=int(atr_length), atr_mult=float(atr_mult), atr_source=atr_source, fast=int(fast),
                      slow=int(slow), trigger=int(trigger), normalize=int(normalize), ma_type=int(ma_type),
                      near_band=near_band, near_low=near_low, lookback=int(lookback), exit_rule=exit_rule,
                      exit_level=exit_level, stop_loss=stop_loss / 100, max_bars=int(max_bars), pyramid=pyramid,
                      entry_size=entry_size, max_entries=int(max_entries),
                      fill="open" if fill_label.startswith("Next") else "close",
                      cost_bps=float(cost_bps), capital=float(capital))
d, eq, trades, stats = run_cached(prices, asdict(settings))
funnel = B.funnel(d, f"{bar_word}s")

st.title("ATR Trend Bands + Normalized MACD backtest")
mode = (f"each buy signal buys {entry_size:.0%} of the account, up to {max_entries} buys at once, and a sell "
        "signal sells them all" if pyramid else "one position at a time, all capital in each trade")
st.caption(f"{label}: {len(prices):,} {bars_desc} bars, {when(prices.index[0])} to {when(prices.index[-1])}. "
           f"Long only, {mode}.")

b, s_ = st.columns(2)
with b.container(border=True):
    st.markdown(f'<div class="rule-label">Buy when, all on the same {bar_word}\'s close</div>', unsafe_allow_html=True)
    st.markdown(f"- The ATR Trend Bands trend is **down**.\n"
                f"- The close is **near the lower band**: in the bottom {near_band:.0%} of the corridor.\n"
                f"- The NMACD line **crosses above its trigger line**.\n"
                f"- The NMACD line **came from near -1**: it was at or below {near_low:g} within the last "
                f"{lookback} bars."
                + (f"\n- While holding, **every new signal buys more**: {entry_size:.0%} of the account each "
                   f"time, up to {max_entries} buys." if pyramid else ""))
with s_.container(border=True):
    st.markdown(f'<div class="rule-label">{"Sell everything when" if pyramid else "Sell when"}</div>',
                unsafe_allow_html=True)
    nmacd_line = f"The NMACD line **hits +1** (closes at or above {exit_level:g})."
    lines = {"nmacd": [f"- {nmacd_line}"],
             "atr": ["- The ATR Trend Bands **turn green**: the trend flips from down to up."],
             "either": [f"- {nmacd_line}",
                        "- Or the ATR Trend Bands **turn green** (the trend flips up), whichever comes first."]}[exit_rule]
    if stop_loss:
        lines.append(f"- Or the price falls {stop_loss}% below the {'average ' if pyramid else ''}entry price "
                     "(stop loss).")
    if max_bars:
        lines.append(f"- Or {max_bars} {unit} have passed{' since the first buy' if pyramid else ''}.")
    st.markdown("\n".join(lines))
fill_text = f"the next {bar_word}'s open" if settings.fill == "open" else f"the signal {bar_word}'s close"
st.caption(f"Orders fill at {fill_text}, with {cost_bps:g} basis points of cost on each side.")

last = d.iloc[-1]
open_lots = trades[trades["Status"] == "Open"] if len(trades) else trades
notes = []
if len(open_lots) == 1:
    t = open_lots.iloc[0]
    notes.append(f"At the last close ({when(d.index[-1])}) the backtest is holding a position bought on "
                 f"{when(t['Entry date'])} at {t['Entry price']:.2f}, currently {spct(t['Return'])}.")
elif len(open_lots) > 1:
    inv = open_lots["Invested ($)"]
    avg_px = inv.sum() / (inv / open_lots["Entry price"]).sum()
    notes.append(f"At the last close ({when(d.index[-1])}) the backtest is holding {len(open_lots)} buys made "
                 f"from {when(open_lots['Entry date'].min())} to {when(open_lots['Entry date'].max())}, at an "
                 f"average price of {avg_px:.2f}, currently {spct(open_lots['P&L ($)'].sum() / inv.sum())}.")
if bool(last["Entry signal"]) and settings.fill == "open" and len(open_lots) < (max_entries if pyramid else 1):
    notes.append(f"The rules fired a buy signal on the last close ({when(d.index[-1])}). In the backtest it would "
                 f"fill at the next {'session' if bar_word == 'day' else bar_word}'s open, which isn't in the data "
                 "yet. This is the rules' output, not advice.")
if notes:
    st.info(" ".join(notes))

# ----------------------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("Total P&L", money(stats["Total P&L ($)"]), spct(stats["Total return"]), delta_color="normal")
c[1].metric("Win rate", "–" if pd.isna(stats["Win rate"]) else f"{stats['Win rate']:.0%}",
            f"{int(round(stats['Win rate'] * stats['Closed trades'])) if stats['Closed trades'] else 0} of "
            f"{stats['Closed trades']} closed trades", delta_color="off")
c[2].metric("Average win / loss", f"{spct(stats['Average win'])} / {spct(stats['Average loss'])}")
c[3].metric("Profit factor", "–" if pd.isna(stats["Profit factor"]) else f"{stats['Profit factor']:.2f}")
c[4].metric("Max drawdown", spct(stats["Max drawdown"]))
c[5].metric("Buy & hold instead", spct(stats["Buy & hold return"]),
            f"max drawdown {stats['Buy & hold max drawdown'] * 100:.0f}%", delta_color="off")
c = st.columns(6)
c[0].metric("Trades", f"{stats['Trades']}", f"in {stats['Positions']} positions" if pyramid else None,
            delta_color="off")
c[1].metric("Average trade", spct(stats["Average trade"]))
c[2].metric("Best / worst trade", f"{spct(stats['Best trade'])} / {spct(stats['Worst trade'])}")
c[3].metric(f"Average {bar_word}s held",
            "–" if pd.isna(stats["Average bars held"]) else f"{stats['Average bars held']:.0f}")
c[4].metric("Time in the market", f"{stats['Time in market']:.0%}")
c[5].metric("Worst dip inside a trade", spct(stats["Worst drawdown inside a trade"]),
            help="Lowest low while holding, against the entry price.")
n_closed = stats["Closed positions"] if pyramid else stats["Closed trades"]
if n_closed < 30:
    what = f"closed positions ({stats['Closed trades']} buys)" if pyramid else "closed trades"
    st.warning(f"Only {n_closed} {what}. With so few, the win rate and P&L can swing a lot from one trade, so "
               "treat them as anecdotes rather than evidence.")

# ----------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------
st.markdown("##### Equity")
bh = prices["Close"] / prices["Close"].iloc[0] * settings.capital
fig = go.Figure()
fig.add_trace(go.Scatter(x=X(bh.index), y=bh, name="Buy & hold", line=dict(color=NEUTRAL, width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=X(eq.index), y=eq, name="Strategy", line=dict(color=ACCENT, width=2.2)))
fig = style_fig(fig, 360)
fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$", tickformat=",.0f")
if intraday:
    fig.update_xaxes(type="category")
st.plotly_chart(fig, width="stretch")

st.markdown("##### Price with ATR Trend Bands, and the Normalized MACD")
x = list(X(d.index))
state = d["Trend"].map({"Up": 1, "Down": -1}).fillna(0).to_numpy()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=x, y=d["Close"], name="Close", line=dict(color=INK, width=1.1)), row=1, col=1)
for band in ("Upper", "Lower"):
    y = d[band].to_numpy()
    for want, color in ((1, UP), (-1, DOWN)):
        xs, ys = segments(x, y, state, want)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Lower"], mode="lines", line=dict(width=0), showlegend=False,
                         hovertemplate="Lower band %{y:,.2f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Upper"], mode="lines", line=dict(width=0), showlegend=False,
                         hovertemplate="Upper band %{y:,.2f}<extra></extra>"), row=1, col=1)
if len(trades):
    fig.add_trace(go.Scatter(x=X(trades["Entry date"]), y=trades["Entry price"], mode="markers", name="Buy",
                             marker=dict(symbol="triangle-up", size=12, color=UP)), row=1, col=1)
    closed = trades[trades["Status"] == "Closed"].drop_duplicates("Exit date")   # buys sold together: one marker
    fig.add_trace(go.Scatter(x=X(closed["Exit date"]), y=closed["Exit price"], mode="markers", name="Sell",
                             marker=dict(symbol="triangle-down", size=12, color=DOWN)), row=1, col=1)
fig.add_trace(go.Bar(x=x, y=d["Hist"], name="Histogram", marker_color="#B8C0C6", opacity=0.6), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["NMACD"], name="NMACD", line=dict(color=DOWN, width=1.6)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Trigger"], name="Trigger", line=dict(color=TRIG, width=1.4)), row=2, col=1)
sig = d[d["Entry signal"]]
fig.add_trace(go.Scatter(x=X(sig.index), y=sig["NMACD"], mode="markers", name="Buy signal",
                         marker=dict(symbol="circle", size=8, color=UP)), row=2, col=1)
for yv, dash in ((1, "dot"), (-1, "dot"), (0, "solid")):
    fig.add_hline(y=yv, line=dict(color=MUTED, width=0.8, dash=dash), row=2, col=1)
fig = style_fig(fig, 640)
if intraday:
    fig.update_xaxes(type="category")
elif bars_desc == "daily" and not (prices.index.dayofweek >= 5).any():
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_yaxes(range=[-1.15, 1.15], row=2, col=1)
st.plotly_chart(fig, width="stretch")
st.caption(f"Bands are green in an uptrend and red in a downtrend. Green dots on the NMACD panel are the "
           f"{bar_word}s all four buy conditions held.")

# ----------------------------------------------------------------------------------------
# Trades and diagnostics
# ----------------------------------------------------------------------------------------
st.markdown("##### Trade log")
if len(trades):
    show = trades.iloc[::-1].copy()
    for col in ("Signal date", "Entry date", "Exit date"):
        show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "Entry price": st.column_config.NumberColumn(format="%.2f"),
        "Invested ($)": st.column_config.NumberColumn(format="dollar"),
        "Exit price": st.column_config.NumberColumn(format="%.2f"),
        "Return": st.column_config.NumberColumn(format="percent"),
        "P&L ($)": st.column_config.NumberColumn(format="dollar"),
        "Worst drawdown in trade": st.column_config.NumberColumn(format="percent"),
        "Band position at signal": st.column_config.NumberColumn(format="%.2f"),
        "NMACD at signal": st.column_config.NumberColumn(format="%.2f"),
        "Recent NMACD low": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download the trade log (CSV)", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
else:
    st.write("No trades with these settings.")

st.markdown("##### Why there are so few trades")
st.dataframe(funnel, hide_index=True, width="stretch")
skipped = (f"With 'Buy more', a signal is skipped only once {max_entries} buys are held or the cash has run out, "
           f"so trades can still be fewer than signal {bar_word}s." if pyramid else
           f"Signal {bar_word}s that come while a position is already open are skipped, so trades can be fewer "
           f"than signal {bar_word}s.")
st.caption(f"Each row keeps only the {bar_word}s that also pass the rows above it. {skipped}")

st.markdown("##### How the result changes with the two vague thresholds")
sens = sensitivity_cached(prices, asdict(settings))
st.dataframe(sens, hide_index=True, width="stretch", column_config={
    "Win rate": st.column_config.NumberColumn(format="percent"),
    "Average trade": st.column_config.NumberColumn(format="percent"),
    "Total return": st.column_config.NumberColumn(format="percent"),
    "Max drawdown": st.column_config.NumberColumn(format="percent")})
st.caption("'Close to the lower band' and 'near -1' have no exact definition, so this reruns the backtest "
           "across several readings of each. Everything else stays as set in the sidebar.")

with st.expander("How the indicators and rules are defined"):
    exit_terms = []
    if exit_rule != "atr":
        exit_terms.append(f'- "Hits +1": the NMACD line closes at or above {exit_level:g}.')
    if exit_rule != "nmacd":
        exit_terms.append('- "Turn green": the ATR Trend Bands trend is up at the close. Buys only happen while '
                          "it's down, so this is the bar where it flips up and the indicator prints an L.")
    buying_more = (f"""
**Buying more.** Each buy signal while holding buys another {entry_size:.0%} of the account value (cash plus
the shares already held, at the fill price), never more than the cash left, up to {max_entries} buys. Any exit
sells all of them at once. The stop loss is measured from the average price of the buys and the time limit
counts from the first buy. Each buy is a row in the trade log, so the win rate and average trade are per buy.
""" if pyramid else "")
    st.markdown(
        f"""
**Normalized MACD** (TradingView's "N MACD"). Fast and slow moving averages of the close
({fast} and {slow} bars; type {ma_type}: 1 = EMA, 2 = WMA, 3 = SMA). Their ratio is turned into a
momentum value, which is rescaled to between -1 and +1 against its own range over the last {normalize} bars.
The trigger line is a {trigger}-bar weighted average of the NMACD line.

**What +1 and -1 mean.** +1 means the fast average is further above the slow one than at any point in the
last {normalize} bars, and -1 means the reverse. Because the scale is relative to the last {normalize} bars,
the line can reach +1 after a modest bounce in a falling market. So "NMACD hit +1" doesn't mean the price is
above where you bought, which is why some exits here are losses.

**ATR Trend Bands** use the same replica as the ATR Trend Bands app, with source {atr_source},
length {atr_length} and multiplier {atr_mult:g}. They're green while the trend is up and red while it's down.

**Turning the words into numbers.**
- "Close to the lower band": the close sits in the bottom {near_band:.0%} of the corridor between the bands.
- "Crosses the trigger from near -1": the NMACD line closes above the trigger after closing at or below it
  the bar before, and was at or below {near_low:g} within the last {lookback} bars.
{chr(10).join(exit_terms)}
{buying_more}
**What the test leaves out:** taxes, borrowing costs, dividends (Netflix pays none) and gaps through a stop,
beyond the fill rules above.
        """
    )
