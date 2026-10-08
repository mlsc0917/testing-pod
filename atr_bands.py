"""ATR Trend Bands [Misu], replicated in Python.

Follows the TradingView script line by line, and the library function it imports
(fontilab.getTrendBands, version 11) as translated line by line in the published ThinkOrSwim
port. Two details can't be checked without the original library source; both are decided here
and flagged in the app:

* Start of the chart. Before the first bar with an ATR value the bands are treated as 0, so that
  bar opens an uptrend with upper = source and lower = source - delta.
* Ties. A close exactly on a band counts as inside the bands (strict comparisons).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCES = ("close", "open", "high", "low", "hl2", "hlc3", "ohlc4")


def source_series(df: pd.DataFrame, name: str) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    return {"close": c, "open": o, "high": h, "low": l, "hl2": (h + l) / 2,
            "hlc3": (h + l + c) / 3, "ohlc4": (o + h + l + c) / 4}[name]


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine's ta.rma: empty until n values exist, seeded with their simple average, then
    each value = (previous x (n - 1) + new) / n."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def trend_bands(src: np.ndarray, delta: np.ndarray) -> dict:
    """The imported getTrendBands logic, bar by bar, keeping every intermediate value."""
    n = len(src)
    cols = {k: np.full(n, np.nan) for k in ("Prev upper", "Prev lower", "Candidate", "Upper", "Lower", "Mid")}
    rule = np.array(["warm-up"] * n, dtype=object)
    moved = np.zeros(n, dtype=bool)                        # did the far band move to the candidate?
    up_prev = low_prev = 0.0                               # start-of-chart assumption (see module notes)
    for i in range(n):
        if not (np.isfinite(delta[i]) and np.isfinite(src[i])):
            continue
        s = src[i]
        s_prev = src[i - 1] if i > 0 and np.isfinite(src[i - 1]) else s
        cols["Prev upper"][i], cols["Prev lower"][i] = up_prev, low_prev
        if s > up_prev:                                    # close above the upper band
            upper = max(up_prev, s, s_prev)
            cand = upper - delta[i]
            keep = cand < low_prev or (cand > low_prev and upper == up_prev)
            lower = low_prev if keep else cand
            rule[i], moved[i] = "Breakout up", not keep
        elif s < low_prev:                                 # close below the lower band
            lower = min(low_prev, s, s_prev)
            cand = lower + delta[i]
            keep = cand > up_prev or (cand < up_prev and lower == low_prev)
            upper = up_prev if keep else cand
            rule[i], moved[i] = "Breakout down", not keep
        else:                                              # inside: both bands frozen
            upper, lower, cand = up_prev, low_prev, np.nan
            rule[i] = "Inside"
        cols["Candidate"][i], cols["Upper"][i], cols["Lower"][i] = cand, upper, lower
        cols["Mid"][i] = (upper + lower) / 2
        up_prev, low_prev = upper, lower
    cols["Rule"], cols["Moved"] = rule, moved
    return cols


def compute(df: pd.DataFrame, source: str = "close", length: int = 5, mult: float = 7.6) -> pd.DataFrame:
    """Every intermediate number the indicator uses, one row per bar."""
    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    prev_close = df["Close"].shift(1)
    out = pd.DataFrame(index=df.index)
    out["Open"], out["High"], out["Low"], out["Close"] = df["Open"], df["High"], df["Low"], df["Close"]
    out["Source"] = source_series(df, source)
    out["High - low"] = df["High"] - df["Low"]
    out["|High - prev close|"] = (df["High"] - prev_close).abs()
    out["|Low - prev close|"] = (df["Low"] - prev_close).abs()
    tr = out[["High - low", "|High - prev close|", "|Low - prev close|"]].max(axis=1)
    tr.iloc[0] = out["High - low"].iloc[0]                 # first bar: no previous close
    out["True range"] = tr
    out["ATR"] = rma(tr.to_numpy(), length)
    out["Delta"] = mult * out["ATR"]
    out["ATR30"] = rma(tr.to_numpy(), 30)

    bands = trend_bands(out["Source"].to_numpy(), out["Delta"].to_numpy())
    for k, v in bands.items():
        out[k] = v
    out["Width"] = out["Upper"] - out["Lower"]
    out["Width / ATR"] = out["Width"] / out["ATR"]

    # Trend state, signals and colours exactly as in the visible Pine script.
    n = len(out)
    mid = out["Mid"].to_numpy()
    mid_prev = np.r_[np.nan, mid[:-1]]
    valid = np.isfinite(mid)
    trend_up = valid & (mid > np.nan_to_num(mid_prev))
    trend_down = valid & (mid < np.nan_to_num(mid_prev))
    state = np.zeros(n, dtype=int)
    for i in range(n):
        state[i] = 1 if trend_up[i] else (-1 if trend_down[i] else (state[i - 1] if i else 0))
    prev_state = np.r_[0, state[:-1]]
    out["Midline rose"] = trend_up
    out["Midline fell"] = trend_down
    out["Trend"] = np.where(state == 1, "Up", np.where(state == -1, "Down", ""))
    out["Signal"] = np.where(trend_up & (prev_state == -1), "L", np.where(trend_down & (prev_state == 1), "S", ""))
    out["Prev ATR"] = out["ATR"].shift(1)
    out["Prev mid"] = out["Mid"].shift(1)
    out["Why"] = [explain(r, mult) for r in out.to_dict("records")]
    return out


# ----------------------------------------------------------------------------------------
# Plain-language explanations
# ----------------------------------------------------------------------------------------
def _f(x: float) -> str:
    return f"{x:,.2f}"


def explain(r: dict, mult: float) -> str:
    """One sentence on why the day's bands and width are what they are (r: one row as a dict)."""
    rule = r["Rule"]
    if rule == "warm-up":
        return "Not enough bars yet to compute the ATR."
    if r["Prev upper"] == 0 and r["Prev lower"] == 0:
        return (f"First bar with an ATR: the bands start here, upper at the source {_f(r['Upper'])} and lower "
                f"{mult:g} x ATR below it. Width {_f(r['Width'])}.")
    if rule == "Inside":
        return (f"Source {_f(r['Source'])} is between the bands ({_f(r['Prev lower'])} to {_f(r['Prev upper'])}), "
                f"so nothing moves. Width stays {_f(r['Width'])} even though today's {mult:g} x ATR is "
                f"{_f(r['Delta'])}.")
    up = rule == "Breakout up"
    if up:
        head = (f"Source {_f(r['Source'])} closed above the upper band {_f(r['Prev upper'])}, so upper moves to "
                f"{_f(r['Upper'])}. ")
        far, far_prev, sign, verb = "Lower", r["Prev lower"], "-", "pulled up"
        anchor, side = r["Upper"], "below the current lower band"
    else:
        head = (f"Source {_f(r['Source'])} closed below the lower band {_f(r['Prev lower'])}, so lower moves to "
                f"{_f(r['Lower'])}. ")
        far, far_prev, sign, verb = "Upper", r["Prev upper"], "+", "pulled down"
        anchor, side = r["Lower"], "above the current upper band"
    if r["Moved"]:
        body = (f"{far} is {verb} to {_f(anchor)} {sign} {_f(r['Delta'])} = {_f(r['Candidate'])} (it was "
                f"{_f(far_prev)}). Width = today's {mult:g} x ATR = {_f(r['Width'])}.")
    else:
        body = (f"{_f(anchor)} {sign} {_f(r['Delta'])} = {_f(r['Candidate'])} would be {side} "
                f"{_f(far_prev)}, and bands never loosen, so {far.lower()} stays. Width = {_f(r['Upper'])} - "
                f"{_f(r['Lower'])} = {_f(r['Width'])}, less than today's {mult:g} x ATR of {_f(r['Delta'])}.")
    tail = {"L": " The midline rose after a downtrend: flip to long (L).",
            "S": " The midline fell after an uptrend: flip to short (S)."}.get(r["Signal"], "")
    return head + body + tail


def walkthrough(df: pd.DataFrame, when, length: int, mult: float) -> str:
    """Step-by-step markdown for one bar."""
    i = df.index.get_loc(when)
    r = df.iloc[i]
    lines = [f"**Step 1: true range.** High - low = {_f(r['High - low'])}"]
    if i == 0:
        lines[0] += ". First bar, so there's no previous close; true range = high - low."
    else:
        lines[0] += (f", |high - previous close| = {_f(r['|High - prev close|'])}, |low - previous close| = "
                     f"{_f(r['|Low - prev close|'])}. True range is the largest: **{_f(r['True range'])}**.")
    if not np.isfinite(r["ATR"]):
        lines.append(f"**Step 2: ATR({length}).** Needs {length} bars of true range; not available yet.")
        return "\n\n".join(lines)
    if not np.isfinite(r["Prev ATR"]):
        lines.append(f"**Step 2: ATR({length}).** The first ATR is the plain average of the first {length} true "
                     f"ranges: **{_f(r['ATR'])}**.")
    else:
        lines.append(f"**Step 2: ATR({length}).** (previous ATR {_f(r['Prev ATR'])} x {length - 1} + today's true "
                     f"range {_f(r['True range'])}) / {length} = **{_f(r['ATR'])}**.")
    lines.append(f"**Step 3: allowed distance.** {mult:g} x {_f(r['ATR'])} = **{_f(r['Delta'])}**. This is the "
                 "width the corridor gets whenever a band is re-placed today.")
    if r["Prev upper"] == 0 and r["Prev lower"] == 0:
        lines.append(f"**Step 4: start of the bands.** This is the first bar with an ATR, so the bands start here: "
                     f"upper = {_f(r['Upper'])}, lower = {_f(r['Upper'])} - {_f(r['Delta'])} = {_f(r['Lower'])}.")
    else:
        where = {"Breakout up": "above the upper band", "Breakout down": "below the lower band",
                 "Inside": "between the bands"}[r["Rule"]]
        lines.append(f"**Step 4: compare with the previous bar's bands.** Source {_f(r['Source'])} against upper "
                     f"{_f(r['Prev upper'])} and lower {_f(r['Prev lower'])}: it is **{where}**.")
        lines.append("**Step 5: new bands.** " + explain(r.to_dict(), mult).split(" The midline")[0])
    step = len(lines) + 1
    lines.append(f"**Step {step}: width.** {_f(r['Upper'])} - {_f(r['Lower'])} = **{_f(r['Width'])}**, which is "
                 f"{r['Width / ATR']:.2f} x today's ATR (the setting is {mult:g}).")
    if np.isfinite(r["Prev mid"]):
        move = "rose" if r["Midline rose"] else ("fell" if r["Midline fell"] else "didn't move")
        lines.append(f"**Step {step + 1}: trend.** Midline (upper + lower) / 2 = {_f(r['Mid'])}, against "
                     f"{_f(r['Prev mid'])} on the previous bar: it {move}. Trend: **{r['Trend'].lower()}**."
                     + {"L": " That's a flip from down to up: **L** signal.",
                        "S": " That's a flip from up to down: **S** signal."}.get(r["Signal"], ""))
    return "\n\n".join(lines)
