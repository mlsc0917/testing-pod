"""Normalized MACD ("N MACD" on TradingView), replicated in Python.

Pine logic of the original script:
    sh, lon  = moving averages of close (fast 13, slow 21; EMA, WMA or SMA)
    ratio    = min(sh, lon) / max(sh, lon)
    Mac      = (2 - ratio if sh > lon else ratio) - 1
    MacNorm  = (Mac - lowest(Mac, 50)) / (highest(Mac, 50) - lowest(Mac, 50) + 0.000001) * 2 - 1
    Trigger  = WMA(MacNorm, 9)
    Hist     = MacNorm - Trigger, clipped to [-1, 1]

MacNorm sits at +1 when the fast/slow gap is at its widest of the last 50 bars and at -1 when it is
at its narrowest (most negative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: pd.Series, n: int) -> pd.Series:
    """Pine's ta.ema: seeded with the simple average of the first n values."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    alpha = 2.0 / (n + 1)
    start = np.flatnonzero(np.isfinite(v))
    if len(start) == 0 or len(v) - start[0] < n:
        return pd.Series(out, index=x.index)
    s = start[0] + n - 1
    out[s] = np.mean(v[start[0]:s + 1])
    for i in range(s + 1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=x.index)


def wma(x: pd.Series, n: int) -> pd.Series:
    """Pine's ta.wma: weights 1..n, the newest bar heaviest."""
    w = np.arange(1, n + 1, dtype=float)
    return x.rolling(n).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def compute(close: pd.Series, fast: int = 13, slow: int = 21, trigger: int = 9, normalize: int = 50,
            ma_type: int = 1) -> pd.DataFrame:
    ma = {1: ema, 2: wma, 3: sma}[ma_type]
    sh, lon = ma(close, fast), ma(close, slow)
    ratio = np.minimum(sh, lon) / np.maximum(sh, lon)
    mac = pd.Series(np.where(sh > lon, 2 - ratio, ratio) - 1, index=close.index)
    mac[sh.isna() | lon.isna()] = np.nan
    if normalize < 2:
        norm = mac
    else:
        lo = mac.rolling(normalize).min()
        hi = mac.rolling(normalize).max()
        norm = (mac - lo) / (hi - lo + 0.000001) * 2 - 1
    trig = wma(norm, trigger)
    hist = (norm - trig).clip(-1, 1)
    return pd.DataFrame({"Fast MA": sh, "Slow MA": lon, "Mac": mac, "NMACD": norm, "Trigger": trig, "Hist": hist})
