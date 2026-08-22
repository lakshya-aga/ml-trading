"""Information-driven bars (AFML chapter 2).

Calendar-time bars sample the market on a clock the market does not follow:
Indian equities transact a large share of their daily volume in the first and
last twenty minutes, so a 5-minute grid oversamples a dead midday and
undersamples the open. Tick, volume and rupee-value bars sample on activity
instead, which brings returns much closer to IID normality — the assumption
almost every downstream estimator relies on.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from afml_india.utils.validation import ensure_datetime_index, ensure_monotonic

TICK_COLUMNS = ("price", "volume")
BAR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "ticks",
    "vwap",
)


def _empty_bars() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(BAR_COLUMNS), dtype=float)
    frame.index = pd.DatetimeIndex([], name="timestamp")
    return frame


def _validate_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    ensure_datetime_index(ticks, "ticks")
    ensure_monotonic(ticks, "ticks")
    missing = [c for c in TICK_COLUMNS if c not in ticks.columns]
    if missing:
        raise KeyError(f"tick frame is missing columns {missing}")
    return ticks


def _aggregate(
    ticks: pd.DataFrame,
    boundaries: np.ndarray,
) -> pd.DataFrame:
    """Aggregate ticks into OHLCV bars given right-inclusive boundary positions.

    ``boundaries`` holds the *exclusive* end position of each bar, so bar ``i``
    covers ``ticks[boundaries[i-1]:boundaries[i]]``.
    """
    if len(boundaries) == 0:
        return _empty_bars()

    price = ticks["price"].to_numpy(dtype=float)
    volume = ticks["volume"].to_numpy(dtype=float)
    value = price * volume
    stamps = ticks.index.to_numpy()

    starts = np.concatenate(([0], boundaries[:-1]))
    rows = []
    for start, stop in zip(starts, boundaries):
        if stop <= start:
            continue
        chunk_p = price[start:stop]
        chunk_v = volume[start:stop]
        chunk_val = value[start:stop]
        total_v = chunk_v.sum()
        rows.append(
            (
                stamps[stop - 1],
                chunk_p[0],
                chunk_p.max(),
                chunk_p.min(),
                chunk_p[-1],
                total_v,
                chunk_val.sum(),
                float(stop - start),
                chunk_val.sum() / total_v if total_v > 0 else chunk_p[-1],
            )
        )
    if not rows:
        return _empty_bars()

    index = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
    frame = pd.DataFrame([r[1:] for r in rows], index=index, columns=list(BAR_COLUMNS))
    if ticks.index.tz is not None and frame.index.tz is None:
        frame.index = frame.index.tz_localize(ticks.index.tz)
    return frame


def _threshold_boundaries(increments: np.ndarray, threshold: float) -> np.ndarray:
    """Positions where the running sum of ``increments`` first crosses ``threshold``.

    The running total resets on each crossing, so a single huge print closes its
    own bar rather than swallowing the next several.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    cumulative = 0.0
    out: list[int] = []
    for i, inc in enumerate(increments):
        cumulative += inc
        if cumulative >= threshold:
            out.append(i + 1)
            cumulative = 0.0
    return np.asarray(out, dtype=int)


def tick_bars(ticks: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Bars closed every ``threshold`` transactions."""
    ticks = _validate_ticks(ticks)
    if len(ticks) == 0:
        return _empty_bars()
    if threshold <= 0:
        raise ValueError("threshold must be a positive number of ticks")
    boundaries = np.arange(threshold, len(ticks) + 1, threshold, dtype=int)
    return _aggregate(ticks, boundaries)


def volume_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Bars closed every ``threshold`` shares traded."""
    ticks = _validate_ticks(ticks)
    if len(ticks) == 0:
        return _empty_bars()
    boundaries = _threshold_boundaries(ticks["volume"].to_numpy(dtype=float), threshold)
    return _aggregate(ticks, boundaries)


def value_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Bars closed every ``threshold`` rupees of turnover.

    Preferred over volume bars across an Indian cross-section, where share
    prices span three orders of magnitude and a fixed share count means very
    different things for a Rs 80 PSU bank and a Rs 80,000 name.
    """
    ticks = _validate_ticks(ticks)
    if len(ticks) == 0:
        return _empty_bars()
    value = ticks["price"].to_numpy(dtype=float) * ticks["volume"].to_numpy(dtype=float)
    boundaries = _threshold_boundaries(value, threshold)
    return _aggregate(ticks, boundaries)


#: AFML calls these dollar bars; the rupee-denominated name is the same thing.
dollar_bars = value_bars


def time_bars(ticks: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Calendar-time bars, kept for benchmarking against the activity-driven ones."""
    ticks = _validate_ticks(ticks)
    if len(ticks) == 0:
        return _empty_bars()
    value = ticks["price"] * ticks["volume"]
    grouped = ticks.assign(_value=value).resample(freq, label="right", closed="right")
    frame = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["volume"].sum(),
            "value": grouped["_value"].sum(),
            "ticks": grouped["price"].count().astype(float),
        }
    ).dropna(subset=["close"])
    frame["vwap"] = np.where(frame["volume"] > 0, frame["value"] / frame["volume"], frame["close"])
    frame.index.name = "timestamp"
    return frame[list(BAR_COLUMNS)]


def bars_from_ohlcv(
    ohlcv: pd.DataFrame,
    kind: str = "value",
    threshold: float | None = None,
    target_bars_per_day: float | None = None,
) -> pd.DataFrame:
    """Build activity-driven bars from OHLCV data by treating each bar as one tick.

    A true tick feed is better, but Indian minute data is far easier to obtain
    than full tick history, and resampling minute bars into rupee-value bars
    already recovers most of the statistical benefit. Each source bar is
    represented by its VWAP-equivalent typical price.
    """
    ensure_datetime_index(ohlcv, "ohlcv")
    ensure_monotonic(ohlcv, "ohlcv")
    if "close" not in ohlcv or "volume" not in ohlcv:
        raise KeyError("ohlcv needs at least 'close' and 'volume' columns")

    if {"high", "low"} <= set(ohlcv.columns):
        price = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
    else:
        price = ohlcv["close"]
    synthetic = pd.DataFrame({"price": price, "volume": ohlcv["volume"]}, index=ohlcv.index)

    if threshold is None:
        threshold = suggest_threshold(
            synthetic, kind=kind, target_bars_per_day=target_bars_per_day or 10
        )

    builders: dict[str, Callable[[pd.DataFrame, float], pd.DataFrame]] = {
        "tick": lambda t, x: tick_bars(t, int(max(1, round(x)))),
        "volume": volume_bars,
        "value": value_bars,
        "rupee": value_bars,
    }
    if kind not in builders:
        raise ValueError(f"unknown bar kind {kind!r}; expected one of {sorted(builders)}")
    return builders[kind](synthetic, threshold)


def suggest_threshold(
    ticks: pd.DataFrame,
    kind: str = "value",
    target_bars_per_day: float = 10.0,
) -> float:
    """Threshold that yields roughly ``target_bars_per_day`` bars.

    Sampling frequency is a real modelling choice: too few bars starves the
    model, too many inflate autocorrelation and make purging bite harder.
    """
    ticks = _validate_ticks(ticks)
    if len(ticks) == 0:
        raise ValueError("cannot suggest a threshold from an empty tick frame")
    if target_bars_per_day <= 0:
        raise ValueError("target_bars_per_day must be positive")

    days = max(1, ticks.index.normalize().nunique())
    if kind == "tick":
        total = float(len(ticks))
    elif kind == "volume":
        total = float(ticks["volume"].sum())
    elif kind in ("value", "rupee"):
        total = float((ticks["price"] * ticks["volume"]).sum())
    else:
        raise ValueError(f"unknown bar kind {kind!r}")
    return total / (days * target_bars_per_day)
