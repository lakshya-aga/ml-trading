"""Adapters that feed Indian tick data into fin-kit's AFML bar builders.

fin-kit's ``get_tick_bars`` / ``get_volume_bars`` / ``get_dollar_bars`` expect a
three-column frame of ``[date_time, price, volume]`` with no index, and their
thresholds are absolute. Both are easy to get wrong on Indian data — timestamps
arrive tz-aware in IST, quote rows have to be stripped out of a mixed tape, and
a dollar-bar threshold tuned on US equities produces either three bars a year or
one per tick on a Rs 200 PSU name. These helpers handle all three.
"""

from __future__ import annotations

import pandas as pd

from afml_india import finkit  # noqa: F401  (side effect: puts mlfinlab on sys.path)
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

#: AFML's rule of thumb: sample about 50 bars per day.
DEFAULT_BARS_PER_DAY = 50


def prepare_tick_frame(
    ticks: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "volume",
    time_col: str | None = None,
    trades_only: bool = True,
    tz_naive: bool = True,
) -> pd.DataFrame:
    """Normalise a tick tape into the ``[date_time, price, volume]`` frame fin-kit wants.

    Parameters
    ----------
    ticks:
        Tick data, either indexed by timestamp or carrying a timestamp column.
    trades_only:
        Drop non-``TRADE`` rows when a ``type`` column is present. Bar
        construction must run on the trade tape: including BID/ASK rows
        double-counts activity and corrupts every threshold.
    tz_naive:
        Strip the timezone. fin-kit compares raw values and mixing tz-aware and
        naive timestamps downstream raises; IST wall-clock time is unambiguous
        here because NSE has no DST.
    """
    frame = ticks.copy()

    if time_col is not None:
        stamps = pd.to_datetime(frame[time_col])
    elif isinstance(frame.index, pd.DatetimeIndex):
        stamps = frame.index.to_series(index=range(len(frame)))
        frame = frame.reset_index(drop=True)
    elif "date_time" in frame.columns:
        stamps = pd.to_datetime(frame["date_time"])
    else:
        raise KeyError(
            "could not find timestamps: pass time_col, or supply a DatetimeIndex "
            "or a 'date_time' column"
        )

    stamps = pd.DatetimeIndex(pd.to_datetime(stamps.to_numpy(), utc=False))
    if tz_naive and stamps.tz is not None:
        stamps = stamps.tz_localize(None)

    out = pd.DataFrame(
        {
            "date_time": stamps,
            "price": pd.to_numeric(frame[price_col].to_numpy(), errors="coerce"),
            "volume": pd.to_numeric(frame[volume_col].to_numpy(), errors="coerce"),
        }
    )

    if trades_only and "type" in frame.columns:
        is_trade = frame["type"].astype(str).str.upper().to_numpy() == "TRADE"
        if is_trade.any():
            dropped = int((~is_trade).sum())
            if dropped:
                logger.info("Dropping %d non-TRADE rows before bar construction", dropped)
            out = out[is_trade]

    before = len(out)
    out = out.dropna(subset=["price", "volume"])
    out = out[(out["price"] > 0) & (out["volume"] > 0)]
    if len(out) < before:
        logger.info("Dropped %d ticks with non-positive or missing price/volume", before - len(out))

    return out.sort_values("date_time").reset_index(drop=True)


def suggest_thresholds(
    ticks: pd.DataFrame,
    bars_per_day: int = DEFAULT_BARS_PER_DAY,
) -> dict[str, float]:
    """Tick, volume and rupee thresholds that yield ~``bars_per_day`` bars.

    Thresholds have to be set per name, not per market: across the NIFTY the
    daily turnover of the largest and smallest members differs by well over an
    order of magnitude, so one shared rupee threshold gives the big names
    hundreds of bars a day and the small ones almost none.
    """
    frame = prepare_tick_frame(ticks)
    if frame.empty:
        raise ValueError("cannot suggest thresholds from an empty tick frame")

    days = max(1, frame["date_time"].dt.normalize().nunique())
    per_day = days * bars_per_day
    value = (frame["price"] * frame["volume"]).sum()
    return {
        "tick": max(1.0, round(len(frame) / per_day)),
        "volume": float(frame["volume"].sum() / per_day),
        "dollar": float(value / per_day),
    }


def _bar_builder(kind: str):
    from mlfinlab.data_structures import (  # noqa: PLC0415
        get_dollar_bars,
        get_tick_bars,
        get_volume_bars,
    )

    builders = {
        "tick": get_tick_bars,
        "volume": get_volume_bars,
        "dollar": get_dollar_bars,
        "rupee": get_dollar_bars,  # same construction, rupee-denominated
    }
    if kind not in builders:
        raise ValueError(f"unknown bar kind {kind!r}; expected one of {sorted(builders)}")
    return builders[kind]


def source_timezone(ticks: pd.DataFrame, time_col: str | None = None) -> str | None:
    """Timezone of a tick tape's timestamps, or ``None`` when naive."""
    if time_col is not None:
        stamps = pd.DatetimeIndex(pd.to_datetime(ticks[time_col]))
    elif isinstance(ticks.index, pd.DatetimeIndex):
        stamps = ticks.index
    elif "date_time" in ticks.columns:
        stamps = pd.DatetimeIndex(pd.to_datetime(ticks["date_time"]))
    else:
        return None
    return str(stamps.tz) if stamps.tz is not None else None


def build_bars(
    ticks: pd.DataFrame,
    kind: str = "dollar",
    threshold: float | None = None,
    bars_per_day: int = DEFAULT_BARS_PER_DAY,
    verbose: bool = False,
    set_index: bool = True,
    restore_tz: bool = True,
    **prepare_kwargs,
) -> pd.DataFrame:
    """Build AFML bars of ``kind`` from a tick tape.

    ``threshold`` defaults to the value :func:`suggest_thresholds` derives for
    this specific tape, which is almost always what you want for a first pass.

    ``restore_tz`` puts the source timezone back on the bar index. fin-kit needs
    naive timestamps internally, but returning naive bars from IST-aware ticks
    makes every later join against the tape fail with "cannot join tz-naive with
    tz-aware", so by default the round trip is invisible.
    """
    valid_kinds = ("tick", "volume", "dollar", "rupee")
    if kind not in valid_kinds:
        raise ValueError(f"unknown bar kind {kind!r}; expected one of {sorted(valid_kinds)}")

    tz = source_timezone(ticks, prepare_kwargs.get("time_col")) if restore_tz else None

    frame = prepare_tick_frame(ticks, **prepare_kwargs)
    if frame.empty:
        raise ValueError("no usable ticks after cleaning")

    if threshold is None:
        threshold = suggest_thresholds(frame, bars_per_day)[
            "dollar" if kind == "rupee" else kind
        ]
    if kind == "tick":
        threshold = int(max(1, round(threshold)))

    logger.info(
        "Building %s bars from %d ticks with threshold %s", kind, len(frame), f"{threshold:,.0f}"
    )
    bars = _bar_builder(kind)(frame, threshold=threshold, verbose=verbose)

    if set_index and "date_time" in bars.columns:
        bars = bars.copy()
        bars["date_time"] = pd.to_datetime(bars["date_time"])
        bars = bars.set_index("date_time").sort_index()
        bars.index.name = "timestamp"
        if tz is not None:
            bars.index = bars.index.tz_localize(tz)
    return bars


def build_all_bars(
    ticks: pd.DataFrame,
    bars_per_day: int = DEFAULT_BARS_PER_DAY,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Tick, volume and rupee bars from one tape, all targeting the same frequency.

    Holding the target bar count fixed is what makes the three comparable: any
    difference in the return distribution is then attributable to the sampling
    clock rather than to sample size.
    """
    tz = source_timezone(ticks)
    frame = prepare_tick_frame(ticks)
    thresholds = suggest_thresholds(frame, bars_per_day)
    out = {}
    for kind in ("tick", "volume", "dollar"):
        bars = build_bars(
            frame,
            kind=kind,
            threshold=thresholds[kind],
            verbose=verbose,
            trades_only=False,
            restore_tz=False,
        )
        if tz is not None:
            bars.index = bars.index.tz_localize(tz)
        out[kind] = bars
    return out
