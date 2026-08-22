"""Bar construction, both the dependency-light path and the fin-kit adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml_india.bars.standard import (
    bars_from_ohlcv,
    suggest_threshold,
    tick_bars,
    time_bars,
    value_bars,
    volume_bars,
)


def test_tick_bars_have_the_requested_tick_count(tick_frame):
    bars = tick_bars(tick_frame, threshold=500)
    assert len(bars) == len(tick_frame) // 500
    assert (bars["ticks"] == 500).all()


def test_volume_bars_meet_their_threshold(tick_frame):
    bars = volume_bars(tick_frame, threshold=20_000)
    # Each closed bar must have reached the threshold; it may overshoot on the
    # tick that closes it, but it can never fall short.
    assert (bars["volume"] >= 20_000).all()


def test_value_bars_meet_their_threshold(tick_frame):
    bars = value_bars(tick_frame, threshold=5_000_000)
    assert (bars["value"] >= 5_000_000).all()


def test_ohlc_relationships_hold(tick_frame):
    for bars in (tick_bars(tick_frame, 400), volume_bars(tick_frame, 20_000)):
        assert (bars["high"] >= bars["low"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()
        assert (bars["low"] <= bars["open"]).all()


def test_vwap_lies_within_the_bar_range(tick_frame):
    bars = value_bars(tick_frame, 5_000_000)
    assert (bars["vwap"] >= bars["low"] - 1e-9).all()
    assert (bars["vwap"] <= bars["high"] + 1e-9).all()


def test_bars_preserve_the_source_timezone(tick_frame):
    assert str(tick_bars(tick_frame, 500).index.tz) == "Asia/Kolkata"


def test_bars_are_sorted_and_within_the_sample(tick_frame):
    bars = value_bars(tick_frame, 5_000_000)
    assert bars.index.is_monotonic_increasing
    assert bars.index[0] >= tick_frame.index[0]
    assert bars.index[-1] <= tick_frame.index[-1]


def test_suggest_threshold_hits_the_target_bar_count(tick_frame):
    target = 20
    threshold = suggest_threshold(tick_frame, kind="value", target_bars_per_day=target)
    bars = value_bars(tick_frame, threshold)
    per_day = len(bars) / tick_frame.index.normalize().nunique()
    assert target * 0.6 <= per_day <= target * 1.4


def test_unsorted_input_is_rejected(tick_frame):
    with pytest.raises(ValueError, match="sorted ascending"):
        tick_bars(tick_frame.iloc[::-1], 100)


def test_missing_columns_are_rejected(tick_frame):
    with pytest.raises(KeyError, match="missing columns"):
        tick_bars(tick_frame[["price"]], 100)


def test_empty_input_returns_an_empty_frame():
    empty = pd.DataFrame({"price": [], "volume": []}, index=pd.DatetimeIndex([], name="timestamp"))
    assert tick_bars(empty, 10).empty


def test_time_bars_respect_the_requested_frequency(tick_frame):
    bars = time_bars(tick_frame, "30min")
    gaps = pd.Series(bars.index).diff().dropna()
    assert gaps.min() >= pd.Timedelta("30min")


def test_bars_from_ohlcv_runs_without_a_tick_feed(ohlcv):
    bars = bars_from_ohlcv(ohlcv, kind="value", target_bars_per_day=1)
    assert len(bars) > 10
    assert (bars["high"] >= bars["low"]).all()


def test_activity_bars_track_turnover_and_time_bars_do_not(tick_frame):
    """The core AFML claim: bar counts should follow activity, not the clock.

    Stated as two separate assertions because the comparison cannot be made as a
    correlation on both sides — a time-bar count is constant across sessions by
    construction, so its correlation with turnover is undefined rather than low.
    """
    turnover = (
        (tick_frame["price"] * tick_frame["volume"]).groupby(tick_frame.index.normalize()).sum()
    )

    value_counts = (
        value_bars(tick_frame, suggest_threshold(tick_frame, "value", 20))
        .index.normalize()
        .value_counts()
        .sort_index()
    )
    clock_counts = time_bars(tick_frame, "20min").index.normalize().value_counts().sort_index()

    value_corr = np.corrcoef(value_counts.to_numpy(), turnover.reindex(value_counts.index))[0, 1]
    assert value_corr > 0.8, f"rupee bars should track turnover, got r={value_corr:.3f}"

    # Coefficient of variation, not correlation: the clock emits the same count
    # whether the session was busy or dead.
    clock_cv = clock_counts.std() / clock_counts.mean()
    value_cv = value_counts.std() / value_counts.mean()
    assert clock_cv < 0.05
    assert value_cv > clock_cv
