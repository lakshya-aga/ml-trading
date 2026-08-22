"""Adapters between Indian tick data and fin-kit's AFML bar builders."""

from __future__ import annotations

import pandas as pd
import pytest

from afml_india import finkit

pytestmark = pytest.mark.skipif(not finkit.available(), reason="fin-kit is not on the path")


def test_prepare_tick_frame_produces_the_expected_schema(tick_frame):
    from afml_india.bars.finkit_bars import prepare_tick_frame

    prepared = prepare_tick_frame(tick_frame)
    assert list(prepared.columns) == ["date_time", "price", "volume"]
    assert prepared["date_time"].dt.tz is None  # fin-kit needs naive stamps
    assert prepared["date_time"].is_monotonic_increasing


def test_quotes_are_dropped_from_the_trade_tape(tick_frame):
    from afml_india.bars.finkit_bars import prepare_tick_frame

    quotes = tick_frame.head(100).copy()
    quotes["type"] = "BID"
    mixed = pd.concat([tick_frame, quotes]).sort_index()
    prepared = prepare_tick_frame(mixed, trades_only=True)
    assert len(prepared) == len(tick_frame)


def test_non_positive_rows_are_dropped(tick_frame):
    from afml_india.bars.finkit_bars import prepare_tick_frame

    dirty = tick_frame.copy()
    dirty.iloc[0, dirty.columns.get_loc("price")] = 0.0
    dirty.iloc[1, dirty.columns.get_loc("volume")] = 0
    assert len(prepare_tick_frame(dirty)) == len(tick_frame) - 2


def test_suggested_thresholds_hit_the_target(tick_frame):
    from afml_india.bars.finkit_bars import build_bars, suggest_thresholds

    thresholds = suggest_thresholds(tick_frame, bars_per_day=20)
    assert set(thresholds) == {"tick", "volume", "dollar"}
    bars = build_bars(tick_frame, kind="dollar", threshold=thresholds["dollar"])
    per_day = len(bars) / tick_frame.index.normalize().nunique()
    assert 12 <= per_day <= 30


def test_build_all_bars_restores_the_source_timezone(tick_frame):
    from afml_india.bars.finkit_bars import build_all_bars

    bars = build_all_bars(tick_frame, bars_per_day=20)
    assert set(bars) == {"tick", "volume", "dollar"}
    for frame in bars.values():
        assert str(frame.index.tz) == "Asia/Kolkata"
        assert frame.index.is_monotonic_increasing


def test_bars_join_cleanly_against_the_source_ticks(tick_frame):
    """The regression this guards: naive bars against tz-aware ticks raise on join."""
    from afml_india.bars.finkit_bars import build_bars

    bars = build_bars(tick_frame, kind="dollar", bars_per_day=20)
    turnover = (tick_frame["price"] * tick_frame["volume"]).groupby(
        tick_frame.index.normalize()
    ).sum()
    counts = bars.index.normalize().value_counts().sort_index()
    joined = counts.to_frame("bars").join(turnover.rename("turnover"), how="inner")
    assert len(joined) > 0


def test_unknown_bar_kind_is_rejected(tick_frame):
    from afml_india.bars.finkit_bars import build_bars

    with pytest.raises(ValueError, match="unknown bar kind"):
        build_bars(tick_frame, kind="nonsense")
