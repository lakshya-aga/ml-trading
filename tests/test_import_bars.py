"""Importing free intraday data into the snapshot format."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_snapshot_from_bars", ROOT / "scripts" / "build_snapshot_from_bars.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_snapshot_from_bars"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _minute_frame(seed: int, px0: float, days: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2026-06-01", periods=days)
    rows = []
    level = px0
    for day in sessions:
        stamps = pd.date_range(f"{day:%Y-%m-%d} 09:15", f"{day:%Y-%m-%d} 15:29", freq="1min")
        n = len(stamps)
        close = level * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
        level = float(close[-1])
        open_ = close * (1 + rng.normal(0, 0.0004, n))
        # Asymmetric wicks, so the typical price is genuinely distinct from the
        # close — a symmetric bar makes the two identical and hides real bugs.
        upper = np.abs(rng.normal(0, 0.0012, n)) * close
        lower = np.abs(rng.normal(0, 0.0006, n)) * close
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": stamps,
                    "open": open_,
                    "high": np.maximum(open_, close) + upper,
                    "low": np.minimum(open_, close) - lower,
                    "close": close,
                    "volume": rng.integers(100, 5000, n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #


def test_kaggle_date_time_pair_is_understood(script):
    frame = _minute_frame(1, 1000.0, days=2)
    kaggle = pd.DataFrame(
        {
            "Date": frame["timestamp"].dt.strftime("%Y-%m-%d"),
            "Time": frame["timestamp"].dt.strftime("%H:%M:%S"),
            "Open": frame["open"],
            "High": frame["high"],
            "Low": frame["low"],
            "Close": frame["close"],
            "Volume": frame["volume"],
        }
    )
    out = script.normalise(kaggle, "kaggle.csv")
    assert len(out) == len(frame)
    assert str(out["timestamp"].dt.tz) == "Asia/Kolkata"


def test_abbreviated_ohlcv_columns_are_understood(script):
    frame = _minute_frame(2, 500.0, days=2)
    terse = frame.rename(
        columns={
            "timestamp": "datetime",
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "volume": "v",
        }
    )
    assert len(script.normalise(terse, "terse.csv")) == len(frame)


def test_a_missing_timestamp_column_is_reported_not_guessed(script):
    frame = _minute_frame(3, 100.0, days=1).drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="no timestamp column"):
        script.normalise(frame, "bad.csv")


def test_missing_ohlcv_is_reported(script):
    frame = _minute_frame(4, 100.0, days=1).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        script.normalise(frame, "bad.csv")


def test_naive_timestamps_are_treated_as_ist(script):
    frame = _minute_frame(5, 100.0, days=1)
    out = script.normalise(frame, "x.csv")
    assert out["timestamp"].iloc[0].hour == 9
    assert out["timestamp"].iloc[0].minute == 15


def test_interval_is_inferred(script):
    assert script.infer_interval(script.normalise(_minute_frame(6, 100.0, 2), "x")) == "1min"


# --------------------------------------------------------------------------- #
# The long-format regression
# --------------------------------------------------------------------------- #


def test_long_format_keeps_every_symbol(script, tmp_path):
    """Symbols share a minute grid, so deduplicating before splitting loses data."""
    frames = []
    for symbol, seed, px in (("AAA", 7, 100.0), ("BBB", 8, 250.0), ("CCC", 9, 900.0)):
        frame = _minute_frame(seed, px, days=4)
        frame["instrument"] = symbol
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    expected = len(frames[0])

    path = tmp_path / "long.csv"
    combined.to_csv(path, index=False)

    sources = script.load_sources(
        path, long_format=True, yahoo=None, interval="1m", start=None, end=None
    )
    assert set(sources) == {"AAA", "BBB", "CCC"}
    for symbol, frame in sources.items():
        assert len(frame) == expected, f"{symbol} lost rows to cross-symbol dedup"


def test_long_format_without_a_symbol_column_is_rejected(script, tmp_path):
    path = tmp_path / "nosym.csv"
    _minute_frame(10, 100.0, days=1).to_csv(path, index=False)
    with pytest.raises(ValueError, match="needs a symbol column"):
        script.load_sources(path, long_format=True, yahoo=None, interval="1m", start=None, end=None)


def test_a_directory_of_per_symbol_files(script, tmp_path):
    directory = tmp_path / "minute"
    directory.mkdir()
    for symbol, seed, px in (("RELIANCE", 11, 1400.0), ("INFY", 12, 1600.0)):
        _minute_frame(seed, px, days=3).to_csv(directory / f"{symbol}.csv", index=False)

    sources = script.load_sources(
        directory, long_format=False, yahoo=None, interval="1m", start=None, end=None
    )
    assert set(sources) == {"RELIANCE", "INFY"}


def test_one_unreadable_file_does_not_kill_the_batch(script, tmp_path):
    directory = tmp_path / "minute"
    directory.mkdir()
    _minute_frame(13, 100.0, days=2).to_csv(directory / "GOOD.csv", index=False)
    (directory / "BAD.csv").write_text("nonsense,columns\n1,2\n")

    sources = script.load_sources(
        directory, long_format=False, yahoo=None, interval="1m", start=None, end=None
    )
    assert set(sources) == {"GOOD"}


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def imported_snapshot(tmp_path_factory):
    """Build a snapshot from minute CSVs by running the script for real."""
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("snapshots")
    for symbol, seed, px in (("RELIANCE", 21, 1400.0), ("INFY", 22, 1600.0)):
        _minute_frame(seed, px, days=45).to_csv(raw / f"{symbol}.csv", index=False)

    result = subprocess.run(
        [sys.executable, "scripts/build_snapshot_from_bars.py", str(raw), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"import failed:\n{result.stdout}\n{result.stderr}")

    from afml_india.data.snapshot import Snapshot, find_snapshot

    return Snapshot(find_snapshot(out))


def test_imported_snapshot_is_readable(imported_snapshot):
    snap = imported_snapshot
    assert set(snap.tickers) == {"RELIANCE", "INFY"}
    assert snap.has_intraday_bars()
    assert snap.manifest["mode"] == "imported-bars"
    assert snap.manifest["intraday_bar_interval"] == "1min"
    # The manifest must say these are bars, not ticks.
    assert "not ticks" in snap.manifest["tick_window"]["note"]


def test_daily_is_resampled_from_the_minute_bars(imported_snapshot):
    daily = imported_snapshot.daily("RELIANCE")
    bars = imported_snapshot.intraday_bars("RELIANCE")
    assert len(daily) == bars.index.normalize().nunique()
    assert (daily["px_high"] >= daily["px_low"]).all()
    assert daily["px_volume"].sum() == pytest.approx(bars["volume"].sum(), rel=1e-9)


def test_pseudo_ticks_feed_the_bar_builders(imported_snapshot):
    from afml_india.bars.finkit_bars import build_all_bars

    ticks = imported_snapshot.pseudo_ticks("RELIANCE")
    assert str(ticks.index.tz) == "Asia/Kolkata"
    assert (ticks["volume"] > 0).all()

    bars = build_all_bars(ticks, bars_per_day=20)
    for kind, frame in bars.items():
        assert len(frame) > 0, kind
        assert (frame["high"] >= frame["low"]).all()
        assert str(frame.index.tz) == "Asia/Kolkata"


def test_bar_resolution_cannot_beat_the_source(imported_snapshot):
    """A hard floor worth asserting: minute input cannot make sub-minute bars."""
    from afml_india.bars.finkit_bars import build_bars

    ticks = imported_snapshot.pseudo_ticks("RELIANCE")
    # An absurdly low threshold would make one bar per input row, no finer.
    bars = build_bars(ticks, kind="dollar", threshold=1.0)
    assert len(bars) <= len(ticks)
    gaps = pd.Series(bars.index).diff().dropna()
    assert gaps.min() >= pd.Timedelta("1min")


def test_pseudo_tick_price_modes(imported_snapshot):
    typical = imported_snapshot.pseudo_ticks("RELIANCE", price="typical")
    close = imported_snapshot.pseudo_ticks("RELIANCE", price="close")
    bars = imported_snapshot.intraday_bars("RELIANCE")
    assert close["price"].iloc[0] == pytest.approx(bars["close"].iloc[0])
    assert not np.allclose(typical["price"], close["price"])


@pytest.mark.slow
def test_full_pipeline_runs_on_imported_minute_data(imported_snapshot):
    from afml_india.features import FeatureConfig
    from afml_india.models import random_forest
    from afml_india.pipeline import PipelineConfig, TickerPipeline

    config = PipelineConfig(
        bars_per_day=25,
        cusum_multiple=0.4,
        vol_lookback=30,
        holding_days=2,
        lookback=10,
        min_train=60,
        n_splits=2,
        features=FeatureConfig(momentum_windows=(5, 20), volatility_windows=(20,)),
    )
    ticks = imported_snapshot.pseudo_ticks("RELIANCE")
    result = TickerPipeline(config).run(
        "RELIANCE", make_model=lambda: random_forest(seed=0), ticks=ticks
    )
    assert result.fold_metrics is not None and len(result.fold_metrics) >= 1
    assert result.fold_metrics["accuracy"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# Yahoo loader (offline: the chunking logic, not the network)
# --------------------------------------------------------------------------- #


def test_yahoo_rejects_an_unknown_interval():
    from afml_india.data.free_sources import yahoo_intraday

    pytest.importorskip("yfinance")
    with pytest.raises(ValueError, match="unknown interval"):
        yahoo_intraday("RELIANCE", interval="3m")


def test_yahoo_chunks_a_1m_request_into_legal_spans(monkeypatch):
    """The whole point of the loader: one 30-day 1m request returns nothing."""
    pytest.importorskip("yfinance")
    import yfinance

    from afml_india.data import free_sources

    calls: list[tuple[str, str]] = []

    def fake_download(ticker, start=None, end=None, interval=None, **kwargs):
        calls.append((start, end))
        index = pd.date_range(f"{start} 09:15", periods=5, freq="1min", tz="Asia/Kolkata")
        return pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 10},
            index=index,
        )

    monkeypatch.setattr(yfinance, "download", fake_download)
    free_sources.yahoo_intraday("RELIANCE", interval="1m")

    assert len(calls) > 1, "a 30-day 1m window must be split into chunks"
    for start, end in calls:
        span = (pd.Timestamp(end) - pd.Timestamp(start)).days
        assert span <= 7, f"chunk {start}..{end} exceeds Yahoo's 7-day 1m limit"
