"""End-to-end pipeline smoke tests against a generated snapshot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from afml_india import finkit

pytestmark = pytest.mark.skipif(not finkit.available(), reason="fin-kit is not on the path")


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    """Build a small synthetic snapshot by running the real fetch script."""
    out = tmp_path_factory.mktemp("snapshots")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_bloomberg_snapshot.py",
            "--offline-demo",
            "--members",
            "2",
            "--tick-days",
            "40",
            "--out",
            str(out),
            "--seed",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    if result.returncode != 0:
        pytest.fail(f"snapshot generation failed:\n{result.stdout}\n{result.stderr}")

    from afml_india.data.snapshot import Snapshot, find_snapshot

    return Snapshot(find_snapshot(out))


def test_snapshot_round_trips(snapshot):
    assert snapshot.is_synthetic
    assert len(snapshot.members) == 2
    assert snapshot.manifest["index"] == "NIFTY Index"

    ticker = snapshot.tickers[0]
    ticks = snapshot.ticks(ticker, "trades")
    assert len(ticks) > 1000
    assert str(ticks.index.tz) == "Asia/Kolkata"
    assert ticks.index.is_monotonic_increasing
    assert (ticks["type"] == "TRADE").all()


def test_quotes_and_trades_are_separate_streams(snapshot):
    ticker = snapshot.tickers[0]
    assert set(snapshot.ticks(ticker, "quotes")["type"].unique()) <= {"BID", "ASK"}


def test_ticker_lookup_accepts_every_spelling(snapshot):
    ticker = snapshot.tickers[0]
    for spelling in (ticker, ticker.replace("_", " "), f"{ticker.replace('_', ' ')} Equity"):
        assert len(snapshot.ticks(spelling, "trades")) > 0


@pytest.mark.slow
def test_pipeline_prepares_a_trainable_dataset(snapshot):
    from afml_india.features import FeatureConfig
    from afml_india.pipeline import PipelineConfig, TickerPipeline

    config = PipelineConfig(
        bars_per_day=50,
        cusum_multiple=0.5,
        vol_lookback=50,
        holding_days=2,
        lookback=12,
        min_train=100,
        n_splits=3,
        features=FeatureConfig(momentum_windows=(5, 20), volatility_windows=(20,)),
    )
    data = TickerPipeline(config).prepare(snapshot.tickers[0], snapshot=snapshot)

    assert data.X.ndim == 3 and data.X.shape[1] == 12
    assert len(data.X) == len(data.y) == len(data.spans)
    assert set(data.y) <= {-1.0, 1.0}
    assert 0.0 < data.d <= 1.0
    # Uniqueness weights must be a real correction, not all ones.
    assert data.sample_weight is not None
    assert 0.0 < data.sample_weight.mean() < 1.0
    # Every sample's span must start at or before its event and end at or after.
    assert (data.spans["start"] <= data.spans.index).all()
    assert (data.spans["end"] >= data.spans.index).all()


@pytest.mark.slow
def test_pipeline_scores_a_baseline(snapshot):
    from afml_india.features import FeatureConfig
    from afml_india.models import random_forest
    from afml_india.pipeline import PipelineConfig, TickerPipeline

    config = PipelineConfig(
        bars_per_day=50,
        cusum_multiple=0.5,
        vol_lookback=50,
        holding_days=2,
        lookback=12,
        min_train=100,
        n_splits=3,
        features=FeatureConfig(momentum_windows=(5, 20), volatility_windows=(20,)),
    )
    pipeline = TickerPipeline(config)
    result = pipeline.evaluate(
        pipeline.prepare(snapshot.tickers[0], snapshot=snapshot),
        make_model=lambda: random_forest(seed=0),
    )
    assert result.fold_metrics is not None and len(result.fold_metrics) >= 1
    assert result.fold_metrics["accuracy"].between(0, 1).all()
    # Predictions are out of sample and each appears exactly once.
    assert not result.predictions.index.duplicated().any()
    assert result.predictions["proba"].between(0, 1).all()


@pytest.mark.slow
def test_a_shuffled_label_scores_no_better_than_chance(snapshot):
    """The strongest negative control available: destroy the signal, expect nothing."""
    import numpy as np

    from afml_india.features import FeatureConfig
    from afml_india.models import random_forest
    from afml_india.models.evaluate import walk_forward_evaluate
    from afml_india.pipeline import PipelineConfig, TickerPipeline

    config = PipelineConfig(
        bars_per_day=50,
        cusum_multiple=0.5,
        vol_lookback=50,
        holding_days=2,
        lookback=12,
        min_train=100,
        n_splits=3,
        features=FeatureConfig(momentum_windows=(5, 20), volatility_windows=(20,)),
    )
    data = TickerPipeline(config).prepare(snapshot.tickers[0], snapshot=snapshot)

    shuffled = np.random.default_rng(0).permutation(data.y)
    folds, _ = walk_forward_evaluate(
        lambda: random_forest(seed=0),
        data.X,
        shuffled,
        data.spans,
    )
    # A leak would let the model recover a shuffled label; it must not.
    assert folds["accuracy"].mean() < 0.62
