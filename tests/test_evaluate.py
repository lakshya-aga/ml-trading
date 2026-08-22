"""Purged walk-forward splitting — the part that decides whether a score means anything."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml_india.models.evaluate import WalkForwardSplit, classification_metrics
from afml_india.models.sequences import SequenceScaler, make_sequences, sequence_span


@pytest.fixture
def spans() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    # Each sample spans backwards 5 bars and forwards 20.
    return pd.DataFrame(
        {"start": index - pd.Timedelta(hours=5), "end": index + pd.Timedelta(hours=20)},
        index=index,
    )


def test_splits_are_chronological(spans):
    splitter = WalkForwardSplit(n_splits=4, min_train=80, embargo=0.0)
    previous_end = -1
    for train_idx, test_idx in splitter.split(spans):
        assert train_idx.max() < test_idx.min()  # never trains on the future
        assert test_idx.min() > previous_end  # folds march forward
        previous_end = test_idx.max()


def test_no_training_sample_overlaps_the_test_window(spans):
    """The whole point of purging, asserted directly."""
    splitter = WalkForwardSplit(n_splits=4, min_train=80, embargo=0.0)
    starts, ends = spans["start"].to_numpy(), spans["end"].to_numpy()

    for train_idx, test_idx in splitter.split(spans):
        window_start, window_end = starts[test_idx].min(), ends[test_idx].max()
        overlapping = (ends[train_idx] >= window_start) & (starts[train_idx] <= window_end)
        assert not overlapping.any()


def test_purging_removes_samples_a_naive_split_would_keep(spans):
    splitter = WalkForwardSplit(n_splits=4, min_train=80, embargo=0.0)
    for train_idx, test_idx in splitter.split(spans):
        assert len(train_idx) < test_idx[0]  # a naive split would take everything before
        break


def test_embargo_widens_the_gap(spans):
    without = WalkForwardSplit(n_splits=3, min_train=80, embargo=0.0)
    with_embargo = WalkForwardSplit(n_splits=3, min_train=80, embargo=0.10)
    a = next(iter(without.split(spans)))[0]
    b = next(iter(with_embargo.split(spans)))[0]
    assert len(b) < len(a)


def test_rolling_window_trains_on_less_than_expanding(spans):
    expanding = list(WalkForwardSplit(n_splits=3, min_train=80, expanding=True).split(spans))
    rolling = list(
        WalkForwardSplit(n_splits=3, min_train=80, expanding=False, train_window=150).split(spans)
    )
    assert expanding and rolling
    assert len(expanding[-1][0]) > len(rolling[-1][0])


def test_rolling_window_length_is_independent_of_min_train(spans):
    """train_window sizes the window; min_train only decides acceptance.

    Conflating the two makes rolling mode unusable: purging eats into the
    window, so a window equal to min_train can never clear it.
    """
    splits = list(
        WalkForwardSplit(n_splits=3, min_train=50, expanding=False, train_window=200).split(spans)
    )
    assert splits, "rolling mode produced no usable folds"
    for train_idx, _ in splits:
        assert len(train_idx) <= 200
        assert len(train_idx) >= 50


def test_too_few_samples_is_rejected():
    tiny = pd.DataFrame(
        {
            "start": pd.date_range("2024-01-01", periods=4),
            "end": pd.date_range("2024-01-02", periods=4),
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    with pytest.raises(ValueError, match="at least"):
        list(WalkForwardSplit(n_splits=5).split(tiny))


def test_spans_require_start_and_end():
    frame = pd.DataFrame({"start": pd.date_range("2024-01-01", periods=10)})
    frame.index = frame["start"]
    with pytest.raises(KeyError, match="end"):
        list(WalkForwardSplit().split(frame))


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #


def test_sequences_end_at_the_event_bar():
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    features = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2}, index=index)
    labels = pd.Series(1, index=index[50:60])

    X, y, kept = make_sequences(features, labels, lookback=10)
    assert X.shape == (10, 10, 2)
    # The final timestep of each window must be the event bar itself.
    for position, stamp in enumerate(kept):
        assert X[position, -1, 0] == pytest.approx(float(features.loc[stamp, "a"]))


def test_labels_without_a_full_window_are_dropped():
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    features = pd.DataFrame({"a": np.arange(50.0)}, index=index)
    labels = pd.Series(1, index=index[:20])
    _, _, kept = make_sequences(features, labels, lookback=10)
    assert kept[0] == index[9]  # the first bar with 10 bars of history


def test_sequence_span_covers_input_and_label():
    bars = pd.date_range("2024-01-01", periods=100, freq="h")
    samples = bars[50:55]
    label_end = pd.Series(samples + pd.Timedelta(hours=20), index=samples)
    spans = sequence_span(samples, bars, lookback=10, label_end=label_end)

    assert (spans["start"] == samples - pd.Timedelta(hours=9)).all()  # backwards
    assert (spans["end"] == label_end).all()  # forwards


def test_scaler_uses_training_statistics_only():
    train = np.random.default_rng(0).normal(0, 1, (100, 8, 3)).astype(np.float32)
    test = train * 10 + 50

    scaler = SequenceScaler(clip=None).fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    assert abs(scaled_train.reshape(-1, 3).mean()) < 0.1
    # Test data is not re-centred; if it were, the scaler had seen it.
    assert abs(scaled_test.reshape(-1, 3).mean()) > 5


def test_scaler_handles_constant_features_and_nans():
    values = np.ones((20, 5, 2), dtype=np.float32)
    values[0, 0, 0] = np.nan
    scaled = SequenceScaler().fit_transform(values)
    assert np.isfinite(scaled).all()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_on_a_perfect_classifier():
    y = np.array([1, 1, -1, -1])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    metrics = classification_metrics(y, proba)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_metrics_survive_a_single_class():
    metrics = classification_metrics(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))
    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == pytest.approx(1.0)
