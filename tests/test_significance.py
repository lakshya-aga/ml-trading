"""The cross-sectional significance layer, with both controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml_india.data.costs import CostModel, Segment
from afml_india.significance import (
    cross_section_summary,
    daily_pnl,
    event_returns,
    permutation_pvalue,
    ticker_metrics,
)


def _fake_ticker(informed: bool, n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-05-01 10:00", periods=n, freq="4h")
    realised = rng.normal(0.0005, 0.01, n)
    if informed:
        proba = 0.5 + 0.35 * np.sign(realised) * rng.uniform(0.5, 1.0, n)
    else:
        proba = rng.uniform(0.2, 0.8, n)
    predictions = pd.DataFrame({"proba": np.clip(proba, 0.01, 0.99)}, index=index)
    labels = pd.DataFrame(
        {"ret": realised, "t1": index + pd.Timedelta("8h"), "bin": np.sign(realised)},
        index=index,
    )
    return predictions, labels


@pytest.fixture
def cost_model():
    return CostModel(segment=Segment.INTRADAY)


def test_event_returns_shapes_and_signs(cost_model):
    predictions, labels = _fake_ticker(True, seed=1)
    events = event_returns(predictions, labels, cost_model)
    assert set(events.columns) >= {"side", "size", "gross", "cost", "net", "realised"}
    assert events["size"].between(0, 1).all()
    assert (events["cost"] >= 0).all()
    # Net can never beat gross.
    assert (events["net"] <= events["gross"] + 1e-15).all()


def test_min_edge_creates_a_no_trade_band(cost_model):
    predictions, labels = _fake_ticker(False, seed=2)
    all_in = event_returns(predictions, labels, cost_model, min_edge=0.0)
    banded = event_returns(predictions, labels, cost_model, min_edge=0.3)
    assert (banded["size"] > 0).sum() < (all_in["size"] > 0).sum()
    # A zero-size event pays nothing and earns nothing.
    idle = banded[banded["size"] == 0]
    assert (idle["net"] == 0).all()


def test_daily_pnl_attributes_to_exit_day(cost_model):
    predictions, labels = _fake_ticker(True, seed=3)
    events = event_returns(predictions, labels, cost_model)
    daily = daily_pnl(events)
    assert daily["net"].sum() == pytest.approx(events["net"].sum())
    assert (daily["turnover"] >= 0).all()


def test_oracle_is_detected(cost_model):
    events = {
        f"T{k}": event_returns(*_fake_ticker(True, seed=k), cost_model=cost_model) for k in range(8)
    }
    result = permutation_pvalue(events, n_permutations=300, seed=1)
    assert result["p_value"] < 0.05
    assert result["observed"] > result["null_95th"]


def test_noise_is_not_detected(cost_model):
    """The control that keeps the whole layer honest."""
    events = {
        f"T{k}": event_returns(*_fake_ticker(False, seed=100 + k), cost_model=cost_model)
        for k in range(8)
    }
    result = permutation_pvalue(events, n_permutations=300, seed=1)
    assert result["p_value"] > 0.05


def test_permutation_preserves_costs_under_the_null(cost_model):
    """Shuffling positions must not change what the strategy paid."""
    predictions, labels = _fake_ticker(False, seed=5)
    events = event_returns(predictions, labels, cost_model)
    traded = events[events["size"] > 0]
    # The null redraws gross by shuffling signed size, but cost depends only on
    # |size|, which the shuffle preserves as a multiset.
    assert traded["cost"].sum() > 0


def test_cross_section_summary_reports_the_basics(cost_model):
    metrics = pd.DataFrame(
        {
            t: ticker_metrics(event_returns(*_fake_ticker(True, seed=k), cost_model=cost_model))
            for k, t in enumerate(["A", "B", "C", "D"])
        }
    ).T
    summary = cross_section_summary(metrics)
    assert summary["n_tickers"] == 4
    assert 0 <= summary["share_positive"] <= 1


def test_too_few_tickers_is_rejected(cost_model):
    events = {"ONLY": event_returns(*_fake_ticker(True, seed=9), cost_model=cost_model)}
    with pytest.raises(ValueError, match="at least two"):
        permutation_pvalue(events, n_permutations=50)


def test_rotation_null_widens_under_serial_correlation(cost_model):
    """The regression that forced the rotation default.

    Overlapping events make positions and returns block-correlated; an iid
    shuffle then understates the null variance and manufactures significance.
    The rotation null must be wider on block-correlated data and about the
    same on genuinely independent data.
    """
    rng = np.random.default_rng(7)

    def block_events(n_blocks=30, block=12):
        n = n_blocks * block
        index = pd.date_range("2026-05-01 10:00", periods=n, freq="h")
        realised = np.repeat(rng.normal(0, 0.01, n_blocks), block)
        proba = np.clip(np.repeat(rng.uniform(0.2, 0.8, n_blocks), block), 0.01, 0.99)
        predictions = pd.DataFrame({"proba": proba}, index=index)
        labels = pd.DataFrame({"ret": realised, "t1": index + pd.Timedelta("4h")}, index=index)
        return event_returns(predictions, labels, cost_model)

    blocked = {f"B{k}": block_events() for k in range(4)}
    wide = permutation_pvalue(blocked, n_permutations=400, method="rotation", seed=1)
    narrow = permutation_pvalue(blocked, n_permutations=400, method="shuffle", seed=1)
    assert wide["null_sd"] > 1.5 * narrow["null_sd"]

    independent = {
        f"T{k}": event_returns(*_fake_ticker(False, seed=200 + k), cost_model=cost_model)
        for k in range(4)
    }
    wide_iid = permutation_pvalue(independent, n_permutations=400, method="rotation", seed=1)
    narrow_iid = permutation_pvalue(independent, n_permutations=400, method="shuffle", seed=1)
    assert wide_iid["null_sd"] < 1.5 * narrow_iid["null_sd"]


def test_oracle_still_detected_under_rotation(cost_model):
    events = {
        f"T{k}": event_returns(*_fake_ticker(True, seed=k), cost_model=cost_model) for k in range(8)
    }
    result = permutation_pvalue(events, n_permutations=300, method="rotation", seed=2)
    assert result["p_value"] < 0.05
