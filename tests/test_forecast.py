"""Fractional differencing inversion and one-step-ahead forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml_india.models.forecast import (
    ForecastConfig,
    FracDiffTransformer,
    expected_returns,
    forecast_errors,
)

torch = pytest.importorskip("torch", reason="the forecaster needs PyTorch")


def test_ffd_weights_lead_with_one():
    transformer = FracDiffTransformer(d=0.4, thresh=1e-3)
    assert transformer.weights[-1] == pytest.approx(1.0)
    assert transformer.width > 1


def test_lower_d_needs_a_longer_window():
    short = FracDiffTransformer(d=0.9, thresh=1e-3).width
    long = FracDiffTransformer(d=0.2, thresh=1e-3).width
    assert long > short


def test_inversion_is_exact(price_series):
    """The property the whole levels-vs-fracdiff comparison rests on."""
    transformer = FracDiffTransformer(d=0.4, thresh=1e-3)
    differenced = transformer.transform(price_series)

    errors = []
    for position in range(transformer.width, len(price_series)):
        stamp = price_series.index[position]
        if stamp not in differenced.index:
            continue
        recovered = transformer.invert_next(price_series.iloc[:position], differenced.loc[stamp])
        errors.append(abs(recovered - price_series.iloc[position]))

    assert len(errors) > 100
    assert max(errors) < 1e-8


def test_inversion_is_exact_for_several_orders(price_series):
    for d in (0.1, 0.5, 1.0):
        transformer = FracDiffTransformer(d=d, thresh=1e-3)
        differenced = transformer.transform(price_series)
        stamp = differenced.index[-1]
        position = price_series.index.get_loc(stamp)
        recovered = transformer.invert_next(price_series.iloc[:position], differenced.loc[stamp])
        assert recovered == pytest.approx(float(price_series.iloc[position]), rel=1e-10)


def test_lower_d_retains_more_memory(price_series):
    low = FracDiffTransformer(d=0.2, thresh=1e-3).memory_retained(price_series)
    high = FracDiffTransformer(d=0.9, thresh=1e-3).memory_retained(price_series)
    assert low > high


def test_d_one_is_stationary_and_forgetful(price_series):
    from statsmodels.tsa.stattools import adfuller

    differenced = FracDiffTransformer(d=1.0, thresh=1e-3).transform(price_series)
    assert adfuller(differenced, maxlag=1, autolag=None)[1] < 0.01
    assert FracDiffTransformer(d=1.0, thresh=1e-3).memory_retained(price_series) < 0.2


def test_log_space_rejects_non_positive_prices():
    series = pd.Series([1.0, 2.0, -1.0], index=pd.date_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="strictly positive"):
        FracDiffTransformer(d=0.4, thresh=1e-1).transform(series)


@pytest.mark.slow
def test_walk_forward_uses_realised_history_only(price_series):
    """A one-step forecast must not change when later data is removed."""
    from afml_india.models.forecast import walk_forward_forecast

    config = ForecastConfig(lookback=8, max_epochs=25, patience=8, seed=0)
    monthly = price_series.resample("ME").mean()
    split = monthly.index[int(len(monthly) * 0.7)]

    full = walk_forward_forecast(monthly, split, config)
    truncated = walk_forward_forecast(monthly.iloc[:-5], split, config)

    shared = full.index.intersection(truncated.index)
    assert len(shared) > 5
    # Same fitted model, same realised inputs, therefore identical predictions.
    np.testing.assert_allclose(full.loc[shared], truncated.loc[shared], rtol=1e-6)


def test_expected_returns_uses_the_previous_price():
    index = pd.date_range("2024-01-31", periods=4, freq="ME")
    prices = pd.Series([100.0, 110.0, 120.0, 130.0], index=index)
    forecast = pd.Series([115.0, 125.0], index=index[2:])
    result = expected_returns(forecast, prices)
    assert result.iloc[0] == pytest.approx(115.0 / 110.0 - 1.0)
    assert result.iloc[1] == pytest.approx(125.0 / 120.0 - 1.0)


def test_flat_forecast_has_undefined_direction():
    index = pd.date_range("2024-01-31", periods=10, freq="ME")
    actual = pd.Series(np.linspace(100, 120, 10), index=index)
    flat = pd.Series(100.0, index=index)
    assert np.isnan(forecast_errors(actual, flat)["directional_accuracy"])


def test_perfect_forecast_scores_perfectly():
    index = pd.date_range("2024-01-31", periods=10, freq="ME")
    actual = pd.Series(np.linspace(100, 120, 10), index=index)
    errors = forecast_errors(actual, actual.copy())
    assert errors["rmse"] == pytest.approx(0.0)
    assert errors["directional_accuracy"] == pytest.approx(1.0)
