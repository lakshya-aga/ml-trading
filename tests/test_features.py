"""Feature construction and the look-ahead guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml_india.features import (
    FeatureConfig,
    assert_no_lookahead,
    momentum_features,
    time_features,
    volatility_features,
)


def test_momentum_features_are_backward_looking(ohlcv):
    features = momentum_features(ohlcv, windows=(5, 20))
    # A w-window feature cannot be defined before w observations exist.
    assert features["ret_20"].iloc[:20].isna().all()
    assert features["ret_20"].iloc[20:].notna().any()


def test_volatility_estimators_are_non_negative(ohlcv):
    features = volatility_features(ohlcv, windows=(20,))
    for col in ("parkinson_20", "garman_klass_20"):
        assert (features[col].dropna() >= 0).all()


def test_volatility_features_need_ohlc(ohlcv):
    with pytest.raises(KeyError):
        volatility_features(ohlcv[["close"]], windows=(20,))


def test_session_features_are_bounded(ohlcv):
    features = time_features(ohlcv)
    assert features["session_progress"].between(0, 1).all()
    assert features["session_sin"].between(-1, 1).all()
    assert features["session_cos"].between(-1, 1).all()


def test_feature_matrix_has_no_nan_or_inf(ohlcv):
    features = FeatureConfig(d=0.4, fracdiff_thresh=1e-3).build(ohlcv)
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy()).all()


def test_fracdiff_block_is_optional(ohlcv):
    with_fd = FeatureConfig(d=0.4, fracdiff_thresh=1e-3).build(ohlcv)
    without = FeatureConfig(d=None).build(ohlcv)
    assert any(c.startswith("fd_") for c in with_fd.columns)
    assert not any(c.startswith("fd_") for c in without.columns)


def test_no_lookahead_in_the_default_recipe(ohlcv):
    """The guarantee the whole framework rests on."""
    assert_no_lookahead(FeatureConfig(d=0.4, fracdiff_thresh=1e-3), ohlcv, n_checks=6)


def test_lookahead_check_catches_a_deliberate_leak(ohlcv, monkeypatch):
    """A negative control: the check must fail on a feature that peeks ahead."""
    import afml_india.features as features_module

    clean = features_module.momentum_features

    def leaky(bars, windows=(5, 20, 50), price_col="close"):
        out = clean(bars, windows, price_col)
        # A centred rolling mean reads the future. This is the classic silent leak.
        out["leak"] = bars[price_col].rolling(11, center=True).mean()
        return out

    monkeypatch.setattr(features_module, "momentum_features", leaky)
    with pytest.raises(AssertionError, match="look-ahead detected"):
        assert_no_lookahead(FeatureConfig(d=None), ohlcv, n_checks=6)
