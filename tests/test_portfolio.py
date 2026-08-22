"""Allocation, backtesting and attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml_india.portfolio import (
    annualised_sharpe,
    attribution,
    backtest_signals,
    concentration,
    decompose_vs_equal_weight,
    default_cost_model,
    equal_weights,
    max_drawdown,
    project_sharpe,
    roy_allocation,
    roy_ratio,
    signal_weights,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    generator = np.random.default_rng(5)
    index = pd.date_range("2020-01-31", periods=60, freq="ME")
    return pd.DataFrame(
        {
            symbol: 100 * np.exp(np.cumsum(generator.normal(0.005, 0.05, 60)))
            for symbol in ("A", "B", "C", "D")
        },
        index=index,
    )


def test_roy_ratio_scales_rise_by_forecast_spread():
    wide = roy_ratio(pd.Series([100.0, 130.0, 110.0]), last_price=100.0)
    narrow = roy_ratio(pd.Series([108.0, 112.0, 110.0]), last_price=100.0)
    # Same terminal forecast, narrower path: more conviction, higher ratio.
    assert narrow > wide


def test_roy_ratio_of_a_flat_forecast_is_zero():
    assert roy_ratio(pd.Series([100.0, 100.0, 100.0]), 100.0) == 0.0


def test_roy_allocation_falls_back_and_says_so():
    forecasts = {s: pd.Series([100.0, 90.0]) for s in ("A", "B")}
    weights = roy_allocation(forecasts, pd.Series({"A": 100.0, "B": 100.0}))
    assert weights.attrs["fallback"] is True
    assert weights.sum() == pytest.approx(1.0)


def test_weight_cap_is_not_undone_by_renormalising():
    """Clip-then-renormalise is the obvious implementation and it is wrong."""
    expected = pd.Series({"A": 0.10, "B": 0.05, "C": 0.02, "D": 0.01})
    weights = signal_weights(expected, mode="long_only", max_weight=0.4)
    assert weights.max() <= 0.4 + 1e-9
    assert weights.sum() == pytest.approx(1.0)


def test_a_single_positive_signal_leaves_the_rest_in_cash():
    expected = pd.Series({"A": 0.05, "B": -0.01, "C": -0.02})
    weights = signal_weights(expected, mode="long_only", max_weight=0.4)
    assert weights["A"] == pytest.approx(0.4)
    assert weights.sum() == pytest.approx(0.4)  # 60% uninvested, not forced into A


def test_long_only_holds_nothing_when_nothing_is_expected_to_rise():
    weights = signal_weights(pd.Series({"A": -0.05, "B": -0.02}), mode="long_only")
    assert weights.sum() == pytest.approx(0.0)


def test_long_short_is_gross_normalised_and_market_neutral_ish():
    expected = pd.Series({"A": 0.10, "B": 0.05, "C": -0.02, "D": -0.08})
    weights = signal_weights(expected, mode="long_short")
    assert weights.abs().sum() == pytest.approx(1.0)
    assert abs(weights.sum()) < 0.5


def test_concentration_reports_effective_n():
    stats = concentration(pd.Series({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}))
    assert stats["effective_n"] == pytest.approx(1.0)
    assert concentration(equal_weights(list("ABCD")))["effective_n"] == pytest.approx(4.0)


def test_contributions_reconcile_to_the_gross_return(panel):
    # A value at date t is the return expected over the period *ending* at t,
    # so perfect foresight is pct_change() itself, not a shifted copy.
    signal = panel.pct_change().iloc[1:]
    result = backtest_signals(panel, signal, mode="long_only")
    np.testing.assert_allclose(
        result.contributions.sum(axis=1).to_numpy(),
        result.gross_returns.to_numpy(),
        rtol=1e-12,
    )
    # Attribution must add up to the compounded-free sum of period contributions.
    assert attribution(result)["contribution"].sum() == pytest.approx(
        result.gross_returns.sum(), rel=1e-9
    )


def test_costs_reduce_the_net_return(panel):
    signal = pd.DataFrame(
        np.random.default_rng(1).normal(0, 0.02, panel.shape),
        index=panel.index,
        columns=panel.columns,
    ).iloc[1:]
    free = backtest_signals(panel, signal, mode="long_only")
    charged = backtest_signals(panel, signal, mode="long_only", cost_model=default_cost_model())
    assert charged.costs.sum() > 0
    assert charged.net_returns.sum() < free.net_returns.sum()


def test_an_informed_signal_beats_equal_weight(panel):
    """A negative control on the machinery: a near-oracle must show positive selection."""
    oracle = panel.pct_change().iloc[1:]
    result = backtest_signals(panel, oracle, mode="long_only")
    decomposition = decompose_vs_equal_weight(result)
    assert decomposition["selection_effect"] > 0
    assert decomposition["selection_t_stat"] > 2


def test_an_uninformed_signal_shows_no_selection(panel):
    noise = pd.DataFrame(
        np.random.default_rng(99).normal(0, 0.02, panel.shape),
        index=panel.index,
        columns=panel.columns,
    ).iloc[1:]
    decomposition = decompose_vs_equal_weight(backtest_signals(panel, noise, mode="long_only"))
    assert abs(decomposition["selection_t_stat"]) < 2.5


def test_equal_weight_mode_ignores_the_signal(panel):
    signal = pd.DataFrame(
        np.random.default_rng(4).normal(0, 0.02, panel.shape),
        index=panel.index,
        columns=panel.columns,
    ).iloc[1:]
    result = backtest_signals(panel, signal, mode="equal")
    assert result.weights.iloc[-1].to_numpy() == pytest.approx(0.25)


def test_lag_shifts_the_book(panel):
    signal = panel.pct_change().iloc[1:]
    prompt = backtest_signals(panel, signal, mode="long_only")
    delayed = backtest_signals(panel, signal, mode="long_only", lag=1)
    assert not np.allclose(prompt.gross_returns, delayed.gross_returns)


def test_sharpe_variants_agree_in_sign():
    returns = pd.Series(np.random.default_rng(8).normal(0.02, 0.03, 48))
    assert annualised_sharpe(returns) > 0
    assert project_sharpe(returns * 100) > 0


def test_max_drawdown_is_negative_and_bounded():
    values = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8])
    drawdown = max_drawdown(values)
    assert -1.0 <= drawdown < 0
    assert drawdown == pytest.approx(0.8 / 1.2 - 1.0)
