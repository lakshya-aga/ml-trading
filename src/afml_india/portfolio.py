"""Portfolio construction and evaluation, following the Stock-Portfolio-Optimiser framing.

That project turned a per-stock price forecast into an allocation via the "Roy
ratio" — expected rise divided by the spread of the forecast — and compared the
result against two equal-weight controls: buy-and-hold, and rebalanced every
period. The same scaffolding is reproduced here so a forecast produced by any
model can be scored the same way.

Keeping the controls is the important part. A forecast-driven portfolio that
does not beat equal-weight buy-and-hold has not demonstrated anything, and on
a rising market that control is a much harder benchmark than it sounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afml_india.data.costs import CostModel, Segment
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Risk metrics
# --------------------------------------------------------------------------- #


def project_sharpe(returns_pct, risk_free_rate: float = 0.02) -> float:
    """Sharpe as computed in the Stock-Portfolio-Optimiser utilities.

    Reproduced exactly for comparability: the risk-free rate is divided by the
    sample length and the result scaled by ``sqrt(n)``, where ``n`` is the
    number of observations rather than the number of periods per year. That
    makes the number depend on sample length, so it is comparable *within* a
    study but not against a conventionally annualised Sharpe. Use
    :func:`annualised_sharpe` when you need the latter.

    Parameters
    ----------
    returns_pct:
        Period returns in **percent**, matching the original implementation.
    """
    data = np.asarray(list(returns_pct), dtype=float)
    n = len(data)
    if n < 2:
        return float("nan")
    mean = data.mean()
    std = data.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(np.sqrt(n) * (mean - risk_free_rate / n) / std)


def annualised_sharpe(
    returns: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.065,
) -> float:
    """Conventional annualised Sharpe on fractional returns.

    The default risk-free rate is 6.5%, roughly the Indian 10-year G-sec — using
    a US treasury rate on a rupee portfolio overstates the Sharpe by a wide
    margin, since the whole nominal return sits on a higher risk-free base.
    """
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return float("nan")
    excess = series - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def max_drawdown(values: pd.Series) -> float:
    """Largest peak-to-trough decline of a portfolio value series, as a fraction."""
    series = pd.Series(values).dropna()
    if series.empty:
        return float("nan")
    running_max = series.cummax()
    return float((series / running_max - 1.0).min())


def performance_summary(
    values: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.065,
) -> pd.Series:
    """Total return, CAGR, volatility, Sharpe and drawdown for a value series."""
    series = pd.Series(values).dropna()
    if len(series) < 2:
        raise ValueError("need at least two observations")
    returns = series.pct_change().dropna()
    years = len(returns) / periods_per_year
    total = float(series.iloc[-1] / series.iloc[0] - 1.0)
    return pd.Series(
        {
            "total_return": total,
            "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 else float("nan"),
            "volatility": float(returns.std(ddof=1) * np.sqrt(periods_per_year)),
            "sharpe": annualised_sharpe(returns, periods_per_year, risk_free_rate),
            "project_sharpe": project_sharpe(returns * 100),
            "max_drawdown": max_drawdown(series),
            "periods": float(len(returns)),
        }
    )


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #


def roy_ratio(forecast: pd.Series, last_price: float) -> float:
    """The project's Roy ratio: expected rise divided by the forecast's range.

    A large predicted rise counts for less when the forecast path is wide,
    which is a crude but genuine risk adjustment — the spread of the recursive
    forecast stands in for the model's uncertainty about that name.

    Returns 0.0 when the forecast is flat (zero range), which would otherwise
    divide by zero and hand the whole portfolio to one stock.
    """
    values = pd.Series(forecast).dropna()
    if values.empty:
        return 0.0
    spread = float(values.max() - values.min())
    if spread <= 0:
        return 0.0
    return float((values.iloc[-1] - last_price) / spread)


def roy_allocation(
    forecasts: dict[str, pd.Series],
    last_prices: pd.Series,
    long_only: bool = True,
) -> pd.Series:
    """Normalised portfolio weights from per-stock Roy ratios.

    ``long_only`` clips negative ratios to zero, matching the original project.
    If every ratio is non-positive — a forecast of broad decline — the function
    falls back to equal weight and says so, rather than returning weights that
    do not sum to one.
    """
    ratios = pd.Series(
        {sym: roy_ratio(series, float(last_prices[sym])) for sym, series in forecasts.items()}
    )
    if long_only:
        ratios = ratios.clip(lower=0.0)
    total = ratios.abs().sum()
    if total <= 0:
        logger.warning("All Roy ratios are non-positive; falling back to equal weight")
        return pd.Series(1.0 / len(ratios), index=ratios.index)
    return ratios / total


def equal_weights(symbols) -> pd.Series:
    """Equal allocation across ``symbols``."""
    symbols = list(symbols)
    return pd.Series(1.0 / len(symbols), index=symbols)


# --------------------------------------------------------------------------- #
# Portfolio simulation
# --------------------------------------------------------------------------- #


def buy_and_hold(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Portfolio value from a one-time allocation, never rebalanced.

    Weights drift with performance, so winners compound their share. This is the
    harder of the two controls in a trending market.
    """
    prices = prices.dropna(how="all").ffill().dropna()
    weights = weights.reindex(prices.columns).fillna(0.0)
    units = weights / prices.iloc[0]
    return (prices * units).sum(axis=1).rename("buy_and_hold")


def rebalanced(
    prices: pd.DataFrame,
    weights: pd.Series,
    cost_model: CostModel | None = None,
) -> pd.Series:
    """Portfolio value with the target weights restored every period.

    ``cost_model`` charges turnover at each rebalance. Leaving it ``None``
    reproduces the original frictionless comparison, but on Indian delivery
    trades a monthly rebalance is far from free — roughly 22 bps of round-trip
    statutory cost on whatever fraction of the book actually turns over — so it
    is worth running both.
    """
    prices = prices.dropna(how="all").ffill().dropna()
    weights = weights.reindex(prices.columns).fillna(0.0)
    returns = prices.pct_change().fillna(0.0)

    value = 1.0
    values, held = [], weights.copy()
    for i, (_, row) in enumerate(returns.iterrows()):
        if i == 0:
            values.append(value)
            continue
        grown = held * (1.0 + row)
        period_return = float(grown.sum() / held.sum() - 1.0) if held.sum() else 0.0
        value *= 1.0 + period_return

        drifted = grown / grown.sum() if grown.sum() else weights
        if cost_model is not None:
            turnover = float((drifted - weights).abs().sum() / 2.0)
            # One-way turnover pays roughly half a round trip.
            value *= 1.0 - turnover * cost_model.round_trip_bps() / 1e4 / 2.0
        held = weights.copy()
        values.append(value)
    return pd.Series(values, index=prices.index, name="rebalanced")


def compare_portfolios(
    prices: pd.DataFrame,
    allocations: dict[str, pd.Series],
    rebalance: bool = False,
    cost_model: CostModel | None = None,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate several allocations over the same prices and summarise them.

    Returns ``(values, summary)`` — the value paths and one summary row per
    allocation.
    """
    values = {}
    for name, weights in allocations.items():
        values[name] = (
            rebalanced(prices, weights, cost_model) if rebalance else buy_and_hold(prices, weights)
        )
    value_frame = pd.DataFrame(values)
    summary = pd.DataFrame(
        {name: performance_summary(series, periods_per_year) for name, series in values.items()}
    ).T
    return value_frame, summary


def control_portfolios(
    prices: pd.DataFrame,
    cost_model: CostModel | None = None,
) -> dict[str, pd.Series]:
    """The two equal-weight controls from the original project."""
    weights = equal_weights(prices.columns)
    return {
        "equal_weight_hold": buy_and_hold(prices, weights),
        "equal_weight_rebalanced": rebalanced(prices, weights, cost_model),
    }


def default_cost_model() -> CostModel:
    """Delivery-segment cost model, the right default for a monthly portfolio."""
    return CostModel(segment=Segment.DELIVERY)
