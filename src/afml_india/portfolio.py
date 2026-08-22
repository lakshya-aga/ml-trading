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

from dataclasses import dataclass

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


def roy_ratios(forecasts: dict[str, pd.Series], last_prices: pd.Series) -> pd.Series:
    """Raw (unnormalised) Roy ratio per stock, for inspection before allocating."""
    return pd.Series(
        {sym: roy_ratio(series, float(last_prices[sym])) for sym, series in forecasts.items()}
    )


def roy_allocation(
    forecasts: dict[str, pd.Series],
    last_prices: pd.Series,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """Normalised portfolio weights from per-stock Roy ratios.

    ``long_only`` clips negative ratios to zero, matching the original project.

    ``max_weight`` caps any single position and redistributes the excess. Worth
    setting on a small universe: the Roy ratio is a bare ratio with no
    diversification term, so when only one forecast is positive it will happily
    allocate the entire book to that name.

    If every ratio is non-positive — a forecast of broad decline — the result
    falls back to equal weight, and ``result.attrs['fallback']`` records it so
    the caller can tell a real allocation from a degenerate one.
    """
    ratios = roy_ratios(forecasts, last_prices)
    if long_only:
        ratios = ratios.clip(lower=0.0)

    total = ratios.abs().sum()
    if total <= 0:
        logger.warning("All Roy ratios are non-positive; falling back to equal weight")
        weights = pd.Series(1.0 / len(ratios), index=ratios.index)
        weights.attrs["fallback"] = True
        weights.attrs["ratios"] = ratios
        return weights

    weights = ratios / total
    if max_weight is not None:
        if max_weight < 1.0 / len(weights):
            raise ValueError(
                f"max_weight={max_weight} is below equal weight "
                f"({1.0 / len(weights):.4f}); no allocation can satisfy it"
            )
        # Iteratively cap and redistribute; converges in a few passes because
        # each pass strictly reduces the number of names above the cap.
        for _ in range(len(weights)):
            over = weights > max_weight + 1e-12
            if not over.any():
                break
            excess = float((weights[over] - max_weight).sum())
            weights[over] = max_weight
            under = ~over
            room = weights[under].sum()
            if room <= 0:
                weights[under] = excess / max(under.sum(), 1)
                break
            weights[under] = weights[under] + excess * weights[under] / room

    weights.attrs["fallback"] = False
    weights.attrs["ratios"] = ratios
    return weights


def concentration(weights: pd.Series) -> pd.Series:
    """Herfindahl index, effective number of positions, and the largest weight.

    Effective N is ``1 / sum(w^2)``: a portfolio of five names with one at 100%
    has an effective N of 1, which is the number worth looking at rather than
    the nominal count.
    """
    w = pd.Series(weights).fillna(0.0)
    hhi = float((w**2).sum())
    return pd.Series(
        {
            "herfindahl": hhi,
            "effective_n": float(1.0 / hhi) if hhi > 0 else float("nan"),
            "max_weight": float(w.max()),
            "n_nonzero": float((w > 1e-12).sum()),
        }
    )


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


# --------------------------------------------------------------------------- #
# Signal backtesting and return attribution
# --------------------------------------------------------------------------- #


@dataclass
class BacktestResult:
    """Output of :func:`backtest_signals`, kept together so attribution is easy."""

    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    contributions: pd.DataFrame
    realised_returns: pd.DataFrame

    @property
    def equity(self) -> pd.DataFrame:
        """Cumulative gross and net equity curves, both starting at 1.0."""
        return pd.DataFrame(
            {
                "gross": (1.0 + self.gross_returns).cumprod(),
                "net": (1.0 + self.net_returns).cumprod(),
            }
        )

    def summary(self, periods_per_year: int = 12, risk_free_rate: float = 0.065) -> pd.Series:
        """Headline performance, gross and net, plus turnover and cost drag."""
        equity = self.equity
        gross = performance_summary(equity["gross"], periods_per_year, risk_free_rate)
        net = performance_summary(equity["net"], periods_per_year, risk_free_rate)
        out = pd.concat(
            [gross.add_prefix("gross_"), net.add_prefix("net_")]
        )
        out["total_costs"] = float(self.costs.sum())
        out["mean_turnover"] = float(self.turnover.mean())
        out["cost_drag"] = float(gross["total_return"] - net["total_return"])
        out["hit_rate"] = float((self.net_returns > 0).mean())
        return out


def signal_weights(
    expected: pd.Series,
    mode: str = "long_only",
    max_weight: float | None = None,
) -> pd.Series:
    """Convert one period's expected returns into portfolio weights.

    Parameters
    ----------
    mode:
        ``"long_only"`` allocates in proportion to positive expected return and
        holds nothing otherwise; ``"long_short"`` goes long the above-median
        names and short the rest, gross exposure normalised to one;
        ``"equal"`` ignores the signal entirely and is the control.

    Notes
    -----
    Long-only returns an all-zero vector when nothing is expected to rise. That
    is a real decision — sit in cash — not an error, and the backtest treats it
    as a flat period rather than silently reverting to equal weight.
    """
    expected = expected.dropna()
    if expected.empty:
        return pd.Series(dtype=float)

    if mode == "equal":
        weights = pd.Series(1.0 / len(expected), index=expected.index)
    elif mode == "long_only":
        positive = expected.clip(lower=0.0)
        total = positive.sum()
        weights = positive / total if total > 0 else positive * 0.0
    elif mode == "long_short":
        centred = expected - expected.median()
        gross = centred.abs().sum()
        weights = centred / gross if gross > 0 else centred * 0.0
    else:
        raise ValueError(f"unknown mode {mode!r}; expected long_only, long_short or equal")

    if max_weight is not None and weights.abs().sum() > 0:
        weights = _cap_and_redistribute(weights, max_weight)
    return weights


def _cap_and_redistribute(weights: pd.Series, max_weight: float) -> pd.Series:
    """Cap each position, pushing the excess onto the names with room.

    Clipping and then renormalising would undo the cap — that is the obvious
    implementation and it is wrong. When no name has room left (a single
    positive signal under a 40% cap, say), the residual stays **uninvested**
    rather than being forced back into the capped position: holding cash is the
    honest reading of "do not put more than 40% in one name".
    """
    capped = weights.copy()
    for _ in range(len(capped) + 1):
        over = capped.abs() > max_weight + 1e-12
        if not over.any():
            break
        excess = float((capped[over].abs() - max_weight).sum())
        capped[over] = np.sign(capped[over]) * max_weight

        # Only names already on the same side of the book can absorb the excess;
        # redistributing into an unsignalled name would be inventing a position.
        room = (max_weight - capped[~over].abs()).clip(lower=0.0)
        available = float(room[capped[~over] != 0].sum())
        if available <= 1e-12:
            break  # nothing can absorb it: the remainder sits in cash
        share = room.where(capped[~over] != 0, 0.0)
        capped[~over] = capped[~over] + np.sign(capped[~over]) * excess * share / available
    return capped


def backtest_signals(
    prices: pd.DataFrame,
    expected: pd.DataFrame,
    mode: str = "long_only",
    max_weight: float | None = None,
    cost_model: CostModel | None = None,
    lag: int = 0,
) -> BacktestResult:
    """Backtest one-step-ahead return forecasts as a rebalanced portfolio.

    Parameters
    ----------
    prices:
        Realised prices, wide (dates x symbols).
    expected:
        Predicted return **for** each date, aligned to ``prices``. A value at
        date ``t`` is the return expected to be earned over the period ending at
        ``t``, so the position it implies must be taken at ``t - 1``.
    lag:
        Extra periods of delay between forming a signal and trading it. Zero
        assumes the forecast for ``t`` is actionable at ``t - 1``, which is
        correct for a one-step-ahead model built from data through ``t - 1``.
        Raise it to test sensitivity to execution delay.
    cost_model:
        Charges turnover each period. ``None`` reports gross returns only.

    Returns
    -------
    BacktestResult
        Weights, gross and net returns, costs, turnover and the per-stock
        contribution to each period's return.
    """
    prices = prices.sort_index()
    realised = prices.pct_change()

    common = expected.index.intersection(realised.index)
    if len(common) == 0:
        raise ValueError("expected returns and prices share no dates")
    expected = expected.loc[common]
    realised = realised.loc[common]

    weight_rows, dates = [], []
    for stamp in common:
        row = expected.loc[stamp].dropna()
        weights = signal_weights(row, mode=mode, max_weight=max_weight)
        weight_rows.append(weights.reindex(prices.columns).fillna(0.0))
        dates.append(stamp)

    weights = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(dates))
    if lag:
        weights = weights.shift(lag).fillna(0.0)

    contributions = weights * realised.reindex(columns=weights.columns).fillna(0.0)
    gross = contributions.sum(axis=1)

    # Turnover is the change in the held book, so the first period's cost is the
    # cost of establishing it.
    previous = weights.shift(1).fillna(0.0)
    turnover = (weights - previous).abs().sum(axis=1) / 2.0

    if cost_model is None:
        costs = pd.Series(0.0, index=weights.index)
    else:
        # One-way turnover pays about half a round trip.
        costs = turnover * cost_model.round_trip_bps() / 1e4 / 2.0

    return BacktestResult(
        weights=weights,
        gross_returns=gross,
        net_returns=gross - costs,
        costs=costs,
        turnover=turnover,
        contributions=contributions,
        realised_returns=realised,
    )


def attribution(result: BacktestResult) -> pd.DataFrame:
    """Per-stock breakdown of where the return came from.

    Contribution is the sum of ``weight * realised return`` over the backtest —
    it adds up to the gross return, which is what makes it an attribution rather
    than a set of unrelated statistics. Reading it is how you find out whether a
    result is broad or is one name carrying everything.
    """
    contributions = result.contributions
    weights = result.weights

    frame = pd.DataFrame(
        {
            "contribution": contributions.sum(),
            "mean_weight": weights.mean(),
            "max_weight": weights.abs().max(),
            "periods_held": (weights.abs() > 1e-12).sum(),
            "hit_rate": (contributions > 0).sum() / (weights.abs() > 1e-12).sum().replace(0, np.nan),
            "mean_return_when_held": (
                result.realised_returns.where(weights.abs() > 1e-12).mean()
            ),
        }
    )
    total = frame["contribution"].sum()
    frame["share_of_gross"] = frame["contribution"] / total if total != 0 else np.nan
    return frame.sort_values("contribution", ascending=False)


def decompose_vs_equal_weight(
    result: BacktestResult,
    equal_result: BacktestResult | None = None,
) -> pd.Series:
    """Split net return into the equal-weight baseline, selection and costs.

    ``net = equal_weight_baseline + selection_effect - costs``

    The selection effect is the part attributable to *deviating* from equal
    weight. It is the only component a forecast can claim credit for, and it is
    routinely much smaller than the baseline — which is the honest reason so
    many signal-driven portfolios fail to justify themselves.
    """
    realised = result.realised_returns
    baseline = realised.mean(axis=1).fillna(0.0)
    selection = result.gross_returns - baseline

    def compound(series: pd.Series) -> float:
        return float((1.0 + series).prod() - 1.0)

    out = pd.Series(
        {
            "equal_weight_baseline": compound(baseline),
            "gross_strategy": compound(result.gross_returns),
            "selection_effect": compound(result.gross_returns) - compound(baseline),
            "total_costs": float(result.costs.sum()),
            "net_strategy": compound(result.net_returns),
            "selection_per_period": float(selection.mean()),
            "selection_t_stat": (
                float(selection.mean() / selection.std(ddof=1) * np.sqrt(len(selection)))
                if selection.std(ddof=1) > 0
                else float("nan")
            ),
        }
    )
    if equal_result is not None:
        out["equal_weight_net"] = compound(equal_result.net_returns)
    return out
