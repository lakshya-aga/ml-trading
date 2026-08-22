"""Cross-sectional significance testing for per-ticker strategies.

The unit of evidence in this project is the **cross-section**, not the best
ticker. With 40 names and a handful of folds each, several tickers will look
good by chance alone; the question a significance layer must answer is whether
the *distribution* of out-of-sample results across the universe is
distinguishable from luck.

Three layers, each feeding the next:

1. :func:`event_returns` — turn one ticker's out-of-sample probabilities into
   per-event strategy returns, net of Indian costs.
2. :func:`ticker_metrics` — per-ticker Sharpe, turnover, hit rate, and a
   t-statistic on net event returns.
3. :func:`permutation_pvalue` — the null: shuffle each ticker's positions
   across its own events and recompute the cross-sectional statistic. Costs,
   return distributions, and event counts are all preserved under the null, so
   the only thing destroyed is the alignment between prediction and outcome —
   which is exactly the thing being tested.

The permutation approach sidesteps the two standard objections to a naive
cross-sectional t-test: event returns are not IID (overlap, fat tails), and
tickers are not independent (they share the market factor). Shuffling within
ticker preserves both structures in the null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from afml_india.data.costs import CostModel
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 1. Event-level PnL
# --------------------------------------------------------------------------- #


def event_returns(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    cost_model: CostModel | None = None,
    min_edge: float = 0.0,
) -> pd.DataFrame:
    """Per-event strategy returns from out-of-sample probabilities.

    AFML bet sizing: side is the sign of ``proba - 0.5`` and size is
    ``|2 * proba - 1|``, so a 50.1% call risks almost nothing and an 80% call
    risks a full unit. Each event is one round trip — entered at the event bar,
    exited at the barrier touch — so the cost model's round trip is charged on
    the size actually traded.

    Parameters
    ----------
    predictions:
        ``walk_forward_evaluate`` output: ``proba`` (and ``fold``) indexed by
        event time. Only out-of-sample rows should ever be passed here.
    labels:
        Triple-barrier output with ``ret`` (the realised event return, sign
        already reflecting the price path) and ``t1``.
    min_edge:
        Ignore events where ``|2p - 1|`` is below this — a no-trade band.
        Raising it trades fewer, more confident bets against fewer data points.
    """
    common = predictions.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("predictions and labels share no events")

    frame = predictions.loc[common, ["proba"]].copy()
    frame["realised"] = labels.loc[common, "ret"].astype(float)
    frame["t1"] = labels.loc[common, "t1"]

    edge = (2.0 * frame["proba"] - 1.0).clip(-1, 1)
    frame["side"] = np.sign(edge).replace(0, 1).astype(int)
    frame["size"] = edge.abs()
    frame.loc[frame["size"] < min_edge, "size"] = 0.0

    frame["gross"] = frame["side"] * frame["size"] * frame["realised"]
    round_trip = 0.0 if cost_model is None else cost_model.round_trip_bps() / 1e4
    frame["cost"] = frame["size"] * round_trip
    frame["net"] = frame["gross"] - frame["cost"]
    return frame


def daily_pnl(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event PnL to a daily series, attributing each event to its exit.

    One unit of capital per event, which overstates capital use when events
    overlap; the per-ticker Sharpe below is therefore a *per-unit-of-bet*
    figure, comparable across tickers, not a fund-level number. Turnover is
    entry plus exit — two legs per event.
    """
    exit_day = pd.DatetimeIndex(
        events["t1"].fillna(pd.Series(events.index, index=events.index))
    ).normalize()
    frame = pd.DataFrame(
        {
            "net": events["net"].to_numpy(),
            "gross": events["gross"].to_numpy(),
            "turnover": 2.0 * events["size"].to_numpy(),
        },
        index=exit_day,
    )
    return frame.groupby(level=0).sum().sort_index()


def ticker_metrics(
    events: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Headline numbers for one ticker's out-of-sample events.

    ``t_stat`` is the statistic that matters most: mean net event return over
    its standard error. The annualised Sharpe is reported because the request
    was for it, but on a short window it is a noisy rescaling of the same
    information and should not be read to two decimal places.
    """
    traded = events[events["size"] > 0]
    daily = daily_pnl(traded) if len(traded) else pd.DataFrame(columns=["net", "turnover"])

    net = traded["net"].to_numpy(dtype=float)
    out = {
        "n_events": float(len(events)),
        "n_traded": float(len(traded)),
        "hit_rate": float((traded["gross"] > 0).mean()) if len(traded) else np.nan,
        "mean_gross": float(traded["gross"].mean()) if len(traded) else np.nan,
        "mean_net": float(net.mean()) if len(net) else np.nan,
        "total_net": float(net.sum()),
        "t_stat": (
            float(net.mean() / net.std(ddof=1) * np.sqrt(len(net)))
            if len(net) > 2 and net.std(ddof=1) > 0
            else np.nan
        ),
        "daily_sharpe_ann": (
            float(daily["net"].mean() / daily["net"].std(ddof=1) * np.sqrt(periods_per_year))
            if len(daily) > 2 and daily["net"].std(ddof=1) > 0
            else np.nan
        ),
        "daily_turnover": float(daily["turnover"].mean()) if len(daily) else 0.0,
        "cost_drag": float(traded["cost"].sum()),
    }
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# 2. Cross-sectional statistics
# --------------------------------------------------------------------------- #


def cross_section_summary(metrics: pd.DataFrame) -> pd.Series:
    """Classical tests on the per-ticker mean net returns.

    Reported for orientation only — both assume independence across tickers,
    which the shared market factor violates. The permutation test below is the
    one to believe.
    """
    means = metrics["mean_net"].dropna()
    out = {
        "n_tickers": float(len(means)),
        "mean_of_means": float(means.mean()),
        "tickers_positive": float((means > 0).sum()),
        "share_positive": float((means > 0).mean()),
    }
    if len(means) > 2 and means.std(ddof=1) > 0:
        t_stat, p_value = stats.ttest_1samp(means, 0.0)
        out["t_stat_across_tickers"] = float(t_stat)
        out["t_pvalue"] = float(p_value)
        try:
            w_stat, w_p = stats.wilcoxon(means)
            out["wilcoxon_pvalue"] = float(w_p)
        except ValueError:
            out["wilcoxon_pvalue"] = np.nan
    return pd.Series(out)


def permutation_pvalue(
    per_ticker_events: dict[str, pd.DataFrame],
    n_permutations: int = 1000,
    statistic: str = "mean_net",
    seed: int = 0,
    method: str = "rotation",
) -> dict:
    """Permutation test of the cross-sectional result against a skill-free null.

    Under the null, each ticker's position sequence ``(side * size)`` is
    decoupled from its realised returns. Everything else — the realised return
    distribution, the sizes traded (hence costs), the event count, the
    cross-ticker dependence structure — is held fixed. Only the pairing of
    prediction with outcome is destroyed.

    **Why the default is a circular shift, not a shuffle.** Triple-barrier
    events overlap, so neighbouring realised returns are strongly correlated —
    and so are neighbouring predictions, because they see near-identical
    inputs. The events are therefore *not exchangeable*: a run of correlated
    positions aligned by luck with a run of correlated returns produces a
    large observed statistic that independent shuffles can almost never
    reproduce. On this project's own synthetic data the iid-shuffle null
    understated the null standard deviation by a factor of two and returned
    p = 0.001 on a random walk. Rotating each ticker's position sequence by a
    random offset (``method="rotation"``) preserves the serial structure of
    both sequences and destroys only their alignment; the test then calibrates
    correctly on clean geometric-Brownian universes (see
    ``scripts/calibrate_significance.py``). ``method="shuffle"`` remains
    available for demonstrating exactly this failure.

    The observed statistic is the cross-ticker mean of per-ticker mean net
    returns (or of per-ticker t-stats with ``statistic="t_stat"``); the
    p-value is the fraction of null draws at least as large. One-sided,
    because the claim under test is positive edge.

    Returns a dict with the observed value, the null draws, the p-value, the
    null standard deviation and 95th percentile, and the method used.
    """
    if method not in ("rotation", "shuffle"):
        raise ValueError(f"unknown method {method!r}; expected 'rotation' or 'shuffle'")
    rng = np.random.default_rng(seed)

    prepared = {}
    for ticker, events in per_ticker_events.items():
        traded = events[events["size"] > 0]
        if len(traded) < 5:
            logger.warning("%s: only %d traded events; excluded from the test", ticker, len(traded))
            continue
        prepared[ticker] = {
            "signed": (traded["side"] * traded["size"]).to_numpy(dtype=float),
            "realised": traded["realised"].to_numpy(dtype=float),
            "cost": traded["cost"].to_numpy(dtype=float),
        }
    if len(prepared) < 2:
        raise ValueError("need at least two tickers with enough traded events")

    def per_ticker_stat(net: np.ndarray) -> float:
        if statistic == "mean_net":
            return float(net.mean())
        if statistic == "t_stat":
            sd = net.std(ddof=1)
            return float(net.mean() / sd * np.sqrt(len(net))) if sd > 0 else 0.0
        raise ValueError(f"unknown statistic {statistic!r}")

    observed = float(
        np.mean(
            [per_ticker_stat(p["signed"] * p["realised"] - p["cost"]) for p in prepared.values()]
        )
    )

    null = np.empty(n_permutations)
    for b in range(n_permutations):
        draws = []
        for p in prepared.values():
            if method == "rotation":
                decoupled = np.roll(p["signed"], rng.integers(1, len(p["signed"])))
            else:
                decoupled = rng.permutation(p["signed"])
            draws.append(per_ticker_stat(decoupled * p["realised"] - p["cost"]))
        null[b] = np.mean(draws)

    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {
        "observed": observed,
        "null": null,
        "p_value": p_value,
        "null_sd": float(null.std(ddof=1)),
        "null_95th": float(np.percentile(null, 95)),
        "n_tickers": len(prepared),
        "n_permutations": n_permutations,
        "statistic": statistic,
        "method": method,
    }
