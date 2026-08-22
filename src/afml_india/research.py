"""One import for an AFML research session.

The intent is that a notebook opens with::

    from afml_india.research import *

and has the whole pipeline in scope: load a snapshot, build bars, filter events,
fractionally difference, label, weight, cross-validate. The AFML algorithms come
straight from fin-kit (``mlfinlab``); this module only re-exports them alongside
the Indian market layer and adds the few glue functions that sit between.

Anything imported here that fin-kit does not provide is defined below and
documented as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afml_india import finkit  # noqa: F401  (side effect: mlfinlab on sys.path)
from afml_india.bars.finkit_bars import (
    build_all_bars,
    build_bars,
    prepare_tick_frame,
    suggest_thresholds,
)
from afml_india.bars.standard import bars_from_ohlcv, time_bars
from afml_india.data.calendar import IST, NSE, NSECalendar
from afml_india.data.costs import CostModel, InstrumentSpec, Segment, apply_costs
from afml_india.data.snapshot import Snapshot, find_snapshot
from afml_india.data.universe import Universe
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# AFML algorithms, re-exported from fin-kit
# --------------------------------------------------------------------------- #
from mlfinlab.cross_validation import PurgedKFold, ml_cross_val_score  # noqa: E402
from mlfinlab.data_structures import (  # noqa: E402
    get_dollar_bars,
    get_ema_dollar_imbalance_bars,
    get_ema_dollar_run_bars,
    get_tick_bars,
    get_time_bars,
    get_volume_bars,
)
from mlfinlab.features.fracdiff import (  # noqa: E402
    frac_diff,
    frac_diff_ffd,
    get_weights,
    get_weights_ffd,
)
from mlfinlab.filters.filters import cusum_filter, z_score_filter  # noqa: E402
from mlfinlab.labeling.labeling import (  # noqa: E402
    add_vertical_barrier,
    drop_labels,
    get_bins,
    get_events,
)
from mlfinlab.labeling.trend_scanning import trend_scanning_labels  # noqa: E402
from mlfinlab.sample_weights.attribution import (  # noqa: E402
    get_weights_by_return,
    get_weights_by_time_decay,
)
from mlfinlab.sampling.bootstrapping import get_ind_matrix, seq_bootstrap  # noqa: E402
from mlfinlab.sampling.concurrent import get_av_uniqueness_from_triple_barrier  # noqa: E402
from mlfinlab.structural_breaks import get_chu_stinchcombe_white_statistics, get_sadf  # noqa: E402
from mlfinlab.util.volatility import get_daily_vol as _get_daily_vol  # noqa: E402

from afml_india._tz import tz_safe  # noqa: E402

# fin-kit loses the timezone when it round-trips a DatetimeIndex through
# ``.values``. Wrapping the affected entry points keeps IST-aware bars usable
# everywhere downstream instead of forcing naive timestamps on the caller.
get_daily_vol = tz_safe(_get_daily_vol)
get_events = tz_safe(get_events)
get_bins = tz_safe(get_bins)
add_vertical_barrier = tz_safe(add_vertical_barrier)
drop_labels = tz_safe(drop_labels)
trend_scanning_labels = tz_safe(trend_scanning_labels)
get_av_uniqueness_from_triple_barrier = tz_safe(get_av_uniqueness_from_triple_barrier)
get_weights_by_return = tz_safe(get_weights_by_return)
get_weights_by_time_decay = tz_safe(get_weights_by_time_decay)
get_ind_matrix = tz_safe(get_ind_matrix)

__all__ = [
    # India layer
    "NSE", "NSECalendar", "IST", "CostModel", "InstrumentSpec", "Segment",
    "apply_costs", "Universe", "Snapshot", "find_snapshot",
    # Bars
    "build_bars", "build_all_bars", "prepare_tick_frame", "suggest_thresholds",
    "bars_from_ohlcv", "time_bars",
    "get_tick_bars", "get_volume_bars", "get_dollar_bars", "get_time_bars",
    "get_ema_dollar_imbalance_bars", "get_ema_dollar_run_bars",
    # Filters / event sampling
    "cusum_filter", "z_score_filter", "sample_events",
    "get_chu_stinchcombe_white_statistics", "get_sadf",
    # Stationarity
    "frac_diff", "frac_diff_ffd", "get_weights", "get_weights_ffd",
    "min_ffd_order", "fracdiff_scan",
    # Labelling
    "get_events", "get_bins", "add_vertical_barrier", "drop_labels",
    "trend_scanning_labels", "get_daily_vol", "triple_barrier_labels",
    # Sample weights
    "get_av_uniqueness_from_triple_barrier", "get_weights_by_return",
    "get_weights_by_time_decay", "get_ind_matrix", "seq_bootstrap",
    # Cross-validation
    "PurgedKFold", "ml_cross_val_score",
    # Diagnostics
    "bar_statistics", "compare_bar_types",
]


# --------------------------------------------------------------------------- #
# Glue that fin-kit does not provide
# --------------------------------------------------------------------------- #


def sample_events(
    close: pd.Series,
    threshold: float | pd.Series | None = None,
    vol_multiple: float = 1.0,
    vol_lookback: int = 100,
    log_prices: bool = True,
) -> pd.DatetimeIndex:
    """CUSUM event sampling on log prices, with a volatility-scaled threshold.

    Two things this handles that calling ``cusum_filter`` directly does not.

    **Log prices.** ``cusum_filter`` accumulates the first difference of
    whatever series it is given. Hand it rupee prices and the running sum is in
    rupees, so a return-scale threshold like 0.02 is crossed within a bar or two
    and effectively every bar becomes an event. AFML applies the filter to log
    prices, where the differences *are* returns; that is what happens here
    unless ``log_prices=False``.

    **A threshold that travels.** A hard-coded threshold gives no events in a
    quiet quarter and thousands in a volatile one. Scaling it by trailing daily
    volatility keeps the event rate roughly stationary — which matters more
    across an Indian cross-section, where a PSU bank and an IT major differ in
    volatility by a factor of three, than it does within a single index series.

    Parameters
    ----------
    close:
        Bar close prices, indexed by timestamp.
    threshold:
        Explicit CUSUM threshold, in return units when ``log_prices`` is True.
        When ``None``, uses ``vol_multiple * median(daily_vol)``.
    vol_multiple:
        Multiplier on median trailing volatility. Higher means fewer, larger
        events. Start around 1-2 and tune to a sensible event count.
    log_prices:
        Apply the filter to ``log(close)`` rather than to ``close``.
    """
    if (close <= 0).any():
        raise ValueError("close contains non-positive prices; clean the series first")

    if threshold is None:
        vol = get_daily_vol(close, lookback=vol_lookback).dropna()
        if vol.empty:
            raise ValueError(
                "could not estimate volatility for the default threshold; "
                "pass threshold= explicitly or supply more bars"
            )
        threshold = float(vol.median()) * vol_multiple
        logger.info("CUSUM threshold set to %.5f (%.1fx median daily vol)", threshold, vol_multiple)

    series = np.log(close) if log_prices else close
    # cusum_filter rebuilds its result as a bare DatetimeIndex and loses the
    # timezone on the way, which then fails to align with the tz-aware bars it
    # came from. Run it on a naive copy and map the hits back by position.
    tz = series.index.tz
    naive = series.copy()
    if tz is not None:
        naive.index = naive.index.tz_localize(None)

    hits = cusum_filter(naive, threshold=threshold)
    positions = naive.index.get_indexer(pd.DatetimeIndex(hits))
    positions = np.unique(positions[positions >= 0])
    events = close.index[positions]

    rate = 100.0 * len(events) / max(1, len(close))
    logger.info("Sampled %d events from %d bars (%.1f%%)", len(events), len(close), rate)
    if rate > 50:
        logger.warning(
            "Over half the bars became events; the threshold is likely too low. "
            "Raise vol_multiple."
        )
    return events


def min_ffd_order(
    series: pd.Series,
    thresh: float = 1e-5,
    max_order: float = 1.0,
    step: float = 0.05,
    p_value: float = 0.05,
    min_corr: float | None = None,
    min_obs: int = 100,
) -> dict:
    """Smallest ``d`` whose fixed-width fracdiff series passes ADF stationarity.

    This is AFML's central stationarity-versus-memory trade-off: differencing
    once (``d=1``) guarantees stationarity but destroys the level information a
    model needs, while ``d=0`` keeps all memory and fails every test. The useful
    answer is almost always a fraction, and finding it per series beats
    assuming one.

    Returns a dict with the chosen ``d``, its ADF statistic and p-value, the
    correlation to the original series, and the full scan as a frame.
    """
    table = fracdiff_scan(series, thresh=thresh, max_order=max_order, step=step)
    passing = table[(table["adf_pvalue"] <= p_value) & (table["n_obs"] >= min_obs)]
    if min_corr is not None:
        passing = passing[passing["corr"] >= min_corr]
    if passing.empty:
        raise ValueError(
            f"no d in [0, {max_order}] produced an ADF p-value <= {p_value} on at "
            f"least {min_obs} observations"
            + (f" with corr >= {min_corr}" if min_corr is not None else "")
            + ". Try a larger max_order, a looser thresh, or a longer series."
        )
    best = passing.iloc[0]
    return {
        "d": float(best["d"]),
        "adf_stat": float(best["adf_stat"]),
        "adf_pvalue": float(best["adf_pvalue"]),
        "corr": float(best["corr"]),
        "n_obs": int(best["n_obs"]),
        "scan": table,
    }


def fracdiff_scan(
    series: pd.Series,
    thresh: float = 1e-5,
    max_order: float = 1.0,
    step: float = 0.05,
) -> pd.DataFrame:
    """ADF statistic and memory retention across a grid of differencing orders.

    The returned frame is what you plot to make the trade-off visible: ADF
    statistic falling through the 5% critical value as ``d`` rises, against
    correlation to the undifferenced series falling away at the same time.

    Read ``n_obs`` alongside the statistics. Fixed-width fracdiff drops the
    leading window, and that window grows as ``d`` shrinks — at small ``d`` and
    a tight ``thresh`` the window can consume most of a short series, so the
    rows are not computed on equal samples. A row surviving on 60 observations
    is not evidence of stationarity; :func:`min_ffd_order` enforces a floor via
    its ``min_obs`` argument for exactly this reason. Loosening ``thresh``
    shortens the windows and widens the usable grid.
    """
    from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415

    series = series.dropna()
    if len(series) < 50:
        raise ValueError(f"need at least 50 observations to scan, got {len(series)}")
    frame = series.to_frame("value")

    rows = []
    for d in np.arange(0.0, max_order + 1e-9, step):
        differenced = frac_diff_ffd(frame, diff_amt=float(d), thresh=thresh)["value"].dropna()
        if len(differenced) < 20:
            continue
        aligned = series.reindex(differenced.index)
        corr = float(np.corrcoef(aligned.to_numpy(), differenced.to_numpy())[0, 1])
        # adfuller returns 5 values with autolag=None and 6 with autolag set.
        result = adfuller(differenced, maxlag=1, regression="c", autolag=None)
        stat, pval, _, nobs, crit = result[:5]
        rows.append(
            {
                "d": round(float(d), 4),
                "adf_stat": stat,
                "adf_pvalue": pval,
                "crit_5pct": crit["5%"],
                "corr": corr,
                "n_obs": nobs,
            }
        )
    return pd.DataFrame(rows)


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex | None = None,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    num_days: int = 5,
    min_ret: float = 0.0,
    vol_lookback: int = 100,
    side: pd.Series | None = None,
    num_threads: int = 1,
    calendar: NSECalendar | None = None,
) -> pd.DataFrame:
    """Triple-barrier labels with an NSE-session vertical barrier.

    fin-kit's ``add_vertical_barrier`` counts calendar days, which silently
    shortens the holding period across Diwali, a Monday holiday or a long
    weekend. This wires in :class:`~afml_india.data.calendar.NSECalendar` so
    ``num_days`` means trading sessions.

    Returns the ``get_bins`` output — ``ret``, ``trgt``, ``bin`` — joined to the
    event's ``t1`` so the sample-weight functions can be called directly on it.
    """
    calendar = calendar or NSE
    target = get_daily_vol(close, lookback=vol_lookback)
    if events is None:
        events = sample_events(close, vol_lookback=vol_lookback)

    events = pd.DatetimeIndex(events)
    events = events.intersection(target.dropna().index)
    if len(events) == 0:
        raise ValueError("no events survive the volatility warm-up; supply more bars")

    vertical = pd.Series(
        calendar.add_sessions(events, num_days), index=events, name="t1"
    )
    # Barriers past the end of the sample cannot resolve; NaT tells get_events
    # to leave them open rather than mislabel them.
    vertical[vertical > close.index[-1]] = pd.NaT

    triple = get_events(
        close=close,
        t_events=events,
        pt_sl=list(pt_sl),
        target=target,
        min_ret=min_ret,
        num_threads=num_threads,
        vertical_barrier_times=vertical,
        side_prediction=side,
        verbose=False,
    )
    bins = get_bins(triple, close)
    return bins.join(triple[["t1"]], how="left")


def bar_statistics(bars: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Distributional diagnostics for one bar series.

    These are the numbers AFML chapter 2 uses to argue for activity-driven
    sampling: lower serial correlation, kurtosis nearer normal, and a
    Jarque-Bera statistic that is at least in a different league.
    """
    from scipy import stats  # noqa: PLC0415

    price = bars[price_col].astype(float)
    rets = np.log(price).diff().dropna()
    if len(rets) < 8:
        raise ValueError(f"need at least 8 returns for statistics, got {len(rets)}")

    jb_stat, jb_p = stats.jarque_bera(rets)
    return pd.Series(
        {
            "n_bars": float(len(bars)),
            "mean_ret": float(rets.mean()),
            "std_ret": float(rets.std()),
            "skew": float(stats.skew(rets)),
            "kurtosis": float(stats.kurtosis(rets)),  # excess; 0 is normal
            "jarque_bera": float(jb_stat),
            "jb_pvalue": float(jb_p),
            "autocorr_1": float(rets.autocorr(lag=1)),
            "bars_per_day": float(len(bars) / max(1, bars.index.normalize().nunique())),
        }
    )


def compare_bar_types(bar_dict: dict[str, pd.DataFrame], price_col: str = "close") -> pd.DataFrame:
    """Side-by-side :func:`bar_statistics` for several bar types."""
    return pd.DataFrame({name: bar_statistics(bars, price_col) for name, bars in bar_dict.items()}).T
