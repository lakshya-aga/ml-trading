"""Feature construction for bar-level models.

The organising constraint is that every feature must be computable from
information available **at the bar's own close**. Financial ML fails most often
not because the model is weak but because a feature leaked; centred rolling
windows, full-sample scaling and forward-filled fundamentals are all silent
leaks. Everything here is strictly backward-looking, and
:func:`assert_no_lookahead` exists to check it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afml_india import finkit  # noqa: F401  (side effect: mlfinlab on sys.path)
from afml_india.utils.logging import get_logger
from afml_india.utils.validation import ensure_monotonic

logger = get_logger(__name__)


def fracdiff_features(
    bars: pd.DataFrame,
    d: float,
    columns: tuple[str, ...] = ("close", "volume"),
    thresh: float = 1e-4,
    log_transform: tuple[str, ...] = ("close", "volume", "value", "cum_dollar_value"),
) -> pd.DataFrame:
    """Fixed-width fractionally differenced versions of selected columns.

    Level series are log-transformed first so the differencing operates on
    returns-like quantities; ``d`` then has the same meaning across a Rs 200 and
    a Rs 20,000 stock. Columns not in ``log_transform`` are differenced as-is.
    """
    from mlfinlab.features.fracdiff import frac_diff_ffd  # noqa: PLC0415

    ensure_monotonic(bars, "bars")
    out = {}
    for col in columns:
        if col not in bars.columns:
            logger.warning("Column %r not in bars; skipping", col)
            continue
        series = bars[col].astype(float)
        if col in log_transform:
            if (series <= 0).any():
                series = series.where(series > 0)
                logger.warning("Column %r has non-positive values; they become NaN", col)
            series = np.log(series)
        differenced = frac_diff_ffd(series.to_frame("v"), diff_amt=float(d), thresh=thresh)["v"]
        out[f"fd_{col}"] = differenced
    if not out:
        raise ValueError(f"none of {columns} are present in bars")
    return pd.DataFrame(out, index=bars.index)


def momentum_features(
    bars: pd.DataFrame,
    windows: tuple[int, ...] = (5, 20, 50),
    price_col: str = "close",
) -> pd.DataFrame:
    """Trailing return, volatility, and normalised-momentum features."""
    price = bars[price_col].astype(float)
    log_price = np.log(price)
    rets = log_price.diff()

    out = {}
    for w in windows:
        out[f"ret_{w}"] = log_price.diff(w)
        out[f"vol_{w}"] = rets.rolling(w).std()
        # Return per unit of its own realised volatility: comparable across
        # names and across regimes in a way a raw return is not.
        out[f"zmom_{w}"] = out[f"ret_{w}"] / (out[f"vol_{w}"] * np.sqrt(w))
    out["ret_1"] = rets
    return pd.DataFrame(out, index=bars.index)


def volatility_features(
    bars: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50),
) -> pd.DataFrame:
    """Range-based volatility estimators, which beat close-to-close at bar scale.

    Parkinson uses the high-low range and Garman-Klass adds the open-close body;
    both extract more information per bar than a close-to-close estimate, which
    matters when the bar count is modest.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise KeyError(f"volatility features need columns {sorted(missing)}")

    o, h, l, c = (bars[x].astype(float) for x in ("open", "high", "low", "close"))
    hl = np.log(h / l)
    co = np.log(c / o)

    out = {}
    for w in windows:
        out[f"parkinson_{w}"] = np.sqrt((hl**2).rolling(w).mean() / (4 * np.log(2)))
        out[f"garman_klass_{w}"] = np.sqrt(
            (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(w).mean().clip(lower=0)
        )
        # Ratio of range vol to close-to-close vol: a crude gap/jump detector.
        cc = np.log(c).diff().rolling(w).std()
        out[f"vol_ratio_{w}"] = out[f"parkinson_{w}"] / cc.replace(0, np.nan)
    return pd.DataFrame(out, index=bars.index)


def microstructure_features(
    bars: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50),
) -> pd.DataFrame:
    """Liquidity and order-flow features derivable from AFML bar columns.

    fin-kit's bars carry ``cum_buy_volume``, ``cum_ticks`` and
    ``cum_dollar_value``, which is enough for a tick-rule order-flow imbalance
    and an Amihud illiquidity ratio without needing the quote stream.
    """
    out = {}
    index = bars.index

    if {"cum_buy_volume", "volume"} <= set(bars.columns):
        volume = bars["volume"].astype(float).replace(0, np.nan)
        buy_share = bars["cum_buy_volume"].astype(float) / volume
        out["buy_share"] = buy_share
        # Signed order-flow imbalance in [-1, 1]; the tick rule's view of
        # whether the bar was accumulation or distribution.
        out["order_imbalance"] = 2.0 * buy_share - 1.0
        for w in windows:
            out[f"order_imbalance_{w}"] = out["order_imbalance"].rolling(w).mean()

    if "cum_ticks" in bars.columns:
        ticks = bars["cum_ticks"].astype(float)
        out["log_ticks"] = np.log(ticks.replace(0, np.nan))
        if "volume" in bars.columns:
            out["avg_trade_size"] = bars["volume"].astype(float) / ticks.replace(0, np.nan)

    if {"close", "volume"} <= set(bars.columns):
        rets = np.log(bars["close"].astype(float)).diff().abs()
        value = (bars["close"].astype(float) * bars["volume"].astype(float)).replace(0, np.nan)
        # Amihud: price impact per rupee traded. Higher means thinner.
        amihud = rets / value
        for w in windows:
            out[f"amihud_{w}"] = np.log1p(amihud.rolling(w).mean() * 1e9)

    if not out:
        raise KeyError("bars carry none of the columns microstructure features need")
    return pd.DataFrame(out, index=index)


def time_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Intraday position and session-progress features.

    Indian intraday volatility is strongly U-shaped, so where a bar sits in the
    session is genuinely informative and costs nothing to compute.
    """
    from afml_india.data.calendar import MARKET_CLOSE, MARKET_OPEN  # noqa: PLC0415

    index = pd.DatetimeIndex(bars.index)
    minutes = np.asarray(index.hour * 60 + index.minute, dtype=float) + index.second / 60.0
    open_min = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
    close_min = MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute
    progress = np.clip((minutes - open_min) / (close_min - open_min), 0.0, 1.0)

    return pd.DataFrame(
        {
            "session_progress": progress,
            # Sine/cosine so the model sees the open and close as adjacent ends
            # of one cycle rather than as distant scalar values.
            "session_sin": np.sin(2 * np.pi * progress),
            "session_cos": np.cos(2 * np.pi * progress),
            "day_of_week": np.asarray(index.dayofweek, dtype=float),
        },
        index=bars.index,
    )


@dataclass(frozen=True)
class FeatureConfig:
    """Declarative description of a feature matrix.

    Holding the recipe as data rather than as call-site keyword arguments is
    what makes :meth:`assert_no_lookahead` possible: the check has to rebuild
    the *same* features on truncated history, and it can only do that if the
    recipe is a value it can replay.

    Parameters
    ----------
    d:
        Fractional differencing order. ``None`` skips the fracdiff block, which
        is the right baseline to compare against.
    """

    d: float | None = None
    fracdiff_columns: tuple[str, ...] = ("close", "volume")
    fracdiff_thresh: float = 1e-4
    momentum_windows: tuple[int, ...] = (5, 20, 50)
    volatility_windows: tuple[int, ...] = (20, 50)
    include_micro: bool = True
    include_time: bool = True
    dropna: bool = True

    def build(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Assemble the full backward-looking feature matrix from ``bars``."""
        blocks = [momentum_features(bars, self.momentum_windows)]

        if self.d is not None:
            blocks.append(
                fracdiff_features(
                    bars, d=self.d, columns=self.fracdiff_columns, thresh=self.fracdiff_thresh
                )
            )
        if {"open", "high", "low", "close"} <= set(bars.columns):
            blocks.append(volatility_features(bars, self.volatility_windows))
        if self.include_micro:
            try:
                blocks.append(microstructure_features(bars, self.volatility_windows))
            except KeyError as exc:
                logger.warning("Skipping microstructure features: %s", exc)
        if self.include_time and isinstance(bars.index, pd.DatetimeIndex):
            blocks.append(time_features(bars))

        features = pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan)

        if self.dropna:
            before = len(features)
            features = features.dropna()
            logger.info(
                "Feature matrix: %d rows x %d features (dropped %d warm-up rows)",
                len(features), features.shape[1], before - len(features),
            )
        return features


def build_feature_matrix(bars: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: ``FeatureConfig(**kwargs).build(bars)``."""
    return FeatureConfig(**kwargs).build(bars)


def assert_no_lookahead(
    config: FeatureConfig,
    bars: pd.DataFrame,
    n_checks: int = 15,
    tolerance: float = 1e-8,
) -> None:
    """Rebuild features on truncated history and check nothing depends on the future.

    Two distinct failures are tested, because a leak can show up as either.

    **Value drift.** A feature that reads ahead computes a different number once
    the future is removed. Rebuilding at random cut points and comparing the
    overlapping rows catches a full-sample statistic or a forward shift.

    **Lost availability.** A centred rolling window does *not* change the values
    it produces — it simply cannot produce the most recent rows, because they
    need data that has not arrived. With ``dropna=True`` those rows vanish, so a
    value comparison alone sees nothing wrong. The check therefore also asserts
    that truncating to bar ``t`` still yields a feature row at ``t`` whenever the
    full build has one.

    Raises
    ------
    AssertionError
        Naming the offending columns, or the cut points where the latest row
        went missing.
    """
    full = config.build(bars)
    if full.empty:
        raise ValueError("feature matrix is empty; nothing to check")

    min_start = max(150, int(0.5 * len(bars)))
    if len(bars) <= min_start + 5:
        logger.warning("Too few bars for a look-ahead check; skipping")
        return

    rng = np.random.default_rng(0)
    candidates = np.arange(min_start, len(bars))
    cuts = sorted(
        int(c) for c in rng.choice(candidates, size=min(n_checks, len(candidates)), replace=False)
    )

    offenders: dict[str, float] = {}
    unavailable: list[pd.Timestamp] = []
    compared = 0

    for cut in cuts:
        truncated = config.build(bars.iloc[:cut])
        if truncated.empty:
            continue

        # 1. Availability: the last truncated bar must still yield a row.
        last_bar = bars.index[cut - 1]
        if last_bar in full.index and last_bar not in truncated.index:
            unavailable.append(last_bar)
            continue

        # 2. Value drift across every overlapping row.
        shared_rows = full.index.intersection(truncated.index)
        shared_cols = full.columns.intersection(truncated.columns)
        if len(shared_rows) == 0 or shared_cols.empty:
            continue
        # The tail is where a leak bites; comparing it all is unnecessary work.
        tail = shared_rows[-min(len(shared_rows), 50) :]
        full_block = full.loc[tail, shared_cols].astype(float)
        trunc_block = truncated.loc[tail, shared_cols].astype(float)
        # Scale by the value's own magnitude, floored at 1, so near-zero
        # features do not trip on floating-point noise.
        scale = full_block.abs().clip(lower=1.0)
        relative = ((full_block - trunc_block).abs() / scale).fillna(0.0)
        compared += 1
        worst = relative.max()
        for col, value in worst[worst > tolerance].items():
            offenders[col] = max(offenders.get(col, 0.0), float(value))

    problems = []
    if offenders:
        detail = ", ".join(f"{c} (max rel. drift {v:.2e})" for c, v in sorted(offenders.items()))
        problems.append(f"values change when the future is removed: {detail}")
    if unavailable:
        problems.append(
            f"the most recent row is unavailable at {len(unavailable)} cut point(s) "
            f"(e.g. {unavailable[0]}) — a feature needs data that has not arrived yet, "
            "which is what a centred rolling window does"
        )
    if problems:
        raise AssertionError("look-ahead detected: " + "; ".join(problems))

    if compared == 0:
        logger.warning("Look-ahead check compared no rows; try more bars or fewer warm-up windows")
    else:
        logger.info("Look-ahead check passed at %d cut points", compared)
