"""End-to-end per-ticker research pipeline.

One ticker in, a scored model out. The pipeline is deliberately a single
configurable object rather than a notebook of loose cells, because the intended
workflow is: get it right on one name, then run the identical configuration
across the point-in-time universe and look at the *distribution* of results.
A strategy that works on one ticker and nowhere else is a strategy that was
fitted to one ticker.

Stages
------
1. Bars — activity-driven sampling from the trade tape (AFML ch. 2).
2. Stationarity — fractional differencing order chosen per ticker (ch. 5).
3. Events — CUSUM filter on log prices (ch. 2).
4. Labels — triple barrier with an NSE-session vertical leg (ch. 3).
5. Weights — average uniqueness, correcting for label overlap (ch. 4).
6. Sequences — backward windows for a recurrent model.
7. Evaluation — purged, embargoed walk-forward (ch. 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from afml_india.data.snapshot import Snapshot
from afml_india.features import FeatureConfig
from afml_india.models.evaluate import WalkForwardSplit, summarise_folds, walk_forward_evaluate
from afml_india.models.sequences import make_sequences, sequence_span
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Every knob for one ticker's run.

    The defaults are a reasonable starting point for a liquid large cap on
    ~20 sessions of tick data. Expect to raise ``bars_per_day`` and
    ``cusum_multiple`` together when you move to a longer history.
    """

    # --- bars -------------------------------------------------------- #
    bar_kind: str = "dollar"
    bars_per_day: int = 50
    bar_threshold: float | None = None

    # --- stationarity ------------------------------------------------ #
    #: ``None`` selects the minimum passing order per ticker; a float pins it.
    #: Note ``0.0`` is *not* "no differencing applied" — it is the identity
    #: filter, which feeds the raw non-stationary log level in as a feature.
    #: Set ``include_fracdiff=False`` to drop the block entirely.
    fracdiff_d: float | None = None
    include_fracdiff: bool = True
    fracdiff_max_order: float = 1.0
    fracdiff_step: float = 0.1
    fracdiff_thresh: float = 1e-4
    fracdiff_min_obs: int = 100

    # --- events ------------------------------------------------------ #
    cusum_multiple: float = 1.0
    vol_lookback: int = 50

    # --- labels ------------------------------------------------------ #
    pt_sl: tuple[float, float] = (1.0, 1.0)
    holding_days: int = 2
    min_ret: float = 0.0

    # --- sequences and evaluation ------------------------------------ #
    lookback: int = 32
    n_splits: int = 4
    embargo: float = 0.02
    min_train: int = 100
    expanding: bool = True
    use_sample_weights: bool = True

    features: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class PipelineResult:
    """Everything one ticker's run produced, for inspection or aggregation."""

    ticker: str
    bars: pd.DataFrame
    events: pd.DatetimeIndex
    labels: pd.DataFrame
    features: pd.DataFrame
    d: float
    X: np.ndarray
    y: np.ndarray
    spans: pd.DataFrame
    sample_weight: np.ndarray | None
    fold_metrics: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> pd.Series:
        """Flat summary row, suitable for concatenating across tickers."""
        base = pd.Series(
            {
                "ticker": self.ticker,
                "n_bars": len(self.bars),
                "n_events": len(self.events),
                "n_labels": len(self.labels),
                "n_samples": len(self.X),
                "n_features": self.X.shape[-1] if self.X.ndim == 3 else np.nan,
                "d": self.d,
                "label_balance": float(np.mean(self.y == np.max(self.y))),
            }
        )
        if self.fold_metrics is not None:
            base = pd.concat([base, summarise_folds(self.fold_metrics)])
        return base


class TickerPipeline:
    """Run the AFML pipeline end to end for a single ticker."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ #
    def prepare(
        self,
        ticker: str,
        ticks: pd.DataFrame | None = None,
        bars: pd.DataFrame | None = None,
        snapshot: Snapshot | None = None,
    ) -> PipelineResult:
        """Run stages 1-6, stopping short of fitting a model.

        Supply exactly one of ``ticks``, ``bars`` or ``snapshot``. Separating
        preparation from evaluation means you can build the dataset once and try
        several models against it without recomputing bars and labels.
        """
        from afml_india.bars.finkit_bars import build_bars  # noqa: PLC0415
        from afml_india.research import (  # noqa: PLC0415
            min_ffd_order,
            sample_events,
            triple_barrier_labels,
        )

        cfg = self.config
        diagnostics: dict[str, Any] = {}

        # --- 1. Bars -------------------------------------------------- #
        if bars is None:
            if ticks is None:
                if snapshot is None:
                    raise ValueError("supply one of ticks=, bars= or snapshot=")
                ticks = snapshot.ticks(ticker, kind="trades")
            bars = build_bars(
                ticks,
                kind=cfg.bar_kind,
                threshold=cfg.bar_threshold,
                bars_per_day=cfg.bars_per_day,
            )
        if len(bars) < cfg.lookback + cfg.min_train:
            raise ValueError(
                f"{ticker}: only {len(bars)} bars, need at least "
                f"{cfg.lookback + cfg.min_train} for lookback={cfg.lookback}. "
                "Lower bars_per_day or supply more sessions."
            )
        close = bars["close"].astype(float)
        diagnostics["bars"] = len(bars)

        # --- 2. Stationarity ------------------------------------------ #
        if not cfg.include_fracdiff:
            d = float("nan")
            diagnostics["fracdiff"] = {"skipped": True}
        elif cfg.fracdiff_d is None:
            selection = min_ffd_order(
                close,
                thresh=cfg.fracdiff_thresh,
                max_order=cfg.fracdiff_max_order,
                step=cfg.fracdiff_step,
                min_obs=cfg.fracdiff_min_obs,
            )
            d = selection["d"]
            diagnostics["fracdiff"] = {k: v for k, v in selection.items() if k != "scan"}
            diagnostics["fracdiff_scan"] = selection["scan"]
            logger.info(
                "%s: d=%.2f (ADF p=%.4f, memory retained %.1f%%)",
                ticker,
                d,
                selection["adf_pvalue"],
                100 * selection["corr"],
            )
        else:
            d = float(cfg.fracdiff_d)

        # --- 3. Events ------------------------------------------------ #
        events = sample_events(
            close, vol_multiple=cfg.cusum_multiple, vol_lookback=cfg.vol_lookback
        )
        if len(events) < 50:
            raise ValueError(
                f"{ticker}: only {len(events)} CUSUM events; lower cusum_multiple "
                "or build more bars."
            )
        diagnostics["events"] = len(events)

        # --- 4. Labels ------------------------------------------------ #
        labels = triple_barrier_labels(
            close,
            events=events,
            pt_sl=cfg.pt_sl,
            num_days=cfg.holding_days,
            min_ret=cfg.min_ret,
            vol_lookback=cfg.vol_lookback,
        )
        labels = labels[labels["bin"] != 0]
        if labels.empty:
            raise ValueError(f"{ticker}: triple-barrier labelling produced no usable labels")
        diagnostics["labels"] = len(labels)
        diagnostics["label_counts"] = labels["bin"].value_counts().to_dict()

        # --- 5. Sample weights ---------------------------------------- #
        sample_weight_series = None
        if cfg.use_sample_weights:
            sample_weight_series = self._uniqueness_weights(labels, close, ticker)

        # --- 6. Features and sequences -------------------------------- #
        feature_config = FeatureConfig(
            **{
                **cfg.features.__dict__,
                "d": d if cfg.include_fracdiff else None,
                "fracdiff_thresh": cfg.fracdiff_thresh,
            }
        )
        features = feature_config.build(bars)
        diagnostics["features"] = features.shape[1]

        y_series = labels["bin"].reindex(features.index).dropna()
        if len(y_series) < cfg.min_train:
            raise ValueError(
                f"{ticker}: only {len(y_series)} labelled bars survive the feature "
                f"warm-up, need {cfg.min_train}. Shorten the feature windows or "
                "build more bars."
            )

        X, y, index = make_sequences(features, y_series, lookback=cfg.lookback)
        spans = sequence_span(
            index, pd.DatetimeIndex(bars.index), cfg.lookback, label_end=labels["t1"]
        )

        weights = None
        if sample_weight_series is not None:
            weights = sample_weight_series.reindex(index).fillna(1.0).to_numpy(dtype=float)

        return PipelineResult(
            ticker=ticker,
            bars=bars,
            events=events,
            labels=labels,
            features=features,
            d=d,
            X=X,
            y=y,
            spans=spans,
            sample_weight=weights,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _uniqueness_weights(
        labels: pd.DataFrame, close: pd.Series, ticker: str
    ) -> pd.Series | None:
        """Average uniqueness per label, or ``None`` if it cannot be computed.

        Overlapping triple-barrier labels are not independent draws. Weighting
        by average uniqueness is AFML chapter 4's correction; without it a model
        trained on 50-bar horizons at 50 bars a day is effectively fitting a
        much smaller sample than its row count suggests.
        """
        from afml_india.research import (  # noqa: PLC0415
            get_av_uniqueness_from_triple_barrier,
        )

        events = labels[["t1"]].dropna()
        if events.empty:
            logger.warning(
                "%s: every label has an open barrier; skipping uniqueness weights", ticker
            )
            return None
        try:
            uniqueness = get_av_uniqueness_from_triple_barrier(
                events, close, num_threads=1, verbose=False
            )
        except Exception as exc:  # noqa: BLE001 - weighting is an enhancement, not a gate
            logger.warning("%s: uniqueness weighting failed (%s); using equal weights", ticker, exc)
            return None
        # fin-kit returns a bare Series here; upstream mlfinlab releases return a
        # frame with a 'tW' column. Accept either.
        if isinstance(uniqueness, pd.DataFrame):
            uniqueness = uniqueness["tW"] if "tW" in uniqueness.columns else uniqueness.iloc[:, 0]
        weights = uniqueness.reindex(labels.index)
        logger.info(
            "%s: mean average-uniqueness %.3f (1.0 would mean no overlap)",
            ticker,
            float(weights.mean()),
        )
        return weights

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        result: PipelineResult,
        make_model,
        splitter: WalkForwardSplit | None = None,
    ) -> PipelineResult:
        """Score ``make_model`` on a prepared result with purged walk-forward CV."""
        cfg = self.config
        splitter = splitter or WalkForwardSplit(
            n_splits=cfg.n_splits,
            embargo=cfg.embargo,
            expanding=cfg.expanding,
            min_train=cfg.min_train,
        )
        fold_metrics, predictions = walk_forward_evaluate(
            make_model,
            result.X,
            result.y,
            result.spans,
            sample_weight=result.sample_weight,
            splitter=splitter,
        )
        result.fold_metrics = fold_metrics
        result.predictions = predictions
        return result

    # ------------------------------------------------------------------ #
    def run(
        self,
        ticker: str,
        make_model,
        ticks: pd.DataFrame | None = None,
        bars: pd.DataFrame | None = None,
        snapshot: Snapshot | None = None,
        splitter: WalkForwardSplit | None = None,
    ) -> PipelineResult:
        """Prepare and evaluate in one call."""
        result = self.prepare(ticker, ticks=ticks, bars=bars, snapshot=snapshot)
        return self.evaluate(result, make_model, splitter=splitter)


def run_universe(
    snapshot: Snapshot,
    make_model,
    config: PipelineConfig | None = None,
    tickers: list[str] | None = None,
    raise_on_error: bool = False,
) -> tuple[pd.DataFrame, dict[str, PipelineResult]]:
    """Run the identical pipeline across a snapshot's tickers.

    This is the step that separates a real result from an artefact. Read the
    *cross-section* of fold scores, not the best ticker: with 50 names and
    4 folds each, a handful will look excellent by chance alone.

    A ticker that fails — too few bars, no events, degenerate labels — is logged
    and skipped rather than aborting the sweep, unless ``raise_on_error``.
    """
    pipeline = TickerPipeline(config)
    tickers = tickers or snapshot.tickers

    rows, results = [], {}
    for n, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] %s", n, len(tickers), ticker)
        try:
            result = pipeline.run(ticker, make_model, snapshot=snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", ticker, exc)
            if raise_on_error:
                raise
            rows.append(pd.Series({"ticker": ticker, "error": str(exc)}))
            continue
        results[ticker] = result
        rows.append(result.summary)

    table = pd.DataFrame(rows).set_index("ticker")
    return table, results
