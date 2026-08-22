#!/usr/bin/env python3
"""Calibrate the cross-sectional permutation test on skill-free universes.

A significance test earns trust by producing uniform p-values where there is
nothing to find. This script runs the full pipeline — CUSUM events, triple
barrier, features, purged walk-forward, event PnL — on universes generated
from pure geometric Brownian motion, then applies the permutation test.

Run it whenever the test or the pipeline changes:

    python scripts/calibrate_significance.py --universes 6

History note: this calibration is what exposed the original iid-shuffle null
as anti-conservative (p = 0.001 on a random walk; overlapping events are not
exchangeable) and validated the circular-shift replacement. It also showed
that the ``--offline-demo`` tick generator carries a small learnable artifact
of its own — consistently positive observed statistics across seeds — which is
why notebook 05's synthetic-data verdict cites this script rather than
treating its own snapshot as a perfect null.
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("afml_india").setLevel(logging.ERROR)

from afml_india.data.costs import CostModel, Segment  # noqa: E402
from afml_india.features import FeatureConfig  # noqa: E402
from afml_india.models import WalkForwardSplit, random_forest  # noqa: E402
from afml_india.models.evaluate import walk_forward_evaluate  # noqa: E402
from afml_india.models.sequences import make_sequences, sequence_span  # noqa: E402
from afml_india.research import sample_events, triple_barrier_labels  # noqa: E402
from afml_india.significance import event_returns, permutation_pvalue  # noqa: E402


def gbm_bars(
    rng: np.random.Generator, n_bars: int = 3200, sub: int = 20, sigma_bar: float = 0.008
) -> pd.DataFrame:
    """Bars from a pure log-martingale: iid Gaussian steps, nothing else."""
    steps = rng.normal(0, sigma_bar / np.sqrt(sub), (n_bars, sub))
    bar_ends = np.cumsum(steps.sum(axis=1))
    log_path = (
        np.concatenate(([0.0], bar_ends[:-1]))[:, None] + np.cumsum(steps, axis=1) + np.log(1000.0)
    )
    px = np.exp(log_path)
    index = pd.date_range("2026-01-01 09:15", periods=n_bars, freq="5min")
    return pd.DataFrame(
        {
            "open": px[:, 0],
            "high": px.max(axis=1),
            "low": px.min(axis=1),
            "close": px[:, -1],
            "volume": rng.lognormal(10, 0.5, n_bars),
            "cum_buy_volume": rng.lognormal(9.3, 0.5, n_bars),
            "cum_ticks": rng.integers(50, 200, n_bars).astype(float),
        },
        index=index,
    )


def score_ticker(
    bars: pd.DataFrame, splitter: WalkForwardSplit, cost_model: CostModel
) -> pd.DataFrame:
    close = bars["close"]
    events = sample_events(close, vol_multiple=0.5, vol_lookback=50)
    labels = triple_barrier_labels(close, events=events, pt_sl=(1, 1), num_days=2, vol_lookback=50)
    labels = labels[labels["bin"] != 0]
    features = FeatureConfig(d=0.1).build(bars)
    y = labels["bin"].reindex(features.index).dropna()
    X, y_arr, idx = make_sequences(features, y, lookback=16)
    spans = sequence_span(idx, pd.DatetimeIndex(bars.index), 16, label_end=labels["t1"])
    _, predictions = walk_forward_evaluate(
        lambda: random_forest(seed=0), X, y_arr, spans, splitter=splitter
    )
    return event_returns(predictions, labels, cost_model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", type=int, default=6)
    parser.add_argument("--tickers", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--method", default="rotation", choices=["rotation", "shuffle"])
    args = parser.parse_args()

    splitter = WalkForwardSplit(n_splits=4, embargo=0.02, expanding=True, min_train=200)
    cost_model = CostModel(segment=Segment.INTRADAY)

    p_values = []
    for universe in range(args.universes):
        rng = np.random.default_rng(1000 + universe)
        events = {
            f"T{k}": score_ticker(gbm_bars(rng), splitter, cost_model) for k in range(args.tickers)
        }
        result = permutation_pvalue(
            events, n_permutations=args.permutations, method=args.method, seed=universe
        )
        p_values.append(result["p_value"])
        print(
            f"universe {universe}: observed={result['observed']:+.6f}  "
            f"p={result['p_value']:.3f}  ({args.method} null)"
        )

    p_values = np.asarray(p_values)
    print(f"\n{args.universes} skill-free universes, {args.method} null:")
    print(f"  p-values : {np.round(p_values, 3).tolist()}")
    print(
        f"  below .05: {(p_values < 0.05).sum()} "
        f"(expected ~{args.universes * 0.05:.1f} if calibrated)"
    )
    if (p_values < 0.05).mean() > 0.2:
        print("  WARNING: the test rejects far too often on pure noise — do not")
        print("  trust its p-values on real data until this is fixed.")
        return 1
    print("  Calibration looks acceptable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
