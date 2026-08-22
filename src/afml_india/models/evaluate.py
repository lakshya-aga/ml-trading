"""Purged walk-forward evaluation for sequence models.

``PurgedKFold`` handles the label-overlap purge, but it does not know about an
LSTM's backward-looking input window, and k-fold with shuffled fold order is a
poor fit for a strategy that will be deployed forward in time. This module does
walk-forward splits with an explicit purge and embargo applied to the *full*
sample span — input window and label horizon together.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from afml_india.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardSplit:
    """Expanding- or rolling-window splits with purge and embargo.

    Parameters
    ----------
    n_splits:
        Number of test folds, laid out consecutively through time.
    embargo:
        Fraction of the sample embargoed *after* each test fold. AFML's point:
        purging alone is not enough, because serial correlation makes the bars
        immediately following a test fold informative about it.
    expanding:
        ``True`` grows the training window from the start of the sample;
        ``False`` keeps it a fixed length ending at the purge boundary.
    """

    n_splits: int = 5
    embargo: float = 0.01
    expanding: bool = True
    min_train: int = 100

    def split(self, spans: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` positional arrays.

        ``spans`` must carry ``start`` and ``end`` columns describing the full
        interval each sample occupies — see
        :func:`afml_india.models.sequences.sequence_span`.
        """
        for col in ("start", "end"):
            if col not in spans.columns:
                raise KeyError(f"spans is missing required column {col!r}")

        n = len(spans)
        if n < self.n_splits * 2:
            raise ValueError(f"need at least {self.n_splits * 2} samples, got {n}")

        starts = spans["start"].to_numpy()
        ends = spans["end"].to_numpy()
        embargo_n = int(n * self.embargo)

        fold_edges = np.linspace(self.min_train, n, self.n_splits + 1).astype(int)
        for i in range(self.n_splits):
            test_start, test_stop = fold_edges[i], fold_edges[i + 1]
            if test_stop - test_start < 1:
                continue
            test_idx = np.arange(test_start, test_stop)

            test_window_start = starts[test_idx].min()
            test_window_end = ends[test_idx].max()

            candidates = np.arange(0, test_start)
            if not self.expanding:
                window = test_start - self.min_train
                candidates = candidates[candidates >= max(0, window)]

            # Purge: drop any training sample whose span overlaps the test span
            # in either direction. A sample whose label resolves inside the test
            # window leaks forward; one whose input window reaches into it leaks
            # backward.
            keep = (ends[candidates] < test_window_start) | (starts[candidates] > test_window_end)
            train_idx = candidates[keep]

            # Embargo: also drop training samples immediately preceding the test
            # fold, which purging by span alone can leave in.
            if embargo_n > 0:
                train_idx = train_idx[train_idx < max(0, test_start - embargo_n)]

            if len(train_idx) < self.min_train:
                logger.warning(
                    "Fold %d has only %d training samples after purge; skipping",
                    i, len(train_idx),
                )
                continue
            yield train_idx, test_idx


def classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
    positive_class=None,
) -> dict[str, float]:
    """Standard binary metrics, weighted when weights are supplied.

    ROC-AUC is reported but should not be the decision criterion on its own:
    with a near-balanced triple-barrier label and heavy overlap it moves very
    little, and precision at the operating threshold is what a bet-sizing layer
    actually consumes.
    """
    classes = np.unique(y_true)
    positive = positive_class if positive_class is not None else classes.max()
    y_binary = (y_true == positive).astype(int)
    predicted = (proba >= 0.5).astype(int)

    out: dict[str, float] = {}
    kw = {"sample_weight": sample_weight}
    out["accuracy"] = float(accuracy_score(y_binary, predicted, **kw))
    out["precision"] = float(precision_score(y_binary, predicted, zero_division=0, **kw))
    out["recall"] = float(recall_score(y_binary, predicted, zero_division=0, **kw))
    out["f1"] = float(f1_score(y_binary, predicted, zero_division=0, **kw))
    if len(np.unique(y_binary)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_binary, proba, **kw))
        out["avg_precision"] = float(average_precision_score(y_binary, proba, **kw))
        out["log_loss"] = float(log_loss(y_binary, np.clip(proba, 1e-6, 1 - 1e-6), **kw))
    else:
        out["roc_auc"] = float("nan")
        out["avg_precision"] = float("nan")
        out["log_loss"] = float("nan")
    out["positive_rate"] = float(predicted.mean())
    out["base_rate"] = float(y_binary.mean())
    return out


def walk_forward_evaluate(
    make_model: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    spans: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    splitter: WalkForwardSplit | None = None,
    scaler_factory: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score a model across purged walk-forward folds.

    ``make_model`` is called fresh per fold so no state carries across, and the
    scaler is refit on each training fold — the whole point of passing a factory
    rather than a fitted object.

    Returns
    -------
    fold_metrics:
        One row per fold.
    predictions:
        Out-of-sample probability per sample, with its fold and true label.
    """
    from afml_india.models.sequences import SequenceScaler  # noqa: PLC0415

    splitter = splitter or WalkForwardSplit()
    scaler_factory = scaler_factory or SequenceScaler

    rows, prediction_frames = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(spans)):
        scaler = scaler_factory()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        weights = None if sample_weight is None else sample_weight[train_idx]
        model = make_model()
        model.fit(X_train, y[train_idx], sample_weight=weights)

        proba = model.predict_proba(X_test)[:, 1]
        test_weights = None if sample_weight is None else sample_weight[test_idx]
        metrics = classification_metrics(y[test_idx], proba, test_weights)
        metrics.update(
            {"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx),
             "test_start": spans.index[test_idx[0]], "test_end": spans.index[test_idx[-1]]}
        )
        rows.append(metrics)
        prediction_frames.append(
            pd.DataFrame(
                {"fold": fold, "proba": proba, "y_true": y[test_idx]},
                index=spans.index[test_idx],
            )
        )
        logger.info(
            "Fold %d: train=%d test=%d  acc=%.3f  auc=%.3f",
            fold, len(train_idx), len(test_idx), metrics["accuracy"], metrics["roc_auc"],
        )

    if not rows:
        raise RuntimeError("no fold produced enough training data; loosen the splitter")

    fold_metrics = pd.DataFrame(rows).set_index("fold")
    predictions = pd.concat(prediction_frames).sort_index()
    return fold_metrics, predictions


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.Series:
    """Mean and dispersion across folds.

    Report the dispersion, not just the mean. A model averaging 0.55 accuracy
    with a fold range of 0.44 to 0.66 has not learned anything stable, and the
    mean alone hides that completely.
    """
    numeric = fold_metrics.select_dtypes(include=[np.number])
    summary = {}
    for col in numeric.columns:
        summary[f"{col}_mean"] = float(numeric[col].mean())
        summary[f"{col}_std"] = float(numeric[col].std())
    summary["n_folds"] = float(len(fold_metrics))
    return pd.Series(summary)
