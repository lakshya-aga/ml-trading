"""Non-sequential baselines to measure the LSTM against.

An LSTM is only worth its complexity if it beats a bagged tree on the same
features, the same labels and the same purged splits. That comparison is the
first thing to run and the most commonly skipped.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FlattenAdapter:
    """Wrap a tabular sklearn estimator so it accepts 3D sequence input.

    Two framings are supported. ``last_only=True`` gives the estimator only the
    event bar, which is the honest tabular baseline. ``last_only=False``
    flattens the whole window, which lets a tree see the same information as the
    LSTM at the cost of ignoring its temporal ordering.
    """

    def __init__(self, estimator, last_only: bool = True) -> None:
        self.estimator = estimator
        self.last_only = last_only
        self.classes_: np.ndarray | None = None

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            return X
        return X[:, -1, :] if self.last_only else X.reshape(len(X), -1)

    def fit(self, X, y, sample_weight=None):
        flat = self._reshape(X)
        self.estimator.fit(flat, y, **({"sample_weight": sample_weight} if sample_weight is not None else {}))
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._reshape(X))

    def predict(self, X):
        return self.estimator.predict(self._reshape(X))


def random_forest(
    n_estimators: int = 300,
    max_depth: int | None = 6,
    seed: int = 0,
    last_only: bool = True,
) -> FlattenAdapter:
    """Bagged trees with AFML's recommended guards against overfitting overlaps.

    ``max_features=1`` and a minimum leaf fraction are López de Prado's
    prescription for financial data: with correlated features and overlapping
    samples, a deep unconstrained forest memorises the sample.
    """
    return FlattenAdapter(
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=1,
            min_weight_fraction_leaf=0.05,
            class_weight="balanced_subsample",
            criterion="entropy",
            oob_score=False,
            random_state=seed,
            n_jobs=-1,
        ),
        last_only=last_only,
    )


def logistic(seed: int = 0, last_only: bool = True, C: float = 0.1) -> FlattenAdapter:
    """Regularised logistic regression — the floor any model must clear."""
    return FlattenAdapter(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        ),
        last_only=last_only,
    )


class _PipelineWithWeights(Pipeline):
    """Pipeline that forwards ``sample_weight`` to its final estimator."""

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        if sample_weight is not None:
            kwargs[f"{self.steps[-1][0]}__sample_weight"] = sample_weight
        return super().fit(X, y, **kwargs)
