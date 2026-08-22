"""Models and purged evaluation for bar-level financial sequences."""

from afml_india.models.baselines import FlattenAdapter, logistic, random_forest
from afml_india.models.evaluate import (
    WalkForwardSplit,
    classification_metrics,
    summarise_folds,
    walk_forward_evaluate,
)
from afml_india.models.sequences import SequenceScaler, make_sequences, sequence_span

__all__ = [
    "make_sequences",
    "sequence_span",
    "SequenceScaler",
    "WalkForwardSplit",
    "walk_forward_evaluate",
    "classification_metrics",
    "summarise_folds",
    "random_forest",
    "logistic",
    "FlattenAdapter",
    "LSTMClassifier",
    "LSTMConfig",
]


def __getattr__(name: str):
    """Import the torch-backed model lazily so sklearn-only users need no torch."""
    if name in ("LSTMClassifier", "LSTMConfig"):
        from afml_india.models import lstm  # noqa: PLC0415

        return getattr(lstm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
