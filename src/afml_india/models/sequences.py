"""Turn a bar-level feature matrix into sequences for a recurrent model.

An LSTM consumes a window of the recent past, which introduces a second overlap
on top of the one AFML already worries about. A triple-barrier label spans from
its event to its barrier touch, so labels overlap *forward*; a sequence spans
``lookback`` bars before its event, so samples overlap *backward*. Both have to
be purged out of a validation split or the reported score is inflated.
:func:`sequence_span` returns the full interval a sample occupies so the
splitter can do that correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afml_india.utils.logging import get_logger

logger = get_logger(__name__)


def make_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    lookback: int = 32,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build ``(n_samples, lookback, n_features)`` windows ending at each label.

    The window for a label at bar ``t`` covers bars ``t - lookback + 1 .. t``
    inclusive, so the last row of every sequence is the event bar itself and no
    future bar is ever included.

    Returns
    -------
    X:
        3D float32 array of sequences.
    y:
        1D array of labels aligned to ``X``.
    index:
        Event timestamps, one per sample.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")

    columns = feature_columns or list(features.columns)
    frame = features[columns].astype(np.float32)

    common = frame.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("features and labels share no timestamps")

    positions = {stamp: i for i, stamp in enumerate(frame.index)}
    values = frame.to_numpy(dtype=np.float32)

    keep_stamps, windows = [], []
    for stamp in common:
        end = positions[stamp]
        start = end - lookback + 1
        if start < 0:
            continue  # not enough history for a full window
        windows.append(values[start : end + 1])
        keep_stamps.append(stamp)

    if not windows:
        raise ValueError(
            f"no label has {lookback} bars of preceding history; "
            f"reduce lookback (features has {len(frame)} rows)"
        )

    X = np.stack(windows).astype(np.float32)
    index = pd.DatetimeIndex(keep_stamps, name=features.index.name)
    y = labels.loc[index].to_numpy()
    dropped = len(common) - len(index)
    if dropped:
        logger.info("Dropped %d labels lacking a full %d-bar window", dropped, lookback)
    logger.info("Sequences: %s (samples, lookback, features)", X.shape)
    return X, y, index


def sequence_span(
    index: pd.DatetimeIndex,
    bar_index: pd.DatetimeIndex,
    lookback: int,
    label_end: pd.Series | None = None,
) -> pd.DataFrame:
    """Full time interval each sample occupies, for purging.

    ``start`` is the first bar in the sample's input window and ``end`` is the
    later of the event bar and its label's barrier touch. A validation sample
    must not overlap any training sample's span in either direction.
    """
    positions = pd.Index(bar_index).get_indexer(index)
    if (positions < 0).any():
        raise ValueError("some sample timestamps are not present in bar_index")

    starts = bar_index[np.maximum(positions - lookback + 1, 0)]
    ends = pd.Series(index, index=index)
    if label_end is not None:
        aligned = label_end.reindex(index)
        ends = ends.where(aligned.isna(), aligned)
    return pd.DataFrame({"start": pd.DatetimeIndex(starts), "end": pd.DatetimeIndex(ends)}, index=index)


class SequenceScaler:
    """Standardise sequence features using training-fold statistics only.

    Fitting a scaler on the whole sample is one of the most common quiet leaks
    in financial deep learning: the mean and standard deviation carry
    information about the validation period into the training data. Statistics
    are computed here over the flattened ``(samples x timesteps)`` axis, which
    keeps each feature on one scale across the window.
    """

    def __init__(self, clip: float | None = 5.0) -> None:
        self.clip = clip
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> SequenceScaler:
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        # A constant feature would otherwise divide by zero and produce inf.
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceScaler must be fitted before transform")
        out = (X - self.mean_) / self.std_
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
