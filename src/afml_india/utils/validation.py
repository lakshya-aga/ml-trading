"""Input validation helpers used across the package.

Financial ML code fails in quiet, expensive ways when an index is unsorted or
timezone-inconsistent, so every public entry point routes its inputs through
these checks.
"""

from __future__ import annotations

import pandas as pd


def ensure_datetime_index(obj: pd.Series | pd.DataFrame, name: str = "input") -> None:
    """Raise if ``obj`` is not indexed by a :class:`~pandas.DatetimeIndex`."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be indexed by a DatetimeIndex, got {type(obj.index).__name__}")


def ensure_monotonic(obj: pd.Series | pd.DataFrame, name: str = "input") -> None:
    """Raise if ``obj`` is not sorted ascending by index.

    Look-ahead bugs in path-dependent code (barriers, bars, CUSUM) are silent
    when the index is out of order, so this is checked rather than sorted for
    the caller.
    """
    if not obj.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted ascending; call .sort_index() first")


def ensure_series(obj: pd.Series | pd.DataFrame, name: str = "input") -> pd.Series:
    """Coerce a single-column frame to a Series, else validate it already is one."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or single-column DataFrame")
        return obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(obj).__name__}")
    return obj
