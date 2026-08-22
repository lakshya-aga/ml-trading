"""Timezone shim for calling fin-kit from tz-aware data.

Several ``mlfinlab`` functions round-trip a ``DatetimeIndex`` through
``.values`` or rebuild it with ``pd.DatetimeIndex(...)``. Both drop the
timezone — ``get_daily_vol`` converts to UTC and then fails to align against its
own tz-aware input, and ``cusum_filter`` silently returns UTC-naive timestamps
that no longer match the bars they came from.

Rather than scatter ``tz_localize(None)`` through the research code — or give up
timezone-aware data, which is the wrong trade for an IST market — the conversion
is contained here: strip on the way in, restore on the way out.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pandas as pd


def index_timezone(obj: Any) -> Any:
    """Timezone of ``obj``'s datetime index, or ``None``."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        index = obj.index
    elif isinstance(obj, pd.DatetimeIndex):
        index = obj
    else:
        return None
    return getattr(index, "tz", None)


def strip_tz(obj: Any) -> Any:
    """Return ``obj`` with any timezone removed from its datetime index or values."""
    if isinstance(obj, pd.DatetimeIndex):
        return obj.tz_localize(None) if obj.tz is not None else obj
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj
        if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
            out = out.copy()
            out.index = out.index.tz_localize(None)
        # A Series of timestamps (a t1 barrier column) needs the same treatment.
        if isinstance(out, pd.Series) and pd.api.types.is_datetime64tz_dtype(out.dtype):
            out = out.dt.tz_localize(None)
        elif isinstance(out, pd.DataFrame):
            tz_cols = [c for c in out.columns if pd.api.types.is_datetime64tz_dtype(out[c].dtype)]
            if tz_cols:
                out = out.copy()
                for col in tz_cols:
                    out[col] = out[col].dt.tz_localize(None)
        return out
    return obj


def restore_tz(obj: Any, tz: Any) -> Any:
    """Re-apply ``tz`` to ``obj``'s datetime index and datetime values."""
    if tz is None:
        return obj
    if isinstance(obj, pd.DatetimeIndex):
        return obj.tz_localize(tz) if obj.tz is None else obj.tz_convert(tz)
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.copy()
        if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is None:
            out.index = out.index.tz_localize(tz)
        if isinstance(out, pd.Series) and pd.api.types.is_datetime64_any_dtype(out.dtype):
            if getattr(out.dtype, "tz", None) is None:
                out = out.dt.tz_localize(tz)
        elif isinstance(out, pd.DataFrame):
            for col in out.columns:
                dtype = out[col].dtype
                if pd.api.types.is_datetime64_any_dtype(dtype) and getattr(dtype, "tz", None) is None:
                    out[col] = out[col].dt.tz_localize(tz)
        return out
    return obj


def tz_safe(func: Callable) -> Callable:
    """Wrap a fin-kit function so tz-aware input goes in and comes back out.

    The timezone is taken from the first argument that carries one, applied to
    every pandas argument on the way in, and restored on the result. Functions
    that return a tuple or dict have each element restored.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tz = None
        for value in (*args, *kwargs.values()):
            tz = index_timezone(value)
            if tz is not None:
                break
        if tz is None:
            return func(*args, **kwargs)

        stripped_args = [strip_tz(a) for a in args]
        stripped_kwargs = {k: strip_tz(v) for k, v in kwargs.items()}
        result = func(*stripped_args, **stripped_kwargs)

        if isinstance(result, tuple):
            return tuple(restore_tz(r, tz) for r in result)
        if isinstance(result, dict):
            return {k: restore_tz(v, tz) for k, v in result.items()}
        return restore_tz(result, tz)

    wrapper.__doc__ = (
        (func.__doc__ or "")
        + "\n\nWrapped by afml_india so tz-aware input is handled; see afml_india._tz."
    )
    return wrapper
